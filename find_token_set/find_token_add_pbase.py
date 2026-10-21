import argparse
import json
import os

# =========================================================
# Helpers
# =========================================================


def load_json_dataset(path):
    """
    Load either:
      1) local JSON file: /path/to/data.json
      2) Hugging Face dataset: LLM-LAT/benign-dataset
      3) Hugging Face dataset with split: LLM-LAT/benign-dataset:train
      4) Hugging Face dataset with config and split: dataset_name:config:train

    Return:
      A list of dict samples compatible with build_prompt_messages().
    """

    # ---------------------------------------------------------
    # Case 1: local JSON file
    # ---------------------------------------------------------
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Dataset at {path} must be a JSON list.")

        return data

    # ---------------------------------------------------------
    # Case 2: Hugging Face dataset
    # ---------------------------------------------------------
    print(f"[Data] '{path}' not found as local file. Trying Hugging Face dataset...")

    parts = path.split(":")

    if len(parts) == 1:
        dataset_name = parts[0]
        config_name = None
        split_name = "train"
    elif len(parts) == 2:
        dataset_name = parts[0]
        config_name = None
        split_name = parts[1]
    elif len(parts) == 3:
        dataset_name = parts[0]
        config_name = parts[1]
        split_name = parts[2]
    else:
        raise ValueError(
            f"Unsupported dataset path format: {path}\n"
            f"Use one of:\n"
            f"  /path/to/data.json\n"
            f"  dataset_name\n"
            f"  dataset_name:split\n"
            f"  dataset_name:config:split"
        )

    from datasets import load_dataset

    if config_name is None:
        ds = load_dataset(dataset_name, split=split_name)
    else:
        ds = load_dataset(dataset_name, config_name, split=split_name)

    print(f"[Data] Loaded HF dataset: {dataset_name}")
    print(f"[Data] Split: {split_name}")
    print(f"[Data] Number of samples: {len(ds)}")
    print(f"[Data] Columns: {ds.column_names}")

    data = []

    for i, ex in enumerate(ds):
        ex = dict(ex)

        # Keep original fields
        item = dict(ex)
        item["id"] = i
        item["source"] = dataset_name

        # Normalize common fields into "instruction",
        # so your existing build_prompt_messages() can read it.
        text = None

        # Common prompt-like fields across instruction datasets
        candidate_fields = [
            "instruction",
            "prompt",
            "question",
            "query",
            "text",
            "input",
            "goal",
            "Behavior",
            "vanilla prompt",
            "refined prompt",
        ]

        for field in candidate_fields:
            if field in ex and ex[field] is not None and str(ex[field]).strip():
                text = str(ex[field]).strip()
                break

        # Some datasets may store conversations/messages
        if text is None:
            if "messages" in ex and isinstance(ex["messages"], list):
                item["messages"] = ex["messages"]
                data.append(item)
                continue

            if "prompt_messages" in ex and isinstance(ex["prompt_messages"], list):
                item["prompt_messages"] = ex["prompt_messages"]
                data.append(item)
                continue

        if text is None:
            # Skip examples without usable prompt text
            continue

        item["instruction"] = text
        data.append(item)

    if len(data) == 0:
        raise ValueError(
            f"No usable prompt-like fields found in HF dataset '{path}'. "
            f"Available columns: {ds.column_names}"
        )

    print(f"[Data] Converted {len(data)} HF samples into internal list format.")

    # Show one example for sanity check
    print("[Data] Example:")
    print(json.dumps(data[0], ensure_ascii=False, indent=2)[:1000])

    return data


def build_prompt_messages(sample):
    """
    Compatible with several common JSON formats:
      1) {"prompt_messages": [...]}
      2) {"messages": [...]}
      3) {"instruction": "..."} / {"input": "..."} / {"prompt": "..."}
    """
    if "prompt_messages" in sample and isinstance(sample["prompt_messages"], list):
        return sample["prompt_messages"]

    if "messages" in sample and isinstance(sample["messages"], list):
        msgs = []
        for m in sample["messages"]:
            role = m.get("role", "")
            if role in ["system", "user"]:
                msgs.append({"role": role, "content": m.get("content", "")})
        if len(msgs) > 0:
            return msgs

    if "instruction" in sample:
        text = sample.get("instruction", "")
    elif "prompt" in sample:
        text = sample.get("prompt", "")
    elif "input" in sample:
        text = sample.get("input", "")
    else:
        return None

    return [{"role": "user", "content": str(text)}]


def ensure_tokenizer_padding(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def get_mask_token_id(tokenizer, mask_token="<|mdm_mask|>"):
    """
    Try tokenizer.mask_token_id first. If absent, try converting a provided mask token.
    """
    if getattr(tokenizer, "mask_token_id", None) is not None:
        return tokenizer.mask_token_id

    tok_id = tokenizer.convert_tokens_to_ids(mask_token)
    if tok_id is None or tok_id < 0 or tok_id == tokenizer.unk_token_id:
        raise ValueError(
            "Cannot determine mask token id. "
            "Please set tokenizer.mask_token_id or provide a valid mask_token."
        )
    return tok_id


def build_prompt_ids(tokenizer, prompt_messages, device):
    """
    Use chat template if available; otherwise fallback to a simple User/Assistant format.
    """
    try:
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        return prompt_ids
    except Exception:
        text = ""
        for m in prompt_messages:
            if m.get("role") == "system":
                continue
            if m.get("role") == "user":
                text += f"User: {m.get('content', '')}\n\n"
        text += "Assistant: "
        prompt_ids = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(device)
        return prompt_ids


def ids_to_tokens(tokenizer, ids):
    toks = []
    tok_len = len(tokenizer)
    for i in ids:
        i = int(i)
        if 0 <= i < tok_len:
            toks.append(tokenizer.convert_ids_to_tokens(i))
        else:
            toks.append(f"<OUT_OF_TOKENIZER_VOCAB:{i}>")
    return toks


# =========================================================
# DLLM iterative update
# =========================================================


def step_update_confidence_based(
    input_ids,
    logits,
    answer_start,
    gen_len,
    step_idx,
    num_steps,
    mask_token_id,
    decode_k_per_step=4,
):
    """
    A generic MDLM/DLLM-style update:
      - among answer region positions that are still [MASK]
      - select top decode_k_per_step by confidence
      - fill them with argmax token

    This is only a generic fallback.
    If your DLLM repo has an official denoising/sampling function,
    replace this function with the repo-native step update.
    """
    if decode_k_per_step <= 0:
        raise ValueError("decode_k_per_step must be positive.")
    seq = input_ids.clone()
    answer_ids = seq[:, answer_start : answer_start + gen_len]
    answer_logits = logits[:, answer_start : answer_start + gen_len, :]

    masked_pos = (answer_ids[0] == mask_token_id).nonzero(as_tuple=False).squeeze(-1)
    if masked_pos.numel() == 0:
        return seq

    probs = F.softmax(answer_logits[0, masked_pos, :], dim=-1)
    conf, pred_ids = probs.max(dim=-1)

    k = min(decode_k_per_step, masked_pos.numel())
    chosen_local = torch.topk(conf, k=k, dim=0).indices
    chosen_pos = masked_pos[chosen_local]
    chosen_ids = pred_ids[chosen_local]

    seq[0, answer_start + chosen_pos] = chosen_ids
    return seq


# =========================================================
# Core DLLM direction builder
# =========================================================


def _build_logit_direction_dllm(
    model_ref,
    model_base,
    tokenizer_ref,
    tokenizer_base,
    dataset,
    device,
    K=256,
    gen_len=64,
    num_steps=16,
    obs_steps=8,
    obs_positions=8,
    temperature=1.0,
    diff_mode="p_diff",
    topk=200,
    eps=1e-6,
    decode_k_per_step=4,
    require_same_vocab=True,
    compute_dtype=None,
    mask_token="<|mdm_mask|>",
):
    """
    DLLM/MDLM version:
      - No teacher-forced response_text
      - Start from prompt + fully masked answer region
      - Run iterative denoising
      - Aggregate early-step, early-position distribution differences
    """
    assert diff_mode in ["p_diff", "pref_logratio"], f"Unknown diff_mode={diff_mode}"

    compute_dtype = compute_dtype or torch.bfloat16

    model_ref.eval().to(device)
    model_base.eval().to(device)

    V_ref = getattr(model_ref.config, "vocab_size", None)
    V_base = getattr(model_base.config, "vocab_size", None)

    assert V_ref == V_base, f"Model vocab mismatch: ref={V_ref}, base={V_base}"
    assert len(tokenizer_ref) == len(tokenizer_base), (
        f"Tokenizer mismatch: ref_tok={len(tokenizer_ref)}, base_tok={len(tokenizer_base)}"
    )

    mask_id_ref = get_mask_token_id(tokenizer_ref, mask_token=mask_token)
    mask_id_base = get_mask_token_id(tokenizer_base, mask_token=mask_token)

    if require_same_vocab and mask_id_ref != mask_id_base:
        raise ValueError(f"Mask token mismatch: ref={mask_id_ref}, base={mask_id_base}")

    diff_global = torch.zeros(V_ref, device=device, dtype=torch.float32)
    p_ref_global = torch.zeros(V_ref, device=device, dtype=torch.float32)
    p_base_global = torch.zeros(V_ref, device=device, dtype=torch.float32)
    cnt = 0

    num_samples = min(K, len(dataset))

    for i in range(num_samples):
        if i % 50 == 0:
            print(f"Processing sample {i}/{num_samples}")

        sample = dataset[i]
        prompt_messages = build_prompt_messages(sample)
        if not prompt_messages:
            continue

        # -----------------------------------------------------
        # 1) Build prompt ids
        # -----------------------------------------------------
        prompt_ids_ref = build_prompt_ids(tokenizer_ref, prompt_messages, device)
        prompt_ids_base = build_prompt_ids(tokenizer_base, prompt_messages, device)

        P_ref = prompt_ids_ref.size(1)
        P_base = prompt_ids_base.size(1)

        # -----------------------------------------------------
        # 2) Initialize fully masked answer region
        # -----------------------------------------------------
        masked_ref = torch.full((1, gen_len), mask_id_ref, dtype=torch.long, device=device)
        masked_base = torch.full((1, gen_len), mask_id_base, dtype=torch.long, device=device)

        input_ids_ref = torch.cat([prompt_ids_ref, masked_ref], dim=1)
        input_ids_base = torch.cat([prompt_ids_base, masked_base], dim=1)

        # -----------------------------------------------------
        # 3) Iterative denoising
        # -----------------------------------------------------
        diff_sum = torch.zeros(V_ref, device=device, dtype=torch.float32)
        p_ref_sum = torch.zeros(V_ref, device=device, dtype=torch.float32)
        p_base_sum = torch.zeros(V_ref, device=device, dtype=torch.float32)
        local_count = 0

        cur_ref = input_ids_ref
        cur_base = input_ids_base

        for step in range(num_steps):
            attn_ref = torch.ones_like(cur_ref)
            attn_base = torch.ones_like(cur_base)

            out_ref = model_ref(input_ids=cur_ref, attention_mask=attn_ref)
            out_base = model_base(input_ids=cur_base, attention_mask=attn_base)

            logits_ref_full = (out_ref.logits / temperature).float()
            logits_base_full = (out_base.logits / temperature).float()

            # Focus only on early answer positions
            max_pos = min(obs_positions, gen_len)

            # IMPORTANT:
            # for DLLM, compare answer-region distributions at current denoising step
            for pos in range(max_pos):
                lr = logits_ref_full[:, P_ref + pos, :]
                lb = logits_base_full[:, P_base + pos, :]

                p_ref = F.softmax(lr, dim=-1)
                p_base = F.softmax(lb, dim=-1)

                # aligned-base
                if diff_mode == "p_diff":
                    diff_step = p_ref - p_base
                else:
                    logp_ref = F.log_softmax(lr, dim=-1)
                    logp_base = F.log_softmax(lb, dim=-1)
                    diff_step = p_ref * (logp_base - logp_ref)

                if step < obs_steps:
                    diff_sum += diff_step.squeeze(0)
                    p_ref_sum += p_ref.squeeze(0)
                    p_base_sum += p_base.squeeze(0)
                    local_count += 1

            # -------------------------------------------------
            # 4) Update both sequences by iterative denoising
            # -------------------------------------------------
            cur_ref = step_update_confidence_based(
                input_ids=cur_ref,
                logits=logits_ref_full,
                answer_start=P_ref,
                gen_len=gen_len,
                step_idx=step,
                num_steps=num_steps,
                mask_token_id=mask_id_ref,
                decode_k_per_step=decode_k_per_step,
            )

            cur_base = step_update_confidence_based(
                input_ids=cur_base,
                logits=logits_base_full,
                answer_start=P_base,
                gen_len=gen_len,
                step_idx=step,
                num_steps=num_steps,
                mask_token_id=mask_id_base,
                decode_k_per_step=decode_k_per_step,
            )

            # Early stop if both are fully decoded
            ref_done = (cur_ref[0, P_ref : P_ref + gen_len] != mask_id_ref).all()
            base_done = (cur_base[0, P_base : P_base + gen_len] != mask_id_base).all()
            if ref_done and base_done:
                break

        if local_count == 0:
            continue

        diff_global += diff_sum / local_count
        p_ref_global += p_ref_sum / local_count
        p_base_global += p_base_sum / local_count
        cnt += 1

    if cnt == 0:
        raise ValueError("No valid samples to build DLLM direction.")

    # ---------------------------------------------------------
    # 5) Global averaging
    # ---------------------------------------------------------
    diff_global = diff_global / cnt
    p_ref_global = p_ref_global / cnt
    p_base_global = p_base_global / cnt

    # ---------------------------------------------------------
    # 6) Top-k filtering by global p_ref
    # ---------------------------------------------------------
    if topk is not None and topk > 0 and topk < diff_global.size(0):
        topk_ids = torch.topk(p_ref_global, k=topk, dim=-1).indices
        mask = torch.zeros_like(diff_global)
        mask.scatter_(0, topk_ids, 1.0)
        diff_global = diff_global * mask

    # ---------------------------------------------------------
    # 7) Standardize nonzero entries
    # ---------------------------------------------------------
    nonzero_mask = diff_global.abs() > 1e-9

    if nonzero_mask.sum() > 1:
        nonzero_vals = diff_global[nonzero_mask]
        centered_vals = nonzero_vals - nonzero_vals.mean()
        standardized_vals = centered_vals / (centered_vals.std() + eps)
        diff_global[nonzero_mask] = standardized_vals

    # ---------------------------------------------------------
    # 8) L2 normalize
    # ---------------------------------------------------------
    v = diff_global / (diff_global.norm() + eps)
    v = v.to(compute_dtype)

    return v.detach(), p_ref_global.detach(), p_base_global.detach()


# =========================================================
# Main
# =========================================================


def compute_and_save_v_dir_dllm(
    # model paths
    base_model_path: str,
    aligned_model_path: str,
    dataset_path: str,
    # output
    output_path: str = "results/token_discovery/v_dir_dllm.pt",
    # compute params
    K: int = 500,
    gen_len: int = 64,
    num_steps: int = 16,
    obs_steps: int = 8,
    obs_positions: int = 8,
    temperature: float = 1.0,
    diff_mode: str = "p_diff",
    topk: int = 200,
    decode_k_per_step: int = 4,
    # tokenizer/model
    device: str = "cuda:0",
    use_bfloat16: bool = True,
    trust_remote_code: bool = True,
    mask_token: str = "<|mdm_mask|>",
):
    """
    Example:
    python find_token_set/find_token_add_pbase.py \
        --base_model_path /path/to/LLaDA-8B-Base \
        --aligned_model_path /path/to/LLaDA-8B-Instruct \
        --dataset_path /path/to/harmful_prompts.json \
        --output_path ./v_dir_dllm.pt \
        --K 500 \
        --gen_len 64 \
        --num_steps 16 \
        --obs_steps 8 \
        --obs_positions 8 \
        --topk 200 \
        --device cuda:0
    """
    print("=" * 60)
    print("DLLM v_dir Computation")
    print("=" * 60)
    print(f"Base model:       {base_model_path}")
    print(f"Aligned model:    {aligned_model_path}")
    print(f"Dataset:          {dataset_path}")
    print(f"Output:           {output_path}")
    print(f"K:                {K}")
    print(f"gen_len:          {gen_len}")
    print(f"num_steps:        {num_steps}")
    print(f"obs_steps:        {obs_steps}")
    print(f"obs_positions:    {obs_positions}")
    print(f"topk:             {topk}")
    print("=" * 60)

    global torch, F
    import torch
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    if min(K, gen_len, num_steps, obs_steps, obs_positions, decode_k_per_step) <= 0:
        raise ValueError("Sample, generation, observation, and decode counts must be positive.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    dtype = torch.bfloat16 if use_bfloat16 else torch.float32

    print("\nLoading tokenizers...")
    shared_tokenizer_path = aligned_model_path

    tokenizer_ref = AutoTokenizer.from_pretrained(
        shared_tokenizer_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    tokenizer_base = AutoTokenizer.from_pretrained(
        shared_tokenizer_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    ensure_tokenizer_padding(tokenizer_ref)
    ensure_tokenizer_padding(tokenizer_base)

    print("\nLoading dataset...")
    dataset = load_json_dataset(dataset_path)
    print(f"Dataset size: {len(dataset)}")

    print(f"\nLoading base model with dtype={dtype}...")
    model_base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )

    print(f"Loading aligned model with dtype={dtype}...")
    model_ref = AutoModelForCausalLM.from_pretrained(
        aligned_model_path,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )

    print("\nComputing DLLM v_dir...")
    v_dir, p_ref_global, p_base_global = _build_logit_direction_dllm(
        model_ref=model_ref,
        model_base=model_base,
        tokenizer_ref=tokenizer_ref,
        tokenizer_base=tokenizer_base,
        dataset=dataset,
        device=device,
        K=K,
        gen_len=gen_len,
        num_steps=num_steps,
        obs_steps=obs_steps,
        obs_positions=obs_positions,
        temperature=temperature,
        diff_mode=diff_mode,
        topk=topk,
        decode_k_per_step=decode_k_per_step,
        compute_dtype=dtype,
        mask_token=mask_token,
    )

    # ---------------------------------------------------------
    # token analysis
    # ---------------------------------------------------------
    v_float = v_dir.float().cpu()
    p_ref_cpu = p_ref_global.float().cpu()
    p_base_cpu = p_base_global.float().cpu()

    top_pos_ids = torch.topk(v_float, k=min(100, v_float.numel())).indices.tolist()
    top_neg_ids = torch.topk(-v_float, k=min(100, v_float.numel())).indices.tolist()
    top_pref_ids = torch.topk(p_ref_cpu, k=min(200, p_ref_cpu.numel())).indices.tolist()
    top_pbase_ids = torch.topk(p_base_cpu, k=min(200, p_base_cpu.numel())).indices.tolist()

    top_pos_tokens = ids_to_tokens(tokenizer_ref, top_pos_ids)
    top_neg_tokens = ids_to_tokens(tokenizer_ref, top_neg_ids)
    top_pref_tokens = ids_to_tokens(tokenizer_ref, top_pref_ids)
    top_pbase_tokens = ids_to_tokens(tokenizer_ref, top_pbase_ids)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    save_dict = {
        "v_dir": v_dir.to(save_dtype).cpu(),
        "p_ref_global": p_ref_global.to(torch.float32).cpu(),
        "p_base_global": p_base_global.to(torch.float32).cpu(),
        "top_positive_token_ids": top_pos_ids,
        "top_positive_tokens": top_pos_tokens,
        "top_negative_token_ids": top_neg_ids,
        "top_negative_tokens": top_neg_tokens,
        "top_pref_token_ids": top_pref_ids,
        "top_pref_tokens": top_pref_tokens,
        "top_pbase_token_ids": top_pbase_ids,
        "top_pbase_tokens": top_pbase_tokens,
        "metadata": {
            "base_model_path": base_model_path,
            "aligned_model_path": aligned_model_path,
            "dataset_path": dataset_path,
            "K": K,
            "gen_len": gen_len,
            "num_steps": num_steps,
            "obs_steps": obs_steps,
            "obs_positions": obs_positions,
            "temperature": temperature,
            "diff_mode": diff_mode,
            "topk": topk,
            "decode_k_per_step": decode_k_per_step,
            "vocab_size": v_dir.shape[0],
            "dtype": str(v_dir.dtype),
            "mask_token": mask_token,
        },
    }

    torch.save(save_dict, output_path)

    print("\n" + "=" * 60)
    print("DLLM v_dir computation completed!")
    print(f"Shape: {v_dir.shape}")
    print(f"Saved to: {output_path}")
    print("=" * 60)

    print("\nv_dir statistics:")
    print(f"  min:  {v_float.min().item():.6f}")
    print(f"  max:  {v_float.max().item():.6f}")
    print(f"  mean: {v_float.mean().item():.6f}")
    print(f"  std:  {v_float.std().item():.6f}")
    print(f"  norm: {v_float.norm().item():.6f}")

    print("\nTop positive tokens:")
    print(top_pos_tokens[:50])

    print("\nTop negative tokens:")
    print(top_neg_tokens[:50])

    print("\nTop p_ref tokens:")
    print(top_pref_tokens[:50])

    print("\nTop p_base tokens:")
    print(top_pbase_tokens[:50])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discover early refusal-associated tokens from aligned/base probability differences."
    )
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--aligned_model_path", required=True)
    parser.add_argument(
        "--dataset_path", required=True, help="Local JSON or HF dataset[:config]:split."
    )
    parser.add_argument("--output_path", default="results/token_discovery/v_dir_dllm.pt")
    parser.add_argument("--K", type=int, default=500)
    parser.add_argument("--gen_len", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=16)
    parser.add_argument("--obs_steps", type=int, default=8)
    parser.add_argument("--obs_positions", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--diff_mode", choices=["p_diff", "pref_logratio"], default="p_diff")
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--decode_k_per_step", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use_bfloat16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask_token", default="<|mdm_mask|>")
    compute_and_save_v_dir_dllm(**vars(parser.parse_args()))
