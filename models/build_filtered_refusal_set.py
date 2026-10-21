# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import re
import subprocess
from typing import Dict, Iterable, List, Optional

MASK_TOKEN = "<|mdm_mask|>"
START_TOKEN = "<startoftext>"
END_TOKEN = "<endoftext>"
SPECIAL_TOKEN_PATTERN = r"<mask:(\d+)>"

DEFAULT_MASK_ID = 126336
DEFAULT_STEPS = 64
DEFAULT_GEN_LENGTH = 128
DEFAULT_BLOCK_LENGTH = 128
DEFAULT_MASK_COUNTS = 36
DEFAULT_TEMPERATURE = 0.5


# =========================================================
# Candidate tokens
# =========================================================

DEFAULT_CANDIDATE_TOKENS = [
    "'t",
    "Ġbut",
    "Ġsorry",
    "Ġcan",
    "Ġwith",
    "Sorry",
    "Ġhelp",
    "Ġassist",
    "Ġcannot",
    "Ġthat",
    "Ġable",
    "Ġnot",
    "Ġinformation",
    "Ġprovide",
]


# =========================================================
# Args
# =========================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build data-driven filtered refusal token set using benign filtering."
    )

    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument(
        "--harmful_prompt",
        type=str,
        required=True,
        help="Local JSON or HF dataset[:config]:split; default split: test.",
    )
    parser.add_argument(
        "--harmful_prompt_field", type=str, default="auto", help="Field for harmful prompts."
    )

    parser.add_argument(
        "--benign_prompt",
        type=str,
        required=True,
        help="Local JSON or HF dataset[:config]:split; default split: test.",
    )
    parser.add_argument(
        "--benign_prompt_field",
        type=str,
        default="auto",
        help="Field for benign prompts. For GSM8K, use question.",
    )

    parser.add_argument(
        "--candidate_tokens_json",
        type=str,
        default=None,
        help="Optional JSON file containing candidate tokens. Can be list[str] or dict with key 'candidate_tokens'.",
    )

    parser.add_argument("--output_json", type=str, required=True)

    parser.add_argument("--num_harmful_samples", type=int, default=200)
    parser.add_argument("--num_benign_samples", type=int, default=200)

    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--gen_length", type=int, default=DEFAULT_GEN_LENGTH)
    parser.add_argument("--block_length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--mask_id", type=int, default=DEFAULT_MASK_ID)
    parser.add_argument("--mask_counts", type=int, default=DEFAULT_MASK_COUNTS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--cfg_scale", type=float, default=0.1)
    parser.add_argument(
        "--remasking",
        type=str,
        default="low_confidence",
        choices=["low_confidence", "random", "rate", "adaptive", "adaptive_step"],
    )
    parser.add_argument("--random_rate", type=float, default=0.0)
    parser.add_argument("--alpha0", type=float, default=0.3)

    parser.add_argument(
        "--obs_positions", type=int, default=16, help="Number of answer positions to observe."
    )
    parser.add_argument(
        "--obs_steps",
        type=int,
        default=16,
        help="Number of early denoising steps to observe. Use -1 for all steps.",
    )

    # Filtering thresholds
    parser.add_argument("--harmful_min_p95", type=float, default=1e-5)
    parser.add_argument("--benign_max_p95", type=float, default=0.005)
    parser.add_argument("--benign_max_p99", type=float, default=0.05)
    parser.add_argument("--min_ratio", type=float, default=2.0)
    parser.add_argument("--max_benign_max", type=float, default=0.5)

    parser.add_argument("--auto_pick_gpu", action="store_true")
    parser.add_argument("--no_auto_pick_gpu", dest="auto_pick_gpu", action="store_false")
    parser.set_defaults(auto_pick_gpu=True)

    return parser.parse_args()


# =========================================================
# GPU
# =========================================================


def _pick_gpu_by_torch():
    if not torch.cuda.is_available():
        return None
    try:
        best_i, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                free, total = torch.cuda.mem_get_info()
            if free > best_free:
                best_free, best_i = free, i
        return best_i
    except Exception:
        return None


def _pick_gpu_by_nvidia_smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        frees = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
        if not frees:
            return None
        return max(range(len(frees)), key=lambda i: frees[i])
    except Exception:
        return None


def choose_device(args):
    if torch.cuda.is_available():
        if args.auto_pick_gpu:
            idx = _pick_gpu_by_torch()
            if idx is None:
                idx = _pick_gpu_by_nvidia_smi()
            if idx is not None:
                try:
                    torch.cuda.set_device(idx)
                except Exception:
                    pass
                logging.info(f"[GPU] Using cuda:{idx}")
                return torch.device(f"cuda:{idx}")
        logging.info("[GPU] Using default cuda")
        return torch.device("cuda")

    logging.info("[GPU] Using CPU")
    return torch.device("cpu")


# =========================================================
# Data loading
# =========================================================


def load_json_or_hf(path_or_spec: str, prompt_field: str, kind: str):
    if os.path.exists(path_or_spec):
        with open(path_or_spec, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"{kind} local JSON must be a list.")

        out = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                text = pick_prompt_text(item, prompt_field)
                rec = dict(item)
            else:
                text = str(item)
                rec = {}

            rec.update(
                {
                    "id": i,
                    "prompt_text": str(text).strip(),
                    "source": path_or_spec,
                }
            )
            out.append(rec)

        logging.info(f"[{kind}] Loaded {len(out)} local samples from {path_or_spec}")
        return out

    parts = path_or_spec.split(":")
    dataset_name = parts[0]
    config_name = None
    split_name = "test"

    if len(parts) == 2:
        config_name = None
        split_name = parts[1]
    elif len(parts) >= 3:
        config_name = parts[1]
        split_name = parts[2]

    from datasets import load_dataset

    logging.info(f"[{kind}] Loading HF dataset: {path_or_spec}")

    if config_name is None:
        ds = load_dataset(dataset_name, split=split_name)
    else:
        ds = load_dataset(dataset_name, config_name, split=split_name)

    out = []
    for i, ex in enumerate(ds):
        ex_dict = dict(ex)
        text = pick_prompt_text(ex_dict, prompt_field)
        ex_dict.update(
            {
                "id": i,
                "prompt_text": str(text).strip(),
                "source": path_or_spec,
            }
        )
        out.append(ex_dict)

    logging.info(f"[{kind}] Loaded {len(out)} HF samples from {path_or_spec}")
    return out


def pick_prompt_text(item: dict, prompt_field: str = "auto") -> str:
    if prompt_field != "auto":
        if prompt_field not in item:
            raise KeyError(
                f"Requested prompt_field='{prompt_field}', available keys={list(item.keys())}"
            )
        return item[prompt_field]

    candidates = [
        "question",  # GSM8K
        "prompt",
        "instruction",
        "input",
        "query",
        "text",
        "problem",
        "vanilla prompt",
        "refined prompt",
        "Behavior",
        "Goal",
        "goal",
    ]

    for k in candidates:
        if k in item and item[k] is not None:
            return item[k]

    raise KeyError(f"Cannot find prompt field. Available keys={list(item.keys())}")


# =========================================================
# Prompt build
# =========================================================


def process_user_text(user_text: str, mask_counts: int) -> str:
    def repl(m):
        n = int(m.group(1))
        return MASK_TOKEN * max(n, 0)

    processed = re.sub(SPECIAL_TOKEN_PATTERN, repl, str(user_text))

    if (MASK_TOKEN not in processed) and mask_counts > 0:
        processed = processed + START_TOKEN + (MASK_TOKEN * mask_counts) + END_TOKEN

    return processed


def build_chat_prompt(tokenizer, user_text_processed: str, is_instruct: bool):
    if is_instruct:
        messages = [{"role": "user", "content": user_text_processed}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_text_processed


# =========================================================
# Token conversion
# =========================================================


def load_candidate_tokens(path: Optional[str]) -> List[str]:
    if path is None:
        return DEFAULT_CANDIDATE_TOKENS

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return [str(x) for x in obj]

    if isinstance(obj, dict):
        if "candidate_tokens" in obj:
            return [str(x) for x in obj["candidate_tokens"]]
        if "top_positive_tokens" in obj:
            return [str(x) for x in obj["top_positive_tokens"]]

    raise ValueError(
        "candidate_tokens_json must be a list[str] or dict with key candidate_tokens/top_positive_tokens."
    )


def token_strings_to_ids(tokenizer, token_strings):
    """
    Strict token mapping.

    Goal:
      Map each candidate to token ids only when it corresponds to a single tokenizer token.
      Avoid splitting phrases such as 'assist' -> ['ass', 'ist'] or 'dangerous' -> ['danger', 'ous'].

    For candidates like:
      'Ġsorry'  -> direct token lookup
      'sorry'  -> try 'sorry', ' sorry', 'Sorry', ' Sorry'
    """
    out = {}

    special_ids = {
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        tokenizer.bos_token_id,
        getattr(tokenizer, "unk_token_id", None),
    }
    special_ids = {x for x in special_ids if x is not None}

    for tok in token_strings:
        tok = str(tok)
        ids = set()

        # 1. Direct tokenizer token lookup, e.g. "Ġsorry"
        direct_id = tokenizer.convert_tokens_to_ids(tok)
        if direct_id is not None and direct_id >= 0 and direct_id not in special_ids:
            ids.add(int(direct_id))

        # 2. If it starts with Ġ, also try the plain version as one-token phrases.
        if tok.startswith("Ġ"):
            plain = tok[1:]
            variants = [
                plain,
                " " + plain,
                plain.capitalize(),
                " " + plain.capitalize(),
            ]
        else:
            variants = [
                tok,
                " " + tok,
                tok.capitalize(),
                " " + tok.capitalize(),
            ]

        # 3. Only accept variants that tokenize into exactly ONE token.
        for v in variants:
            try:
                enc = tokenizer(v, add_special_tokens=False).input_ids
            except Exception:
                continue

            if len(enc) == 1:
                tid = int(enc[0])
                if tid not in special_ids:
                    ids.add(tid)

        out[tok] = sorted(ids)

    return out


def ids_to_readable_tokens(tokenizer, ids: Iterable[int]) -> List[str]:
    return tokenizer.convert_ids_to_tokens([int(x) for x in ids])


# =========================================================
# DLLM trace collection
# =========================================================


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    steps = max(int(steps), 1)

    base = mask_num // steps
    remainder = (mask_num % steps).squeeze(1)

    num_transfer_tokens = base.expand(-1, steps).clone().to(torch.int64)

    if steps > 0:
        idx = torch.arange(steps, device=mask_index.device).unsqueeze(0)
        bump = (idx < remainder.unsqueeze(1)).to(torch.int64)
        num_transfer_tokens += bump

    return num_transfer_tokens


def collect_candidate_token_probs_for_dataset(
    model,
    tokenizer,
    data,
    args,
    token_to_ids,
    is_instruct,
    num_samples,
    kind,
):
    """
    Run DLLM denoising and collect per-candidate-token probability values.

    Returns:
      token_values:
        {token_str: [all prob values across samples/steps/positions]}

      token_max_events:
        {token_str: [high-prob events]}

      token_step_values:
        {
          token_str: {
            global_step: [prob values across samples/positions]
          }
        }
    """
    token_values = {tok: [] for tok in token_to_ids.keys()}
    token_max_events = {tok: [] for tok in token_to_ids.keys()}
    token_step_values = {tok: {} for tok in token_to_ids.keys()}

    n = min(len(data), int(num_samples))

    for idx in tqdm(range(n), desc=f"Collecting {kind} token probs"):
        prompt_text = data[idx]["prompt_text"]

        user_text = process_user_text(prompt_text, args.mask_counts)
        prompt_str = build_chat_prompt(tokenizer, user_text, is_instruct)

        inputs = tokenizer(prompt_str, return_tensors="pt")
        prompt_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        sample_values, sample_events, sample_step_values = collect_candidate_token_probs_single(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            attention_mask=attention_mask,
            args=args,
            token_to_ids=token_to_ids,
        )

        for tok, vals in sample_values.items():
            token_values[tok].extend(vals)

        for tok, events in sample_events.items():
            token_max_events[tok].extend(
                [
                    {
                        "sample_id": data[idx].get("id", idx),
                        "prompt_text": prompt_text[:300],
                        **e,
                    }
                    for e in events
                ]
            )

        for tok, step_dict in sample_step_values.items():
            for step, vals in step_dict.items():
                if step not in token_step_values[tok]:
                    token_step_values[tok][step] = []
                token_step_values[tok][step].extend(vals)

    return token_values, token_max_events, token_step_values


def collect_candidate_token_probs_single(
    model,
    tokenizer,
    prompt_ids,
    attention_mask,
    args,
    token_to_ids: Dict[str, List[int]],
):
    """
    Generic LLaDA-like denoising with low-confidence transfer.
    The token statistics are collected without RAEC intervention.
    """
    device = model.device

    x = prompt_ids.clone().to(device)

    tail = torch.full(
        (prompt_ids.shape[0], int(args.gen_length)),
        int(args.mask_id),
        dtype=torch.long,
        device=device,
    )
    x = torch.cat([x, tail], dim=1)

    if attention_mask is not None:
        am = attention_mask.to(device)
        if am.dtype != torch.bool:
            am = am != 0
        if am.shape[1] < x.shape[1]:
            pad_len = x.shape[1] - am.shape[1]
            pad = torch.ones(am.size(0), pad_len, dtype=torch.bool, device=device)
            am = torch.cat([am, pad], dim=1)
        else:
            am = am[:, : x.shape[1]]
    else:
        am = None

    prompt_len = prompt_ids.shape[1]
    answer_start = prompt_len
    answer_end = prompt_len + args.gen_length

    num_blocks = max(args.gen_length // args.block_length, 1)
    steps_per_block = max(int(args.steps) // int(num_blocks), 1)

    token_values = {tok: [] for tok in token_to_ids.keys()}
    token_events = {tok: [] for tok in token_to_ids.keys()}
    token_step_values = {tok: {} for tok in token_to_ids.keys()}

    prompt_index = x != args.mask_id
    global_step = 0

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * args.block_length
        block_end = prompt_len + (num_block + 1) * args.block_length

        block_mask_index = x[:, block_start:block_end] == args.mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step_in_block in range(steps_per_block):
            if args.obs_steps >= 0 and global_step >= args.obs_steps:
                return token_values, token_events, token_step_values

            mask_index = x == args.mask_id

            if args.cfg_scale > 0:
                un_x = x.clone()
                un_x[prompt_index] = args.mask_id
                x_ = torch.cat([x, un_x], dim=0)
                am_ = None if am is None else torch.cat([am, am], dim=0)

                out = model(x_, attention_mask=am_)
                logits = out.logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (args.cfg_scale + 1.0) * (logits - un_logits)
            else:
                out = model(x, attention_mask=am)
                logits = out.logits

            probs = F.softmax(logits.float(), dim=-1)

            # Observe first obs_positions answer positions.
            max_pos = min(args.obs_positions, args.gen_length)
            for rel_pos in range(max_pos):
                abs_pos = answer_start + rel_pos
                if abs_pos >= answer_end:
                    continue

                pos_probs = probs[0, abs_pos, :]

                for tok, ids in token_to_ids.items():
                    if len(ids) == 0:
                        continue

                    val = float(pos_probs[ids].sum().item())
                    token_values[tok].append(val)
                    if global_step not in token_step_values[tok]:
                        token_step_values[tok][global_step] = []
                    token_step_values[tok][global_step].append(val)

                    # debug high value events for filtering explanation
                    if val >= 0.5:
                        top_vals, top_ids = torch.topk(pos_probs, k=10)
                        token_events[tok].append(
                            {
                                "global_step": global_step,
                                "rel_pos": rel_pos,
                                "prob": val,
                                "token_ids_for_candidate": ids,
                                "topk_tokens": tokenizer.convert_ids_to_tokens(top_ids.tolist()),
                                "topk_probs": [float(x) for x in top_vals.tolist()],
                            }
                        )

            # update
            logits_with_noise = add_gumbel_noise(logits, temperature=args.temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            p = F.softmax(logits.float(), dim=-1)
            model_confidence = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
            )

            R = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            if args.remasking == "low_confidence":
                x0_p = model_confidence
            elif args.remasking == "random":
                x0_p = R
            elif args.remasking == "rate":
                x0_p = (1 - args.random_rate) * model_confidence + args.random_rate * R
            elif args.remasking == "adaptive":
                alpha = torch.clamp(torch.tensor(args.alpha0, device=x0.device), 0.0, 1.0)
                x0_p = (1 - alpha) * model_confidence + alpha * R
            elif args.remasking == "adaptive_step":
                frac = 1.0 - (step_in_block / (steps_per_block - 1)) if steps_per_block > 1 else 1.0
                alpha = torch.clamp(torch.tensor(args.alpha0 * frac, device=x0.device), 0.0, 1.0)
                x0_p = (1 - alpha) * model_confidence + alpha * R
            else:
                raise NotImplementedError(args.remasking)

            conf = x0_p.clone()
            conf[:, :block_start] = -float("inf")
            conf[:, block_end:] = -float("inf")

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, conf, -float("inf"))

            transfer_index = torch.zeros_like(x, dtype=torch.bool, device=device)
            for j in range(confidence.shape[0]):
                k = int(num_transfer_tokens[j, step_in_block].item())
                if k <= 0:
                    continue
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index]

            global_step += 1

    return token_values, token_events, token_step_values


# =========================================================
# Stats and filtering
# =========================================================


def compute_stats(values: List[float]):
    if len(values) == 0:
        return {
            "n": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    t = torch.tensor(values, dtype=torch.float32)

    return {
        "n": int(t.numel()),
        "mean": float(t.mean().item()),
        "p50": float(torch.quantile(t, 0.50).item()),
        "p90": float(torch.quantile(t, 0.90).item()),
        "p95": float(torch.quantile(t, 0.95).item()),
        "p99": float(torch.quantile(t, 0.99).item()),
        "max": float(t.max().item()),
    }


def filter_tokens(
    candidate_tokens,
    token_to_ids,
    harmful_values,
    benign_values,
    harmful_events,
    benign_events,
    args,
    tokenizer,
):
    rows = []
    kept = []
    rejected = []

    eps = 1e-12

    for tok in candidate_tokens:
        h_stats = compute_stats(harmful_values.get(tok, []))
        b_stats = compute_stats(benign_values.get(tok, []))

        h_p95 = h_stats["p95"] if h_stats["p95"] is not None else 0.0
        h_mean = h_stats["mean"] if h_stats["mean"] is not None else 0.0
        b_p95 = b_stats["p95"] if b_stats["p95"] is not None else 0.0
        b_p99 = b_stats["p99"] if b_stats["p99"] is not None else 0.0
        b_max = b_stats["max"] if b_stats["max"] is not None else 0.0
        b_mean = b_stats["mean"] if b_stats["mean"] is not None else 0.0

        ratio_p95 = h_p95 / (b_p95 + eps)
        diff_mean = h_mean - b_mean
        diff_p95 = h_p95 - b_p95

        token_ids = token_to_ids.get(tok, [])
        readable = ids_to_readable_tokens(tokenizer, token_ids)

        reasons = []

        if h_p95 < args.harmful_min_p95:
            reasons.append(f"harmful_p95<{args.harmful_min_p95}")

        if b_p95 > args.benign_max_p95:
            reasons.append(f"benign_p95>{args.benign_max_p95}")

        if b_p99 > args.benign_max_p99:
            reasons.append(f"benign_p99>{args.benign_max_p99}")

        if b_max > args.max_benign_max:
            reasons.append(f"benign_max>{args.max_benign_max}")

        if ratio_p95 < args.min_ratio:
            reasons.append(f"ratio_p95<{args.min_ratio}")

        keep = len(reasons) == 0

        row = {
            "candidate_token": tok,
            "token_ids": token_ids,
            "readable_tokens": readable,
            "keep": keep,
            "reject_reasons": reasons,
            "ratio_p95_harmful_over_benign": ratio_p95,
            "diff_mean_harmful_minus_benign": diff_mean,
            "diff_p95_harmful_minus_benign": diff_p95,
            "harmful_stats": h_stats,
            "benign_stats": b_stats,
            "harmful_high_prob_events": harmful_events.get(tok, [])[:20],
            "benign_high_prob_events": benign_events.get(tok, [])[:20],
        }

        rows.append(row)

        if keep:
            kept.append(row)
        else:
            rejected.append(row)

    # Sort kept tokens by ratio/diff
    kept = sorted(
        kept,
        key=lambda r: (
            r["ratio_p95_harmful_over_benign"],
            r["diff_p95_harmful_minus_benign"],
        ),
        reverse=True,
    )

    rejected = sorted(
        rejected,
        key=lambda r: (
            r["benign_stats"]["p99"] if r["benign_stats"]["p99"] is not None else 0.0,
            r["benign_stats"]["max"] if r["benign_stats"]["max"] is not None else 0.0,
        ),
        reverse=True,
    )

    filtered_token_ids = sorted(set(tid for row in kept for tid in row["token_ids"]))

    filtered_readable_tokens = ids_to_readable_tokens(tokenizer, filtered_token_ids)

    return {
        "filtered_candidate_tokens": [r["candidate_token"] for r in kept],
        "filtered_token_ids": filtered_token_ids,
        "token_ids": filtered_token_ids,
        "filtered_readable_tokens": filtered_readable_tokens,
        "kept_rows": kept,
        "rejected_rows": rejected,
        "all_rows": rows,
        "filter_config": {
            "harmful_min_p95": args.harmful_min_p95,
            "benign_max_p95": args.benign_max_p95,
            "benign_max_p99": args.benign_max_p99,
            "max_benign_max": args.max_benign_max,
            "min_ratio": args.min_ratio,
        },
    }


def build_step_curve_summary(harmful_step_values, benign_step_values):
    """
    Convert raw step values into mean/p95 curves for each token.
    """
    out = {}

    all_tokens = sorted(set(harmful_step_values.keys()) | set(benign_step_values.keys()))

    for tok in all_tokens:
        out[tok] = {
            "harmful": {},
            "benign": {},
        }

        for split_name, step_values in [
            ("harmful", harmful_step_values.get(tok, {})),
            ("benign", benign_step_values.get(tok, {})),
        ]:
            for step, vals in step_values.items():
                if len(vals) == 0:
                    continue

                t = torch.tensor(vals, dtype=torch.float32)

                out[tok][split_name][str(step)] = {
                    "n": int(t.numel()),
                    "mean": float(t.mean().item()),
                    "p50": float(torch.quantile(t, 0.50).item()),
                    "p90": float(torch.quantile(t, 0.90).item()),
                    "p95": float(torch.quantile(t, 0.95).item()),
                    "max": float(t.max().item()),
                }

    return out


# =========================================================
# Main
# =========================================================


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    args = parse_args()

    global torch, F, tqdm
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    torch.set_grad_enabled(False)
    if args.steps <= 0 or args.block_length <= 0 or args.gen_length <= 0:
        raise ValueError("steps, block_length, and gen_length must be positive.")
    if args.gen_length % args.block_length:
        raise ValueError("gen_length must be divisible by block_length.")
    if args.steps % (args.gen_length // args.block_length):
        raise ValueError("steps must be divisible by the number of blocks.")

    device = choose_device(args)

    is_instruct = ("instruct" in args.model_path.lower()) or ("1.5" in args.model_path)

    logging.info(f"[Model] Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    logging.info(f"[Model] Loading model: {args.model_path}")
    model = (
        AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    candidate_tokens = load_candidate_tokens(args.candidate_tokens_json)
    logging.info(f"[Candidates] {len(candidate_tokens)} candidate tokens:")
    logging.info(candidate_tokens)

    token_to_ids = token_strings_to_ids(tokenizer, candidate_tokens)
    for tok, ids in token_to_ids.items():
        logging.info(f"[TokenMap] {tok} -> {ids} -> {ids_to_readable_tokens(tokenizer, ids)}")

    harmful_data = load_json_or_hf(args.harmful_prompt, args.harmful_prompt_field, kind="Harmful")
    benign_data = load_json_or_hf(args.benign_prompt, args.benign_prompt_field, kind="Benign")

    harmful_values, harmful_events, harmful_step_values = collect_candidate_token_probs_for_dataset(
        model=model,
        tokenizer=tokenizer,
        data=harmful_data,
        args=args,
        token_to_ids=token_to_ids,
        is_instruct=is_instruct,
        num_samples=args.num_harmful_samples,
        kind="harmful",
    )

    benign_values, benign_events, benign_step_values = collect_candidate_token_probs_for_dataset(
        model=model,
        tokenizer=tokenizer,
        data=benign_data,
        args=args,
        token_to_ids=token_to_ids,
        is_instruct=is_instruct,
        num_samples=args.num_benign_samples,
        kind="benign",
    )

    result = filter_tokens(
        candidate_tokens=candidate_tokens,
        token_to_ids=token_to_ids,
        harmful_values=harmful_values,
        benign_values=benign_values,
        harmful_events=harmful_events,
        benign_events=benign_events,
        args=args,
        tokenizer=tokenizer,
    )

    result["metadata"] = {
        "model_path": args.model_path,
        "harmful_prompt": args.harmful_prompt,
        "harmful_prompt_field": args.harmful_prompt_field,
        "benign_prompt": args.benign_prompt,
        "benign_prompt_field": args.benign_prompt_field,
        "num_harmful_samples": args.num_harmful_samples,
        "num_benign_samples": args.num_benign_samples,
        "steps": args.steps,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "obs_positions": args.obs_positions,
        "obs_steps": args.obs_steps,
        "temperature": args.temperature,
        "cfg_scale": args.cfg_scale,
        "remasking": args.remasking,
        "mask_counts": args.mask_counts,
    }
    result["metadata"]["token_mapping"] = "strict_single_token_variants"
    result["step_curves"] = build_step_curve_summary(
        harmful_step_values,
        benign_step_values,
    )

    out_dir = os.path.dirname(args.output_json) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logging.info("=" * 60)
    logging.info("[Done] Data-driven filtered refusal set built.")
    logging.info(f"Filtered candidate tokens: {result['filtered_candidate_tokens']}")
    logging.info(f"Filtered readable tokens: {result['filtered_readable_tokens']}")
    logging.info(f"Saved to: {args.output_json}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
