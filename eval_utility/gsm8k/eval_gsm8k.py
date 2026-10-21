# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import re
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

sys.path.insert(0, PROJECT_ROOT)


# =====================
# Defaults
# =====================

DEFAULT_GEN_LENGTH = 256
DEFAULT_STEPS = 128
DEFAULT_BLOCK_LENGTH = 256
DEFAULT_MASK_ID = 126336
DEFAULT_TEMPERATURE = 0.5


# =====================
# Args
# =====================


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLaDA/DLLM on GSM8K using RAEC.")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_folder", type=str, default=None)
    parser.add_argument("--output_json", type=str, required=True)

    parser.add_argument(
        "--test_data_path",
        type=str,
        default=None,
        help="Optional local GSM8K-style JSON file. If not provided, load gsm8k main test split.",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of GSM8K test samples. Default: use all.",
    )

    parser.add_argument("--cache_dir", type=str, default=None)

    # DLLM decoding args
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--gen_length", type=int, default=DEFAULT_GEN_LENGTH)
    parser.add_argument("--block_length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--mask_id", type=int, default=DEFAULT_MASK_ID)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)

    parser.add_argument("--cfg_scale", type=float, default=0.0)

    parser.add_argument(
        "--remasking",
        type=str,
        default="low_confidence",
        choices=["low_confidence", "random", "rate", "adaptive", "adaptive_step"],
    )

    parser.add_argument("--random_rate", type=float, default=0.0)
    parser.add_argument("--alpha0", type=float, default=0.3)

    # Prompt
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Use tokenizer.apply_chat_template. Recommended for Instruct model.",
    )
    parser.add_argument(
        "--no_chat_template",
        dest="use_chat_template",
        action="store_false",
    )
    parser.set_defaults(use_chat_template=True)

    parser.add_argument(
        "--question_suffix",
        type=str,
        default=(
            "\nPlease solve the problem step by step. "
            "At the end, write the answer in the format: "
            "The final answer is: <number>"
        ),
    )

    parser.add_argument(
        "--answer_prompt",
        type=str,
        default="The final answer is:",
    )

    # RAEC
    parser.add_argument(
        "--refusal_token_set", default=None, help="Optional refusal token-set JSON."
    )
    parser.add_argument(
        "--compliance_token_set", default=None, help="Optional compliance token-set JSON."
    )
    parser.add_argument(
        "--raec_enable",
        action="store_true",
        help="Enable Refusal-Aware Early Commitment.",
    )
    parser.add_argument(
        "--raec_early_steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--raec_early_positions",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--raec_refusal_mass_threshold",
        type=float,
        default=0.00589,
    )
    parser.add_argument(
        "--raec_margin",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--raec_topk",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--raec_min_streak",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--raec_max_commits_per_sample",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--raec_commit_mode",
        type=str,
        default="best_refusal",
        choices=["best_refusal", "top1_if_refusal"],
    )

    # Misc
    parser.add_argument("--debug_print", action="store_true")

    parser.add_argument("--auto_pick_gpu", dest="auto_pick_gpu", action="store_true")
    parser.add_argument("--no_auto_pick_gpu", dest="auto_pick_gpu", action="store_false")
    parser.set_defaults(auto_pick_gpu=True)

    return parser.parse_args()


# =====================
# GPU Utilities
# =====================


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


def pick_best_gpu_index():
    idx = _pick_gpu_by_torch()

    if idx is not None:
        return idx

    return _pick_gpu_by_nvidia_smi()


# =====================
# Data
# =====================


def load_gsm8k_data(test_data_path=None, max_samples=None, question_suffix=""):
    data = []

    if test_data_path is not None and os.path.exists(test_data_path):
        with open(test_data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            raise ValueError(f"Local JSON must be a list, got {type(raw)}")

        for i, item in enumerate(raw):
            question = item.get("question") or item.get("instruction") or item.get("prompt") or ""

            answer = item.get("answer") or item.get("output") or item.get("target") or ""

            data.append(
                {
                    "id": item.get("id", i),
                    "question": str(question),
                    "ground_truth": str(answer),
                }
            )

    else:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split="test")

        for i, item in enumerate(ds):
            data.append(
                {
                    "id": i,
                    "question": item["question"],
                    "ground_truth": item["answer"],
                }
            )

    if max_samples is not None:
        data = data[: int(max_samples)]

    for item in data:
        item["instruction"] = item["question"] + question_suffix

    return data


# =====================
# Prompt
# =====================


def build_prompt(tokenizer, instruction, use_chat_template=True):
    if use_chat_template:
        messages = [
            {
                "role": "user",
                "content": instruction,
            }
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return instruction


def get_tokenized_input(prompt, tokenizer, device):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = inputs["input_ids"].to(device)

    if "attention_mask" in inputs:
        attention_mask = inputs["attention_mask"].to(device)
    else:
        attention_mask = torch.ones_like(input_ids, device=device)

    return input_ids, attention_mask


# =====================
# Answer Extraction
# =====================


def normalize_number_str(x):
    if x is None:
        return None

    x = str(x).strip()
    x = x.replace(",", "")

    try:
        val = float(x)
    except Exception:
        return None

    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))

    return str(val)


def extract_answer_number(text, answer_prompt="The final answer is:"):
    if text is None:
        return None

    text = str(text).replace(",", "")

    # GSM8K ground truth format: #### 42
    if "####" in text:
        tail = text.split("####")[-1]
        nums = re.findall(r"-?\d+\.?\d*", tail)

        if nums:
            return normalize_number_str(nums[-1])

    # Expected model output format
    if answer_prompt and answer_prompt in text:
        tail = text.split(answer_prompt)[-1]
        nums = re.findall(r"-?\d+\.?\d*", tail)

        if nums:
            return normalize_number_str(nums[0])

    # Relaxed matching
    lower_text = text.lower()

    relaxed_patterns = [
        r"final answer is[:\s]+(-?\d+\.?\d*)",
        r"answer is[:\s]+(-?\d+\.?\d*)",
        r"answer[:\s]+(-?\d+\.?\d*)",
    ]

    for pat in relaxed_patterns:
        m = re.search(pat, lower_text)

        if m:
            return normalize_number_str(m.group(1))

    # Fallback: last number
    nums = re.findall(r"-?\d+\.?\d*", text)

    if nums:
        return normalize_number_str(nums[-1])

    return None


# =====================
# Model
# =====================


def load_model_and_tokenizer(args, device):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )

    model = (
        AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            cache_dir=args.cache_dir,
        )
        .to(device)
        .eval()
    )

    if args.lora_folder:
        from peft import PeftModel

        logging.info(f"Loading LoRA from {args.lora_folder}")
        model = PeftModel.from_pretrained(model, args.lora_folder)
        model = model.merge_and_unload()
        model = model.to(device).eval()

    return model, tokenizer


# =====================
# Inference
# =====================


def generate_one(
    model,
    tokenizer,
    item,
    args,
    compliance_ids,
    refusal_ids,
):
    prompt_text = build_prompt(
        tokenizer=tokenizer,
        instruction=item["instruction"],
        use_chat_template=args.use_chat_template,
    )

    input_ids, attention_mask = get_tokenized_input(
        prompt=prompt_text,
        tokenizer=tokenizer,
        device=model.device,
    )

    output_ids = generate_llada(
        input_ids=input_ids,
        attention_mask=attention_mask,
        model=model,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        mask_id=args.mask_id,
        tokenizer=tokenizer,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        random_rate=args.random_rate,
        alpha0=args.alpha0,
        debug_print=args.debug_print,
        # Trace recording is disabled during utility evaluation.
        return_trace=False,
        trace_topk=0,
        trace_positions=0,
        # Still needed internally by RAEC
        refusal_token_ids=refusal_ids,
        compliance_token_ids=compliance_ids,
        # RAEC
        raec_enable=args.raec_enable,
        raec_early_steps=args.raec_early_steps,
        raec_early_positions=args.raec_early_positions,
        raec_refusal_mass_threshold=args.raec_refusal_mass_threshold,
        raec_margin=args.raec_margin,
        raec_topk=args.raec_topk,
        raec_min_streak=args.raec_min_streak,
        raec_max_commits_per_sample=args.raec_max_commits_per_sample,
        raec_commit_mode=args.raec_commit_mode,
    )

    prompt_len = input_ids.shape[1]

    response = tokenizer.batch_decode(
        output_ids[:, prompt_len:],
        skip_special_tokens=True,
    )[0].strip()

    return response, prompt_text


# =====================
# Main
# =====================


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    args = parse_args()

    global torch, generate_llada, load_token_sets
    import torch
    from tqdm import tqdm

    from utility.generate_function_llada_refusal_method import generate_llada
    from utility.token_sets import load_token_sets

    # Device
    if torch.cuda.is_available():
        if args.auto_pick_gpu:
            best_idx = pick_best_gpu_index()

            if best_idx is not None:
                try:
                    torch.cuda.set_device(best_idx)
                except Exception:
                    pass

                device = torch.device(f"cuda:{best_idx}")

                try:
                    with torch.cuda.device(device):
                        free, total = torch.cuda.mem_get_info()

                    logging.info(
                        f"[GPU] Auto-picked cuda:{best_idx} "
                        f"(free {free / 1024 / 1024 / 1024:.1f} GB / "
                        f"total {total / 1024 / 1024 / 1024:.1f} GB)"
                    )

                except Exception:
                    logging.info(f"[GPU] Auto-picked cuda:{best_idx}")

            else:
                device = torch.device("cuda")
                logging.info("[GPU] Auto-pick failed. Using default cuda.")

        else:
            device = torch.device("cuda")
            logging.info("[GPU] Auto-pick disabled. Using default cuda.")

    else:
        device = torch.device("cpu")
        logging.info("[GPU] CUDA unavailable. Using CPU.")

    # Data
    data = load_gsm8k_data(
        test_data_path=args.test_data_path,
        max_samples=args.max_samples,
        question_suffix=args.question_suffix,
    )

    logging.info(f"Loaded {len(data)} GSM8K samples.")

    # Model
    model, tokenizer = load_model_and_tokenizer(args, device)

    # RAEC cue ids
    compliance_ids, refusal_ids = load_token_sets(
        tokenizer, args.refusal_token_set, args.compliance_token_set
    )

    logging.info(f"Number of compliance ids: {len(compliance_ids)}")
    logging.info(f"Number of refusal ids: {len(refusal_ids)}")
    logging.info(f"RAEC enabled: {args.raec_enable}")

    results = []
    correct = 0
    total = 0

    with torch.no_grad():
        for item in tqdm(data, desc="Evaluating GSM8K with RAEC"):
            response, final_prompt = generate_one(
                model=model,
                tokenizer=tokenizer,
                item=item,
                args=args,
                compliance_ids=compliance_ids,
                refusal_ids=refusal_ids,
            )

            gt_answer = extract_answer_number(
                item["ground_truth"],
                answer_prompt=args.answer_prompt,
            )

            pred_answer = extract_answer_number(
                response,
                answer_prompt=args.answer_prompt,
            )

            is_correct = (
                gt_answer is not None and pred_answer is not None and gt_answer == pred_answer
            )

            correct += int(is_correct)
            total += 1

            rec = {
                "id": item["id"],
                "question": item["question"],
                "instruction": item["instruction"],
                "ground_truth": item["ground_truth"],
                "response": response,
                "gt_answer": gt_answer,
                "pred_answer": pred_answer,
                "correct": bool(is_correct),
                "final_prompt": final_prompt,
            }

            results.append(rec)

            logging.info(
                f"[{total}] correct={is_correct} "
                f"gt={gt_answer} pred={pred_answer} "
                f"running_acc={correct / max(total, 1) * 100:.2f}%"
            )

    accuracy = correct / max(total, 1) * 100

    summary = {
        "score": accuracy,
        "score_str": f"{accuracy:.2f}",
        "correct": correct,
        "total": total,
        "config": {
            "model_path": args.model_path,
            "lora_folder": args.lora_folder,
            "test_data_path": args.test_data_path,
            "max_samples": args.max_samples,
            "steps": args.steps,
            "gen_length": args.gen_length,
            "block_length": args.block_length,
            "mask_id": args.mask_id,
            "temperature": args.temperature,
            "cfg_scale": args.cfg_scale,
            "remasking": args.remasking,
            "random_rate": args.random_rate,
            "use_chat_template": args.use_chat_template,
            "question_suffix": args.question_suffix,
            "answer_prompt": args.answer_prompt,
            "raec_enable": args.raec_enable,
            "raec_early_steps": args.raec_early_steps,
            "raec_early_positions": args.raec_early_positions,
            "raec_refusal_mass_threshold": args.raec_refusal_mass_threshold,
            "raec_margin": args.raec_margin,
            "raec_topk": args.raec_topk,
            "raec_min_streak": args.raec_min_streak,
            "raec_max_commits_per_sample": args.raec_max_commits_per_sample,
            "raec_commit_mode": args.raec_commit_mode,
        },
    }

    output = {
        "summary": summary,
        "results": results,
    }

    out_dir = os.path.dirname(args.output_json) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    logging.info(f"Accuracy: {accuracy:.2f}%")
    logging.info(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
