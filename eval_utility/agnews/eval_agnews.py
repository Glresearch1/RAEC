# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import re
import subprocess
import sys

# =========================================================
# Project root
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

sys.path.insert(0, PROJECT_ROOT)


# =========================================================
# Defaults
# =========================================================

DEFAULT_GEN_LENGTH = 64
DEFAULT_STEPS = 32
DEFAULT_BLOCK_LENGTH = 64
DEFAULT_MASK_ID = 126336
DEFAULT_TEMPERATURE = 0.5


# =========================================================
# Args
# =========================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LLaDA/DLLM on AGNews using existing generate_llada() with optional RAEC."
    )

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_folder", type=str, default=None)
    parser.add_argument("--output_json", type=str, required=True)

    parser.add_argument(
        "--test_data_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "agnews_test_data.json"),
        help="Path to local AGNews-style JSON. If missing, load ag_news test split and save here.",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Limit number of AGNews test samples.",
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
        "--instruction",
        type=str,
        default=(
            "Categorize the news article given in the input into one of the 4 categories:\n\n"
            "World\nSports\nBusiness\nSci/Tech\n\n"
            "Only output one category name from: World, Sports, Business, Sci/Tech.\n\n"
            "Input:"
        ),
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


# =========================================================
# GPU Utilities
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


def pick_best_gpu_index():
    idx = _pick_gpu_by_torch()

    if idx is not None:
        return idx

    return _pick_gpu_by_nvidia_smi()


# =========================================================
# Data
# =========================================================


def load_test_data(args):
    if args.test_data_path and os.path.exists(args.test_data_path):
        with open(args.test_data_path, "r", encoding="utf-8") as f:
            input_data_lst = json.load(f)

        print(f"Loaded test data from {args.test_data_path}")

    else:
        from datasets import load_dataset

        dataset = load_dataset("ag_news", cache_dir=args.cache_dir)

        input_data_lst = []

        for index, example in enumerate(dataset["test"]):
            if args.max_samples is not None and index >= int(args.max_samples):
                break

            instance = {
                "id": index,
                "instruction": args.instruction,
                "input": example["text"],
                "label": int(example["label"]),
            }

            input_data_lst.append(instance)

        if args.test_data_path:
            out_dir = os.path.dirname(args.test_data_path) or "."
            os.makedirs(out_dir, exist_ok=True)

            with open(args.test_data_path, "w", encoding="utf-8") as f:
                json.dump(input_data_lst, f, indent=4, ensure_ascii=False)

            print(f"Saved test data to {args.test_data_path}")

    if args.max_samples is not None:
        input_data_lst = input_data_lst[: int(args.max_samples)]

    return input_data_lst


# =========================================================
# Prompt
# =========================================================


def build_prompt(data, tokenizer, use_chat_template=True):
    instruction = data["instruction"]
    input_text = data.get("input", "")

    if input_text:
        user_content = instruction + " " + input_text
    else:
        user_content = instruction

    if use_chat_template:
        messages = [
            {
                "role": "user",
                "content": user_content,
            }
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return prompt_text

    return user_content


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


# =========================================================
# Model
# =========================================================


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


# =========================================================
# Inference
# =========================================================


def generate_one(
    model,
    tokenizer,
    data,
    args,
    compliance_ids,
    refusal_ids,
):
    prompt_text = build_prompt(
        data=data,
        tokenizer=tokenizer,
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
        # Needed internally by RAEC
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


# =========================================================
# Evaluation
# =========================================================


def normalize_pred_text(pred):
    if pred is None:
        return ""

    pred = str(pred).strip()

    # Flatten multiline explanations before matching category names.
    pred = pred.replace("\n", " ")

    return pred


def predict_label_from_text(pred):
    """
    AGNews labels:
        0: World
        1: Sports
        2: Business
        3: Sci/Tech
    """

    pred = normalize_pred_text(pred)

    # Match the more specific science and technology category first.
    sci_patterns = [
        r"\bSci/Tech\b",
        r"\bSci-Tech\b",
        r"\bScience/Technology\b",
        r"\bScience and Technology\b",
        r"\bTechnology\b",
        r"\bTech\b",
    ]

    label_patterns = {
        0: [
            r"\bWorld\b",
        ],
        1: [
            r"\bSports\b",
            r"\bSport\b",
        ],
        2: [
            r"\bBusiness\b",
            r"\bFinance\b",
            r"\bFinancial\b",
            r"\bEconomy\b",
            r"\bMarket\b",
        ],
        3: sci_patterns,
    }

    # Accept category names embedded in phrases such as "Category: Business".
    for label in [3, 0, 1, 2]:
        for pat in label_patterns[label]:
            if re.search(pat, pred, flags=re.IGNORECASE):
                return label

    return None


def is_correct_prediction(pred, label):
    pred_label = predict_label_from_text(pred)

    if pred_label is None:
        return False, None

    return int(pred_label) == int(label), pred_label


def label_id_to_name(label):
    mapping = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech",
    }

    return mapping.get(int(label), str(label))


# =========================================================
# Main
# =========================================================


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
    input_data_lst = load_test_data(args)
    logging.info(f"Loaded {len(input_data_lst)} AGNews samples.")

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
        for data in tqdm(input_data_lst, desc="Evaluating AGNews with DLLM/RAEC"):
            pred, final_prompt = generate_one(
                model=model,
                tokenizer=tokenizer,
                data=data,
                args=args,
                compliance_ids=compliance_ids,
                refusal_ids=refusal_ids,
            )

            label = int(data["label"])
            is_correct, pred_label = is_correct_prediction(pred, label)

            correct += int(is_correct)
            total += 1

            rec = dict(data)
            rec["output"] = pred
            rec["pred_label"] = pred_label
            rec["pred_label_name"] = (
                label_id_to_name(pred_label) if pred_label is not None else None
            )
            rec["label_name"] = label_id_to_name(label)
            rec["correct"] = "true" if is_correct else "false"
            rec["final_prompt"] = final_prompt

            results.append(rec)

            logging.info(
                f"[{total}] correct={is_correct} "
                f"gt={label_id_to_name(label)} "
                f"pred={rec['pred_label_name']} "
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
            "instruction": args.instruction,
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
