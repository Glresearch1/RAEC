"""Run LLaDA generation with optional Refusal-Aware Early Commitment (RAEC)."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MASK_TOKEN = "<|mdm_mask|>"
START_TOKEN = "<startoftext>"
END_TOKEN = "<endoftext>"
SPECIAL_TOKEN_PATTERN = r"<mask:(\d+)>"
RAEC_FIELDS = (
    "raec_enable",
    "raec_early_steps",
    "raec_early_positions",
    "raec_refusal_mass_threshold",
    "raec_margin",
    "raec_topk",
    "raec_min_streak",
    "raec_max_commits_per_sample",
    "raec_commit_mode",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path", required=True, help="Hugging Face model ID or local checkpoint."
    )
    parser.add_argument(
        "--attack_prompt", required=True, help="JSON list or HF dataset[:config[:split]]."
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--attack_method", choices=["zeroshot", "DIJA", "PAD", "other"], default="zeroshot"
    )
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=128)
    parser.add_argument("--mask_id", type=int, default=126336)
    parser.add_argument(
        "--mask_counts", type=int, default=0, help="Append optional masks inside the user prompt."
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--cfg_scale", type=float, default=0.1)
    parser.add_argument(
        "--remasking",
        choices=["low_confidence", "random", "rate", "adaptive", "adaptive_step"],
        default="low_confidence",
    )
    parser.add_argument("--random_rate", type=float, default=0.0)
    parser.add_argument("--alpha0", type=float, default=0.3)
    parser.add_argument("--fill_all_masks", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--prompt_format", choices=["auto", "chat", "plain"], default="auto")
    parser.add_argument(
        "--prompt_field",
        choices=["auto", "vanilla", "refined"],
        default="auto",
        help="Override automatic original/attack-prompt selection.",
    )
    parser.add_argument("--debug_print", action="store_true")
    parser.add_argument(
        "--no_trace", action="store_true", help="Omit detailed trajectories and their summaries."
    )
    parser.add_argument("--trace_topk", type=int, default=10)
    parser.add_argument("--trace_positions", type=int, default=16)
    parser.add_argument("--refusal_token_set", help="Optional JSON token IDs or string cues.")
    parser.add_argument("--compliance_token_set", help="Optional JSON token IDs or string cues.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate data and save a preview without loading a model.",
    )
    parser.add_argument("--raec_enable", action="store_true")
    parser.add_argument("--raec_early_steps", type=int, default=8)
    parser.add_argument("--raec_early_positions", type=int, default=8)
    parser.add_argument("--raec_refusal_mass_threshold", type=float, default=0.00589)
    parser.add_argument("--raec_margin", type=float, default=0.001)
    parser.add_argument("--raec_topk", type=int, default=10)
    parser.add_argument("--raec_min_streak", type=int, default=2)
    parser.add_argument("--raec_max_commits_per_sample", type=int, default=2)
    parser.add_argument(
        "--raec_commit_mode", choices=["best_refusal", "top1_if_refusal"], default="best_refusal"
    )
    args = parser.parse_args(argv)
    positive = [
        "steps",
        "block_length",
        "trace_topk",
        "trace_positions",
        "raec_topk",
        "raec_min_streak",
    ]
    nonnegative = [
        "gen_length",
        "mask_counts",
        "temperature",
        "raec_early_steps",
        "raec_early_positions",
        "raec_max_commits_per_sample",
        "raec_margin",
    ]
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive.")
    for name in nonnegative:
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be nonnegative.")
    if args.gen_length and args.gen_length % args.block_length:
        parser.error("--gen_length must be divisible by --block_length.")
    if args.gen_length and args.steps < args.gen_length // args.block_length:
        parser.error("--steps must be at least the number of answer blocks.")
    for name in ["random_rate", "alpha0", "raec_refusal_mass_threshold"]:
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name} must lie in [0, 1].")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max_samples must be positive.")
    return args


def load_attack_data(source):
    """Read local rows unchanged, or convert a Hugging Face split to dictionaries."""
    path = Path(source).expanduser()
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            rows = json.load(stream)
    else:
        if (
            path.suffix.lower() in {".json", ".jsonl"}
            or path.is_absolute()
            or source.startswith(".")
        ):
            raise FileNotFoundError(f"Prompt file not found: {source}")
        parts = source.split(":")
        if not 1 <= len(parts) <= 3 or any(not part for part in parts):
            raise ValueError("Use dataset[:config[:split]] for Hugging Face inputs.")
        from datasets import load_dataset

        kwargs = {"split": parts[2] if len(parts) == 3 else "train"}
        ds = load_dataset(parts[0], *([parts[1]] if len(parts) >= 2 else []), **kwargs)
        rows = [dict(row) for row in ds]
    if not isinstance(rows, list) or not rows:
        raise ValueError("The prompt dataset must be a nonempty JSON list.")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Prompt row {index} must be a JSON object.")
    return rows


def _first_text(item, names):
    for name in names:
        value = item.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def pick_two_prompt_fields(item):
    instruction = _first_text(item, ["instruction"])
    task_input = _first_text(item, ["input"])
    if instruction and task_input:
        vanilla = f"{instruction}\n\nInput:\n{task_input}"
    else:
        vanilla = (
            instruction
            or task_input
            or _first_text(
                item,
                [
                    "vanilla prompt",
                    "prompt",
                    "question",
                    "text",
                    "goal",
                    "Goal",
                    "Behavior",
                    "behavior",
                    "forbidden_prompt",
                    "rewrite_prompt",
                ],
            )
        )
    refined = _first_text(
        item,
        [
            "refined prompt",
            "refined_prompt",
            "refined_goal",
            "jailbreak_prompt",
            "rewritten_prompt",
            "Refined_behavior",
        ],
    )
    return vanilla, refined


def should_use_refined(args, attack_prompt_path, refined_text):
    """Prefer attack rewrites for DIJA or a dataset whose name includes 'refine'."""
    field = getattr(args, "prompt_field", "auto")
    if field == "vanilla":
        return False
    if field == "refined":
        if not refined_text.strip():
            raise ValueError(
                "--prompt_field refined requires a refined prompt in every selected row."
            )
        return True
    path_flag = "refine" in Path(attack_prompt_path).name.lower()
    return bool(refined_text.strip()) and (args.attack_method.casefold() == "dija" or path_flag)


def process_user_text(user_text, mask_counts):
    """Expand inline span masks and preserve DIJA's prompt-infilling format."""
    processed = re.sub(
        SPECIAL_TOKEN_PATTERN, lambda match: MASK_TOKEN * int(match.group(1)), user_text
    )
    if MASK_TOKEN not in processed and mask_counts > 0:
        processed += START_TOKEN + MASK_TOKEN * mask_counts + END_TOKEN
    return processed


def prepare_rows(rows, args):
    selected = rows[: args.max_samples] if args.max_samples else rows
    prepared = []
    for index, item in enumerate(selected):
        vanilla, refined = pick_two_prompt_fields(item)
        use_refined = should_use_refined(args, args.attack_prompt, refined)
        chosen = refined if use_refined else vanilla
        if not chosen:
            raise ValueError(f"Prompt row {index} has no usable text for {args.attack_method}.")
        prepared.append(
            (item, vanilla, refined, use_refined, process_user_text(chosen, args.mask_counts))
        )
    return prepared


def build_chat_prompt(tokenizer, text, is_instruct):
    if is_instruct:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return text


def summarize_refusal_trace(trace, refusal_token_ids=None, compliance_token_ids=None):
    """Summarize zero-based steps in the leading positions recorded by the trace."""
    summary = {
        "first_refusal_topk_step": None,
        "first_refusal_top1_step": None,
        "first_refusal_commit_step": None,
        "first_compliance_commit_step": None,
        "refusal_commit_delay": None,
        "compliance_beats_refusal": None,
        "refusal_top1_but_never_committed": None,
        "raec_triggered": False,
        "raec_first_step": None,
        "raec_forced_commit_count": 0,
        "raec_forced_positions": [],
        "raec_forced_tokens": [],
    }
    if not trace:
        return summary
    refusal_ids = set(refusal_token_ids or ())
    compliance_ids = set(compliance_token_ids or ())
    for step in trace:
        global_step = step["global_step"]
        for event in step.get("raec_events", []):
            summary["raec_triggered"] = True
            if summary["raec_first_step"] is None:
                summary["raec_first_step"] = event.get("global_step", global_step)
            summary["raec_forced_commit_count"] += 1
            summary["raec_forced_positions"].append(event.get("relative_answer_pos"))
            summary["raec_forced_tokens"].append(event.get("commit_token"))
        for position in step.get("positions", []):
            masked = position.get("is_masked_before_step", False)
            if (
                masked
                and position.get("best_refusal_rank") is not None
                and summary["first_refusal_topk_step"] is None
            ):
                summary["first_refusal_topk_step"] = global_step
            if (
                masked
                and position.get("is_refusal_top1")
                and summary["first_refusal_top1_step"] is None
            ):
                summary["first_refusal_top1_step"] = global_step
            committed = position.get("committed_id")
            if committed in refusal_ids and summary["first_refusal_commit_step"] is None:
                summary["first_refusal_commit_step"] = global_step
            if committed in compliance_ids and summary["first_compliance_commit_step"] is None:
                summary["first_compliance_commit_step"] = global_step
    first_refusal = summary["first_refusal_commit_step"]
    first_compliance = summary["first_compliance_commit_step"]
    first_top1 = summary["first_refusal_top1_step"]
    if first_top1 is not None and first_refusal is not None:
        summary["refusal_commit_delay"] = first_refusal - first_top1
    if first_compliance is not None:
        summary["compliance_beats_refusal"] = (
            first_refusal is None or first_compliance < first_refusal
        )
    summary["refusal_top1_but_never_committed"] = first_top1 is not None and first_refusal is None
    return summary


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rows = prepare_rows(load_attack_data(args.attack_prompt), args)
    if args.dry_run:
        save_json(
            args.output_json,
            {
                "mode": "dry_run",
                "model_loaded": False,
                "num_samples": len(rows),
                "config": vars(args),
                "samples": [
                    {
                        "id": item.get("id", index),
                        "used_prompt_type": "refined" if use_refined else "vanilla",
                        "processed_user_text": processed,
                        "num_prompt_masks": processed.count(MASK_TOKEN),
                    }
                    for index, (item, _, _, use_refined, processed) in enumerate(rows)
                ],
            },
        )
        logging.info(
            "Validated %d prompts; saved preview to %s. No model was loaded.",
            len(rows),
            args.output_json,
        )
        return

    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    from utility.generate_function_llada_refusal_method import generate_llada
    from utility.token_sets import load_token_sets

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    dtype = getattr(torch, args.dtype)
    logging.info("Loading %s on %s with %s", args.model_path, device, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=dtype)
        .to(device)
        .eval()
    )
    compliance_ids, refusal_ids = load_token_sets(
        tokenizer, args.refusal_token_set, args.compliance_token_set
    )
    is_instruct = args.prompt_format == "chat" or (
        args.prompt_format == "auto"
        and ("instruct" in args.model_path.lower() or "1.5" in args.model_path)
    )
    raec_config = {field: getattr(args, field) for field in RAEC_FIELDS}
    metadata = vars(args).copy()
    metadata.update(
        {
            "refusal_token_ids": sorted(refusal_ids),
            "compliance_token_ids": sorted(compliance_ids),
            "torch_version": str(torch.__version__),
        }
    )
    save_json(str(args.output_json) + ".config.json", metadata)
    results = []
    with torch.no_grad():
        for item, vanilla, refined, use_refined, processed in tqdm(rows, desc="Generating"):
            prompt = build_chat_prompt(tokenizer, processed, is_instruct)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            output = generate_llada(
                input_ids=input_ids,
                attention_mask=attention_mask,
                model=model,
                tokenizer=tokenizer,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                mask_id=args.mask_id,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking,
                random_rate=args.random_rate,
                alpha0=args.alpha0,
                fill_all_masks=args.fill_all_masks,
                debug_print=args.debug_print,
                attack_method="pad" if args.attack_method == "PAD" else "none",
                return_trace=not args.no_trace,
                trace_topk=args.trace_topk,
                trace_positions=args.trace_positions,
                refusal_token_ids=refusal_ids,
                compliance_token_ids=compliance_ids,
                **raec_config,
            )
            output_ids, trace = output if isinstance(output, tuple) else (output, None)
            response = tokenizer.batch_decode(
                output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
            )[0]
            if args.attack_method == "DIJA":
                response = response.split("assistant\n")[0]
            record = {
                "vanilla prompt": vanilla,
                "refined prompt": refined,
                "used_prompt_type": "refined" if use_refined else "vanilla",
                "final_prompt": prompt,
                "response": response,
                "trace": trace,
                "trace_summary": summarize_refusal_trace(trace, refusal_ids, compliance_ids)
                if trace is not None
                else None,
                "raec_config": raec_config,
            }
            for field in [
                "id",
                "BehaviorID",
                "source",
                "category",
                "goal",
                "refined_goal",
                "Behavior",
                "Refined_behavior",
                "target",
                "label",
                "focus",
                "type",
            ]:
                if field in item:
                    record[field] = item[field]
            results.append(record)
            if args.debug_print:
                logging.info("Response: %s", response)
    save_json(args.output_json, results)
    logging.info("Saved %d responses to %s", len(results), args.output_json)


if __name__ == "__main__":
    main()
