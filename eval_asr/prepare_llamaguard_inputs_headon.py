# -*- coding: utf-8 -*-
import argparse
import json
import os
from typing import Any, Dict, List


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def choose_prompt(record: Dict) -> str:
    used_prompt_type = record.get("used_prompt_type", "").strip().lower()
    vanilla_prompt = record.get("vanilla prompt", "") or ""
    refined_prompt = record.get("refined prompt", "") or ""

    if used_prompt_type == "refined" and refined_prompt.strip():
        return refined_prompt
    if used_prompt_type == "vanilla" and vanilla_prompt.strip():
        return vanilla_prompt

    # fallback
    if refined_prompt.strip():
        return refined_prompt
    if vanilla_prompt.strip():
        return vanilla_prompt
    if record.get("prompt"):
        return str(record["prompt"])
    if record.get("instruction"):
        return str(record["instruction"]) + (
            " " + str(record["input"]) if record.get("input") else ""
        )
    return str(record.get("question", record.get("final_prompt", "")))


def main():
    parser = argparse.ArgumentParser(
        description="Convert generation results into LlamaGuard input format."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Input JSON file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output flattened JSON file",
    )
    args = parser.parse_args()

    data = load_json(args.input_json)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON list.")

    converted: List[Dict] = []

    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            print(f"[Warning] Skip non-dict record at index {idx}")
            continue

        prompt = choose_prompt(record)
        response = record.get("response", record.get("output", ""))

        item = {
            "sample_id": idx,
            "prompt": prompt,
            "output": response,
            **({"condition": record["condition"]} if "condition" in record else {}),
        }
        converted.append(item)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    save_json(converted, args.output_file)

    print("=" * 80)
    print(f"Original samples: {len(data)}")
    print(f"Converted samples: {len(converted)}")
    print(f"Saved to: {args.output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
