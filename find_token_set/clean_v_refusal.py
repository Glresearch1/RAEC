import argparse
import json
import os
import re


def normalize_token_for_grouping(tok: str) -> str:
    """
    Normalize only for display; preserve the raw tokens in the final set.
    Examples:
      Ġsorry -> sorry
      Sorry  -> sorry
    """
    t = tok.strip()
    if t.startswith("Ġ"):
        t = t[1:]
    t = t.strip()
    return t.lower()


def is_drop_token(tok: str) -> bool:
    """
    Conservatively drop special tokens, punctuation, numbers, and fragments.
    """
    if tok is None:
        return True

    t = tok.strip()

    # Empty strings.
    if len(t) == 0:
        return True

    # Special tokens.
    special_tokens = {
        "<|endoftext|>",
        "<|eot_id|>",
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
        "Ċ",  # Common newline token.
    }
    if t in special_tokens:
        return True

    # Standalone punctuation and symbols.
    punctuation_only = {
        ",",
        ".",
        ":",
        ";",
        "!",
        "?",
        '"',
        "'",
        "`",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "%",
        "+",
        "-",
        "*",
        "/",
        "\\",
        "|",
        "&",
        "#",
        "@",
        "^",
        "~",
        "=",
    }
    if t in punctuation_only:
        return True

    # Ignore the BPE space prefix while applying the remaining checks.
    core = t[1:] if t.startswith("Ġ") else t

    # Numeric tokens.
    if re.fullmatch(r"\d+", core):
        return True

    # Single-letter tokens.
    if re.fullmatch(r"[A-Za-z]", core):
        return True

    # Drop common contraction fragments that are weak refusal cues on their own.
    if core in {"'m", "'s", "'re", "'ve", "'ll", "'d"}:
        return True

    return False


def build_refusal_token_set(
    top_positive_tokens,
    top_pref_tokens,
    top_positive_k: int = 100,
    top_pref_k: int = 200,
):
    """
    Intersect the top positive-direction tokens with the top aligned-probability
    tokens, then apply conservative token cleaning.
    """
    top_positive_main = top_positive_tokens[:top_positive_k]
    top_pref_filter = top_pref_tokens[:top_pref_k]

    top_pref_filter_set = set(top_pref_filter)

    # Preserve the positive-direction ranking when intersecting the lists.
    refusal_candidates_raw = [tok for tok in top_positive_main if tok in top_pref_filter_set]

    drop_tokens_removed = []
    refusal_token_set_cleaned = []

    for tok in refusal_candidates_raw:
        if is_drop_token(tok):
            drop_tokens_removed.append(tok)
        else:
            refusal_token_set_cleaned.append(tok)

    return {
        "top_positive_main": top_positive_main,
        "top_pref_filter": top_pref_filter,
        "refusal_candidates_raw": refusal_candidates_raw,
        "drop_tokens_removed": drop_tokens_removed,
        "refusal_token_set_cleaned": refusal_token_set_cleaned,
        "tokens": refusal_token_set_cleaned,
        "candidate_tokens": refusal_token_set_cleaned,
        "normalized_groups_preview": {
            tok: normalize_token_for_grouping(tok) for tok in refusal_token_set_cleaned
        },
    }


def save_results(results: dict, output_path: str):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Saved refusal token set results to:")
    print(output_path)
    print("=" * 60)

    print("\nRefusal candidates before cleaning:")
    print(results["refusal_candidates_raw"])

    print("\nDropped tokens:")
    print(results["drop_tokens_removed"])

    print("\nFinal refusal token set:")
    print(results["refusal_token_set_cleaned"])

    print(f"\nFinal size: {len(results['refusal_token_set_cleaned'])}")


def main(
    checkpoint_path: str,
    output_path: str = "results/token_discovery/refusal_token_set.json",
    top_positive_k: int = 500,
    top_pref_k: int = 500,
):
    """
    Read token rankings from a checkpoint produced by find_token_add_pbase.py.
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    print(ckpt.keys())

    top_positive_tokens = ckpt["top_positive_tokens"]
    top_negative_tokens = ckpt["top_negative_tokens"]
    top_pref_tokens = ckpt["top_pref_tokens"]

    # =========================
    # Build the refusal token set.
    # =========================
    results = build_refusal_token_set(
        top_positive_tokens=top_positive_tokens,
        top_pref_tokens=top_pref_tokens,
        top_positive_k=top_positive_k,
        top_pref_k=top_pref_k,
    )

    # Preserve negative-direction tokens for optional compliance-set analysis.
    results["top_negative_tokens"] = top_negative_tokens

    save_results(results, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean refusal-associated token rankings.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_path", default="results/token_discovery/refusal_token_set.json")
    parser.add_argument("--top_positive_k", type=int, default=500)
    parser.add_argument("--top_pref_k", type=int, default=500)
    main(**vars(parser.parse_args()))
