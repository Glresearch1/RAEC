"""Load optional, tokenizer-specific refusal and compliance token sets."""

import json
from pathlib import Path


def _load_ids(tokenizer, path):
    from utility.generate_function_llada_refusal_method import _ids_from_phrases

    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    exact = False
    if isinstance(data, dict):
        if "token_ids" in data:
            data = data["token_ids"]
            exact = True
        elif "tokens" in data:
            data = data["tokens"]
        else:
            raise ValueError(f"{path}: expected 'token_ids' or 'tokens'.")
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: token sets must be nonempty JSON lists.")
    if all(isinstance(item, int) and not isinstance(item, bool) for item in data):
        ids = set(data)
    elif not exact and all(isinstance(item, str) for item in data):
        ids = _ids_from_phrases(tokenizer, data)
    else:
        raise ValueError(f"{path}: use either integer IDs or string cues, without mixing types.")
    if not ids or any(token_id < 0 or token_id >= len(tokenizer) for token_id in ids):
        raise ValueError(f"{path}: token IDs must lie within this tokenizer's vocabulary.")
    return ids


def load_token_sets(tokenizer, refusal_token_set=None, compliance_token_set=None):
    """Return (compliance_ids, refusal_ids), preserving the original built-in defaults.

    Integer lists and {"token_ids": [...]} specify exact IDs. String lists and
    {"tokens": [...]} use the same phrase expansion as the built-in cue lists.
    Token IDs are specific to a model tokenizer and must not be reused blindly.
    """
    from utility.generate_function_llada_refusal_method import build_cue_id_sets

    compliance_ids, refusal_ids = build_cue_id_sets(tokenizer)
    if refusal_token_set:
        refusal_ids = _load_ids(tokenizer, refusal_token_set)
    if compliance_token_set:
        compliance_ids = _load_ids(tokenizer, compliance_token_set)
    return compliance_ids, refusal_ids
