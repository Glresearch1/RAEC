"""LLaDA masked generation with Refusal-Aware Early Commitment (RAEC).

RAEC adds a bounded number of refusal-token commitments during the early
sampling window. It uses the same forward pass as the baseline sampler.
"""

import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


def _ids_from_phrases(tokenizer, phrases: Iterable[str]) -> set:
    ids = set()

    vocab = tokenizer.get_vocab()

    for ph in phrases:
        ph = str(ph).strip()
        if not ph:
            continue

        # Case 1: ph is already a vocab token, e.g., "Ġsorry"
        if ph in vocab:
            ids.add(int(vocab[ph]))

        # Case 2: phrase-style tokenization
        variants = {ph, " " + ph}

        for v in variants:
            toks = tokenizer(v, add_special_tokens=False).input_ids
            for t in toks:
                ids.add(int(t))

    return ids


def build_cue_id_sets(
    tokenizer,
    compliance_cues: Optional[Iterable[str]] = None,
    refusal_cues: Optional[Iterable[str]] = None,
) -> Tuple[set, set]:
    """
    Build token-id sets for compliance-like cues and refusal-like cues.

    Important:
        For intervention, refusal cues should be relatively clean.
        Avoid too-general cues such as "no" and "not" if utility is sensitive.

    Returns:
        compliance_ids, refusal_ids
    """

    if compliance_cues is None:
        compliance_cues = [
            "sure",
            "Sure",
            "here",
            "Here",
            "heres",
            "Here's",
            "certainly",
            "Certainly",
            "yes",
            "Yes",
            "okay",
            "Okay",
            "step",
            "Step",
            "first",
            "First",
            "instructions",
            "Instructions",
            "guide",
            "Guide",
            "tutorial",
            "Tutorial",
            "method",
            "Method",
            "process",
            "Process",
            "how",
            "How",
            "to",
            "To",
        ]

    if refusal_cues is None:
        # Cleaner set for RAEC-style intervention.
        # Deliberately avoid very broad tokens like "no" and "not".

        refusal_cues = [
            "able",
            "Ġbut",
            "'t",
            "Ġhelp",
            "Ġinformation",
            "ĠBut",
            "Ġprovide",
            "Ġable",
            "But",
            "Ġcannot",
            "but",
            "Ġassist",
            "ĠInformation",
            "ĠHelp",
            "Ġdangerous",
            "Ġunable",
            "help",
            "Help",
            "Ġsorry",
            "Ġillegal",
            "Information",
            "Ġharmful",
            "Ġrefuse",
            "ĠIllegal",
            "Sorry",
            "information",
            "ĠSorry",
            "ĠProvide",
            "Cannot",
            "Unable",
            "ĠAssist",
            "cannot",
            "ĠCannot",
            "Provide",
            "Illegal",
            "sorry",
            "ĠUnable",
            "provide",
            "unable",
            "ĠDangerous",
            "ĠAble",
            "Assist",
            "illegal",
        ]

    compliance_ids = _ids_from_phrases(tokenizer, compliance_cues)
    refusal_ids = _ids_from_phrases(tokenizer, refusal_cues)

    return compliance_ids, refusal_ids


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


def _collect_position_trace(
    probs: torch.Tensor,  # [B, L, V]
    x0: torch.Tensor,  # [B, L]
    x: torch.Tensor,  # [B, L]
    mask_index: torch.Tensor,  # [B, L]
    transfer_index: torch.Tensor,  # [B, L]
    tokenizer,
    track_positions: list[int],
    topk: int,
    refusal_token_ids: Optional[set],
    compliance_token_ids: Optional[set],
):
    trace_positions = []

    vocab_size = probs.shape[-1]
    k = min(int(topk), vocab_size)

    refusal_token_ids = set() if refusal_token_ids is None else set(refusal_token_ids)
    compliance_token_ids = set() if compliance_token_ids is None else set(compliance_token_ids)

    for pos in track_positions:
        pos_probs = probs[0, pos]

        top_vals, top_ids = torch.topk(pos_probs, k=k)
        top_ids_list = [int(x) for x in top_ids.tolist()]
        top_vals_list = [float(x) for x in top_vals.tolist()]

        top_tokens = tokenizer.convert_ids_to_tokens(top_ids_list)

        top1_id = int(top_ids_list[0])
        top1_token = top_tokens[0]
        top1_prob = float(top_vals_list[0])

        # Refusal mass
        refusal_mass = 0.0
        best_refusal_prob = 0.0
        best_refusal_rank = None
        best_refusal_id = None
        best_refusal_token = None

        if refusal_token_ids:
            ref_ids = list(refusal_token_ids)
            refusal_mass = float(pos_probs[ref_ids].sum().item())

            for rank, tid in enumerate(top_ids_list, start=1):
                if tid in refusal_token_ids:
                    best_refusal_rank = rank
                    best_refusal_id = int(tid)
                    best_refusal_token = tokenizer.convert_ids_to_tokens([tid])[0]
                    best_refusal_prob = float(pos_probs[tid].item())
                    break

        # Compliance mass
        compliance_mass = 0.0
        best_compliance_prob = 0.0
        best_compliance_rank = None
        best_compliance_id = None
        best_compliance_token = None

        if compliance_token_ids:
            cmp_ids = list(compliance_token_ids)
            compliance_mass = float(pos_probs[cmp_ids].sum().item())

            for rank, tid in enumerate(top_ids_list, start=1):
                if tid in compliance_token_ids:
                    best_compliance_rank = rank
                    best_compliance_id = int(tid)
                    best_compliance_token = tokenizer.convert_ids_to_tokens([tid])[0]
                    best_compliance_prob = float(pos_probs[tid].item())
                    break

        committed_this_step = bool(transfer_index[0, pos].item())

        if committed_this_step:
            committed_id = int(x0[0, pos].item())
            committed_token = tokenizer.convert_ids_to_tokens([committed_id])[0]
            committed_is_refusal = committed_id in refusal_token_ids if refusal_token_ids else False
            committed_is_compliance = (
                committed_id in compliance_token_ids if compliance_token_ids else False
            )
        else:
            committed_id = None
            committed_token = None
            committed_is_refusal = None
            committed_is_compliance = None

        trace_positions.append(
            {
                "pos": int(pos),
                "is_masked_before_step": bool(mask_index[0, pos].item()),
                "topk_ids": top_ids_list,
                "topk_tokens": top_tokens,
                "topk_probs": top_vals_list,
                "top1_id": top1_id,
                "top1_token": top1_token,
                "top1_prob": top1_prob,
                "predicted_id": int(x0[0, pos].item()),
                "predicted_token": tokenizer.convert_ids_to_tokens([int(x0[0, pos].item())])[0],
                "committed_this_step": committed_this_step,
                "committed_id": committed_id,
                "committed_token": committed_token,
                "committed_is_refusal": committed_is_refusal,
                "committed_is_compliance": committed_is_compliance,
                "refusal_mass": refusal_mass,
                "best_refusal_rank": best_refusal_rank,
                "best_refusal_id": best_refusal_id,
                "best_refusal_token": best_refusal_token,
                "best_refusal_prob": best_refusal_prob,
                "is_refusal_top1": bool(top1_id in refusal_token_ids)
                if refusal_token_ids
                else False,
                "compliance_mass": compliance_mass,
                "best_compliance_rank": best_compliance_rank,
                "best_compliance_id": best_compliance_id,
                "best_compliance_token": best_compliance_token,
                "best_compliance_prob": best_compliance_prob,
                "is_compliance_top1": bool(top1_id in compliance_token_ids)
                if compliance_token_ids
                else False,
            }
        )

    return trace_positions


# ===================== RAEC Helper =====================


def _apply_refusal_aware_early_commit(
    *,
    probs: torch.Tensor,  # [B, L, V]
    x: torch.Tensor,  # [B, L]
    x0: torch.Tensor,  # [B, L]
    mask_index: torch.Tensor,  # [B, L]
    transfer_index: torch.Tensor,  # [B, L]
    tokenizer,
    prompt_len: int,
    global_step: int,
    refusal_token_ids: Optional[set],
    compliance_token_ids: Optional[set],
    raec_streak: torch.Tensor,
    raec_early_steps: int,
    raec_early_positions: int,
    raec_refusal_mass_threshold: float,
    raec_margin: float,
    raec_topk: int,
    raec_min_streak: int,
    raec_max_commits_per_sample: int,
    raec_commit_mode: str,
    current_forced_commit_count: int,
):
    """
    Refusal-Aware Early Commitment.

    This method only modifies:
        1. x0
        2. transfer_index
        3. raec_streak
        4. current_forced_commit_count

    It commits a refusal token only if:
        - current global step is inside early window;
        - position is inside early answer positions;
        - current position is still masked;
        - refusal mass exceeds threshold;
        - refusal mass exceeds compliance mass by margin;
        - a refusal token appears in top-k;
        - persistence / streak condition is satisfied.

    Returns:
        x0,
        transfer_index,
        raec_streak,
        new_forced_commit_count,
        events
    """

    events = []

    if refusal_token_ids is None or len(refusal_token_ids) == 0:
        return x0, transfer_index, raec_streak, current_forced_commit_count, events

    if int(global_step) >= int(raec_early_steps):
        return x0, transfer_index, raec_streak, current_forced_commit_count, events

    if int(current_forced_commit_count) >= int(raec_max_commits_per_sample):
        return x0, transfer_index, raec_streak, current_forced_commit_count, events

    if probs.shape[0] != 1:
        raise ValueError("RAEC supports batch size 1 only.")

    device = probs.device

    ref_ids = torch.tensor(
        sorted(list(refusal_token_ids)),
        dtype=torch.long,
        device=device,
    )

    if compliance_token_ids is not None and len(compliance_token_ids) > 0:
        cmp_ids = torch.tensor(
            sorted(list(compliance_token_ids)),
            dtype=torch.long,
            device=device,
        )
    else:
        cmp_ids = None

    gen_start = int(prompt_len)
    gen_end = min(
        int(prompt_len) + int(raec_early_positions),
        x.shape[1],
    )

    if gen_end <= gen_start:
        return x0, transfer_index, raec_streak, current_forced_commit_count, events

    # The public generator validates the single-sample RAEC contract.
    b = 0

    candidate_positions = []
    candidate_scores = []
    candidate_details = []

    for pos in range(gen_start, gen_end):
        # Only consider positions that are still masked before this step.
        if not bool(mask_index[b, pos].item()):
            raec_streak[b, pos] = 0
            continue

        # If normal decoding already selected this position, do not override it here.
        if bool(transfer_index[b, pos].item()):
            continue

        pos_probs = probs[b, pos]

        refusal_mass = float(pos_probs[ref_ids].sum().item())

        if cmp_ids is not None:
            compliance_mass = float(pos_probs[cmp_ids].sum().item())
        else:
            compliance_mass = 0.0

        k = min(int(raec_topk), pos_probs.shape[-1])
        top_vals, top_ids = torch.topk(pos_probs, k=k)
        top_ids_list = [int(t) for t in top_ids.tolist()]

        refusal_in_topk = any(tid in refusal_token_ids for tid in top_ids_list)
        top1_is_refusal = top_ids_list[0] in refusal_token_ids

        trigger = (
            refusal_mass >= float(raec_refusal_mass_threshold)
            and refusal_mass >= compliance_mass + float(raec_margin)
            and refusal_in_topk
        )

        if str(raec_commit_mode) == "top1_if_refusal":
            trigger = trigger and top1_is_refusal

        if trigger:
            raec_streak[b, pos] += 1
        else:
            raec_streak[b, pos] = 0

        if int(raec_streak[b, pos].item()) >= int(raec_min_streak):
            score = refusal_mass - compliance_mass
            candidate_positions.append(pos)
            candidate_scores.append(score)
            candidate_details.append(
                {
                    "refusal_mass": refusal_mass,
                    "compliance_mass": compliance_mass,
                    "top1_id": int(top_ids_list[0]),
                    "top1_token": tokenizer.convert_ids_to_tokens([int(top_ids_list[0])])[0],
                    "streak": int(raec_streak[b, pos].item()),
                }
            )

    if not candidate_positions:
        return x0, transfer_index, raec_streak, current_forced_commit_count, events

    remaining_budget = int(raec_max_commits_per_sample) - int(current_forced_commit_count)
    if remaining_budget <= 0:
        return x0, transfer_index, raec_streak, current_forced_commit_count, events

    # Prefer positions with strongest refusal advantage.
    order = sorted(
        range(len(candidate_positions)),
        key=lambda i: candidate_scores[i],
        reverse=True,
    )

    selected = order[:remaining_budget]

    for idx in selected:
        pos = candidate_positions[idx]
        pos_probs = probs[b, pos]

        if str(raec_commit_mode) == "top1_if_refusal":
            commit_id = int(torch.argmax(pos_probs).item())
        else:
            # Commit the most probable refusal token.
            ref_probs = pos_probs[ref_ids]
            best_local = int(torch.argmax(ref_probs).item())
            commit_id = int(ref_ids[best_local].item())

        x0[b, pos] = commit_id
        transfer_index[b, pos] = True
        current_forced_commit_count += 1

        try:
            commit_token = tokenizer.convert_ids_to_tokens([commit_id])[0]
        except Exception:
            commit_token = str(commit_id)

        event = {
            "global_step": int(global_step),
            "pos": int(pos),
            "relative_answer_pos": int(pos - prompt_len),
            "commit_id": int(commit_id),
            "commit_token": commit_token,
            "score": float(candidate_scores[idx]),
            "mode": str(raec_commit_mode),
        }

        event.update(candidate_details[idx])
        events.append(event)

        if int(current_forced_commit_count) >= int(raec_max_commits_per_sample):
            break

    return x0, transfer_index, raec_streak, current_forced_commit_count, events


def _align_sequence_mask(mask, batch_size, sequence_length, device, *, fill, name):
    """Align a two-dimensional mask to the generated sequence length."""
    if mask is None:
        return None
    if mask.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, sequence].")
    mask = mask.to(device=device, dtype=torch.bool)
    if mask.shape[0] != batch_size:
        if name == "protected_index" and mask.shape[0] == 1:
            mask = mask.expand(batch_size, -1)
        else:
            raise ValueError(f"{name} has an incompatible batch dimension.")
    if mask.shape[1] < sequence_length:
        padding = torch.full(
            (batch_size, sequence_length - mask.shape[1]),
            fill,
            device=device,
            dtype=torch.bool,
        )
        mask = torch.cat([mask, padding], dim=1)
    return mask[:, :sequence_length]


def _validate_generation_options(
    prompt,
    steps,
    gen_length,
    block_length,
    temperature,
    cfg_scale,
    remasking,
    random_rate,
    alpha0,
    attack_method,
    return_trace,
    trace_topk,
    trace_positions,
    raec_enable,
    refusal_token_ids,
    raec_early_steps,
    raec_early_positions,
    raec_refusal_mass_threshold,
    raec_margin,
    raec_topk,
    raec_min_streak,
    raec_max_commits_per_sample,
    raec_commit_mode,
):
    if prompt.ndim != 2 or prompt.shape[0] < 1 or prompt.shape[1] < 1:
        raise ValueError("prompt must have nonempty shape [batch, sequence].")
    if prompt.dtype != torch.long:
        raise ValueError("prompt token IDs must have dtype torch.long.")
    for name, value, minimum in (
        ("steps", steps, 1),
        ("gen_length", gen_length, 0),
        ("block_length", block_length, 1),
        ("trace_topk", trace_topk, 1),
        ("trace_positions", trace_positions, 0),
        ("raec_early_steps", raec_early_steps, 0),
        ("raec_early_positions", raec_early_positions, 0),
        ("raec_topk", raec_topk, 1),
        ("raec_min_streak", raec_min_streak, 1),
        ("raec_max_commits_per_sample", raec_max_commits_per_sample, 0),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if gen_length and gen_length % block_length:
        raise ValueError("gen_length must be divisible by block_length.")
    for name, value in (
        ("temperature", temperature),
        ("cfg_scale", cfg_scale),
        ("random_rate", random_rate),
        ("alpha0", alpha0),
        ("raec_refusal_mass_threshold", raec_refusal_mass_threshold),
        ("raec_margin", raec_margin),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if temperature < 0 or cfg_scale < 0:
        raise ValueError("temperature and cfg_scale must be nonnegative.")
    if not 0 <= random_rate <= 1:
        raise ValueError("random_rate must be between 0 and 1.")
    if not 0 <= raec_refusal_mass_threshold <= 1 or raec_margin < 0:
        raise ValueError("RAEC requires a threshold in [0, 1] and a nonnegative margin.")
    if remasking not in {"low_confidence", "random", "rate", "adaptive", "adaptive_step"}:
        raise ValueError(f"Unsupported remasking strategy: {remasking!r}.")
    if str(attack_method).lower() not in {"none", "pad"}:
        raise ValueError("Sampler attack_method must be 'none' or 'pad'.")
    if raec_commit_mode not in {"best_refusal", "top1_if_refusal"}:
        raise ValueError("Unsupported RAEC commit mode.")
    if raec_enable and prompt.shape[0] != 1:
        raise ValueError("RAEC supports batch size 1 only.")
    if raec_enable and not refusal_token_ids:
        raise ValueError("RAEC requires a nonempty refusal_token_ids set.")


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    steps=64,
    gen_length=128,
    block_length=128,
    temperature=0.5,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    random_rate=0.0,
    alpha0: float = 0.3,
    fill_all_masks: bool = False,
    debug_print: bool = False,
    attention_mask: Optional[torch.Tensor] = None,
    attack_method: str = "none",
    pad_anchors: Optional[Iterable[str]] = None,
    pad_positions: Optional[Iterable[int]] = None,
    pad_in_uncond: bool = True,
    protected_index: Optional[torch.Tensor] = None,
    return_trace: bool = False,
    trace_topk: int = 10,
    trace_positions: int = 16,
    refusal_token_ids: Optional[set] = None,
    compliance_token_ids: Optional[set] = None,
    raec_enable: bool = False,
    raec_early_steps: int = 8,
    raec_early_positions: int = 8,
    raec_refusal_mass_threshold: float = 0.00589,
    raec_margin: float = 0.001,
    raec_topk: int = 10,
    raec_min_streak: int = 2,
    raec_max_commits_per_sample: int = 2,
    raec_commit_mode: str = "best_refusal",
):
    """Generate tokens using LLaDA sampling and optional RAEC.

    Existing prompt masks are decoded together with appended masks, matching
    the DIJA infilling path. Otherwise, appended answer blocks are decoded in
    sequence. ``fill_all_masks`` explicitly requests a full-sequence schedule.
    With ``gen_length=0``, existing sequence masks are decoded in place.

    RAEC supports one sample per call. Baseline sampling supports batches;
    position traces describe the first sample. Protected positions are never
    committed by either sampling path. The return value includes the prompt.
    """
    if tokenizer is None:
        raise ValueError("A tokenizer is required for PAD and token traces.")
    _validate_generation_options(
        prompt,
        steps,
        gen_length,
        block_length,
        temperature,
        cfg_scale,
        remasking,
        random_rate,
        alpha0,
        attack_method,
        return_trace,
        trace_topk,
        trace_positions,
        raec_enable,
        refusal_token_ids,
        raec_early_steps,
        raec_early_positions,
        raec_refusal_mass_threshold,
        raec_margin,
        raec_topk,
        raec_min_streak,
        raec_max_commits_per_sample,
        raec_commit_mode,
    )
    if not isinstance(mask_id, int) or mask_id < 0:
        raise ValueError("mask_id must be a nonnegative integer.")

    device = next(model.parameters()).device
    x = prompt.clone().to(device)
    prompt_len = prompt.shape[1]
    if gen_length:
        tail = torch.full((prompt.shape[0], gen_length), mask_id, dtype=torch.long, device=device)
        x = torch.cat([x, tail], dim=1)

    am = _align_sequence_mask(
        attention_mask, x.shape[0], x.shape[1], device, fill=True, name="attention_mask"
    )
    protected = _align_sequence_mask(
        protected_index, x.shape[0], x.shape[1], device, fill=False, name="protected_index"
    )
    prompt_index = x != mask_id
    uncond_prompt_index = prompt_index

    if str(attack_method).lower() == "pad":
        anchors = (
            list(pad_anchors) if pad_anchors is not None else ["Step 1:", "Step 2:", "Step 3:"]
        )
        if pad_positions is None:
            gap = max(int(max(gen_length // (len(anchors) + 1), 1) // 1.5), 1)
            offsets = [(i + 1) * gap for i in range(len(anchors))]
        else:
            offsets = list(pad_positions)
        for relative_pos, text in zip(offsets, anchors):
            ids = tokenizer(text, add_special_tokens=False).input_ids
            start = prompt_len + int(relative_pos)
            end = start + len(ids)
            if start >= prompt_len and end <= x.shape[1]:
                anchor_ids = (
                    torch.tensor(ids, dtype=torch.long, device=device)
                    .unsqueeze(0)
                    .expand(x.shape[0], -1)
                )
                if protected is None:
                    x[:, start:end] = anchor_ids
                else:
                    x[:, start:end] = torch.where(
                        protected[:, start:end], x[:, start:end], anchor_ids
                    )
        if not pad_in_uncond:
            uncond_prompt_index = x != mask_id

    prompt_mask_positions = (prompt == mask_id).nonzero(as_tuple=False)
    if fill_all_masks or gen_length == 0:
        blocks = [(0, x.shape[1])]
    elif prompt_mask_positions.numel():
        first_mask = int(prompt_mask_positions[:, 1].min().item())
        blocks = [(first_mask, x.shape[1])]
    else:
        blocks = [
            (prompt_len + offset, prompt_len + offset + block_length)
            for offset in range(0, gen_length, block_length)
        ]
    if steps < len(blocks) or steps % len(blocks):
        raise ValueError("steps must be divisible by the number of decoding blocks.")
    steps_per_block = steps // len(blocks)
    trace = [] if return_trace else None
    streak = torch.zeros_like(x, dtype=torch.long) if raec_enable else None
    forced_commit_count = 0
    token_sets_checked = False
    track_positions = list(range(prompt_len, min(prompt_len + trace_positions, x.shape[1])))

    for block_id, (block_start, block_end) in enumerate(blocks):
        editable = torch.zeros_like(x, dtype=torch.bool)
        editable[:, block_start:block_end] = True
        if protected is not None:
            editable &= ~protected
        schedule = get_num_transfer_tokens((x == mask_id) & editable, steps_per_block)

        for step in range(steps_per_block):
            global_step = block_id * steps_per_block + step
            mask_index = x == mask_id
            if cfg_scale > 0:
                un_x = x.clone()
                un_x[uncond_prompt_index] = mask_id
                model_input = torch.cat([x, un_x], dim=0)
                model_am = None if am is None else torch.cat([am, am], dim=0)
                output = model(
                    model_input,
                    attention_mask=model_am,
                    output_hidden_states=False,
                    return_dict=True,
                )
                logits, un_logits = torch.chunk(output.logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(
                    x, attention_mask=am, output_hidden_states=False, return_dict=True
                ).logits

            if not token_sets_checked:
                for name, ids in (
                    ("refusal_token_ids", refusal_token_ids),
                    ("compliance_token_ids", compliance_token_ids),
                ):
                    if ids and any(
                        not isinstance(tid, int) or tid < 0 or tid >= logits.shape[-1]
                        for tid in ids
                    ):
                        raise ValueError(f"{name} contains an invalid vocabulary index.")
                token_sets_checked = True

            x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
            probs = F.softmax(logits, dim=-1)
            model_confidence = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            random_confidence = torch.rand(x0.shape, device=device)
            if remasking == "low_confidence":
                confidence = model_confidence
            elif remasking == "random":
                confidence = random_confidence
            elif remasking == "rate":
                confidence = (1 - random_rate) * model_confidence + random_rate * random_confidence
            else:
                fraction = 1.0
                if remasking == "adaptive_step" and steps_per_block > 1:
                    fraction = 1 - step / (steps_per_block - 1)
                alpha = torch.tensor(
                    alpha0 * fraction, device=device, dtype=model_confidence.dtype
                ).clamp(0, 1)
                confidence = (1 - alpha) * model_confidence + alpha * random_confidence

            eligible = mask_index & editable
            confidence = confidence.masked_fill(~eligible, -float("inf"))
            x0 = torch.where(mask_index, x0, x)
            transfer_index = torch.zeros_like(x, dtype=torch.bool)
            for batch_id in range(x.shape[0]):
                count = min(
                    int(schedule[batch_id, step].item()), int(eligible[batch_id].sum().item())
                )
                if count > 0:
                    selected = torch.topk(confidence[batch_id], k=count).indices
                    transfer_index[batch_id, selected] = True

            step_events = []
            if raec_enable:
                x0, transfer_index, streak, forced_commit_count, step_events = (
                    _apply_refusal_aware_early_commit(
                        probs=probs,
                        x=x,
                        x0=x0,
                        mask_index=eligible,
                        transfer_index=transfer_index,
                        tokenizer=tokenizer,
                        prompt_len=prompt_len,
                        global_step=global_step,
                        refusal_token_ids=refusal_token_ids,
                        compliance_token_ids=compliance_token_ids,
                        raec_streak=streak,
                        raec_early_steps=raec_early_steps,
                        raec_early_positions=raec_early_positions,
                        raec_refusal_mass_threshold=raec_refusal_mass_threshold,
                        raec_margin=raec_margin,
                        raec_topk=raec_topk,
                        raec_min_streak=raec_min_streak,
                        raec_max_commits_per_sample=raec_max_commits_per_sample,
                        raec_commit_mode=raec_commit_mode,
                        current_forced_commit_count=forced_commit_count,
                    )
                )

            if return_trace:
                committed_positions = transfer_index[0].nonzero(as_tuple=False).squeeze(1).tolist()
                trace.append(
                    {
                        "block_id": block_id,
                        "step_in_block": step,
                        "global_step": global_step,
                        "block_start": block_start,
                        "block_end": block_end,
                        "num_masked_before_step": int(mask_index.sum().item()),
                        "committed_positions": committed_positions,
                        "committed_tokens": tokenizer.convert_ids_to_tokens(
                            [int(x0[0, pos].item()) for pos in committed_positions]
                        ),
                        "raec_enabled": bool(raec_enable),
                        "raec_forced_commit_count_so_far": forced_commit_count,
                        "raec_events": step_events,
                        "positions": _collect_position_trace(
                            probs=probs,
                            x0=x0,
                            x=x,
                            mask_index=mask_index,
                            transfer_index=transfer_index,
                            tokenizer=tokenizer,
                            track_positions=track_positions,
                            topk=trace_topk,
                            refusal_token_ids=refusal_token_ids,
                            compliance_token_ids=compliance_token_ids,
                        ),
                    }
                )
            x[transfer_index] = x0[transfer_index]
        if debug_print:
            print(
                f"[Block {block_id + 1}/{len(blocks)}] RAEC commitments: {forced_commit_count}",
                flush=True,
            )

    return (x, trace) if return_trace else x


@torch.no_grad()
def generate_llada(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    model,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    mask_id: int = 126336,
    *,
    tokenizer=None,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    random_rate: float = 0.0,
    alpha0: float = 0.3,
    fill_all_masks: bool = False,
    debug_print: bool = False,
    attack_method: str = "none",
    pad_anchors: Optional[Iterable[str]] = None,
    pad_positions: Optional[Iterable[int]] = None,
    pad_in_uncond: bool = True,
    protected_index: Optional[torch.Tensor] = None,
    return_trace: bool = False,
    trace_topk: int = 10,
    trace_positions: int = 16,
    refusal_token_ids: Optional[set] = None,
    compliance_token_ids: Optional[set] = None,
    raec_enable: bool = False,
    raec_early_steps: int = 8,
    raec_early_positions: int = 8,
    raec_refusal_mass_threshold: float = 0.00589,
    raec_margin: float = 0.001,
    raec_topk: int = 10,
    raec_min_streak: int = 2,
    raec_max_commits_per_sample: int = 2,
    raec_commit_mode: str = "best_refusal",
):
    """Call :func:`generate` using the evaluation runner's argument names."""
    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=input_ids,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        mask_id=mask_id,
        cfg_scale=cfg_scale,
        remasking=remasking,
        random_rate=random_rate,
        alpha0=alpha0,
        fill_all_masks=fill_all_masks,
        debug_print=debug_print,
        attention_mask=attention_mask,
        attack_method=attack_method,
        pad_anchors=pad_anchors,
        pad_positions=pad_positions,
        pad_in_uncond=pad_in_uncond,
        protected_index=protected_index,
        return_trace=return_trace,
        trace_topk=trace_topk,
        trace_positions=trace_positions,
        refusal_token_ids=refusal_token_ids,
        compliance_token_ids=compliance_token_ids,
        raec_enable=raec_enable,
        raec_early_steps=raec_early_steps,
        raec_early_positions=raec_early_positions,
        raec_refusal_mass_threshold=raec_refusal_mass_threshold,
        raec_margin=raec_margin,
        raec_topk=raec_topk,
        raec_min_streak=raec_min_streak,
        raec_max_commits_per_sample=raec_max_commits_per_sample,
        raec_commit_mode=raec_commit_mode,
    )
