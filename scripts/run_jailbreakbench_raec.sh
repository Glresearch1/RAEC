#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
raec_python="${RAEC_PYTHON:-python3}"

"$raec_python" models/jailbreakbench_llada_refusal_method.py \
  --model_path GSAI-ML/LLaDA-8B-Instruct \
  --attack_prompt refine_prompt/DIJA_JBB_refined_Qwen.json \
  --output_json outputs/jailbreakbench_raec.json \
  --attack_method DIJA \
  --steps 64 \
  --gen_length 128 \
  --block_length 128 \
  --temperature 0.5 \
  --cfg_scale 0.1 \
  --remasking low_confidence \
  --mask_counts 0 \
  --raec_enable \
  --raec_early_steps 8 \
  --raec_early_positions 8 \
  --raec_refusal_mass_threshold 0.00589 \
  --raec_margin 0.001 \
  --raec_topk 10 \
  --raec_min_streak 2 \
  --raec_max_commits_per_sample 2 \
  --raec_commit_mode best_refusal \
  "$@"
