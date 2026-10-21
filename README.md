Code for **Beyond Token Positions: Safety Alignment Across Denoising Steps in Diffusion Language Models** (EMNLP 2026)

RAEC is a training-free decoding method for masked diffusion language models. It preserves refusal evidence that appears early and persists across denoising steps by committing a small number of refusal tokens before later decoding can overwrite that evidence. It uses the model logits already computed for standard decoding and adds no model forward passes. This paper identifies **shallow-step alignment**: both the denoising step of commitment and the response position influence safety.

## Repository layout

```text
models/jailbreakbench_llada_refusal_method.py   # Main generation CLI
utility/generate_function_llada_refusal_method.py
                                              # LLaDA decoder and RAEC
refine_prompt/                                # Prepared benchmark prompts
data/example_prompts.json                     # Benign schema example
data/agnews_test_data.json                    # AGNews evaluation input
find_token_set/                               # Candidate discovery and cleaning
eval_asr/                                     # Guard evaluation and summaries
eval_utility/                                 # AGNews and GSM8K
scripts/run_jailbreakbench_raec.sh             # Portable main experiment command
requirements.txt                              # Inference dependencies
```

## Setup

Use Python 3.10 or newer and a PyTorch installation compatible with your CUDA environment. A CUDA GPU with enough memory for the 8B model and its activations is recommended. Model weights are downloaded separately from Hugging Face or loaded from a local path.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python models/jailbreakbench_llada_refusal_method.py --help
```

Local JSON inference uses the base requirements. Install `requirements-data.txt` for Hugging Face dataset inputs or GSM8K downloads. Optional vLLM judging should use a separate CUDA environment with `requirements-eval.txt`, because vLLM selects its own compatible PyTorch version.

The loader uses Hugging Face `trust_remote_code=True`, as required by the LLaDA model implementation.

## Generate responses with RAEC

Run commands from the repository root. The bundled JBB attack prompts are in `refine_prompt/DIJA_JBB_refined_Qwen.json` (100 records). Use this exact filename for the bundled DIJA-JailbreakBench snapshot.

```bash
python models/jailbreakbench_llada_refusal_method.py \
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
  --raec_commit_mode best_refusal
```

RAEC is enabled by `--raec_enable`. To run a vanilla baseline, use the same model, prompts, and decoding parameters and omit that flag. Save the baseline to a different `--output_json` path.

Validate input selection and output paths without downloading or loading the model:

```bash
python models/jailbreakbench_llada_refusal_method.py \
  --model_path GSAI-ML/LLaDA-8B-Instruct \
  --attack_prompt data/example_prompts.json \
  --output_json outputs/input_preview.json \
  --attack_method zeroshot --mask_counts 0 \
  --max_samples 2 --dry_run
```

## Input format

Local input must be a JSON list of records. A two-prompt record contains the original behavior and, optionally, a prepared attack prompt:

```json
[
  {
    "id": "example-1",
    "vanilla prompt": "Explain why the sky appears blue.",
    "refined prompt": "Explain why the sky appears blue in simple language."
  }
]
```

## RAEC decision rule

For each still-masked response position, RAEC computes the summed probability of refusal tokens (`m_ref`) and compliance tokens (`m_cmp`). A position is eligible only if it is in the early step and response-position windows and is not already selected by the standard commitment rule. It must satisfy all three conditions:

```text
m_ref >= refusal_mass_threshold
m_ref >= m_cmp + margin
at least one refusal token occurs among the top-K vocabulary candidates
```

After these conditions persist for `raec_min_streak` consecutive steps, `best_refusal` commits the most probable token within the refusal set. At most `raec_max_commits_per_sample` additional commitments are made in one response. Standard decoding fills all other selected positions. The optional `top1_if_refusal` mode additionally requires the full-vocabulary top prediction itself to be a refusal token.

| Argument | Meaning | Supplied example | Paper default |
| --- | --- | --- | --- |
| `--raec_early_steps` | First global denoising steps eligible for RAEC | 8 | 8 |
| `--raec_early_positions` | First generated response positions eligible for RAEC | 8 | 8 |
| `--raec_refusal_mass_threshold` | Minimum refusal mass | 0.00589 | 0.00589 for LLaDA |
| `--raec_margin` | Required advantage over compliance mass | 0.001 | 0.001 |
| `--raec_topk` | Candidate-set size | 10 | 10 |
| `--raec_min_streak` | Consecutive eligible steps | 2 | 3 |
| `--raec_max_commits_per_sample` | Additional commitment budget per response | 2 | 2 |
| `--raec_commit_mode` | Token-selection rule | `best_refusal` | Most probable refusal token |

## Outputs and evaluation

The paper evaluates complete responses with Llama-Guard-3-8B on StrongREJECT, JailbreakBench, HarmBench, and their DIJA variants, and measures utility on GSM8K and AGNews. Additional appendix experiments use Beaver-Dam-7B.

## Citation

```bibtex
@article{wang2026beyond,
  title={Beyond Token Positions: Safety Alignment Across Denoising Steps in Diffusion Language Models},
  author={Wang, Guoli and Shi, Haonan and Ouyang, Tu and Wang, An},
  journal={arXiv preprint arXiv:2609.00495},
  year={2026}
}
```