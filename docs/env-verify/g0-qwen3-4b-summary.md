# G0 Qwen3-4B Curated Holdout Summary

Run date: 2026-07-26 KST

## Environment

- Python: 3.11.9
- Torch: 2.11.0+cu128
- Transformers: 5.14.1
- PEFT: 0.19.1
- bitsandbytes: 0.49.2
- GPU: NVIDIA GeForce RTX 5070 Ti, capability (12, 0)
- T0 torch/bitsandbytes/QLoRA smoke: PASS

## Training

- Base model: Qwen/Qwen3-4B
- Adapter: `checkpoints/g0-law-curated-holdout-qwen3-4b/lora_adapter`
- Data: `data/processed/law_sft_curated_holdout.jsonl`
- Rows: 3000 generated, 2968 used after truncation filtering
- Eval holdout excluded positive IDs: 145
- Steps: 742 optimizer steps, 2 epochs
- Final logged loss region: roughly 0.015-0.024
- Adapter SHA256: `720ee4cfbb2c10180d05f234aa45b7b108ee4296eb8af8df9209f4a60e36aae0`

## Faithbench

Result: `docs/env-verify/g0-qwen3-4b-faithbench-result.json`

| Model | selection_exact | gold_recall | faithfulness_mean | leak_rate |
|---|---:|---:|---:|---:|
| Qwen3-4B base few-shot | 0.088 | 0.088 | 0.088 | 0.000 |
| Qwen3-4B FT zero-shot | 0.904 | 0.904 | 0.912 | 0.000 |
| Qwen2.5-7B base few-shot | 0.344 | 0.352 | 0.356 | 0.300 |

Unseen-only answerable split, n=23:

| Model | selection_exact | gold_recall | distractor_cite_rate |
|---|---:|---:|---:|
| Qwen3-4B base few-shot | 0.087 | 0.087 | 0.000 |
| Qwen3-4B FT zero-shot | 0.913 | 0.913 | 0.000 |
| Qwen2.5-7B base few-shot | 0.435 | 0.435 | 0.043 |

## Partial Span

Result: `docs/env-verify/g0-qwen3-4b-partial-result.json`

| Model | partial_exact | span_f1 | span_precision | span_recall | selected_gold |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B base few-shot | 0.416 | 0.508 | 0.488 | 0.592 | 0.656 |
| Qwen3-4B FT zero-shot | 0.656 | 0.770 | 0.705 | 0.920 | 0.968 |
| Qwen2.5-7B base few-shot | 0.248 | 0.323 | 0.322 | 0.355 | 0.400 |

## Notes

- The first faithbench run completed and saved JSON/transcript, but hit a Windows console encoding error after saving while printing a final em dash line. The evaluator scripts now force UTF-8 stdout/stderr to avoid repeat failures.
- During evaluation, an unrelated Python GPU job (`scripts/eval_real_public_fpr.py`) was also active; it was left untouched.
- The current claim should be framed as citation-grounded statutory QA improvement on the curated holdout, not generic legal reasoning superiority.
