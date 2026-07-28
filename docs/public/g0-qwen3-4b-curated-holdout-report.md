# G0 Qwen3-4B Curated Holdout Report

Run date: 2026-07-26 KST

Status: reproducible local G0 result, suitable for an alpha evidence package.

## Claim Boundary

This run supports a narrow claim:

> A Qwen3-4B LoRA model trained on the project recipe strongly outperformed a larger generic
> Qwen2.5-7B baseline on citation-grounded Korean statutory QA in this curated closed-corpus holdout.

It does not claim broad legal reasoning superiority, production legal advice quality, or benchmark
standard status. The useful asset is the deterministic measurement surface: source selection,
supported citation, refusal behavior, leakage checks, and tight-span extraction.

## Setup

- Corpus: `data/processed/laws.json`
- Corpus scope: Constitution, Civil Act, Criminal Act, Personal Information Protection Act,
  Electronic Financial Transactions Act
- Closed-set corpus size: 3303 law entries
- Answerable eval: `eval/questions.laws.curated.json`, n=125
- Partial-span eval: `eval/questions.partial.laws.curated.json`, n=125
- Unanswerable eval: `eval/questions.unanswerable.laws.curated.json`, n=20
- Holdout positive IDs excluded from training: 145
- Training rows generated: 3000
- Training rows used after truncation filtering: 2968
- Seed: 3407

## Models

| Name | Condition |
|---|---|
| `base_small_fewshot` | `Qwen/Qwen3-4B`, few-shot |
| `ft_small_zeroshot` | `Qwen/Qwen3-4B` + LoRA adapter, zero-shot |
| `base_large_fewshot` | `Qwen/Qwen2.5-7B-Instruct`, few-shot |

Adapter:

- Path: `checkpoints/g0-law-curated-holdout-qwen3-4b/lora_adapter`
- SHA256: `720ee4cfbb2c10180d05f234aa45b7b108ee4296eb8af8df9209f4a60e36aae0`

## FaithBench Result

Source:

- `docs/env-verify/g0-qwen3-4b-faithbench-result.json`
- Transcript: `docs/env-verify/g0-qwen3-4b-faithbench-result-transcript.jsonl`
- Freeze manifest: `docs/env-verify/g0-qwen3-4b-freeze-manifest.md`

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

## Partial-Span Result

Source:

- `docs/env-verify/g0-qwen3-4b-partial-result.json`

| Model | partial_exact | span_f1 | span_precision | span_recall | selected_gold |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B base few-shot | 0.416 | 0.508 | 0.488 | 0.592 | 0.656 |
| Qwen3-4B FT zero-shot | 0.656 | 0.770 | 0.705 | 0.920 | 0.968 |
| Qwen2.5-7B base few-shot | 0.248 | 0.323 | 0.322 | 0.355 | 0.400 |

## Interpretation

The strongest result is not "small beats large" in general. It is that a small specialized
grounding model can dominate a larger generic baseline on this project's deterministic statutory
citation tasks:

- It selected the exact gold source much more often: 0.904 vs 0.344.
- It refused all unanswerable items without leakage in this run: leak_rate 0.000 vs 0.300.
- It also improved tight-span extraction: partial_exact 0.656 vs 0.248, span_f1 0.770 vs 0.323.
- The unseen answerable split stayed strong: selection_exact 0.913 on n=23.

This is commercially useful because it supports a positioning shift from "build a generally better
legal LLM" to "own the Korean legal grounding measurement and verification layer."

## Reproduce

Environment check:

```powershell
python scripts/smoke.py --skip-eval-demos
```

Training:

```powershell
python scripts/02_train_sft.py --config configs/train_law_curated_holdout_qwen3_4b.yaml
```

Resume from the newest saved training checkpoint:

```powershell
python scripts/02_train_sft.py `
  --config configs/train_law_curated_holdout_qwen3_4b.yaml `
  --resume-from latest
```

FaithBench evaluation:

```powershell
python scripts/train/run_g0_faithbench.py `
  --small Qwen/Qwen3-4B `
  --large Qwen/Qwen2.5-7B-Instruct `
  --adapter checkpoints/g0-law-curated-holdout-qwen3-4b/lora_adapter `
  --corpus data/processed/laws.json `
  --questions eval/questions.laws.curated.json `
  --unanswerable-file eval/questions.unanswerable.laws.curated.json `
  --k 5 `
  --out docs/env-verify/g0-qwen3-4b-faithbench-result.json
```

Partial-span evaluation:

```powershell
python scripts/train/run_g0_partial.py `
  --small Qwen/Qwen3-4B `
  --large Qwen/Qwen2.5-7B-Instruct `
  --adapter checkpoints/g0-law-curated-holdout-qwen3-4b/lora_adapter `
  --corpus data/processed/laws.json `
  --items eval/questions.partial.laws.curated.json `
  --k 5 `
  --out docs/env-verify/g0-qwen3-4b-partial-result.json
```

Transcript rescore verification:

```powershell
python scripts/eval/score_predictions.py rescore `
  --transcript docs/env-verify/g0-qwen3-4b-faithbench-result-transcript.jsonl `
  --corpus data/processed/laws.json `
  --expect docs/env-verify/g0-qwen3-4b-faithbench-result.json
```

## Limitations

- The corpus is a 5-law closed set, not all Korean law.
- The evaluation set is still small for public benchmark claims.
- The curated set mixes manual and templated rows; headline public claims should continue to
  disclose composition and eventually move to a larger human-audited holdout.
- The result has not yet been audited by external legal experts.
- Adapter/checkpoint artifacts are local generated outputs and are not packaged in git.

## Next Work

1. Freeze the G0 evidence bundle and tag the code state used to reproduce it.
2. Expand the human-audited holdout to 300-500 answerable and partial-span items.
3. Export an HRET/HAE-RAE compatible benchmark card.
4. Build a small RAG demo that exposes citation IDs and refusal behavior as product-facing signals.
5. Run a 7-14B instruction baseline and one Korean-specialized open model for stronger public comparison.
