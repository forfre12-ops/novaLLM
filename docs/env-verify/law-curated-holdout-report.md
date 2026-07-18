# Law Curated Holdout Report

실행일: 2026-07-18

## Summary

5개 법령 코퍼스에서 사람이 고른 curated seed를 만들고, 해당 평가 ID를 SFT positive/context 풀에서 제외한
`g0-law-curated-holdout` 어댑터를 학습한 뒤 재평가했다.

결론:

- `selection_exact` 축에서 1.5B FT가 7B base보다 유의하게 높다.
- `partial-span` 축에서도 1.5B FT가 7B base보다 높다.
- unanswerable leak 제어는 FT가 가장 안정적이다.
- 다만 `pilot split` 기준 unseen은 4개뿐이므로, 이것은 정식 최종 G0가 아니라 curated-holdout seed 결과다.

## Curated Eval Files

- answerable: `eval/questions.laws.curated.json` (`30`)
- partial-span: `eval/questions.partial.laws.curated.json` (`30`)
- unanswerable: `eval/questions.unanswerable.laws.curated.json` (`20`)
- builder: `scripts/data/build_curated_law_eval.py`

Builder validation:

```powershell
python scripts/data/build_curated_law_eval.py --corpus data/processed/laws.json
python scripts/eval/faithbench.py --corpus data/processed/laws.json --questions eval/questions.laws.curated.json --unanswerable-file eval/questions.unanswerable.laws.curated.json --dump 1
python scripts/eval/faithbench_partial.py --corpus data/processed/laws.json --items eval/questions.partial.laws.curated.json --dump 1
```

## Holdout Training

```powershell
python scripts/data/gen_law_sft.py `
  --corpus data/processed/laws.json `
  --out data/processed/law_sft_curated_holdout.jsonl `
  --max-full 1200 `
  --max-tight 1200 `
  --refusal-ratio 0.2 `
  --exclude-questions eval/questions.laws.curated.json `
  --exclude-questions eval/questions.partial.laws.curated.json

python scripts/02_train_sft.py --config configs/train_law_curated_holdout.yaml
```

- generated rows: `3000`
- excluded positive/context IDs: `36`
- trained rows after truncation filter: `2954`
- dropped all-masked examples: `46`
- epochs: `2`
- effective batch: `1 x 8 = 8`
- peak VRAM: `5.72 GB`
- adapter: `checkpoints/g0-law-curated-holdout/lora_adapter`

## FaithBench Result

Command:

```powershell
python scripts/train/run_g0_faithbench.py `
  --adapter checkpoints/g0-law-curated-holdout/lora_adapter `
  --corpus data/processed/laws.json `
  --questions eval/questions.laws.curated.json `
  --unanswerable-file eval/questions.unanswerable.laws.curated.json `
  --k 5 `
  --out docs/env-verify/law-curated-holdout-faithbench-result.json
```

| model | selection_exact | gold_recall | distractor_cite_rate | answerable_no_citation_rate | answerable_refused_rate | refusal_rate | leak_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_small_fewshot` | 0.100 | 0.100 | 0.200 | 0.467 | 0.000 | 0.750 | 0.250 |
| `ft_small_zeroshot` | 0.733 | 0.733 | 0.000 | 0.167 | 0.133 | 1.000 | 0.000 |
| `base_large_fewshot` | 0.333 | 0.333 | 0.033 | 0.533 | 0.000 | 0.500 | 0.500 |

Paired exact McNemar:

- FT vs 7B base: `n_pairs=30`, FT-only `15`, base-only `3`, Holm-adjusted `p=0.0075`.
- FT vs 1.5B base: `n_pairs=30`, FT-only `19`, base-only `0`, Holm-adjusted `p=0.0`.

## Partial-Span Result

Command:

```powershell
python scripts/train/run_g0_partial.py `
  --adapter checkpoints/g0-law-curated-holdout/lora_adapter `
  --corpus data/processed/laws.json `
  --items eval/questions.partial.laws.curated.json `
  --k 5 `
  --out docs/env-verify/law-curated-holdout-partial-result.json
```

| model | partial_exact | span_f1 | span_precision | span_recall | selected_gold |
|---|---:|---:|---:|---:|---:|
| `base_small_fewshot` | 0.033 | 0.060 | 0.100 | 0.047 | 0.133 |
| `ft_small_zeroshot` | 0.533 | 0.526 | 0.521 | 0.533 | 0.867 |
| `base_large_fewshot` | 0.333 | 0.413 | 0.446 | 0.414 | 0.500 |

## Interpretation

This is the first result where the earlier "small beats large" hypothesis survives both:

- a hand-selected curated seed,
- and a partial-span precision check,
- while holding out the curated eval IDs from SFT positive/context rows.

The claim should still be scoped carefully. The question set is only `30 + 30 + 20`, and all data comes from 5 laws.
The next public-safe phrasing is:

> A 1.5B Korean law citation grounder can be trained to outperform a 7B base model on a curated closed-set citation
> seed, under deterministic citation scoring, while reducing unanswerable leakage.

Do not call this the final benchmark until the curated set reaches at least `100` answerable + `100` partial items and
multi-law coverage expands beyond the current 5-law corpus.

