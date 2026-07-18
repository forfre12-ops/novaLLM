# G0 Curated-Holdout Update

Date: 2026-07-18

This note updates the earlier `G0 = SPLIT` judgment with the new curated-holdout seed result.
It does not replace the final G0 verdict yet.

## Setup

- corpus: `data/processed/laws.json` (`5` laws, `3303` entries)
- answerable: `eval/questions.laws.curated.json` (`30`)
- partial-span: `eval/questions.partial.laws.curated.json` (`30`)
- unanswerable: `eval/questions.unanswerable.laws.curated.json` (`20`)
- held-out eval IDs from SFT positive/context rows: `36`
- adapter: `checkpoints/g0-law-curated-holdout/lora_adapter`
- scorer: `citation_verify v0.2`

## FaithBench

| model | selection_exact | gold_recall | distractor_cite_rate | refusal_rate | leak_rate |
|---|---:|---:|---:|---:|---:|
| `base_small_fewshot` | 0.100 | 0.100 | 0.200 | 0.750 | 0.250 |
| `ft_small_zeroshot` | 0.733 | 0.733 | 0.000 | 1.000 | 0.000 |
| `base_large_fewshot` | 0.333 | 0.333 | 0.033 | 0.500 | 0.500 |

Paired exact McNemar:

- FT vs 7B base: FT-only `15`, base-only `3`, Holm-adjusted `p=0.0075`.
- FT vs 1.5B base: FT-only `19`, base-only `0`, Holm-adjusted `p=0.0`.

## Partial-Span

| model | partial_exact | span_f1 | span_precision | span_recall | selected_gold |
|---|---:|---:|---:|---:|---:|
| `base_small_fewshot` | 0.033 | 0.060 | 0.100 | 0.047 | 0.133 |
| `ft_small_zeroshot` | 0.533 | 0.526 | 0.521 | 0.533 | 0.867 |
| `base_large_fewshot` | 0.333 | 0.413 | 0.446 | 0.414 | 0.500 |

## Interpretation

This is stronger than the earlier split signal: the small FT now beats the 7B base on both selection and
partial-span axes while also reducing unanswerable leakage.

Keep the claim scoped:

> A 1.5B Korean law citation grounder outperformed a 7B base model on a curated 5-law closed-set seed under
> deterministic citation scoring.

Do not present this as the final benchmark until the curated set expands to at least `100` answerable and `100`
partial-span items, with broader law coverage.

