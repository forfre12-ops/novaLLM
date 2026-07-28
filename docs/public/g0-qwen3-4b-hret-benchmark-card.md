# HRET/HAE-RAE Benchmark Card: G0 Korean Statutory Citation Grounding

Card status: alpha export draft

Source release: `g0-qwen3-4b-alpha`

Primary report: `docs/public/g0-qwen3-4b-curated-holdout-report.md`

Freeze manifest: `docs/env-verify/g0-qwen3-4b-freeze-manifest.md`

## Intended Use

This card describes the G0 Korean statutory citation grounding task in a form that
can be reviewed for future HRET/HAE-RAE integration.

The benchmark is designed to measure whether a model can answer Korean statutory
questions using only provided legal evidence, cite the correct statutory source,
quote supported text, refuse out-of-corpus questions, and avoid leakage from
parametric memory.

It is not intended to evaluate broad legal reasoning, legal advice quality, or
general Korean language ability.

## Task Identity

- Proposed task name: `korean_statutory_citation_grounding_g0`
- Short name: `kscg_g0`
- Domain: Korean statutory QA
- Language: Korean
- Evaluation style: closed-corpus, evidence-provided, deterministic scoring
- Judge model required: no
- Current maturity: alpha evidence package

## Corpus

- Corpus path: `data/processed/laws.json`
- Corpus SHA256: `c25df3c753bfa177058cb9a8656e89aa0331a9c8ead8cf5a05c2b940a3808f30`
- Corpus size: 3303 entries
- Scope:
  - Constitution
  - Civil Act
  - Criminal Act
  - Personal Information Protection Act
  - Electronic Financial Transactions Act
- Source: normalized Korean statutory text from the National Law Information OpenAPI path
  used by this project
- License note: statutory text is treated as non-copyrightable legal text under the
  project documentation; provenance must remain attached to exported datasets

## Evaluation Files

| Split | File | N | SHA256 |
|---|---|---:|---|
| answerable | `eval/questions.laws.curated.json` | 125 | `9c69fccc5f243e8e2017b778a71cc3e7b364be1808839fc3c5167b8d180313a8` |
| partial-span | `eval/questions.partial.laws.curated.json` | 125 | `ad69dffa14a06ee2e1825b7d8c381679b65806b6960de732188754b20a46873b` |
| unanswerable | `eval/questions.unanswerable.laws.curated.json` | 20 | `e2220e51577f77de6ad53cd941b85d8ab6e5f3059dd3a9c0a4d16744ac8dbf91` |

Composition note:

- answerable: 55 curated rows, 70 auto-templated rows
- partial-span: 55 curated rows, 70 auto-templated rows
- unanswerable: 20 curated out-of-corpus rows
- headline public claims should disclose this composition until a larger human-audited
  holdout is available

## Instance Contract

Each answerable or partial-span instance should expose:

```json
{
  "id": "stable instance id",
  "question": "Korean statutory question",
  "context": [
    {"id": "statutory source id", "text": "source text"}
  ],
  "gold": ["gold statutory source id"],
  "gold_span": "optional exact substring for partial-span scoring",
  "split": "answerable | partial | unanswerable",
  "provenance": {
    "source": "curated | auto",
    "corpus_sha256": "..."
  }
}
```

For unanswerable instances, `gold` is empty and the expected behavior is refusal
without citing unsupported sources or copying unavailable legal content.

## Prompting Contract

The model receives a Korean question and a bounded evidence context. The expected
answer format is citation-bearing Korean text that quotes only supported source
text and includes source IDs.

For unanswerable questions, the model should refuse because the provided evidence
does not contain the answer.

Few-shot and zero-shot conditions must be reported separately.

## Metrics

Core deterministic metrics:

| Metric | Meaning |
|---|---|
| `selection_exact` | exact match between cited source IDs and gold source IDs |
| `gold_recall` | fraction of gold sources recovered in citations |
| `faithfulness_mean` | supported citation behavior under the deterministic scorer |
| `distractor_cite_rate` | rate of citing non-gold distractor sources |
| `answerable_no_citation_rate` | answerable outputs that do not include valid citations |
| `answerable_refused_rate` | answerable outputs that incorrectly refuse |
| `refusal_rate` | unanswerable outputs that refuse |
| `leak_rate` | unanswerable outputs that leak unsupported legal content |
| `partial_exact` | exact tight-span success for partial-span items |
| `span_precision` | character-span precision against expected quote |
| `span_recall` | character-span recall against expected quote |
| `span_f1` | character-span F1 |
| `selected_gold` | partial-span items where the model selected the gold source |

Recommended reporting:

- report answerable, partial-span, and unanswerable axes separately
- report `selection_exact` with `answerable_no_citation_rate`
- report leakage metrics alongside refusal metrics
- avoid single-number leaderboard summaries at alpha stage
- include transcript files for paired analysis

## G0 Alpha Results

FaithBench:

| Model | selection_exact | gold_recall | faithfulness_mean | leak_rate |
|---|---:|---:|---:|---:|
| Qwen3-4B base few-shot | 0.088 | 0.088 | 0.088 | 0.000 |
| Qwen3-4B FT zero-shot | 0.904 | 0.904 | 0.912 | 0.000 |
| Qwen2.5-7B base few-shot | 0.344 | 0.352 | 0.356 | 0.300 |

Partial-span:

| Model | partial_exact | span_f1 | span_precision | span_recall | selected_gold |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B base few-shot | 0.416 | 0.508 | 0.488 | 0.592 | 0.656 |
| Qwen3-4B FT zero-shot | 0.656 | 0.770 | 0.705 | 0.920 | 0.968 |
| Qwen2.5-7B base few-shot | 0.248 | 0.323 | 0.322 | 0.355 | 0.400 |

Unseen-only answerable split, n=23:

| Model | selection_exact | gold_recall | distractor_cite_rate |
|---|---:|---:|---:|
| Qwen3-4B base few-shot | 0.087 | 0.087 | 0.000 |
| Qwen3-4B FT zero-shot | 0.913 | 0.913 | 0.000 |
| Qwen2.5-7B base few-shot | 0.435 | 0.435 | 0.043 |

## Result Artifacts

- `docs/env-verify/g0-qwen3-4b-summary.md`
- `docs/env-verify/g0-qwen3-4b-faithbench-result.json`
- `docs/env-verify/g0-qwen3-4b-faithbench-result-transcript.jsonl`
- `docs/env-verify/g0-qwen3-4b-partial-result.json`
- `docs/env-verify/g0-qwen3-4b-freeze-manifest.md`

## Reproduction Commands

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

```powershell
python scripts/eval/score_predictions.py rescore `
  --transcript docs/env-verify/g0-qwen3-4b-faithbench-result-transcript.jsonl `
  --corpus data/processed/laws.json `
  --expect docs/env-verify/g0-qwen3-4b-faithbench-result.json
```

## HRET Integration Notes

This card intentionally avoids depending on a specific HRET custom-task API shape.
The next integration step should map the instance contract and metrics above to the
current HRET registry/config interface after checking the upstream toolkit.

Suggested package skeleton:

```text
faithbench_legal_ko/
  README.md
  dataset_card.md
  sample.jsonl
  metrics.py
  citation_verify.py
  task_config.yaml
```

## Limitations

- Alpha-scale holdout, not a public benchmark standard.
- Closed 5-law corpus, not all Korean statutes.
- Mixed curated and auto-templated rows.
- No external legal expert audit yet.
- Adapter artifacts are not included in git.
- The HRET mapping has not yet been tested against upstream HRET APIs.

## Release Claim

Use:

> G0 alpha exports a deterministic Korean statutory citation-grounding task card
> for future HRET/HAE-RAE integration, with reproducible Qwen3-4B LoRA evidence
> on a curated 5-law closed-corpus holdout.

Avoid:

- "HRET has adopted this benchmark."
- "This is an official HAE-RAE task."
- "This benchmark proves legal reasoning ability."
- "This dataset is large enough for a public legal leaderboard."
