# Citation Fingerprint for Korean Legal LLMs

> A deterministic, LLM-judge-free profile of how Korean LLMs cite provided legal evidence.

## TL;DR

`nova-llm` is not claiming that a small model is generally better than a larger model.

The useful artifact is a **citation fingerprint**: a set of deterministic probes that splits
Korean legal grounding into separate behaviors:

- selecting the right source among distractors
- quoting only text supported by the source
- quoting the tight span requested by the question
- refusing when the answer is not in the provided evidence
- detecting uncited parametric leakage
- separating open-book grounding from closed-book memorization

The first public prototype used a closed-set corpus of 93 entries from the Constitution of Korea.
The current local seed has expanded to 5 statutes and 3303 closed-set entries. Neither path uses
an LLM judge: every citation is checked mechanically against the source text.

## Why This Exists

Most grounding evaluations collapse several behaviors into one score. A model can look good
because it learned an output format, copied a whole paragraph, or memorized a famous legal text.

This project tries to make those failure modes visible. The core claim is:

> Korean legal grounding should be measured as a profile, not as a single leaderboard number.

## Current Prototype

Constitution-only prototype corpus:

- `data/seed/constitution.json`
- Constitution of Korea, Articles 1-39
- 93 closed-set article/paragraph entries

Multi-law local seed corpus:

- `data/processed/laws.json` (ignored generated artifact)
- Constitution, Civil Act, Criminal Act, Personal Information Protection Act, Electronic Financial Transactions Act
- 3303 closed-set article/paragraph entries

Question sets:

- `eval/questions.constitution.json` — 93 content questions with no explicit article ID hints
- `eval/questions.partial.constitution.json` — 14 tight-span questions
- `eval/questions.laws.curated.json` — 100 curated multi-law answerable questions
- `eval/questions.partial.laws.curated.json` — 100 curated multi-law tight-span questions
- `eval/questions.unanswerable.laws.curated.json` — 20 curated out-of-corpus questions

Scorers:

- `citation_verify.py` — verifies `「quoted text」[source ID]` against the closed-set corpus
- `faithbench.py` — evaluates source selection, supported citation, refusal, and leakage
- `faithbench_partial.py` — evaluates character-span precision/recall/F1 for tight quotations
- `faithbench_stats.py` — reports Wilson/Newcombe intervals and paired exact McNemar when transcripts exist

## The Fingerprint

The same three model conditions were tested:

- base small: `Qwen/Qwen2.5-1.5B-Instruct`, few-shot
- fine-tuned small: `Qwen/Qwen2.5-1.5B-Instruct` + local LoRA adapter, zero-shot
- base large: `Qwen/Qwen2.5-7B-Instruct`, few-shot

### 1. Source Selection And Supported Citation

Run:

```powershell
python scripts/train/run_g0_faithbench.py `
  --questions eval/questions.constitution.json `
  --k 5 --near --closed-book `
  --out docs/env-verify/g0-faithbench-v02-result.json
```

Result file:

- `docs/env-verify/g0-faithbench-v02-result.json`
- transcript: `docs/env-verify/g0-faithbench-v02-result-transcript.jsonl`

| Model | selection_exact | faithfulness | leak_rate | answerable_no_citation |
|---|---:|---:|---:|---:|
| base 1.5B few-shot | 0.215 | 0.220 | 0.625 | 0.140 |
| FT 1.5B zero-shot | 0.742 | 0.806 | 0.000 | 0.022 |
| base 7B few-shot | 0.387 | 0.387 | 0.250 | 0.548 |

Paired exact McNemar on the same 93 answerable instances:

| Comparison | FT-only correct | baseline-only correct | Difference | Holm-adjusted |
|---|---:|---:|---:|---|
| FT 1.5B vs base 7B | 40 | 7 | +0.355 | significant |
| FT 1.5B vs base 1.5B | 52 | 3 | +0.527 | significant |

Interpretation:

- The fine-tuned small model is much better at following the benchmark's citation format and selecting a supported source.
- This is useful, but it is not the whole story.

### 2. Closed-Book Memorization Check

The same answerable questions were asked without providing evidence.

| Model | closed-book verbatim recall | open-book selection_exact | grounding gain |
|---|---:|---:|---:|
| base 1.5B | 0.022 | 0.215 | +0.193 |
| FT 1.5B | 0.032 | 0.742 | +0.710 |
| base 7B | 0.043 | 0.387 | +0.344 |

Interpretation:

- The open-book scores are not explained by simple verbatim memorization of the Constitution.
- The fine-tuned model's gain mainly appears when evidence is provided.

### 3. Tight Span Citation

Run:

```powershell
python scripts/train/run_g0_partial.py `
  --k 5 --near `
  --out docs/env-verify/g0-partial-v02-result.json
```

Result file:

- `docs/env-verify/g0-partial-v02-result.json`

| Model | partial_exact | span_f1 | span_precision | span_recall | selected_gold |
|---|---:|---:|---:|---:|---:|
| base 1.5B few-shot | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| FT 1.5B zero-shot | 0.286 | 0.448 | 0.360 | 0.714 | 0.714 |
| base 7B few-shot | 0.429 | 0.394 | 0.393 | 0.405 | 0.571 |

Interpretation:

- The fine-tuned model selects the right source more often and has higher average span F1/recall.
- The 7B base has better precision and higher `partial_exact`.
- Tight-span citation is therefore a **split result**, not a small-model win.

## What We Claim

This prototype supports these claims:

1. A deterministic Korean legal citation scorer can be built without an LLM judge.
2. The scorer can expose different failure modes across models.
3. Fine-tuning strongly improves source-citation behavior on this benchmark.
4. A single "small beats large" headline is not robust enough.

## Multi-Law Seed Update

After the Constitution-only fingerprint, a 5-law 30/30 curated holdout seed was run with the
curated IDs excluded from SFT positive/context rows. This was stronger than the original split
result, but it was still a seed, not the final benchmark.

Result files from that earlier 30/30 seed run:

- `docs/env-verify/law-curated-holdout-report.md`
- `docs/env-verify/law-curated-holdout-faithbench-result.json`
- `docs/env-verify/law-curated-holdout-partial-result.json`

Selection result:

| Model | selection_exact | gold_recall | distractor_cite_rate | refusal_rate | leak_rate |
|---|---:|---:|---:|---:|---:|
| base 1.5B few-shot | 0.100 | 0.100 | 0.200 | 0.750 | 0.250 |
| FT 1.5B zero-shot | 0.733 | 0.733 | 0.000 | 1.000 | 0.000 |
| base 7B few-shot | 0.333 | 0.333 | 0.033 | 0.500 | 0.500 |

Tight-span result:

| Model | partial_exact | span_f1 | span_precision | span_recall | selected_gold |
|---|---:|---:|---:|---:|---:|
| base 1.5B few-shot | 0.033 | 0.060 | 0.100 | 0.047 | 0.133 |
| FT 1.5B zero-shot | 0.533 | 0.526 | 0.521 | 0.533 | 0.867 |
| base 7B few-shot | 0.333 | 0.413 | 0.446 | 0.414 | 0.500 |

Scoped claim:

> A 1.5B Korean law citation grounder outperformed a 7B base model on a curated 5-law
> closed-set seed under deterministic citation scoring.

The tracked eval set has since expanded to 100 answerable and 100 partial-span items. The model
scores above must be rerun on that expanded set, ideally with a 2-4B preregistered grounder,
before they are presented as a final benchmark.

## What We Do Not Claim

This prototype does **not** claim:

- that 1.5B models are generally better than 7B models
- that the benchmark is ready as a public standard
- that the current 5-law seed is enough
- that exact substring citation measures semantic faithfulness completely
- that tight-span behavior is solved

## Reproduce

Environment used:

- Windows 11
- RTX 5070 Ti 16GB, Blackwell sm_120
- PyTorch `2.11.0+cu128`
- bitsandbytes `0.49.2`
- SDPA attention, no `flash-attn`, no `unsloth`

Install:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-core.txt
```

Self-check:

```powershell
python scripts/eval/citation_verify.py --demo
python scripts/eval/faithbench.py --demo
python scripts/eval/faithbench_partial.py --demo
```

Run the fingerprint:

```powershell
python scripts/train/run_g0_faithbench.py `
  --questions eval/questions.constitution.json `
  --k 5 --near --closed-book `
  --out docs/env-verify/g0-faithbench-v02-result.json

python scripts/eval/faithbench_stats.py `
  --result docs/env-verify/g0-faithbench-v02-result.json

python scripts/train/run_g0_partial.py `
  --k 5 --near `
  --out docs/env-verify/g0-partial-v02-result.json
```

## License

- Code: Apache-2.0
- Benchmark question sets and result artifacts: CC BY 4.0
- Korean legal text: public legal text under Korean copyright law

See `LICENSE`, `LICENSE-DATA`, and `NOTICE`.

## Next

The next step is not another local benchmark variant. The current work should harden the
multi-law seed into a publishable evaluation package:

1. Regenerate holdout SFT with the expanded 100/100 eval IDs excluded.
2. Run the preregistered 2-4B grounder condition, not only the 1.5B smoke adapter.
3. Publish the fingerprint as a small reproducible technical note with checksums/provenance.
4. Package the scorer for external evaluation tooling such as HRET.
