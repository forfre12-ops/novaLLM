# G0 Qwen3-4B Alpha

Tag: `g0-qwen3-4b-alpha`

Release type: alpha evidence package

## Summary

This alpha release freezes the first Qwen3-4B G0 evidence package for deterministic
Korean statutory citation grounding.

The result supports a narrow claim: a Qwen3-4B LoRA grounder outperformed a larger
generic Qwen2.5-7B baseline on citation-grounded Korean statutory QA in this curated
5-law closed-corpus holdout. It does not claim broad legal reasoning superiority or
production legal advice readiness.

## Highlights

- Added resumable LoRA checkpoint support for SFT training.
- Recorded Qwen3-4B LoRA G0 curated Korean statutory holdout results.
- Published a public G0 alpha report and freeze manifest with SHA256 evidence hashes.
- Tagged the reproducible evidence state as `g0-qwen3-4b-alpha`.

## Key Metrics

| Metric | Qwen3-4B base | Qwen3-4B + LoRA | Qwen2.5-7B base |
|---|---:|---:|---:|
| FaithBench selection_exact | 0.088 | 0.904 | 0.344 |
| FaithBench leak_rate | 0.000 | 0.000 | 0.300 |
| Partial partial_exact | 0.416 | 0.656 | 0.248 |
| Partial span_f1 | 0.508 | 0.770 | 0.323 |

## Evidence

- Public report: `docs/public/g0-qwen3-4b-curated-holdout-report.md`
- Freeze manifest: `docs/env-verify/g0-qwen3-4b-freeze-manifest.md`
- Summary: `docs/env-verify/g0-qwen3-4b-summary.md`
- FaithBench result: `docs/env-verify/g0-qwen3-4b-faithbench-result.json`
- FaithBench transcript: `docs/env-verify/g0-qwen3-4b-faithbench-result-transcript.jsonl`
- Partial-span result: `docs/env-verify/g0-qwen3-4b-partial-result.json`

## Verification

Observed before publishing:

- `compileall`: PASS
- `tests/test_train_resume.py`: 3/3 PASS
- `scripts/smoke.py --skip-eval-demos`: PASS
- `git diff --check`: PASS

## Limitations

- The corpus is a 5-law closed set, not all Korean law.
- The eval set is still small for public benchmark standard claims.
- The curated set mixes manual and templated rows.
- The result has not yet been externally audited by legal experts.
- Adapter/checkpoint artifacts are local generated outputs and are not included in git.

## Recommended Next Work

1. Expand the human-audited holdout to 300-500 answerable and partial-span items.
2. Export an HRET/HAE-RAE compatible benchmark card.
3. Add one Korean-specialized open model and one 7-14B instruction baseline.
4. Build a product-facing RAG demo that exposes citation IDs, refusal behavior, and leak checks.
