# G0 Qwen3-4B Freeze Manifest

Date: 2026-07-29 KST

Purpose: make the 2026-07-26 G0 Qwen3-4B result easy to verify, commit, tag, and
package without overstating the claim.

## Baseline State

- Pre-freeze git revision recorded in result metadata:
  `e5bf88ec530e32268c514665cf865ab56e4c18b0`
- Current working tree includes code, docs, and result artifacts that should be
  committed before a public alpha tag.
- Recommended tag name after commit: `g0-qwen3-4b-alpha`

## Evidence Bundle

| File | SHA256 |
|---|---|
| `docs/env-verify/g0-qwen3-4b-summary.md` | `c72eea0f808c294355433d6334b4ce2facf0795b757cb6b570d2e60fe728424c` |
| `docs/env-verify/g0-qwen3-4b-faithbench-result.json` | `d11cb2537dfe6c11e03de04ec5e0960bd3bce886dfba0b5c8c7a029ca03186c8` |
| `docs/env-verify/g0-qwen3-4b-faithbench-result-transcript.jsonl` | `27b459fcd56395c3a2e2a0519eeac0b815f5bc9c7c8701192e8a4785cb860ce1` |
| `docs/env-verify/g0-qwen3-4b-partial-result.json` | `185761fc5a8865bc71da9ec5213dcbf1ae37fda15ebe355a98b58fb844221aed` |
| `docs/public/g0-qwen3-4b-curated-holdout-report.md` | `bb757eb3bd2182d366d42f420af846b2cd309e6975b39b831aab6d81a4e5551e` |

## Input Bundle

| File | SHA256 |
|---|---|
| `data/processed/laws.json` | `c25df3c753bfa177058cb9a8656e89aa0331a9c8ead8cf5a05c2b940a3808f30` |
| `eval/questions.laws.curated.json` | `9c69fccc5f243e8e2017b778a71cc3e7b364be1808839fc3c5167b8d180313a8` |
| `eval/questions.partial.laws.curated.json` | `ad69dffa14a06ee2e1825b7d8c381679b65806b6960de732188754b20a46873b` |
| `eval/questions.unanswerable.laws.curated.json` | `e2220e51577f77de6ad53cd941b85d8ab6e5f3059dd3a9c0a4d16744ac8dbf91` |
| `configs/train_law_curated_holdout_qwen3_4b.yaml` | `b629d328cd07933040089c4b92028d2a73bc01b5f6d8ae56d8f0d07095dc7a64` |

## Local Generated Artifacts

These should not be committed unless a release policy changes:

- `checkpoints/g0-law-curated-holdout-qwen3-4b/lora_adapter`
- `checkpoints/g0-law-curated-holdout-qwen3-4b/checkpoint-*`
- `checkpoints/resume-smoke/checkpoint-*`

The G0 adapter SHA256 is recorded in the summary and result JSON:

- `720ee4cfbb2c10180d05f234aa45b7b108ee4296eb8af8df9209f4a60e36aae0`

## Commit Units

Recommended order:

1. `train: add resumable lora checkpoints`
   - `scripts/02_train_sft.py`
   - `tests/test_train_resume.py`
   - `scripts/smoke.py`
   - `configs/train_resume_smoke.yaml`
   - `data/processed/resume_smoke_demo.jsonl`

2. `eval: record g0 qwen3 4b results`
   - `docs/env-verify/g0-qwen3-4b-*.json`
   - `docs/env-verify/g0-qwen3-4b-*.jsonl`
   - `docs/env-verify/g0-qwen3-4b-*.log`
   - `docs/env-verify/g0-qwen3-4b-summary.md`
   - `configs/train_law_curated_holdout_qwen3_4b.yaml`
   - `scripts/train/run_g0_faithbench.py`
   - `scripts/train/run_g0_partial.py`

3. `docs: publish g0 alpha evidence package`
   - `README.md`
   - `docs/public/g0-qwen3-4b-curated-holdout-report.md`
   - `docs/public/citation-fingerprint.md`
   - `docs/env-verify/g0-qwen3-4b-freeze-manifest.md`
   - `docs/strategy.md`
   - `docs/next-runbook.md`

Keep `docs/env-verify/perf.json` in a separate commit if it reflects machine-local environment
verification rather than the G0 evidence bundle.

## Verification Commands

```powershell
.venv\Scripts\python.exe -m compileall -q scripts tests
.venv\Scripts\python.exe tests\test_train_resume.py
.venv\Scripts\python.exe scripts\smoke.py --skip-eval-demos
git diff --check
```

Latest observed status:

- compileall: PASS
- resume tests: 3/3 PASS
- smoke: PASS
- diff check: PASS, with CRLF conversion warnings only

## Public Claim Text

Use:

> G0 alpha shows that a Qwen3-4B LoRA grounder can outperform a larger generic
> Qwen2.5-7B baseline on deterministic Korean statutory citation grounding,
> refusal, and tight-span extraction in a curated 5-law closed-corpus holdout.

Avoid:

- "The small model is better than large legal models."
- "This is a complete Korean legal benchmark."
- "This proves legal reasoning quality."
- "This model is ready for legal advice."
