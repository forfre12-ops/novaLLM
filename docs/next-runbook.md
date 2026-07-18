# Next Runbook

아침에 재개할 때 필요한 최소 실행 순서다. 현재 repo는 키 없이 가능한 parser/fixture/CI scaffolding까지
완료되어 있다.

## 0. Check Clean State

```powershell
git pull --ff-only
git status --short --branch
```

## 1. Local Smoke

```powershell
python -m compileall -q scripts
python scripts/eval/citation_verify.py --demo
python scripts/eval/faithbench.py --demo
python scripts/eval/faithbench_partial.py --demo
python scripts/eval/power_analysis.py --base 0.387 --target 0.742
python scripts/data/fetch_law.py --smoke
python scripts/data/law_corpus.py --demo
python scripts/data/verify_law_corpus.py
python scripts/smoke.py --skip-eval-demos
```

## 2. Add Law API Key

```powershell
copy .env.example .env
notepad .env
```

`.env`에 `LAW_API_KEY`를 넣은 뒤 현재 셸에도 주입한다.

```powershell
$env:LAW_API_KEY="<open.law.go.kr OC>"
```

## 3. Search Candidate Laws

```powershell
python scripts/data/fetch_law.py --query 헌법 --raw-out data/raw/law_search_헌법.json
python scripts/data/fetch_law.py --query 민법 --raw-out data/raw/law_search_민법.json
python scripts/data/fetch_law.py --query 형법 --raw-out data/raw/law_search_형법.json
python scripts/data/fetch_law.py --query 개인정보보호법 --raw-out data/raw/law_search_개인정보보호법.json
python scripts/data/fetch_law.py --query 전자금융거래법 --raw-out data/raw/law_search_전자금융거래법.json
```

## 4. Build Manifest

```powershell
python scripts/data/plan_law_fetch.py `
  --in data/raw/law_search_*.json `
  --out data/raw/law_manifest.json

python scripts/data/plan_law_fetch.py `
  --in data/raw/law_search_*.json `
  --law-id 001444 `
  --law-id 001706 `
  --law-id 001692 `
  --law-id 011357 `
  --law-id 010199 `
  --out data/raw/law_manifest_core.json

python scripts/data/bulk_fetch_laws.py `
  --manifest data/raw/law_manifest_core.json `
  --dry-run
```

`--dry-run` 출력이 괜찮으면 실수집으로 전환한다.

## 5. Bulk Fetch And Normalize

```powershell
python scripts/data/bulk_fetch_laws.py `
  --manifest data/raw/law_manifest.json `
  --raw-dir data/raw/laws `
  --corpus-dir data/processed/laws
```

## 6. Merge Corpus And Generate Smoke Questions

```powershell
python scripts/data/merge_law_corpora.py `
  --in data/processed/laws/*.json `
  --out data/processed/laws.json

python scripts/data/validate_law_corpus.py `
  --corpus data/processed/laws.json

python scripts/data/gen_law_questions.py `
  --corpus data/processed/laws.json `
  --out eval/questions.laws.smoke.json `
  --partial-out eval/questions.partial.laws.smoke.json

python scripts/data/gen_unanswerable_questions.py `
  --corpus data/processed/laws.json `
  --out eval/questions.unanswerable.laws.smoke.json
```

## 7. Verify Eval Builders

```powershell
python scripts/eval/faithbench.py `
  --corpus data/processed/laws.json `
  --questions eval/questions.laws.smoke.json `
  --unanswerable-file eval/questions.unanswerable.laws.smoke.json `
  --dump 2

python scripts/eval/faithbench_partial.py `
  --corpus data/processed/laws.json `
  --items eval/questions.partial.laws.smoke.json `
  --dump 2
```

## 8. Generate SFT Smoke Data

```powershell
python scripts/data/gen_law_sft.py `
  --corpus data/processed/laws.json `
  --out data/processed/law_sft.jsonl `
  --max-full 1200 `
  --max-tight 1200 `
  --refusal-ratio 0.2
```

Optional GPU smoke training:

```powershell
python scripts/02_train_sft.py --config configs/train_law_sft.yaml
```

## 9. Next Decision

- If corpus parsing fails: patch `scripts/data/law_corpus.py` against the saved raw response.
- If corpus parses but questions are weak: keep smoke only, then author curated questions.
- If smoke passes: run GPU evaluation with `scripts/train/run_g0_faithbench.py --corpus data/processed/laws.json`.
- If SFT smoke data validates: train `configs/train_law_sft.yaml`, then rerun the law runner smoke.

Do not market a model win from smoke questions. Smoke questions only prove the pipeline runs.
