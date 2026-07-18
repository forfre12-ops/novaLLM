# Law Corpus v0.1 Report

실행일: 2026-07-18

## Summary

국가법령정보 OpenAPI(`LAW_API_KEY`)로 5개 핵심 법령을 수집하고 `law-corpus-v0.1` 형식으로
정규화했다. 원문과 병합 코퍼스는 대용량/재생성 가능 산출물이므로 git에는 커밋하지 않는다.

## Laws

| law | law_id | mst | effective_date |
|---|---:|---:|---:|
| 대한민국헌법 | 001444 | 61603 | 19880225 |
| 민법 | 001706 | 284415 | 20260317 |
| 형법 | 001692 | 284025 | 20260312 |
| 개인정보 보호법 | 011357 | 270351 | 20251002 |
| 전자금융거래법 | 010199 | 280277 | 20251216 |

## Artifacts

Local ignored artifacts:

- `data/raw/law_manifest_core.json`
- `data/raw/laws/*.json`
- `data/processed/laws/*.json`
- `data/processed/laws.json`
- `eval/questions.laws.smoke.json`
- `eval/questions.partial.laws.smoke.json`
- `eval/questions.unanswerable.laws.smoke.json`

## Counts

- merged laws: `5`
- merged entries: `3303`
- generated answerable smoke questions: `500`
- generated partial smoke questions: `353`
- generated unanswerable smoke questions: `10`

## Validation

```powershell
python scripts/data/merge_law_corpora.py --in data/processed/laws/*.json --out data/processed/laws.json
python scripts/data/validate_law_corpus.py --corpus data/processed/laws.json --min-entries 200
python scripts/data/gen_law_questions.py --corpus data/processed/laws.json --out eval/questions.laws.smoke.json --partial-out eval/questions.partial.laws.smoke.json --limit 500
python scripts/data/gen_unanswerable_questions.py --corpus data/processed/laws.json --out eval/questions.unanswerable.laws.smoke.json
python scripts/eval/faithbench.py --corpus data/processed/laws.json --questions eval/questions.laws.smoke.json --unanswerable-file eval/questions.unanswerable.laws.smoke.json --dump 1
python scripts/eval/faithbench_partial.py --corpus data/processed/laws.json --items eval/questions.partial.laws.smoke.json --dump 1
```

All commands passed locally.

## Caveat

The generated smoke questions are not a formal benchmark. They prove the data/eval path runs over a
multi-law closed set. Formal G0 still needs curated questions and model reevaluation.
