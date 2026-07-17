# Data Pipeline

이 문서는 국가법령정보 OpenAPI 키가 들어온 뒤 다법령 closed-set 코퍼스를 만드는 경로를 고정한다.
키가 없어도 fixture 기반 검증은 항상 돌아야 한다.

## Current Gate

- `LAW_API_KEY` 없음: URL 구성, parser fixture, scorer smoke까지만 실행한다.
- `LAW_API_KEY` 있음: search → lawService → normalized corpus → merged corpus → smoke questions 순서로 진행한다.

## 1. Smoke Without Key

```powershell
python scripts/data/fetch_law.py --smoke
python scripts/data/law_corpus.py --demo
python scripts/data/verify_law_corpus.py
```

## 2. Search Laws

```powershell
$env:LAW_API_KEY="your-open-law-oc"
python scripts/data/fetch_law.py `
  --query 헌법 `
  --raw-out data/raw/law_search_헌법.json
```

검색 결과에서 `MST` 또는 `ID`를 확인한 뒤 본문을 받는다.

여러 검색 결과를 bulk fetch 대상으로 만들려면 manifest를 생성한다.

```powershell
python scripts/data/plan_law_fetch.py `
  --in data/raw/law_search_*.json `
  --out data/raw/law_manifest.json
```

## 3. Fetch And Normalize One Law

```powershell
python scripts/data/fetch_law.py `
  --mst <MST> `
  --raw-out data/raw/law_service_헌법.json `
  --corpus-out data/processed/law_corpus_헌법.json
```

본문 조회는 `--response-type JSON`이 기본이다. API 응답이 JSON으로 불안정하면 XML로 바꿔도
동일한 정규화 파서를 탄다.

```powershell
python scripts/data/fetch_law.py `
  --mst <MST> `
  --response-type XML `
  --raw-out data/raw/law_service_헌법.xml `
  --corpus-out data/processed/law_corpus_헌법.json
```

정규화 결과는 scorer가 바로 읽는 `articles` mapping과 provenance용 `entries`를 함께 가진다.

```json
{
  "schema_version": "law-corpus-v0.1",
  "closed_set": true,
  "articles": {
    "예시법 제2조 ①": "인용문은 원문 그대로 옮긴다."
  },
  "entries": [
    {
      "id": "예시법 제2조 ①",
      "law_name": "예시법",
      "law_id": "DEMO",
      "article_number": "제2조",
      "paragraph_number": "①",
      "text": "인용문은 원문 그대로 옮긴다.",
      "effective_date": "20260718",
      "source_url": "..."
    }
  ]
}
```

## 4. Merge Multiple Laws

manifest가 있으면 여러 법령을 한 번에 원문/정규화 코퍼스로 저장할 수 있다.

```powershell
# 키 없이 경로와 URL만 확인
python scripts/data/bulk_fetch_laws.py `
  --manifest data/raw/law_manifest.json `
  --dry-run

# 키가 있으면 실수집
python scripts/data/bulk_fetch_laws.py `
  --manifest data/raw/law_manifest.json `
  --raw-dir data/raw/laws `
  --corpus-dir data/processed/laws
```

필요하면 bulk도 XML fallback을 사용한다.

```powershell
python scripts/data/bulk_fetch_laws.py `
  --manifest data/raw/law_manifest.json `
  --response-type XML `
  --raw-dir data/raw/laws `
  --corpus-dir data/processed/laws
```

```powershell
python scripts/data/merge_law_corpora.py `
  --in data/processed/laws/*.json `
  --out data/processed/laws.json

python scripts/data/validate_law_corpus.py `
  --corpus data/processed/laws.json `
  --min-entries 200
```

ID 충돌이 있고 본문이 다르면 병합은 실패한다. 같은 ID/같은 본문은 중복으로 허용하지 않는다.

## 5. Generate Smoke Questions

```powershell
python scripts/data/gen_law_questions.py `
  --corpus data/processed/laws.json `
  --out eval/questions.laws.smoke.json `
  --partial-out eval/questions.partial.laws.smoke.json

python scripts/data/gen_unanswerable_questions.py `
  --corpus data/processed/laws.json `
  --out eval/questions.unanswerable.laws.smoke.json
```

주의: 이 질문셋은 정식 벤치가 아니라 수집 직후 runner를 태우기 위한 deterministic smoke artifact다.
정식 G0 재판정에는 별도 curated 질문셋과 검정력 산정이 필요하다.

## 6. Run Scorers

```powershell
python scripts/eval/citation_verify.py --corpus data/processed/laws.json --demo
python scripts/eval/faithbench.py --corpus data/processed/laws.json --questions eval/questions.laws.smoke.json --dump 2
python scripts/eval/faithbench.py --corpus data/processed/laws.json --questions eval/questions.laws.smoke.json --unanswerable-file eval/questions.unanswerable.laws.smoke.json --dump 2
python scripts/eval/faithbench_partial.py --corpus data/processed/laws.json --items eval/questions.partial.laws.smoke.json --dump 2
```

## Next Real Work

1. `LAW_API_KEY` 발급
2. 5~10개 법령의 `MST` 수집
3. `data/processed/laws.json` 생성
4. smoke questions로 runner end-to-end 확인
5. curated 다법령 질문셋 작성 및 정식 G0 재판정

검정력 planning은 아래 명령으로 갱신한다.

```powershell
python scripts/eval/power_analysis.py --base 0.387 --target 0.742
```

현재 요약은 `docs/env-verify/power-report.md`에 둔다.
