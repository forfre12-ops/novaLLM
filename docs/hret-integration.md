# HRET Integration Plan

HRET/HAE-RAE 연동은 `nova-llm`의 채택 플라이휠을 여는 외부 인프라 경로다.

## Source Check

- HRET GitHub: https://github.com/HAE-RAE/haerae-evaluation-toolkit
- HRET paper: https://arxiv.org/abs/2503.22968

확인한 점:

- HRET는 한국어 LLM 평가 표준화와 재현성을 목표로 하는 open-source toolkit이다.
- README 기준 HRET는 string-match, partial-match, LLM-as-a-judge 등 여러 평가 방식을 지원한다.
- 논문 기준 HRET는 registry-based framework이며 데이터 ingestion, inference, reporting을 표준화한다.
- 따라서 `faithbench`는 "새 leaderboard"보다 HRET에 들어갈 수 있는 deterministic custom task/metric으로
  포지셔닝하는 편이 자연스럽다.

## Fit

`nova-llm`이 제공할 수 있는 고유 슬롯:

- Korean legal closed-set grounding
- deterministic citation verification
- no LLM judge required for core metrics
- source selection, tight span, refusal/leak, closed-book recall split
- per-instance transcript for paired statistics

HRET 쪽 가치:

- 기존 한국어 평가 사용자에게 배포 가능
- registry/config 기반 재현성에 기대어 "1인 벤치" 신뢰 약점을 줄일 수 있음
- HAE-RAE/KMMLU 인접 생태계와 자연스럽게 연결 가능

## Proposed Package Shape

초기 PR은 모델 러너 전체가 아니라 dataset/metric 최소 모듈로 제안한다.

```text
faithbench_legal_ko/
  README.md
  dataset_card.md
  sample.jsonl
  metrics.py
  citation_verify.py
  task_config.yaml
```

Minimal fields:

```json
{
  "id": "민법 제1조",
  "question": "내용 기반 질문",
  "context": [
    {"id": "민법 제1조", "text": "..."},
    {"id": "민법 제2조", "text": "..."}
  ],
  "gold": ["민법 제1조"],
  "gold_span": "optional exact substring"
}
```

Metric outputs:

```json
{
  "selection_exact": 1,
  "faithfulness": 1.0,
  "span_precision": 0.0,
  "span_recall": 0.0,
  "span_f1": 0.0,
  "clean_refusal": 0,
  "leaked": 0
}
```

## Pre-PR Checklist

1. 다법령 corpus로 `N>=200` answerable 생성
2. 사람이 검수한 curated question subset 분리
3. public sample과 private holdout 정책 분리
4. `python scripts/smoke.py` PASS
5. scorer version 고정
6. result transcript 예시 포함
7. "small beats large" 주장 제거

## Open Questions

- HRET의 현재 custom metric API가 decorator registry인지 config-only인지 실제 repo를 clone해서 확인해야 한다.
- dataset을 HRET repo 안에 넣을지, Hugging Face Dataset으로 두고 loader만 제공할지 결정해야 한다.
- 법령 텍스트 snapshot을 얼마나 포함할지, 또는 OpenAPI provenance만 둘지 결정해야 한다.

## Next Action

`LAW_API_KEY`로 다법령 sample을 만든 뒤, 별도 branch에서 HRET module skeleton을 작성한다.
HRET PR은 정식 G0 재판정 전이라도 "deterministic legal citation task prototype"으로 열 수 있다.
