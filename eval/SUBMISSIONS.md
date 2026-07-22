# 제출·재현 규격 (K-FaithBench faithbench 축)

모델을 우리 하드웨어에서 돌릴 필요 없이, **답변 파일만 제출**하면 결정적으로 채점된다.
채점에는 LLM judge가 없다 — 모든 인용은 closed-set 원문에 대해 기계적으로 대조된다.

## 1. 인스턴스 받기

벤치는 `instance_id`가 박힌 인스턴스 JSONL을 배포한다(직접 생성도 가능):

```powershell
python scripts/eval/faithbench.py `
  --corpus data/processed/laws.json `
  --questions eval/questions.laws.curated.json `
  --unanswerable-file eval/questions.unanswerable.laws.curated.json `
  --k 5 --near --out eval/instances.laws.curated.jsonl
```

각 인스턴스 행: `{instance_id, split, gold, question, context_ids, messages}`.
`instance_id`는 `(split, gold, question, context_ids)`의 sha1로, 동일 `(corpus, seed, k, near)`면 재현된다.

## 2. 답변 생성

각 인스턴스의 `messages`(system+user)를 당신의 모델에 넣어 답변을 받는다.
출력은 `「원문 인용」[조항ID]` 형식을 따라야 하며, 근거가 없으면 거절 문장으로 답한다.

## 3. 예측 제출

`instance_id`별 답변을 JSONL로 저장한다(모든 인스턴스에 답변 필수):

```json
{"instance_id": "0a1b2c3d4e5f6071", "answer": "「...」[대한민국헌법 제1조 ①]"}
```

## 4. 채점

```powershell
python scripts/eval/score_predictions.py predictions `
  --instances eval/instances.laws.curated.jsonl `
  --predictions my_answers.jsonl --model my-model `
  --corpus data/processed/laws.json --out result.json
```

`result.json`(축별 지표)과 `result-transcript.jsonl`(인스턴스별 원문답변+정오답 감사용)이 생성된다.

## 5. 공표 결과 재검증 (GPU 불요)

공표된 결과가 스코어러로 재현되는지 누구나 CPU로 확인할 수 있다:

```powershell
python scripts/eval/score_predictions.py rescore `
  --transcript docs/env-verify/g0-faithbench-v02-result-transcript.jsonl `
  --corpus data/seed/constitution.json `
  --expect docs/env-verify/g0-faithbench-v02-result.json
```

이 재현 검증은 `scripts/smoke.py`(CI)에 게이트로 포함되어 있어, 스코어러 규칙이 바뀌면 CI가 깨진다.

> 주: tight-span(부분 인용) 축은 `faithbench_partial.py`로 별도 채점된다. 위 제출 흐름은 현재
> selection/leak 축(faithbench)만 다룬다.
