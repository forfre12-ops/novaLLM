# G0 종합 판정 — "무엇이 검증됐고, 무엇이 안 됐나" (단일 출처)

> 흩어진 실험 리포트(pilot·faithbench·stats·partial·retrain)를 하나로 묶은 결론.
> 장비: RTX 5070 Ti 16GB Blackwell · 코퍼스: 헌법 제1~39조 closed-set(93항목) · 채점: 결정적(LLM-judge 없음).

## G0가 검증하려던 것

전략(`docs/strategy.md`)의 하중 가정 ①:
**"근거충실도(faithfulness)는 모델 크기에 상대적으로 독립적이라, 도메인 타겟 post-training한
소형 모델이 대형 base와 경쟁/우위할 수 있다."**

G0는 이 가정을 저비용 fail-fast로 찌른다. 판정과 무관하게 **측정 자산(벤치)은 크기 논쟁과
독립적으로 생존**한다.

## v0.2 위생 재평가 결과

`991c8b2` 이후 다음 위생 수정을 반영해 새 파일로 재평가했다.

- few-shot 정답 노출 제거: 허구 예시법 사용
- scorer v0.2: 줄바꿈 인용·문자 변형 정규화
- leak 재정의: 무인용 실질답변도 leak 후보로 집계
- per-instance transcript 저장: paired McNemar 가능
- closed-book 암기 프로브 추가
- 결과: `docs/env-verify/g0-faithbench-v02-result.json`
- transcript: `docs/env-verify/g0-faithbench-v02-result-transcript.jsonl`

### 본판 faithbench (K=5, near distractor, 93 answerable + 8 unanswerable)

| 모델 | selection_exact | faithfulness | leak_rate | answerable_no_citation |
|---|---:|---:|---:|---:|
| base 1.5B few-shot | 0.215 | 0.220 | 0.625 | 0.140 |
| **FT 1.5B zero-shot** | **0.742** | **0.806** | **0.000** | **0.022** |
| base 7B few-shot | 0.387 | 0.387 | 0.250 | 0.548 |

paired exact McNemar(overall, 같은 93문항):

| 비교 | FT-only | base-only | diff | Holm 후 |
|---|---:|---:|---:|---|
| FT 1.5B vs base 7B | 40 | 7 | +0.355 | 유의 |
| FT 1.5B vs base 1.5B | 52 | 3 | +0.527 | 유의 |

closed-book verbatim recall:

| 모델 | closed-book recall | open selection_exact | grounding gain |
|---|---:|---:|---:|
| base 1.5B | 0.022 | 0.215 | +0.193 |
| FT 1.5B | 0.032 | 0.742 | +0.710 |
| base 7B | 0.043 | 0.387 | +0.344 |

해석: 헌법 암기만으로 설명되지는 않는다. 제공 근거를 활용하는 open-book gain이 크다.

### 부분-span v0.2 (tight 인용, 14문항)

partial few-shot도 허구 예시법으로 교체한 뒤 새 파일로 재평가했다.

- 결과: `docs/env-verify/g0-partial-v02-result.json`

| 모델 | partial_exact | span_f1 | span_precision | span_recall | selected_gold |
|---|---:|---:|---:|---:|---:|
| base 1.5B few-shot | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| FT 1.5B zero-shot | 0.286 | **0.448** | 0.360 | **0.714** | **0.714** |
| base 7B few-shot | **0.429** | 0.394 | **0.393** | 0.405 | 0.571 |

해석: partial은 단일 결론으로 정리하기 어렵다. FT는 gold 선택과 평균 span_f1/recall이 높고,
base 7B는 `partial_exact`와 precision이 높다. 즉 **엄격 축에서는 우위가 metric-dependent**다.

## 정직한 판정 (2개 주장)

**주장 A — "파인튜닝이 근거 기반 인용 행동을 가르친다": 강하게 지지.**

FT 1.5B는 base 1.5B를 본판 faithbench에서 큰 폭으로 앞선다. leak도 0으로 안정적이다.

**주장 B — "소형 FT가 대형 base(7B)를 이긴다": 축 의존적.**

- 본판 selection_exact에서는 v0.2 위생 수정 후 오히려 더 강하게 지지된다.
- 하지만 tight 부분인용에서는 `partial_exact` 기준 base 7B가 앞서고, span_f1 기준 FT가 앞선다.
- 따라서 외부 서사는 **"소형이 대형을 이긴다"가 아니라 "측정 스위트가 축별 결함을 드러낸다"**가 맞다.

## 측정 자산이 한 역할 ("own the ruler" 실증)

이 G0의 진짜 성과는 소형 우위 여부가 아니라, **측정 자산이 반복해서 제 역할을 했다**는 것:

- 파일럿의 1.0 만점을 faithbench가 변별했다.
- faithbench의 소형 우위를 partial span 채점이 더 세밀하게 분해했다.
- leak 재정의가 base 모델의 무인용 유출 후보를 드러냈다.
- closed-book 프로브가 헌법 암기 confound를 수치화했다.
- transcript + McNemar가 paired 통계 판정을 가능하게 했다.

**더 어려운 잣대마다 앞선 잣대가 숨긴 것을 드러냈다.** 이것이 전략의 핵심 베팅("측정을 소유하라")의 실증이다.

## 전략 함의

실측상 "소형이 대형을 항상 이긴다"는 말은 금지한다. 대신 다음 서사를 채택한다:

> 한국어 법령 근거충실도는 단일 점수가 아니라 선택·인용 span·거절·leak·암기 인출로 분해해 봐야 한다.
> nova-llm은 이 축들을 LLM-judge 없이 결정적으로 측정하는 스위트다.

다음 행동은 벤치 추가 정교화가 아니라 외부 게이트다. 2026-07-18 기준 GitHub 초기 공개물은
`d784c81`로 출하했다. 남은 핵심 병목은 다법령 확장과 정식 G0 재판정이다.

1. 법령 API 키로 다법령 N 확장
2. GitHub release/HF Dataset/한글 테크노트 중 하나로 공개물을 배포 단위로 고정
3. HRET/HAERAE 생태계 기여 경로 확인
4. 그 뒤에 Qwen3-4B 재판정

## 재현

```powershell
# 자기검증
python scripts/eval/citation_verify.py --demo
python scripts/eval/faithbench.py --demo
python scripts/eval/faithbench_partial.py --demo

# v0.2 본판 재평가(GPU)
python scripts/train/run_g0_faithbench.py `
  --questions eval/questions.constitution.json `
  --k 5 --near --closed-book `
  --out docs/env-verify/g0-faithbench-v02-result.json

# paired 통계
python scripts/eval/faithbench_stats.py --result docs/env-verify/g0-faithbench-v02-result.json

# partial v0.2 재평가(GPU)
python scripts/train/run_g0_partial.py --k 5 --near --out docs/env-verify/g0-partial-v02-result.json
```

## 산출물

- 본판 v0.2: `g0-faithbench-v02-result.json`, `g0-faithbench-v02-result-transcript.jsonl`
- partial v0.2: `g0-partial-v02-result.json`
- 공식 판정: `g0-verdict.md`
- 측정 코드: `scripts/eval/{citation_verify,faithbench,faithbench_partial,faithbench_stats}.py`
- 러너: `scripts/train/run_g0_{faithbench,partial}.py`
