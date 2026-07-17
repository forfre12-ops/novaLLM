# G0 종합 판정문 (분할 판정) — 2026-07-17

> 이 문서는 G0 게이트의 **공식 판정**이다. v0.2 위생 재평가까지 반영한다.
> 핵심은 "소형이 대형을 이긴다"를 확정하는 것이 아니라, 관대 선택축과 엄격 부분-span축이
> 서로 다른 실패모드를 드러낸다는 점이다.

## 1. 사전등록 사양 대비 현재 실측의 지위

실행계획 TA-12(정식 G0)의 사전등록 임계는 **파인튜닝된 2~4B grounder vs base 7~14B,
primary=citation exact-match, margin Δ≥+2pt & bootstrap/paired 95% CI 하한>0**이다.

현재 실측은 여전히 사양과 어긋난다.

| 항목 | 사전등록 | 실측 | 판정 |
|---|---|---|---|
| 모델 크기 | 2~4B(Qwen3-4B/Mi:dm Mini) | **Qwen2.5-1.5B** | 사양 밖 |
| 학습 데이터 | teacher 합성 파일럿 1~2만 | 결정적 수작업 소규모 | 사양 밖(단 오염 0로 클린) |
| 통계 | paired/CI | v0.2에서 paired McNemar 확보 | 일부 충족 |
| 코퍼스 | 다법령 확장 예정 | 헌법 제1~39조 | 사양 밖 |

따라서 현재 실측은 **정식 G0가 아니라 G0-pre+위생 재판정**이다.

## 2. v0.2 위생 재평가 결과

반영된 위생 수정:

- few-shot 정답 노출 제거(허구 예시법)
- scorer v0.2(줄바꿈 인용·문자 변형 정규화)
- per-instance transcript 저장
- paired exact McNemar + Holm 보정
- closed-book 암기 프로브
- leak 재정의(무인용 실질답변 포함)

### 관대 축 — faithbench(selection_exact)

`docs/env-verify/g0-faithbench-v02-result.json`

- FT 1.5B: `selection_exact 0.742`
- base 7B: `selection_exact 0.387`
- paired McNemar: FT-only 40, base-only 7, diff +0.355, Holm 후 유의
- unseen 18문항: FT 0.778 > base 7B 0.444
- closed-book recall은 FT 0.032, base 7B 0.043으로 낮음 → 단순 암기 설명은 약함

관대 선택축에서는 소형 FT 우위가 v0.2에서도 유지된다.

### 엄격 축 — faithbench_partial(char-span)

`docs/env-verify/g0-partial-v02-result.json`

- FT 1.5B: `partial_exact 0.286`, `span_f1 0.448`, `span_precision 0.360`
- base 7B: `partial_exact 0.429`, `span_f1 0.394`, `span_precision 0.393`

엄격 부분-span축은 **metric-dependent split**이다. FT는 평균 F1/recall과 gold 선택이 높고,
base 7B는 partial_exact/precision이 높다. 이 축에서는 "소형 FT 우위"를 단정할 수 없다.

## 3. 판정

**G0 = SPLIT.**

- 관대 선택축: 소형 FT 우위 지지
- leak/거절축: FT 안정적
- closed-book: 암기만으로 설명되지는 않음
- 부분-span축: metric-dependent, 단정 불가
- 사전등록 크기/코퍼스 조건: 아직 미충족

따라서 전략 문서의 미달 분기를 부분 발동한다.

**서사는 "소형이 대형을 이긴다"가 아니라 "측정을 소유한다"로 축소한다.**

## 4. 마케팅 금지선

재판정 전까지 다음 표현을 README·모델카드·리포트·커뮤니티 게시에 사용 금지:

- "1.5B/소형이 7B/대형을 이긴다" 계열 단정
- "근거충실도에서 대형보다 우위" 단정
- partial 결과를 한 방향으로 과대해석

허용되는 서사:

> **"우리 잣대는 우리 모델의 장점과 약점을 모두 드러낸다."**
> 선택축에서는 FT가 강하고, 부분-span precision에서는 base 7B가 더 낫다.
> 이 차이를 LLM-judge 없이 결정적으로 보여주는 것이 nova-llm의 자산이다.

## 5. 재판정 조건

정식 G0 재판정은 아래 조건을 먼저 만족해야 한다.

1. Qwen3-4B 또는 Mi:dm Mini 2~4B로 사전등록 크기 밴드 충족
2. 다법령 closed-set 코퍼스로 N 확장
3. primary를 `selection_exact` 단독이 아니라 `selection_exact + span_precision/span_f1 + leak` 복합으로 고정
4. paired exact McNemar/Holm 보정 및 transcript 공개
5. closed-book 프로브를 기본 포함

## 6. 벤치 동결 규칙

- `faithbench v0.1`, `faithbench_partial v0.1`, `citation_verify v0.2`를 현 단계 기준으로 동결한다.
- 다음 벤치 변형(의미 채점·다중 gold·요약 등)은 **외부 공개물 1개** 이후에만 착수한다.
- G1 전에는 벤치 정교화보다 **다법령 수집·공개·생태계 기여**를 우선한다.
