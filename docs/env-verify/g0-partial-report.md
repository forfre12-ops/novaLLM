# 부분-span G0 리포트 — tight 인용에서 가정이 뒤집히다 (중요)

**일자**: 2026-07-17 · **장비**: RTX 5070 Ti 16GB Blackwell · **채점**: `faithbench_partial`(char-level span P/R/F1, 결정적)

## 실험 — 마지막 validity 구멍을 닫다

faithbench 본판(선택 태스크)까지도 남아 있던 약점: **gold 답변이 조문 전체를 「」로 감싸므로,
substring 검증이 '제공 근거를 통째 복사'만 해도 충족**된다(faithfulness가 관대). 이 벤치는
질문이 조문의 **특정 clause**를 겨냥하고, 채점이 인용 span과 gold_span의 **문자 단위
precision/recall/F1**을 재 통째복사를 페널티한다.

- 셋: `eval/questions.partial.constitution.json` 14문항(질문+gold_span, 전 항목 exact substring)
- 지표: span_precision(인용이 얼마나 정확히 관련부만), span_recall(관련부를 얼마나 덮음),
  partial_exact(gold 조문 선택 + span_f1≥0.7 + distractor 미인용)
- 비교: base 1.5B(few-shot) vs FT 1.5B(zero-shot, g0-pilot 어댑터) vs base 7B(few-shot)

## 결과 (n=14, k=5, near)

| 모델 | partial_exact | span_f1 | span_precision | span_recall | selected_gold |
|------|:---:|:---:|:---:|:---:|:---:|
| base 1.5B (few-shot) | 0.071 | 0.277 | 0.292 | 0.349 | 0.571 |
| FT 1.5B (zero-shot) | 0.286 | 0.448 | **0.360** | 0.714 | 0.714 |
| **base 7B (few-shot)** | 0.286 | **0.529** | **0.471** | 0.743 | **0.786** |

## 해석 (정직) — 이번엔 가정이 성립하지 않는다

**핵심 반전:**
1. **tight 부분인용에선 base 7B ≥ FT 1.5B.** span_f1(0.529 vs 0.448)·span_precision(0.471 vs 0.360)·
   selected_gold(0.786 vs 0.714) 모두 base 7B가 앞선다. partial_exact만 동률(0.286). 즉
   faithbench(선택 태스크)에서 확인됐던 "소형 FT > 대형 base"가 **더 엄격한 tight-인용에선 사라진다.**
2. **FT의 병목은 precision** — FT는 recall 0.714(관련부는 잘 덮음)인데 precision 0.360(너무 많이 인용).
   **FT가 조문을 과잉 인용(통째로 뱉음)한다.**
3. **원인은 학습 데이터 자체** — FT(g0-pilot 어댑터)는 gold 답변이 **조문 전체를 「」로 감싼** 데이터
   (`gen_grounded_sft.py`, `run_g0_pilot.py`)로 학습됐다. 그래서 **"근거를 통째 복사"를 배웠고**, 그
   행동이 faithbench(recall 중심 선택)에선 유리했지만 partial-span(precision 요구)에선 **역효과**다.

**이 발견의 전략적 의미(가장 중요):**
- faithbench에서의 FT 우위는 부분적으로 **학습 데이터 아티팩트**(통째복사 학습)였다. 더 어려운
  벤치가 이를 **노출**했다 — 측정 자산이 제 역할을 했다("own the ruler"의 실증).
- **진짜 근거충실도(관련부만 정확 인용)를 이기려면, 학습 데이터의 gold를 조문 전체가 아니라
  partial-span으로 바꿔야 한다.** 현 SFT 레시피는 precision을 가르치지 않는다.
- 즉 "소형이 대형을 이긴다"는 **레시피 의존적**이다. 현 레시피로는 쉬운 축에서만 이긴다.

**증명되지 않은 것(caveat):**
- **n=14 극소표본** — 방향성 신호일 뿐. 통계적 결론 불가.
- **span 채점의 한계** — 문자 겹침은 통째복사엔 강하나 **의미적 적절성**(엉뚱한 관련부 인용)은
  못 잡는다. 여전히 형식·위치 채점이다.
- **단일 법령·단일 형식.**

## 결론 & 다음 액션

측정을 더 엄격히 하자 전략 핵심 가정이 **이 축에선 뒤집혔다**. 이것은 실패가 아니라 **측정 자산의
가치 실증**이다 — 쉬운 벤치가 숨긴 학습 아티팩트(과잉 인용)를 잡아냈다. 구체 액션:

1. **학습 데이터 partial-span화 (최우선)** — `gen_grounded_sft.py`의 gold를 조문 전체가 아니라
   질문에 해당하는 **부분 인용 + 근거 문장**으로 재설계. precision을 명시적으로 학습.
2. 재학습한 grounder로 partial-span·faithbench 재평가 → 개선 여부 확인.
3. (N 확대) 다법령 코퍼스로 partial 셋 확장 — API 키 필요(user-gate).

산출물: `docs/env-verify/g0-partial-result.json`, `eval/questions.partial.constitution.json`,
`scripts/eval/faithbench_partial.py`, `scripts/train/run_g0_partial.py`.
