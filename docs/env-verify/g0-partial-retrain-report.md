# 부분-span 재학습 리포트 — tight-gold 개입: 기전 확인, 격차는 미해소

**일자**: 2026-07-17 · **장비**: RTX 5070 Ti 16GB Blackwell · **채점**: `faithbench_partial`(결정적)

## 가설과 개입

`g0-partial-report.md`의 진단: 구-FT(g0-pilot)가 tight 인용에서 precision이 낮은(과잉 인용) 이유는
**학습 gold가 조문 전체를 「」로 감싼 것**. 개입: 데이터만 tight-gold로 바꿔 재학습.

- 데이터: `gen_partial_sft.py` — 조문을 절(마침표+쉼표) 단위로 쪼갠 tight-gold. partial 107(tight 46) +
  refusal 84, 평가셋 9조항 held-out, self-consistency PASS.
- 학습: **`scripts/02_train_sft.py`(정합화된 파이프라인)로 실학습** — base Qwen2.5-1.5B(구-FT와 동일),
  동일 레시피(r16/q,k,v,o/4ep/lr2e-4). 811s, peak VRAM **5.53GB**, loss→0.001.
- 평가: 동일 14문항 부분-span, base는 few-shot·FT는 zero-shot.

## 결과 (n=14, k=5, near)

| 모델 | partial_exact | span_f1 | span_precision | span_recall | selected_gold |
|------|:---:|:---:|:---:|:---:|:---:|
| base 1.5B (few-shot) | 0.071 | 0.277 | 0.292 | 0.349 | 0.571 |
| 구-FT (통째복사 학습) | 0.286 | 0.448 | 0.360 | 0.714 | 0.714 |
| **신-FT (tight-gold 학습)** | 0.286 | 0.451 | **0.386** | 0.643 | 0.643 |
| base 7B (few-shot) | 0.286 | **0.529** | **0.471** | 0.743 | 0.786 |

## 해석 (정직)

**확인된 것:**
1. **기전 확인** — tight-gold로 바꾸자 span_precision이 **0.360 → 0.386(+0.026)**. 통째복사 gold를
   없애니 과잉 인용이 줄었다. "학습 데이터 형태가 인용 tightness를 좌우한다"는 진단이 **방향으로 맞다.**
2. **02_train_sft.py 실GPU 검증** — 정합화한 파이프라인(수동 루프·grad ckpt·config)으로 실제 학습
   완주(811s, 5.53GB). 파이프라인 정합화 작업이 end-to-end로 동작함을 실증.

**해소되지 않은 것:**
1. **span_f1은 사실상 불변**(0.448 → 0.451) — precision(+0.026)이 **recall 하락(0.714 → 0.643)**으로
   상쇄됐다. 더 짧게 인용하다 보니 관련부를 덜 덮는 trade-off.
2. **base 7B 여전히 우위** — f1 0.529·precision 0.471·selected_gold 0.786 모두 신-FT보다 높다.
   partial_exact는 셋 다 0.286 동률.
3. **selected_gold 소폭 하락**(0.714 → 0.643) — hint-locator 질문 학습이 조문 선택엔 약간 역효과.

**왜 격차를 못 좁혔나(정직한 추정):**
- **tight 신호 희석** — 학습 partial 107건 중 tight는 46건뿐(나머지 61은 단절 조문=전체). 통째복사
  습관을 덮을 만큼 강하지 않았다.
- **질문 style 불일치** — 학습은 '{힌트}로 시작하는 부분 인용'인데 eval은 내용 기반 질문. tight 인용
  *행동*은 일부 전이됐으나(precision↑) 내용→정확한 span *선택*은 전이 약함.
- **n=14 극소** — ±0.03 차이는 노이즈 범위. 강한 결론 불가.

## 결론

**진단은 옳았으나(precision 방향 개선) 단일 데이터 tweak으로는 base 7B 격차를 못 좁혔다.**
"소형 FT가 대형 base를 이긴다"는 **tight-precision 축에서 재학습 후에도 미지지**. 측정 자산이 또
제 역할을 했다 — 그럴듯한 수정의 효과가 **작고 상쇄됨**을 정량으로 드러냈다.

다음 후보:
1. **tight 신호 강화** — 전 조항 clause-split로 tight 비율↑, 단절 조문엔 통째복사 gold 제거.
2. **content-질문 학습 데이터** — hint-locator 대신 내용 질문↔정확 span 쌍(teacher/수작업 필요).
3. **정직한 축소** — 소형이 tight-precision에서 7B를 못 이기면, 전략 서사를 "소형이 이긴다"에서
   "측정을 소유한다"로 축소(전략 문서 ⑨의 예정된 fallback).

산출물: `docs/env-verify/g0-partial-retrain-result.json`, `checkpoints/g0-partial`(git 제외),
`scripts/data/gen_partial_sft.py`, `configs/train_partial.yaml`.
