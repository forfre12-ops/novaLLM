# G0 종합 판정 — "무엇이 검증됐고, 무엇이 안 됐나" (단일 출처)

> 흩어진 5개 실험 리포트(pilot·faithbench·stats·partial·retrain)의 **단일 결론**.
> 장비: RTX 5070 Ti 16GB Blackwell · 코퍼스: 헌법 제1~39조 closed-set(93항목) · 채점: 결정적(LLM-judge 없음).

## G0가 검증하려던 것

전략(`docs/strategy.md`)의 하중 가정 ①: **"근거충실도(faithfulness)는 모델 크기에 상대적으로 독립적이라,
도메인 타겟 post-training한 소형 모델이 대형 base와 경쟁/우위할 수 있다."** G0는 이 가정을 저비용
fail-fast로 찌른다. 판정 무관하게 **측정 자산(벤치)은 크기 논쟁과 독립적으로 생존**한다.

## 실험 흐름 — 측정을 단계적으로 엄격화

| # | 실험 | 벤치/개입 | 핵심 결과 | "소형>대형" |
|---|------|-----------|-----------|:---:|
| 1 | 파일럿 | 근거 1조문, 형식·복사 | before 0.0 → after 1.0 (형식 습득) | — (자기비교) |
| 2 | 파일럿 교차 | 1조문, base 7B 비교 | FT 1.5B 1.0 ≥ base 7B 0.944 (1문항차) | ✅ 약함 |
| 3 | **faithbench** | K=5조문 선택(distractor) | FT 0.778 ≫ base 7B 0.444, unseen 0.857 > 0.571 | ✅ 지지 |
| 4 | 유의성 | Wilson/두-비율 | overall p=0.029, unseen p=0.040 (CI 하한 ≈0) | ⚠️ 경계 |
| 5 | **부분-span** | tight 인용(span P/R/F1) | base 7B f1 0.529 ≥ FT 0.448 | ❌ 뒤집힘 |
| 6 | tight-gold 재학습 | 학습 gold를 tight로 | precision 0.360→0.386, f1 불변, 7B 우위 유지 | ❌ 미해소 |

## 정직한 판정 (2개 주장)

**주장 A — "파인튜닝이 근거충실 인용을 가르친다": ✅ 결정적 증명.**
- 전 축에서 FT 1.5B가 base 1.5B를 압도(faithbench overall Δ+0.591, p<0.001). QLoRA post-training이
  "제공 근거에서 「원문」[ID] 형식으로 인용하는 기술"을 확실히 가르친다. 암기 아닌 전이(unseen ≥ seen).

**주장 B — "소형 FT가 대형 base(7B)를 이긴다": ⚠️ 축·레시피 의존적, 강건하지 않음.**
- **쉬운 축(조문 선택)**: 유의하나 경계(p≈0.03, CI 하한 ≈0, n 작음).
- **어려운 축(tight 부분인용)**: **뒤집힘.** base 7B가 precision·f1에서 우위. 원인 = FT가 통째복사 학습
  으로 과잉 인용. tight-gold 재학습은 precision을 방향으로만 개선(+0.026), f1 격차 미해소.
- 즉 소형 우위는 **벤치가 쉬울 때만** 성립. 실사용에 가까운 tight 인용에선 미지지.

## 측정 자산이 한 역할 ("own the ruler" 실증)

이 G0의 진짜 성과는 소형 우위 여부가 아니라, **측정 자산이 반복해서 제 역할을 했다**는 것:
- 쉬운 벤치(파일럿)가 준 1.0 만점을 → faithbench가 변별(base 1.5B 0.037로 붕괴).
- faithbench의 소형 우위를 → 부분-span이 아티팩트(과잉인용)로 노출.
- 그럴듯한 수정(tight-gold)의 효과가 작음을 → stats·재평가가 정량화.
- **더 어려운 잣대마다 앞선 잣대가 숨긴 것을 드러냈다.** 이것이 전략의 핵심 베팅("측정을 소유하라")의 실증.

## 전략 함의 (실측 기반 권고)

실측이 주장 B를 강하게 지지하지 않으므로, **서사를 "소형이 대형을 이긴다"에서 "재현가능·결정적 측정을
소유한다"로 축소**하는 것이 정직하다(전략 문서 ⑨의 예정된 fallback과 정렬). 마케팅에 "2.4B가 대형을
이겼다"를 넣지 않는다(CLAUDE.md 하드제약과도 일치). 무게중심을 **모델 레이스 → 측정 자산 강화**로.

## 재현 (측정 스위트)

결정적 채점 프리미티브 + 벤치 + 러너. 전부 LLM-judge 없이 참/거짓.

```powershell
# 채점 프리미티브 자기검증
python scripts/eval/citation_verify.py --demo
python scripts/eval/faithbench.py --demo
python scripts/eval/faithbench_partial.py --demo
# 교차비교(GPU)
python scripts/train/run_g0_faithbench.py --questions eval/questions.constitution.json --k 5 --near
python scripts/train/run_g0_partial.py --k 5 --near
# 유의성
python scripts/eval/faithbench_stats.py --result docs/env-verify/g0-faithbench-result.json
```

## 산출물

- 리포트: `g0-pilot-report.md`, `g0-faithbench-report.md`, `g0-partial-report.md`, `g0-partial-retrain-report.md`
- 결과 json: `g0-*-result.json` (전부 by_split/유의성 포함)
- 측정 코드: `scripts/eval/{citation_verify,faithbench,faithbench_partial,faithbench_stats}.py`
- 러너: `scripts/train/run_g0_{faithbench,partial,compare,pilot}.py`
- 질문셋: `eval/questions.constitution.json`(93), `eval/questions.partial.constitution.json`(14 tight)
