# 진짜 G0 리포트 — faithbench(distractor 선택) 교차모델 비교

**일자**: 2026-07-17 · **장비**: RTX 5070 Ti 16GB Blackwell · **채점**: `citation_verify`(결정적, LLM-judge 없음)

## 실험 — 파일럿 대비 무엇이 강화됐나

파일럿(`g0-pilot-report.md`)의 한계를 정면으로 보완:

| 파일럿 (약점) | faithbench (강화) |
|---|---|
| 근거 **1조문**만 제공 → verbatim '복사'만 하면 1.0 | **K=5조문(gold 1 + distractor 4)** 섞어 제공 → 올바른 조문 '선택'해야 정답 |
| 질문이 조항ID 명시 → target_cited 자명 | **내용 기반 질문**(조항ID 미노출) |
| distractor 없음 | `--near`로 **같은 조의 인접 항**을 하드 distractor로 |
| 거절율 3모델 모두 1.0 포화 | leak/선택 지표로 변별 확보 |

- 벤치: answerable 27(내용질문) + unanswerable 8, `k=5 --near`, seed 3407
- 비교: base 1.5B(few-shot) vs **FT 1.5B(zero-shot, g0-pilot 어댑터)** vs base 7B(few-shot)
- 핵심 지표 `selection_exact` = gold만 정확·충실 인용(distractor·환각 0). 파일럿엔 없던 '선택' 지표.

## 결과 — 전체 (answerable 27, unanswerable 8)

| 모델 | selection_exact | gold_recall | distractor_cite | faithfulness | leak_rate |
|------|:---:|:---:|:---:|:---:|:---:|
| base 1.5B (few-shot) | 0.037 | 0.111 | 0.593 | 0.302 | 0.125 |
| **FT 1.5B (zero-shot)** | **0.778** | 0.778 | 0.148 | 0.815 | **0.0** |
| base 7B (few-shot) | 0.444 | 0.556 | 0.148 | 0.537 | 0.0 |

## 결과 — unseen 7조항만 (FT 학습 친숙도 편향 제거)

faithbench gold 27개 중 20개는 파일럿 학습에 포함(seen), 7개는 held-out(unseen). 암기 편향을
제거하려면 unseen만 봐야 한다. (pilot split 재현, seed 3407.)

| 모델 | selection_exact | gold_recall | distractor_cite |
|------|:---:|:---:|:---:|
| base 1.5B (few-shot) | 0.0 (0/7) | 0.0 | 0.429 |
| **FT 1.5B (zero-shot)** | **0.857 (6/7)** | 0.857 | 0.143 |
| base 7B (few-shot) | 0.571 (4/7) | 0.714 | 0.143 |

**FT의 seen vs unseen**: selection_exact seen 0.75(15/20) vs **unseen 0.857(6/7)** — unseen이 더 높음.

## 해석 (정직)

**증명된 것 (파일럿보다 강한 신호):**
1. **벤치가 변별한다** — 파일럿에선 3모델이 1.0 근처로 포화했으나, 더 어려운 선택 태스크에선
   base 1.5B가 `selection_exact 0.037`로 붕괴, base 7B 0.444, FT 1.5B 0.778로 **넓게 벌어짐**.
2. **전략 핵심 가정 지지** — 5배 작은 FT 1.5B(0.778)가 대형 base 7B(0.444)를 선택적 근거충실도에서
   **큰 마진(Δ+0.334)**으로 앞섬. 파일럿의 1문항 차(0.944 vs 1.0)와 질적으로 다른 격차.
3. **암기 아님, 기술 전이** — FT의 unseen 점수(0.857)가 seen(0.75)보다 **오히려 높다**. 암기라면
   seen이 높아야 하므로, 이 우위는 학습 친숙도가 아니라 "제공된 근거에서 올바른 조문을 선택·인용하는
   기술"의 전이다. **편향 제거 후에도 FT(0.857) > base 7B(0.571) 유지.**
4. **leak 억제** — unanswerable에서 FT·base7B는 leak 0, base 1.5B만 0.125(1/8). FT는 distractor
   인용율도 0.148로 base 7B와 동급(대형 대비 열위 없음).

**증명되지 않은 것 (과대해석 금지):**
- ⚠️ **n 소표본** — 특히 unseen은 **n=7**(6/7 vs 4/7 = 2문항 차)로 통계적으로 약함. 전체 27도 검정력
  산정 전. 강한 결론엔 **검정력 N 산정(TB-04p)·다법령·human anchor** 필요.
- ⚠️ **단일 법령·단일 형식** — 헌법 하나, 「」[] 인용 형식 고정. 다른 법령·모호질의·다중 gold·긴 조문의
  **부분 span 인용**은 미검증(현 gold는 짧은 조문 통째 인용이라 substring 판정이 여전히 관대).
- ⚠️ **거절 차원 포화** — FT·base7B 모두 refusal 1.0. leak은 base 1.5B에서만 변별 → 더 어려운
  적대셋(토픽 근접 distractor로 over-citation 유도)이 필요.
- ⚠️ **비대칭 조건** — FT는 이 도메인에 post-training됨, base 7B는 아님(few-shot만). 이는 결함이 아니라
  전략의 주장 자체("faithfulness는 post-training이 좌우 → 소형이 경쟁")이나, "1.5B가 범용으로 7B보다
  낫다"는 주장은 **아님**. 주장 범위는 "도메인 타겟 post-training한 1.5B > 미튜닝 7B few-shot".

## 결론

전략 핵심 베팅("소형 FT가 대형 base를 근거충실도에서 이긴다")이 **파일럿보다 어려운 선택 태스크에서,
암기 편향을 제거한 unseen에서도 유지됨을 실측**. 방향성은 명확하나 n이 작아 아직 '신호'이지 '증명'은
아니다. 다음: (1) 검정력 기반 N 산정 + 다법령 코퍼스(국가법령정보 API) (2) 부분 span 인용·적대
distractor로 난이도 상향 (3) human anchor 소표본 상관검정.

산출물: `docs/env-verify/g0-faithbench-result.json`(전체 + by_split), 벤치 `scripts/eval/faithbench.py`,
러너 `scripts/train/run_g0_faithbench.py`.
