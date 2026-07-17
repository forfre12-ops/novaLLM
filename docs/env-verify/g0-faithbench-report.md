# 진짜 G0 리포트 — faithbench(distractor 선택) 교차모델 비교

**일자**: 2026-07-17 · **장비**: RTX 5070 Ti 16GB Blackwell · **채점**: `citation_verify`(결정적, LLM-judge 없음)

## 실험 — 파일럿 대비 무엇이 강화됐나

파일럿(`g0-pilot-report.md`)의 한계를 정면으로 보완했다.

| 파일럿 (약점) | faithbench (강화) |
|---|---|
| 근거 **1조문**만 제공 → verbatim 복사만 하면 1.0 | **K=5조문(gold 1 + distractor 4)** 섞어 제공 → 올바른 조문을 선택해야 정답 |
| 질문이 조항ID 명시 → target_cited 자명 | **내용 기반 질문**(조항ID 미노출) |
| 27개 curated 질문만 사용 | **헌법 seed 전체 93개 조·항 curated 질문셋** 사용 |
| distractor 없음 | `--near`로 **같은 조의 인접 항**을 하드 distractor로 우선 |

- 벤치: answerable 93 + unanswerable 8, `k=5 --near`, seed 3407
- 질문셋: `eval/questions.constitution.json` (코퍼스 93개와 1:1, missing/extra 0, ID 힌트 lint 0)
- 비교: base 1.5B(few-shot) vs **FT 1.5B(zero-shot, g0-pilot 어댑터)** vs base 7B(few-shot)
- 핵심 지표 `selection_exact` = gold만 정확·충실 인용(distractor·환각 0)

## 결과 — 전체 (answerable 93, unanswerable 8)

| 모델 | selection_exact | gold_recall | distractor_cite | faithfulness | leak_rate |
|------|:---:|:---:|:---:|:---:|:---:|
| base 1.5B (few-shot) | 0.151 | 0.215 | 0.441 | 0.339 | 0.25 |
| **FT 1.5B (zero-shot)** | **0.742** | **0.742** | 0.204 | **0.806** | **0.0** |
| base 7B (few-shot) | 0.591 | 0.613 | **0.043** | 0.613 | **0.0** |

전체 기준 FT 1.5B는 base 7B보다 `selection_exact`가 **+0.151** 높다(약 69/93 vs 55/93).
base 7B는 distractor 인용률이 더 낮지만, gold 조항을 정확히 선택·인용하는 비율은 FT가 더 높다.

## 결과 — unseen 18조항만 (FT 학습 친숙도 편향 제거)

pilot split(seed 3407)을 재현하면 answerable 93개 중 75개는 파일럿 학습에 노출된 seen,
18개는 held-out unseen이다. 암기 편향을 보려면 unseen만 따로 봐야 한다.

| 모델 | selection_exact | gold_recall | distractor_cite |
|------|:---:|:---:|:---:|
| base 1.5B (few-shot) | 0.167 (3/18) | 0.222 | 0.5 |
| **FT 1.5B (zero-shot)** | **0.778 (14/18)** | **0.778** | 0.167 |
| base 7B (few-shot) | 0.444 (8/18) | 0.444 | **0.0** |

**FT의 seen vs unseen**: selection_exact seen 0.733(55/75) vs unseen 0.778(14/18).
unseen에서도 FT 1.5B가 base 7B보다 **+0.334** 높다.

## 유의성 — 이 신호는 얼마나 강한가 (정직화)

같은 인스턴스에서의 **unpaired** Wilson/Newcombe 근사(α=0.05, `scripts/eval/faithbench_stats.py`).
paired McNemar는 더 강력하므로 아래는 **보수적** 추정이다(paired는 인스턴스별 저장 후 재실행 필요).

| 비교 | split | Δ(FT−base) | 95% CI | p | 판정 |
|---|---|:---:|:---:|:---:|:---:|
| FT 1.5B vs base 7B | overall(93) | +0.151 | (0.015, 0.279) | 0.029 | 유의(경계) |
| FT 1.5B vs base 7B | unseen(18) | +0.333 | (0.016, 0.572) | 0.040 | 유의(경계) |
| FT 1.5B vs base 7B | seen(75) | +0.107 | (−0.042, 0.249) | 0.161 | 미유의 |
| FT 1.5B vs base 1.5B | overall(93) | +0.591 | (0.461, 0.689) | <0.001 | 강한 유의 |

- **파인튜닝 효과는 압도적** — FT vs base 1.5B는 전 split에서 p<0.001. QLoRA가 근거충실 인용을 명확히 가르친다.
- **소형 FT > 대형 7B는 유의하나 경계** — overall·unseen 모두 p≈0.03–0.04로 α=0.05를 통과하나 차이 CI 하한이 0에 근접(0.015/0.016). 실재 신호이나 **강하지 않다**.
- **seen에선 미유의(p=0.16), unseen에선 유의** — 우위가 암기(seen)에서 오지 않음을 재확인하나 unseen n=18로 표본이 작다.
- **G0 판정**: "신호 있음, 진행 가능"이나 "소형이 대형을 이긴다"를 강하게 마케팅하기엔 이르다. **다법령으로 N을 키워 CI를 좁히는 것**이 다음 필수 단계.

## 해석 (정직)

**증명된 것:**
1. **벤치가 더 넓은 표본에서도 변별한다** — 27문항에서 93문항으로 키워도 base 1.5B, FT 1.5B, base 7B 사이의 간격이 유지된다.
2. **전략 핵심 가정 지지** — 5배 작은 FT 1.5B가 대형 base 7B를 선택적 근거충실도에서 앞선다. 전체는 +0.151, unseen은 +0.334다.
3. **암기만으로 보기 어렵다** — FT의 unseen 점수가 seen보다 낮지 않다. 제공된 근거에서 올바른 조항을 선택·인용하는 기술이 전이된 신호다.
4. **거절/leak은 FT와 base 7B 모두 안정적** — unanswerable 8개에서 둘 다 leak 0. base 1.5B는 leak 0.25.

**증명되지 않은 것:**
- **단일 법령 한계** — 여전히 헌법 제1~39조만 쓴다. 다법령·긴 조문·서로 비슷한 법률 간 distractor는 미검증이다.
- **unseen도 아직 작다** — 18문항은 7문항보다 나아졌지만 강한 통계 결론에는 부족하다. 다법령으로 N을 키워야 한다.
- **비대칭 조건** — FT는 해당 도메인에 post-training됐고, base 7B는 few-shot만 받았다. 주장은 “도메인 타겟 post-training한 소형 모델이 근거충실도 축에서 대형 base와 경쟁/우위 가능”이지, “1.5B가 범용으로 7B보다 낫다”가 아니다.
- **부분 span·요약 충실도 미측정** — 현재 채점은 `「원문」[ID]` exact substring에 강하다. 부분 인용, 요약, 다중 gold 답변은 별도 설계가 필요하다.

## 결론

전략 핵심 베팅("소형 FT가 대형 base를 근거충실도에서 이긴다")이 93개 curated 질문셋에서도 유지됐다.
기존 27문항 결과보다 표본은 커졌고, unseen에서도 우위가 남았다. 다음 단계는 이 신호를 헌법 단일
코퍼스 밖으로 옮기는 것이다.

우선순위:
1. 국가법령정보 API 기반 다법령 closed-set 코퍼스 구축
2. `eval/questions.<law>.json` 형태의 법령별 curated 질문셋 추가
3. 같은 `run_g0_faithbench.py --questions ... --k 5 --near`로 N 확장
4. 부분 span 인용·근접 주제 distractor·다중 gold 질문 추가

산출물: `docs/env-verify/g0-faithbench-result.json`, `eval/questions.constitution.json`,
벤치 `scripts/eval/faithbench.py`, 러너 `scripts/train/run_g0_faithbench.py`.
