# GroundLM 2026 제출 스코프·결정 문서 (내부)

> 목적: EMNLP 2026 GroundLM 워크숍(grounded LMs) 제출 여부/범위를 **역산 일정**으로 결정한다.
> 이 프로젝트에서 유일하게 **외부 시계가 걸린** 아이템이라 GPU 세션 일정 결정의 트리거다.

> ⚠️ **외부 사실 확인 선행:** 아래 마감·트랙 정보는 2026-07 웹 실사(abstract 수준)에서 얻은 것으로,
> **제출 전 워크숍 CFP 원문으로 재확인 필수**(Track2 ARR commitment ~2026-08-05, 통지 8월 중순,
> 워크숍 10-29, archival 4–8p 또는 non-archival로 추정). 날짜가 다르면 아래 일정을 조정한다.

## 왜 이 워크숍인가 (장점 정합)

- 주제가 **grounding/factuality/citation 평가 프레임워크**로 정확히 우리 자산(A1 결정적 채점기)과 일치.
- **언어 무관 훅**("closed-set에서 char-span exact-match가 NLI/judge를 비용·재현·순환성에서 이긴다")으로
  한국어 특정성 없이 영어권 방법론 청중에 소구 → ⑧ 채널-청중 미스매치 회피.
- 인접 클러스터(ALCE=NLI, RAGTruth=human-annot, LRAGE/KCL/K-FinHallu=LLM-judge)와의 **방법론적 빈칸**을
  논문 thesis로 세울 수 있음. 소형모델은 CaLM 근거로 **'검증자(verifier)'**로 프레이밍(소형-우위 단정 회피).

## 지금 artifact만으로 성립하는 주장 (GPU 불요, 이미 커밋됨)

1. **결정적·LLM-judge-free 법령 인용 채점기** — citation_verify/faithbench/faithbench_partial, 동결(v0.2/v0.1)
   + `check_scorer_frozen` CI 게이트. closed-set이 char-span exact-match를 가능케 하는 설계 논증.
2. **무인용 leak 유형학**(parametric_verbatim vs ungrounded) — 3303조항 closed-set 역검증. judge 없이
   암기 유출을 결정적으로 분리하는, closed-set만 가능한 채점.
3. **gold-ablation in-domain 거절 프로브** — 표면 단서 없이 grounding 규율을 측정(leak축 N 20→75).
4. **오프라인 재현·제출 인터페이스** — score_predictions(rescore/predictions) + instance_id + SUBMISSIONS.md.
   "누구나 GPU 없이 공표 수치를 CPU로 재검증" = judge 벤치가 구조적으로 못 하는 재현성.
5. **genuine 큐레이션 코어 55/55** (5법령 균형) + 검정력 충족(관찰효과 +0.355에 필요 N=30).
6. **정직성 규율** — G0=SPLIT 공개 판정, provenance 태깅(30genuine+70auto), 마케팅 금지선.

→ 이 6개만으로 **"방법론·측정 자산"** 논문(모델 승리 주장 없이 성립)이 가능하다. **이것이 fallback thesis.**

## 4B GPU 런이 있어야 강해지는 주장 (오너 게이트)

- **소형 grounder(2~4B) vs 대형 base의 정식 head-to-head 결과** — 확장 55/55 clean 세트 + confound-safe
  파이프라인(enable_thinking·closed-book 법령무관·gold-ablation) 위에서. "소형이 검증자로서 경쟁력" 실측.
- 이건 논문을 **"측정 자산 + 실증 결과"**로 격상하나, **없어도 fallback thesis는 선다.**

## 역산 일정 (Track2 ARR ~8/5 가정)

| 시점 | 할 일 | 게이트 |
|---|---|---|
| D-14 (~7/22, 지금) | artifact 정비 완료(위 1~6 전부 커밋됨) · GPU 세션 여부 결정 | ✅ 대부분 완료 |
| D-10 | **go/no-go**: (a) 4B 런 포함 vs (b) fallback(방법론) 단독 | 🙋 오너 결정 |
| D-7 | (a 선택 시) 정식 G0 GPU 런 1회 실행·결과 커밋 | 🖥️ GPU |
| D-4 | 초고 작성(재현번들·표는 result JSON에서 자동생성 — fingerprint 카드 생성기 필요, pre_public #4) | 🤖 |
| D-1 | 중립성 리뷰·right-of-reply 확인·리네이밍 반영(FaithBench 충돌) | 🙋 |
| D-0 | ARR 제출 | 🙋 |

## 결정 포인트 (오너)

1. **워크숍 제출 자체를 할 것인가?** — 안 해도 자산은 그대로. 하면 G2(제3자/기관 신뢰) 부트스트랩.
2. **4B 런을 이 마감에 태울 것인가?** — fallback(방법론 단독)으로도 제출 가능하므로 **필수 아님**.
   GPU 세션이 마감 내 가능하면 (a), 아니면 (b)로 제출하거나 다음 회차로 연기.
3. **선결 의존**: 리네이밍(pre_public #1, FaithBench/K-HALU 충돌)·fingerprint 자동생성(#4)은 제출 전 필요.

## 폴백 기준

- GPU 세션이 D-7까지 불가 → **fallback thesis(방법론·측정 자산)로 제출** 또는 다음 grounding 벤치 회차로 연기.
- 리네이밍/중립성 리뷰 미완 → **제출 보류**(정직성·이름충돌이 자산 신뢰를 훼손하므로 서두르지 않는다).
