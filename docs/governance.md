# 측정 거버넌스 헌장 (own the ruler)

이 벤치의 신뢰는 "중립·재현·정직"에서 나온다. 이 문서는 그 규율을 **말이 아니라 코드로 강제**되는
정책으로 고정한다. 각 정책은 실재하는 게이트를 가리킨다(스모크/CI에서 실행). HRET·GroundLM 심사자,
그리고 벤치를 돌리는 제3자가 정확히 이 문서를 묻는다.

## 1. 버전 관리 — 무엇이 "벤치 버전"인가

벤치 결과의 유효성은 아래 **튜플**로만 정의된다:

    bench_version = (faithbench_version, faithbench_partial_version, citation_verify_version,
                     corpus_sha256, questions_sha256)

- 스코어러 버전은 각 코드 상수(`FAITHBENCH_VERSION`, `PARTIAL_VERSION`, `SCORER_VERSION`)에 있고,
  현재 **faithbench v0.3 / partial v0.1 / citation_verify v0.2**로 동결.
- `corpus_sha256`·`questions_sha256`는 result의 `meta`(run_meta.py)에 기록된다.
- **채점 규칙을 바꾸면 버전 bump + golden 재생성이 강제된다** — `scripts/eval/check_scorer_frozen.py`가
  전체 aggregate(leak 유형학 포함)를 golden과 byte-exact 비교해 스모크에서 깨진다.
- **모델 비교 주장은 동결된 버전 기준에서만** 유효(g0-verdict.md §6). 버전이 다르면 수치를 섞지 않는다.

## 2. public / holdout 분리

- 순위는 **비공개 holdout**으로 매긴다. 공개셋은 예시·재현용이다.
- curated eval은 `eval/curated_law_seed.json`에서 결정적으로 재생성되며, 학습 데이터(holdout SFT)에
  eval ID가 **하나도** 들어가지 않아야 한다.
- 강제 게이트: `scripts/data/verify_curated_holdout.py` — (a) tracked eval == seed 재생성(drift=0),
  (b) holdout SFT에 curated ID gold/context 누출=0. 코퍼스 없으면 SKIP, 있으면 하드 게이트.

## 3. 오염(contamination) 방지

- **provenance 전 추적**: 코퍼스 각 조항은 law_id·시행일·source_url·raw_sha256을 갖는다.
  `scripts/data/verify_provenance_chain.py`가 raw_sha256을 디스크 raw 파일과 대조(제3자 재검증).
- **오염원 배제**: AI-Hub/모두의말뭉치/나무위키/KoAlpaca 등 비상업·GPT유래 데이터는 코퍼스·학습에서 배제.
- **표면 단서 제거**: leak 축은 표면 法名 단서가 아니라 제공근거 판독으로만 풀리게 설계
  (gold-ablation 프로브: 정답 조문을 근거에서 제거해 in-domain 거절을 측정).
- **디컨탐**: 공개 전 n-gram 디컨탐 + 오염0 증빙(로드맵).

## 4. 갱신 케이던스 (정직한 지속성)

1인 프로젝트가 지키지 못할 "분기 갱신" 같은 달력 약속은 하지 않는다. 대신 **이벤트-트리거**:

- **법령 개정**이 감지되면(시행일 기반 diff) 영향 문항을 무효화·갱신한다.
- **벤치 포화/오염**이 관측되면 새 버전(holdout 교체)을 동결·아카이브한다.
- 각 버전은 동결 후 아카이브하며, 이전 버전 수치와 섞지 않는다.
- 유지 노동은 감사·컨설팅 매출이 자기펀딩하도록 매출 엔진과 연결(전략 ⑨).

## 5. 재현 (reproducibility)

- **재현 스크립트 100% 공개.** 결과는 result JSON + transcript로 남기고,
  `scripts/eval/score_predictions.py rescore`로 **GPU 없이 CPU에서 수초 만에 재도출**된다.
- 스모크에 재현성 게이트: 공표 v02 결과가 현 스코어러로 재도출됨을 매 CI에서 검증.
- 리포트 표는 `scripts/eval/fingerprint_report.py`로 result JSON에서 결정적 생성(수기 전사 금지).
- 제3자 제출: `eval/SUBMISSIONS.md` — instance_id로 답변만 제출하면 결정적 채점.
- (로드맵) 비-sm_120 CPU 폴백 재현 컨테이너.

## 6. 실명 랭킹 — measurement, not accusation

경쟁·상용 모델을 실명으로 채점할 때:

- **단정·선정 표현 금지.** "X 스택이 샌다"가 아니라 "**X 설정·Y 버전에서 Z 케이스가 재현됨(스크립트 첨부)**"의
  사실·재현 서술로만.
- **지는 항목을 정직 병기**한다(우리 모델이 지는 축도 그대로 공개).
- **발행 전 vendor right-of-reply**: 대상 벤더에 사전 통지하고 반론을 반영/병기한다(아래 템플릿).
- 개인 실명 노출 대신 프로젝트 브랜드로 발행.

### right-of-reply 통지 템플릿

```
제목: [프로젝트명] 벤치 결과 사전 통지 및 반론 요청 — {모델명} {버전}

안녕하세요. {프로젝트명}은 한국 법령 인용 근거충실도를 LLM-judge 없이 결정적으로 측정하는
오픈 벤치입니다. {날짜} 공개 예정 리포트에 귀사 모델 {모델명}({버전})의 측정 결과가 포함됩니다.

- 측정 설정: 코퍼스 sha256 {corpus_sha}, 질문셋 sha256 {questions_sha}, 스코어러 {버전 튜플},
  seed/k/near {값}. 재현 번들: {링크/커맨드}.
- 결과 요약(축별): {selection_exact / leak_rate / partial 등}. 귀사가 강한 항목과 약한 항목을 모두 병기했습니다.
- 이는 특정 설정에서의 재현 가능한 측정이며 제품 전반에 대한 단정이 아닙니다.

발행 전 {N일} 내에 방법론 이의나 반론을 주시면 리포트에 반영·병기하겠습니다.
재현 스크립트로 직접 검증하실 수 있습니다: {커맨드}
```

## 7. 유지 주체·법적 방패

- 발행은 개인 실명이 아니라 프로젝트 브랜드로.
- 공동 유지 주체(HRET/랩 공동서명) 확보를 지속성 목표로 둔다.
- 방법론·스크립트 완전 투명이 이의·비방 소지를 선제 제거한다.
