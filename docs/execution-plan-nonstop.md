# Autopilot: 한국어 LLM 전략 — 무중단 자율 실행 계획 (T0·A·B·C 종합, rev2)

> 본 문서는 4개 트랙(T0 환경 / A 빌드·정직화 / B 유명세 / C 환금)의 단락 분해에 검증 피드백을 반영한 **rev1**을, 다시 **2차 비평(missing·unrealistic_autonomy·weak_success_criteria·gate_logic_issues)을 전량 검토해 타당한 지적만 반영**한 최종본(rev2)이다. 핵심 개정: (a) **G0가 검증할 아티팩트(파인튜닝된 grounder)가 존재하도록** 미니 파인튜닝을 G0 앞으로 이동, (b) 모든 게이트에 **사전등록(pre-registered) 수치 임계** 부여, (c) **이중 위치 게이트 분리 명명**(G2-adopt/G2-build, G3-approve/G3-close), (d) **teacher 라이선스·풀스케일 데이터·human 채점·safety eval·예산 원장·미러 백업·롤백·이식 컨테이너·검정력** 누락 단락 신설, (e) **external '60m' 낙관 추정을 정직한 hours-scale ETA 모델로 대체**하고 도표에서도 무중단 구간이 짧음을 명시. 이 사용자의 autopilot plan.md 관례(단락=1커밋, 측정가능 성공조건, worktree 병렬 최대5, fail-fast 게이트)를 따른다.

---

## ① 개요

### 목표
한국 법령 closed-set 기반 **grounding(인용 충실도)·leak 측정 자산**을 소유해 (1) 소형 오픈웨이트(Nova-Ko-Grounder 2~4B) 릴리스로 명성을 트리거하고, (2) 그 신뢰 신호를 **부품(OEM leak-verify) + 사설 감사**라는 매출 엔진으로 환금한다. **명성 엔진과 매출 엔진은 절대 한 빌드에 묶지 않는다**(hard_constraint).

### '무중단'의 정직한 정의 — 경계 (도표 차원에서 축소 명시)
| 구분 | 의미 | 대상 |
|---|---|---|
| 🤖 **claude-auto (무중단 실행 본체)** | 다음 게이트까지 **확인질문 없이 연속** 단락 commit+진행 | 로컬 코드·cargo check·docs 정직화·python 스캐폴딩·**mock smoke**·ruff·grep 검증 |
| 🌐 **external (알람 후 대기)** | Claude 실행 가능하나 **GPU 장시간·대용량 다운로드·API 과금**으로 wall-clock 수십분~**수시간~수일** → 무중단 흐름에서 분리, **완료/정지 알람 필수** | cu128 휠·모델 가중치 다운로드, bulk fetch, QLoRA 학습, 폐쇄 API 채점 |
| 🙋 **user-gate (개입 필요)** | 사람만 가능 → 폰 알람 후 정지. **대부분의 환금·유명세 트리거가 여기** | HF 토큰·OpenAPI 키 발급, teacher 라이선스 결정, bnb sm_120 실패 시 피벗, **릴리스 공개**, 가격·파트너·계약, **WDAC 우회 테스트(WSL2/CI)**, 브랜치 화해·크레이트 경계, **human 채점자 섭외**, **게이트 최종 판정 승인** |

> **정직 고지 — 무중단 본체의 실제 지속시간**: claude-auto로 확인 없이 연속 실행되는 구간은 **정직화(TA-01/02)+P1 코드픽스(TA-03a/04a/05a)+스캐폴딩(TA-07/09/11a) = 대략 6~9시간의 순수 로컬 작업**이 사실상 전부다. 이 지점 이후 하류는 **거의 모두 T0-08(external QLoRA)·bulk fetch·user-gate에 걸린다.** ②실행그래프의 병렬선은 이 짧은 본체와 그 뒤의 external/user 대기를 **시각적으로 구분(🤖 실선 / 🌐·🙋 점선)**한다. "돈이 들어오는 순간"과 "명성이 발화하는 순간"은 무중단 대상이 아니며, 이를 도표에서도 숨기지 않는다.

### 성공 정의
- **단기(무중단 본체)**: T0 환경 green → 정직성 폭탄 0건 → RAG citation 배선 → P1 동시성 코드 수정(cargo check) → 법령 fetcher·합성기 코드 → **G0-pre 프리프로브 발화**.
- **게이트 성공**: 각 게이트는 **사전등록 수치 임계**로 판정(④표). 통과/미달과 무관하게 판정 자체가 성공 — 측정 자산은 어느 게이트 미달에도 생존.

---

## ② 실행 그래프 (트랙 의존·병렬·게이트 / 🤖 실선=무중단, 🌐·🙋 점선=대기)

```
[T0 환경 — 모든 학습/데이터 트랙의 하드 선행]
 T0-00 driver+toolchain preflight ─(G-env, fail-fast)─┐
   │ 통과: cu128/sm_120 진행   │ 미달: SDPA+양자화백엔드 별도판정 or 클라우드GPU(user-gate)
   ▼
 T0-01 venv/ruff/preflight_drive
   ├─▶ T0-02 torch cu128 [🌐 GATE·alarm] ─┬─▶ T0-03 reqs-core(분리·cu128 assert)
   │                                       └┈▶ T0-05 bnb 4bit [🌐/🙋fail·alarm]
   ├─▶ T0-04 HF토큰 [🙋 run시작 前 프리체크]
   ├─▶ T0-10 전역 예산 원장 + 백업정책 [🤖]      ← 신설(SPOF·예산 SPOF 방어)
   └─▶ T0-11 코퍼스·가중치 F: 미러 백업 훅 [🤖]   ← 신설
                    T0-06 accel/ SDPA폴백 ─┐
   T0-07 model load smoke ◀── T0-04,05    ├┈▶ T0-08 QLoRA micro smoke [🌐 alarm·infra-gate]
   T0-09 verify_env 오케스트레이터 ◀── 01,02,03,04,05,06,07,10,11

──────────────────────────────────────────────────────────── 🤖 무중단 본체(실선, ~6~9h) 시작
[A 0-3mo]  정직화(TA-01/02) ⟂ 동시성/RAG(TA-03~05) ⟂ 데이터(TA-06~10)  ← worktree 병렬(≤5)
 정직화: TA-01, TA-02  [🤖 완전 독립·즉시 병렬]
 코드픽스: TA-03a→TA-05a (handlers/rag.rs 공유→직렬) / TA-04a 독립
        └┈ 검증부 TA-03b/04b/05b [🙋 WSL2·CI — WDAC 우회]
 teacher: TA-06T teacher 선정·라이선스 검증 [🤖 grep+🙋 라이선스 결정]  ← 신설(환금 법적 토대)
 데이터: TA-06[🙋키]→TA-08[🌐]  · TA-07[🤖 depends[]]  · TA-09[🤖mock]→TA-10[🌐]
 ┈┈ G0-pre 저비용 프리프로브(공개셋, 사전등록 primary metric) ┈┈ 코퍼스 풀투자 前 발화
 TA-00 저비용 G0-pre 프리프로브 [🌐 alarm]
 학습config: TA-11a[🤖] · TA-11c 미니 grounder QLoRA(파일럿 데이터) [🌐] ← 신설: G0가 평가할 실아티팩트
   ▼
 TA-12 G0 정식 판정 [🙋 GATE·판정 승인·alarm]  ◀── TA-10,TA-11c,T0-08
   │ 통과: '소형 grounder가 base 7~14B 동급이상' 서사 허용 → B   │ 미달: '측정을 소유한다'로 축소 + base 상향(16GB 실현성 재확인)

──────────────────────────────────────────────────────────── 🌐·🙋 점선(external/게이트 대기)
[B 3-6mo]  ※ 진입 전제 = G0 판정 완료 + (릴리스는 safety+경쟁력 게이트 별도)
 TB-00F 풀스케일 합성데이터(파일럿→릴리스급) [🌐 수시간~수일·alarm] ← 신설: TB-0T가 smoke데이터로 학습되던 dangling 해소
   → TB-0T Nova-Ko-Grounder 릴리스급 QLoRA 학습 [🌐 수시간~수일·alarm]
 TB-01 K-LeakBench 하네스[🤖] → TB-02a sm_120 추론게이트 → TB-02b 오픈채점[🌐alarm]
   ▼ TB-03 최소리포트+게시 [🙋 GATE G1·alarm]
        │ 통과: 확장  │ 미달: 감사서비스로 무게중심 이동
   ── G1 통과 시에만 아래 비싼 저작·폐쇄API 진행(gate 배선) ──
 TB-04 K-FaithBench 스키마·스코어러[🤖] → TB-04p 검정력·최소N 산정[🤖]  ← 신설
   → TB-04b 데이터저작(N=검정력산정값)[🤖, ★G1 뒤 게이트] → TB-05H human 채점 수집[🙋]  ← 신설
   → TB-05 holdout·judge메타[🤖]
   → TB-06a 오픈baseline[🌐] · TB-06b 폐쇄baseline[🙋키·예산·alarm, ★G1 뒤]
 TB-07 모델카드[🤖] → TB-07S safety/bias/toxicity eval[🤖→🙋 공개판정]  ← 신설
   → TB-08 HF업로드[🙋 GATE·safety+경쟁력] ; TB-09 테크리포트[🤖 depends TB-07]
 TB-13 이식가능 재현 컨테이너(CPU/비-sm_120 폴백)[🤖]  ← 신설: 제3자 재현 가능화
 TB-10 HRET/KMMLU 교차+공동서명 [🌐/🙋 GATE G2-adopt·alarm]  ◀── TB-08,TB-13
 TB-11 법령 인용검증 데모[🤖 ← A 법령코퍼스]
 TB-12 브릿지 감사 템플릿 [🌐 GATE G3-close 준비·alarm]

──────────────────────────────────────────────────────────── 🙋 하드 선행(대형 통합)
[C 6-12mo]  ※ 진입 전제 = G0·G1 통과 신호
 TC-00 브랜치 화해(main WireVerifier + semcache work)+크레이트 경계 [🙋 대형작업·다세션·alarm]
 TC-00c 모델/API 크리덴셜 프로비저닝 [🙋]
   ▼
 TC-01 4계층 통합 스코어러[🤖 ※3-브랜치 머지 후 반복 빌드] → TC-02 leak-verify 크레이트 추출[🤖]
   → TC-03 OEM 데이터시트 · TC-04 감사 CLI(mock) · TC-06 온프렘번들 · TC-07 사설코퍼스
 TC-05 감사 리포트 템플릿(+법률검토 게이트) · TC-08 holdout거버넌스 → TC-09 예산 오케스트레이터[← G2-adopt]
   ── G2-adopt 미달 시 TC-06·TC-10 투자 차단(gate 배선) ──
 TC-10 파트너 콜래터럴 초안[🤖 alarm] → TC-11 가격/타겟 승인[🙋 GATE G3-approve] → TC-12 BD실행[🌐 월단위·GATE G3-close]
```

**병렬 선언**: A트랙 무중단 본체는 **최대 5 worktree 병렬** — 그룹① 정직화(TA-01,02) / 그룹② 코드픽스(TA-03a→05a 직렬 + TA-04a) / 그룹③ 데이터코드(TA-07,09) / 그룹④ 학습config(TA-11a) / 그룹⑤ teacher grep(TA-06T-a). handlers/rag.rs·nexusflow-server 공유 단락은 직렬화. **이 5병렬은 ~6~9h 구간에만 유효하며, 이후는 external/user 직렬 대기임을 명시.**

---

## ③ 트랙별 단락 테이블

### T0 — 환경·인프라 (F:\nova-llm, 모든 학습/데이터 트랙 하드 선행)

| id | 제목 | 성공조건(측정가능·사전등록 임계) | 의존 | 자율 | 시간 | 게이트 |
|---|---|---|---|---|---|---|
| **T0-00** | driver+빌드툴체인 fail-fast preflight | `nvidia-smi` driver≥570 + `where cl`/`where nvcc` 감지 → 부재 시 **SDPA 라우팅 로그**. **단, bnb sm_120 양자화 백엔드는 별도 판정(T0-05)로 분기** — cl/nvcc 부재≠양자화 해결. 2.5GB 다운로드 前 exit | — | 🤖 | 15m | **G-env** |
| T0-01 | venv+ruff+preflight_drive | `python -m venv .venv` + `ruff check` 에러0 + `preflight_drive.py`가 F: 마운트·쓰기·여유>50GB exit0 | T0-00 | 🤖 | 30m | |
| T0-02 | PyTorch cu128 + sm_120 실증 + **처리율 계측** | `verify_torch.py`: is_available=True, capability=(12,0), fp16 matmul + **tokens/s·GB/s 벤치 1회 → docs/env-verify/perf.json**(external ETA 모델의 입력) | T0-01 | 🌐 | ~40m | **✅alarm** |
| T0-03 | reqs-core 분리 설치 + import 스모크 | requirements-core.txt만 설치 + `torch.__version__`에 **'+cu128' assert**(클robber 방지) + 전 모듈 import exit0 | T0-02 | 🌐 | 30m | |
| T0-04 | HF 토큰·캐시(HF_HOME=F) | `huggingface-cli whoami` 반환 + `.env` git check-ignore PASS. run 시작 前 프리체크 | T0-01 | 🙋 | 15m | ✅alarm |
| T0-05 | bnb sm_120 4bit forward + **tolerance bound** | `verify_bnb.py`: Linear4bit cuda forward 에러0 + **round-trip 상대오차 < 5e-2**(NaN-only 통과 금지). **실패=user-gate 피벗**(torchao/HQQ/full-LoRA) | T0-03 | 🌐/🙋 | 30m | ✅alarm |
| T0-06 | unsloth/flash-attn OR SDPA 정식폴백 | `verify_accel.py`: (A)unsloth OR (B)`attn_implementation='sdpa'` 동작 — 채택경로 로그 후 exit0 | T0-03,T0-05 | 🌐 | 30m | |
| T0-07 | 2.3~4B 4bit 로드 스모크+VRAM + **처리율** | Qwen3-4B(Apache)/Mi:dm-mini-2.3B(MIT) 4bit 로드 + max_mem<16GB + generate 비어있지않음 + **generate tokens/s 계측→perf.json** | T0-04,T0-05 | 🌐 | 45m | |
| T0-08 | QLoRA 마이크로 스모크(no-OOM) + **유의 감소** | `smoke_qlora.py` r=8~16 ≥**50step** OOM0 + **loss 이동평균 초기 대비 ≥10% 하락**(10step 노이즈 금지) + peak<16GB | T0-06,T0-07 | 🌐 | 50m | infra→G0 ✅alarm |
| **T0-10** ⭐신설 | 전역 예산 원장 + 백업정책 | `budget_ledger.jsonl`(누적 지출 단일 장부) + **글로벌 상한·트랙 간 합산 kill-switch** 단위테스트 PASS(per-segment만이던 결함 해소). 모든 API/GPU 과금 단락이 append 강제 assert | T0-01 | 🤖 | 40m | |
| **T0-11** ⭐신설 | 코퍼스·가중치 F: 미러 백업 훅 | `mirror_backup.py`: 데이터 단락 산출물을 **2차 위치(F: 외 로컬/외장)로 rsync 미러** + 무결성 SHA-256 매니페스트. F: 단일점(SPOF) 완화. 복원 스모크 PASS | T0-01 | 🤖 | 40m | |
| T0-09 | reqs.lock + verify_env 오케스트레이터 | `pip freeze>requirements.lock` + `verify_env.py`가 preflight→torch→stack→hf→bnb→accel→ledger→backup 순차 ALL PASS(개별실패 표기) + ENV_SETUP.md | 01~07,10,11 | 🤖 | 30m | |

> 검증반영: G-env에 **bnb 양자화 백엔드 별도 판정** 명시(SDPA≠양자화 해결), T0-05 **tolerance bound(<5e-2)**, T0-08 **≥50step·≥10% 하락**(10step 노이즈 제거), **T0-02/07에 처리율 계측 추가**(external ETA 근거), **T0-10 예산 원장·T0-11 미러 백업 신설**.

### A — 0~3개월 빌드 & 정직화

| id | 제목 | 성공조건 | 의존 | 자율 | 시간 | 게이트 |
|---|---|---|---|---|---|---|
| **TA-00** | 저비용 G0-pre 프리프로브(공개셋) | 공개 한국어 faithfulness 셋으로 2~4B vs 7~14B 조기 비교 — **사전등록 primary metric(citation exact-match) 단일 지표**에서 소형-우위 판정(bootstrap 95% CI). **다지표 중 ≥1 cherry-pick 금지** | T0-08 | 🌐 | ~40m | G0-pre ✅alarm |
| TA-01 | 정직성① at-rest/KMS 허위광고 삭제 | README·SECURITY·화이트페이퍼에서 'AES-256-GCM at-rest'·'KMS/rotation' 실동작 광고 grep **0건** + **삭제 diff 사람 승인 알람**(grep 프록시 한계 보완). 코드 미접촉 | — | 🤖 | 45m | ✅alarm |
| TA-02 | 정직성② dead-infra 광고 0건 | RAPTOR·multi-vector·GraphRAG·self-RAG재생성·USING INVERTED/GIN·BM25F·양자화75%·HNSW DELETE 광고 grep **0건** + 실재범위만 남았는지 **diff 사람 승인**. 실재(하이브리드·형태소FTS BM25·HNSW filtered) | — | 🤖 | 45m | ✅alarm |
| TA-03a | RAG [n] 인용 프롬프트+파서 코드 | cargo check0 + 두 프롬프트에 '[1]처럼 표기' grep + parse_citation_indices 완성 | — | 🤖 | 30m | |
| TA-03b | citation 파서 단위테스트 실PASS | `'...[1][2]'→citationIds=[1,2],cited=true` 테스트 PASS **(WSL2 1회 or CI)** | TA-03a | 🙋 | 15m | ✅alarm |
| TA-04a | graph_rag 락순서 통일(코드) | 3함수 **nodes→reverse_edges 통일** + cargo check0 + 코드리뷰 | — | 🤖 | 35m | |
| TA-04b | 동시성 hang-0 실측 | A upsert × B add_relation N회 동시→타임아웃내 완주 hang0 PASS **(WSL2/CI)** | TA-04a | 🙋 | 20m | ✅alarm |
| TA-05a | negative_cache 테넌트격리(코드) | make_key/register/filter를 (collection,tenant,question_hash)로 + handle_rag_feedback에 tenant+doc_id 소유검증 + cargo check0 | TA-03a | 🤖 | 40m | |
| TA-05b | poisoning 차단 회귀 실PASS | 테넌트A down이 B 검색서 제외 안 됨 회귀 PASS **(WSL2/CI)** | TA-05a | 🙋 | 15m | ✅alarm |
| **TA-06T** ⭐신설 | teacher 모델 선정·라이선스 검증 | 후보 teacher(예: Qwen2.5-72B Apache 등 **상용 파생물+공개릴리스+OEM 재배포 허용** 라이선스) 표 작성 + **라이선스 조항 grep+원문 링크** + `teacher_allowlist.json` 하드가드(OpenAI/Anthropic 차단은 유지, **NC teacher도 차단**). **최종 채택=🙋 라이선스 결정 승인** — 환금 트랙 법적 토대 | — | 🤖→🙋 | 40m | ✅alarm |
| TA-06 | 법령/DART OpenAPI 키 발급 | docs/data-sources.md(발급URL·KOGL1·오염원회피) + .env placeholder. 사용자가 law.go.kr OC·opendart 키 입력 | — | 🙋 | 25m | ✅alarm |
| TA-07 | 법령 fetcher+provenance(코드) | `10_fetch_legal_corpus.py --smoke`(번들샘플·키 불요): 조문청킹 jsonl + provenance 6필드 100% + 화이트리스트 assert. depends[]로 무중단 | — | 🤖 | 45m | |
| TA-08 | 법령 코퍼스 bulk fetch | manifest.json(**목표건수 명시·provenance100%·오염0·KOGL1**) 커밋. 원본 gitignore + T0-11 미러. **rate-limit 종속 → wall-clock은 ETA 모델(perf.json+요청간격)로 추정·표기** | TA-06,TA-07 | 🌐 | 수십분~수시간 | |
| TA-09 | 합성 SFT 생성기(mock smoke) | `11_synthesize_sft.py --smoke`(순수 mock·표준라이브러리만): positive([n])·negative·환각 각≥1 + citation substring exact-match + **teacher는 TA-06T allowlist assert** | TA-07,TA-06T | 🤖 | 45m | |
| TA-10 | 파일럿 합성 1~2만 실행 | sft_pilot.jsonl 검증통과 1~2만 + manifest(exact-match통과율·오염0·비율·계보·**teacher모델·토큰비용→ledger 기록**). **진짜 teacher 추론(GPU/API)이므로 wall-clock·예산은 ETA/ledger로 표기, mock 아님 명시** | TA-08,TA-09,TA-06T | 🌐 | 수시간(ETA 표기) | |
| TA-11a | QLoRA config 2~4B + license_guard | train_config.yaml(Qwen3-4B/Mi:dm-mini) + license_guard가 NC모델(EXAONE4.0/Kanana-2/Tri-21B) 차단 assert PASS. GPU-free 커밋 | — | 🤖 | 30m | |
| **TA-11c** ⭐신설 | 미니 grounder QLoRA(파일럿 데이터) | 파일럿 SFT로 2~4B **실제 파인튜닝** 1에폭 완주 + adapter 저장 — **G0가 평가할 grounder 아티팩트를 G0 前에 생성**(base만 평가하던 게이트 결함 해소). smoke급임을 명시(릴리스급은 TB-0T) | TA-10,TA-11a,T0-08 | 🌐 | 수시간(ETA) | ✅alarm |
| TA-12 | **G0 정식 판정** | g0_report.md: **파인튜닝된 미니 grounder(2~4B) vs base 7~14B** K-FaithBench 프로토타입 — **사전등록 primary=citation exact-match, margin Δ≥+2pt & bootstrap 95% CI 하한>0** 통과. 판정=🙋 승인(자동 임계+사람 확정). 미달=서사축소+base상향 | TA-10,TA-11c,T0-08 | 🙋 | 60m | **G0** ✅alarm |

> 검증반영: **TA-06T teacher 라이선스 검증 신설**(환금 법적 토대), **TA-11c 미니 파인튜닝을 G0 앞으로**(G0가 base가 아닌 grounder 평가 — gate_logic 핵심 결함 해소), TA-00/TA-12에 **사전등록 단일 primary metric+margin+CI**(cherry-pick·비수치 제거), TA-01/02에 **diff 사람 승인**(grep 프록시 보완), TA-10을 **mock 아닌 실 teacher 추론+예산 ledger**로 정직 표기, TA-08/10 **ETA 모델 표기**.

### B — 3~6개월 유명세 트리거 (진입 전제: G0 판정 완료)

| id | 제목 | 성공조건 | 의존 | 자율 | 시간 | 게이트 |
|---|---|---|---|---|---|---|
| **TB-00F** ⭐신설 | 풀스케일 합성데이터(릴리스급) | 파일럿(1~2만)→릴리스급(예: 10만+) 확장 실행 + manifest(exact-match·오염0·teacher·ledger). **TB-0T가 smoke데이터로 학습되던 dangling 해소**. wall-clock·예산 ETA/ledger 표기 | TA-10,TA-06T | 🌐 | 수시간~수일 | ✅alarm |
| **TB-0T** | Nova-Ko-Grounder 릴리스급 QLoRA 학습 | **풀스케일 데이터로** 2~4B QLoRA distillation ≥N에폭 완주(수렴 로그) + adapter/merged 저장. **60m smoke가 아닌 릴리스급 — 실 wall-clock 수시간~수일 ETA 표기** | TB-00F,TA-11a,T0-08 | 🌐 | 수시간~수일 | ✅alarm |
| TB-01 | K-LeakBench 하네스+leak스코어러 | pytest PASS(≥10 probe) + holdout/public 물리분리 누출0 + **결정적** leak 판정 단위테스트 green | T0-09 | 🤖 | 45m | |
| TB-02a | sm_120 추론 게이트 | 모델 1종 로드+추론 성공(다운로드 前 툴체인 확인) | TB-01,T0-08 | 🌐 | 20m | ✅alarm |
| TB-02b | 오픈모델 3~4종 로컬 채점 | Qwen3/DeepSeek-R1-distill/Llama/한국sLLM per-probe leak-rate JSON + 1커맨드 재현 + 폐쇄API호출0 로그 | TB-02a | 🌐 | ETA 표기 | ✅alarm |
| TB-03 | K-LeakBench 최소리포트+게시 | measurement-not-accusation 중립린트+**사람 중립성 리뷰 1회**+right-of-reply+재현번들 tar. 법인브랜드 게시=승인. **G1 임계: 게시 후 72h 내 (HF 다운로드 or GitHub star/issue) ≥사전등록 N OR 제3자 서면 문의 ≥1** | TB-02b | 🙋 | 40m | **G1** ✅alarm |
| TB-04 | K-FaithBench v1 스키마+기계채점기 | 스키마 PASS + citation exact-match·refusal 스코어러 단위테스트 green | T0-09 | 🤖 | 40m | |
| **TB-04p** ⭐신설 | 검정력·최소 N 산정 | '소형이 이긴다' 검증에 필요한 **최소 표본 N을 검정력 분석(예: power 0.8, 탐지 margin Δ)으로 산정** + `power_report.md`. ≥50 임의값 대체 — 벤치 신뢰도 근거 | TB-04 | 🤖 | 40m | |
| TB-04b | K-FaithBench 데이터 저작(N=산정값) | **TB-04p 산정 N** 객관채점 항목 로드 + 스키마 검증 PASS(스코어러와 분리 커밋). **★G1 통과 뒤 실행(fail-fast)** | TB-04p,**G1통과** | 🤖 | 45m~ | |
| **TB-05H** ⭐신설 | human 채점 수집 프로세스 | **채점자 ≥2명 섭외(🙋)·가이드라인·이중채점·Cohen's κ 산출** → human_agreement 필드 실제 채움. '필드 채움'으로 위장하던 user-gate 명시화 | TB-04b | 🙋 | 프로세스(일단위) | ✅alarm |
| TB-05 | holdout 격리+judge오차메타+디컨탐 | holdout↔공개 물리분리 누출0 + **TB-05H의 κ·human_agreement 반영** + n-gram 디컨탐 green | TB-05H | 🤖 | 45m | |
| TB-06a | 오픈 baseline 리더보드 | 오픈모델 로컬 채점 리더보드 JSON(예산0) | TB-05,**G1통과** | 🌐 | ETA 표기 | ✅alarm |
| TB-06b | 폐쇄 baseline(예산상한·ledger) | GPT/Claude/Trillion/Safeguard 버전당1회+캐시, 자기열세항목≥1 병기, **T0-10 전역 ledger에 append+글로벌 kill-switch**. 키·예산=사용자. **★G1 뒤** | TB-06a,**G1통과** | 🙋 | 40m | ✅alarm |
| TB-07 | Nova-Ko-Grounder 모델카드 | 카드 린트 PASS + eval표=TB-06 수치일치 + base non-NC(Apache/MIT) + teacher 라이선스 명시 + SHA-256. 소형-우위는 **G0 통과 시에만** 기재 | TB-06b,TB-0T | 🤖 | 40m | |
| **TB-07S** ⭐신설 | safety/bias/toxicity eval | 공개 릴리스 前 **독성·편향·거절 안전성 eval**(한국어 셋) 실행 + `safety_report.md` + **사전등록 임계 미달 시 공개 차단**. 명성 역풍 무게이트 결함 해소. 공개판정=🙋 | TB-0T,TB-07 | 🤖→🙋 | 50m | safety ✅alarm |
| **TB-13** ⭐신설 | 이식가능 재현 컨테이너 | sm_120 없는 제3자용 **CPU/비-sm_120 폴백 Docker + 결정적 시드** — 재현번들 tar만으로 불가하던 G2-adopt 재현을 실제 가능화. 타 하드웨어 재현 스모크 PASS | TB-01,TB-04 | 🤖 | 55m | |
| TB-08 | HF 업로드+다운로드훅 | HF 페이지 라이브+카드 렌더+다운로드 스모크. **공개 전제: G0 통과(경쟁력) AND TB-07S safety 통과** — 미충족 시 공개 차단(비경쟁·역풍 모델 릴리스 방지). 결정=사용자 | TB-07,TB-07S | 🙋 | 30m | ✅alarm |
| TB-09 | 'Grounding Distillation' 테크리포트 | 리포트 빌드 + 재현번들 fixture e2e PASS + 표/그림 커밋데이터 재생성(하드코딩0). arXiv=별도 user-gate. depends TB-07(라이브HF 불요) | TB-06b,TB-07 | 🤖 | 55m | |
| TB-10 | HRET/KMMLU/HAERAE 교차+공동서명 | HRET 재현+오염미검출 + 상관표 + **TB-13 컨테이너로 제3자 재현 경로 제공**. **G2-adopt 임계: 독립 재현 ≥2건 OR 공동서명 기관 ≥1** | TB-08,TB-13 | 🌐/🙋 | ETA 표기 | **G2-adopt** ✅alarm |
| TB-11 | 법령 closed-set 인용검증 데모 | 조문 exact-match 정확도 + **테스트셋 크기 N 명시 + false-negative 상한을 N 기반 신뢰구간으로 보고(절대 0 주장 금지)** + provenance 로그 + 1커맨드 PASS | TB-04,TA-08 | 🤖 | 45m | |
| TB-12 | 브릿지 감사 템플릿+dry-run | 템플릿 + 샘플RAG dry-run 리포트 PASS. 실계약=외부(G3-close) | TB-03 | 🌐 | 50m | G3-close 준비 |

> 검증반영: **TB-00F 풀스케일 데이터 신설**(smoke데이터 학습 dangling), TB-0T를 **릴리스급·수시간~수일**로 정직화, **TB-04p 검정력 산정**, **TB-04b/06b를 G1 뒤로 게이트**(fail-fast 일관성), **TB-05H human 채점 프로세스 신설**, **TB-07S safety eval+TB-08 릴리스 게이트**(역풍 차단), **TB-13 이식 컨테이너**(G2 재현 가능화), TB-11 **N 명시·CI**(false-negative0 반증불가 제거), TB-06b **전역 ledger**.

### C — 6~12개월 확장 & 환금 (진입 전제: G0·G1 통과 신호)

| id | 제목 | 성공조건 | 의존 | 자율 | 시간 | 게이트 |
|---|---|---|---|---|---|---|
| **TC-00** | 브랜치 화해 + 크레이트 경계 결정 | main WireVerifier + semcache work + 격리 work **3자 통합 머지**(rerere 자동해결 확인, 의미충돌 보고) + 크레이트 위치 결정. **MEMORY에 '블로커'로 기록된 대형 통합 — 60m 승인이 아니라 다세션·반복 빌드 작업임을 명시. 실소요 수시간~수일 ETA** | — | 🙋 | 대형(일단위) | 하드선행 ✅alarm |
| **TC-00c** | 모델/API 크리덴셜 프로비저닝 | baseline 오픈모델 HF 승인+다운로드 + 폐쇄 API키·예산(models/base 채움·**ledger 등록**) | — | 🙋 | 30m | ✅alarm |
| TC-01 | 4계층 leak-verify 통합 스코어러 | cargo build 성공 + 교차테넌트 프로브 leak-0 통합테스트 + 4계층 단일경로 assert. **TC-00 3-브랜치 머지 직후 반복 빌드 전제 — 단발 60m 성공 가정 아님, 빌드 실패 시 롤백 절차(⑥)** | TC-00 | 🤖 | 반복 빌드 | |
| TC-02 | leak-verify 독립 크레이트 추출 | `cargo build -p leak-verify` 단독 + `cargo tree` bench/weights 의존 엣지0 + 심볼 문서화 | TC-01 | 🤖 | 45m | |
| TC-03 | OEM 데이터시트+호환성 매트릭스 | dead-infra 금지어 grep0 + API 문서가 TC-02 공개심볼 전량 커버. 게시=user-gate | TC-02 | 🤖 | 30m | |
| TC-04 | 사설 감사 하네스 CLI(mock-only) | `audit.run --endpoint <mock>` e2e 리포트 + 예산상한 abort 테스트 PASS + **전역 ledger append**. 실채점=측정프리미티브(←TB-04/05)+TC-00c 후 게이트 | TC-02,TB-05 | 🤖 | 60m | |
| TC-05 | 감사 리포트 템플릿(+법률검토) | 렌더 + right-of-reply·judge오차 섹션 + 금지어 lint0 + **사람 중립성 리뷰**. 실명벤더 발송 前 법률검토 게이트 | TC-04 | 🤖 | 30m | 법률 ✅alarm |
| TC-06 | 온프렘 데모 번들 | 로컬 기동 + leak-verify+감사 e2e 무오류 + dead-infra 금지어0. **G2-adopt 통과 후 착수** | TC-01,TC-04,**G2-adopt** | 🤖 | 45m | |
| TC-07 | 사설 코퍼스 인제스트 파이프라인 | 전 레코드 provenance100% + 오염원 시드 거부 PASS + 테넌트격리 assert | — | 🤖 | 45m | |
| TC-08 | 벤치 거버넌스 holdout+디컨탐 | 시드오염 검출 PASS + holdout↔공개 disjoint assert | — | 🤖 | 45m | |
| TC-09 | 예산상한 자동채점 오케스트레이터 | 자동채점 + 예산초과→오픈모델 대체 + 캐시히트 중복API0 + **전역 ledger 연동**. **G2-adopt 통과에 게이트(빌드 작업이지 게이트 자체 아님 — G2-build 태그 제거)** | TC-08,TC-00c,**G2-adopt** | 🤖 | 45m | (← G2-adopt) |
| TC-10 | 파트너 콜래터럴 초안(LOI/SOW/NDA) | 초안3종+가격 placeholder+검토 체크리스트+measurement-not-accusation·익명화 배타조항 grep. **G2-adopt 통과 후** | TC-03,TC-05,TC-07,**G2-adopt** | 🤖 | 40m | ✅alarm |
| TC-11 | 가격/타겟/아웃리치 확정 승인 | 사용자 승인 가격시트+타겟+아웃리치 기록 | TC-10 | 🙋 | 20m | **G3-approve** ✅alarm |
| TC-12 | BD 실행·환금 마일스톤 | **G3-close 임계: 서명된 유료 감사 계약 ≥1 OR 서면 부품 LOI ≥1**('대화 진입'은 성공 아님). 조달완결은 12개월 이후. 실소요 월단위 | TC-11,TC-04,TC-06,TC-07 | 🌐 | 월단위 | **G3-close** ✅alarm |

> 검증반영: TC-00을 **대형·일단위 통합**으로 정직화(60m 아님), TC-01 **반복 빌드+롤백 전제**, **TC-09의 G2 태그 제거**(빌드는 게이트가 아니라 G2-adopt에 종속), TC-04/09 **전역 ledger**, TC-12 **G3-close='대화'가 아닌 서명 계약/LOI**로 성공바 강화.

---

## ④ fail-fast 게이트 운영 (전 게이트 사전등록 수치 임계)

| 게이트 | 위치·**사전등록 임계** | 통과 시 자동 진행 | 미달 시 자동 피벗 |
|---|---|---|---|
| **G-env** (T0-00/05) | driver≥570 + cu128 fp16 matmul + sm_120. **cl/nvcc 부재는 SDPA로 라우팅되나, bnb sm_120 양자화 백엔드(T0-05)는 별도 통과 필수** — 미해결 시 게이트 통과라도 하류(T0-08) 하드실패 | cu128·bnb·QLoRA 진행 | driver 미지원→2.5GB 前 정지+알람. **bnb sm_120 불가→torchao/HQQ/full-LoRA 피벗(user-gate) or 클라우드GPU** |
| **G0-pre** (TA-00) | **사전등록 단일 primary=citation exact-match**, 2~4B가 7~14B 대비 bootstrap 95% CI 하한>0. **다지표 ≥1 cherry-pick 금지** | 신호 有→코퍼스 풀투자(TA-06~10) | 신호 弱→코퍼스 투자 축소, base 상향 검토 |
| **G0** (TA-12) | **파인튜닝된 미니 grounder(TA-11c) vs base 7~14B**, primary=citation exact-match, **margin Δ≥+2pt & 95% CI 하한>0**(≥N은 TB-04p 검정력 기준). 판정=자동임계+🙋 확정 | '소형 grounder 동급이상' 서사 → B | '측정을 소유한다'로 축소 + **base 상향은 16GB QLoRA 실현성(r-low/seq-low/offload) 재확인, 불가 시 클라우드GPU(user-gate)** |
| **G1** (TB-03) | **게시 후 72h 내 (HF 다운로드 or GitHub star/issue) ≥사전등록 N OR 제3자 서면 문의 ≥1** | 확장(TB-04b~10) + **비싼 저작·폐쇄API(TB-04b/06b) 언블록** | 벤치전략 재고 → 감사서비스로 이동, **저작·폐쇄API 차단** |
| **G2-adopt** (TB-10) | **독립 재현 ≥2건(TB-13 컨테이너 기반) OR 공동서명 기관 ≥1** — 진짜 채택 게이트 | 온프렘·파트너·자동채점(TC-06/09/10) 투자 언블록 | 3~6개월 재현 0 → '표준화' 폐기, 감사·부품 집중(TC-06/09/10 차단) |
| **G3-approve** (TC-11) | 사용자 가격/타겟 승인 완료 | BD 실행 착수 | 매출 가정 재설계 |
| **G3-close** (TC-12) | **서명된 유료 계약 ≥1 OR 서면 LOI ≥1**('대화'≠성공) | 매출 반복·조달(12개월+) | 명성 신호 부족 시 G1/G2 회귀 |
| **safety** (TB-07S) | 독성·편향·거절 eval **사전등록 임계 통과** | HF 공개(TB-08) 허용 | **공개 차단** — 역풍 리스크 모델 릴리스 금지 |

---

## ⑤ 알람 포인트 (폰) — 시점별 사용자 액션 + ETA·신뢰도 산출식

| 시점 | 알람 내용 | 사용자 액션 |
|---|---|---|
| **시작** | 트랙명·단락수·영역·**무중단 본체 예상 지속(~6~9h)과 이후 external/user 대기 비중 명시** | 확인만 |
| **단락 완료** | commit hash·소요·남은단락 + **진척률%·ETA·신뢰도(아래 산출식)** | 없음(자동 다음) |
| **게이트 도달** | G-env/G0-pre/G0/G1/G2-adopt/G3-approve/G3-close/safety 판정(사전등록 임계 대비 수치) + 분기 | 서사·전략 분기 확인(G0/G1 마케팅 문구 승인) |
| **개입 필요** 🙋 | ①HF토큰 ②법령/DART키 ③bnb피벗 ④WSL2/CI 테스트 허용 ⑤HF 공개 결정 ⑥TC-00 브랜치 화해 ⑦크리덴셜·예산 ⑧가격/파트너 ⑨**teacher 라이선스 결정** ⑩**human 채점자 섭외** ⑪**게이트 최종 판정 승인** | 자산 제공 후 재개 |
| **external 완료/정지** 🌐 | cu128·다운로드·bulk fetch·**풀스케일 합성·릴리스급 학습**·폐쇄채점 완료 or stall(3회 소진) + **예산 ledger 누적치** | stall 원인 확인 |
| **skip** | 3회 실패 사유+stderr 마지막줄 | 검토 |
| **완주** | HTML 리포트 sendDocument | 검토 |
| **TELEGRAM_DEAD** | notifications.log+1회 재시도 후 ⚠ 마커+일시정지 | 텔레그램 복구 |

> **ETA 산출식(정직 근거)**: external ETA = (데이터량 / 처리율). 처리율은 **T0-02(matmul·GB/s)·T0-07(generate tokens/s)·T0-08(step/s)에서 계측해 docs/env-verify/perf.json에 기록**. bulk fetch는 (목표건수 × 요청간격 / rate-limit) — rate-limit 종속성으로 **상한 미상 시 '≥X, 상한 미정' 정직 표기**. 학습 ETA = (데이터량 × 에폭 / step throughput). **추정 근거가 없으면 ETA를 내지 않고 '측정 후 산출'로 표기**(허위 ETA 금지).
> **신뢰도(confidence) 산출식**: confidence = f(성공조건 객관성[사전등록 수치=高/grep프록시=中/사람판정=低], 잔여 external 의존 수, 게이트 근접도). 각 단락에 3구간(高/中/低)으로 표기 — **정성 라벨이 아닌 위 3요소의 명시 규칙**.
> **정직 고지**: '개입 필요' 11종은 무중단 자동화 불가 지점이다. ⑤~⑪(릴리스·화해·프로비저닝·환금·teacher·human·판정승인)은 **fame/매출의 실제 트리거**이며 Claude가 대신 눌러줄 수 없다.

---

## ⑥ 무중단 실행 규칙

- **단락 사이 '다음 진행할까요?' 확인질문 금지** — claude-auto 단락은 자동 commit(conventional prefix) + 다음 진행. TodoWrite 갱신으로 갈음.
- **단락 commit 직후** `scripts/autopilot_main_ff.ps1`로 **로컬 main FF만**(자동 push 0). staged 0 강제.
- 독립 그룹은 worktree Agent 병렬(최대5) 자동 spawn, 의존 그룹 직렬. 서버 띄우는 단락은 항상 단독. **병렬은 무중단 본체(~6~9h)에만 유효, 이후 external/user 직렬.**
- **external 단락**은 무중단 흐름에서 분리 — 시작 알람 후 백그라운드, 완료/stall 알람으로 복귀. **매 external 완료 시 예산 ledger 누적치 알람.** 뒤 의존 단락은 대기(다른 독립 그룹은 계속).
- **user-gate 단락**은 폰 알람 후 정지하되 **의존 없는 다른 claude-auto 그룹은 계속** 무중단(예: TA-06 키 대기 중 TA-01/02/03a/07/11a 병렬).
- **불량 auto-commit 롤백 절차(신설)**: cargo check는 통과했으나 하류 통합에서 깨지는 commit(예: TA-05a→TC-01) 발견 시 — (1) 해당 commit `git revert`(reset 아닌 revert로 이력 보존), (2) main FF 재적용, (3) 원인 단락 재개, (4) skip 알람. **partial-state 회복 경로를 게이트와 분리 명시.**

**예외 — 사용자 필수 개입(무중단 중단, hard_constraint):**
- main 직접 push / force push / --no-verify / 자동 머지 금지
- PROGRESS·MEMORY·SHARED_CONTEXT·CLAUDE.md 수정 금지(사후 postmerge만)
- fast-path 끄기 금지 / Vultr SSH 금지(GPT 트랙) / bench_*.py 실행 금지
- `_ds_*`·데이터 삭제 금지
- **OpenAI/Anthropic 출력 학습 금지 + NC teacher 금지**(TA-06T allowlist 하드가드) / **NC 모델 파인튜닝 금지**(license_guard) / **오염원(AI-Hub/모두의말뭉치/나무위키/KoAlpaca) 회피**(provenance 게이트)
- **소형-우위 서사 G0 실측 前 마케팅 금지** / **dead-infra 광고 금지** / **safety eval(TB-07S) 미통과 공개 금지**
- 릴리스 공개·arXiv·커뮤니티 게시·실명 벤더 발송 = 발행 前 사용자 승인 + right-of-reply
- **전역 예산 ledger 글로벌 상한 초과 = 즉시 정지**(per-segment kill-switch만으로 불충분)

---

## ⑦ 지금 당장 시작 단락 — 첫 무중단 연속 블록 (T0 환경 → G-env → A 정직화 병렬)

이 블록은 **user-gate 없이** 연속 실행 가능한 첫 무중단 구간이다(HF 토큰만 run 시작 시 1회 프리체크). **실제 지속 ~6~9h, 이후 external/user 대기가 지배적임을 사용자에게 시작 알람에서 고지.**

```
[Step 0 — run 시작 프리체크(1회 알람)]
  · HF_TOKEN 존재 확인(T0-04 전진) → 미주입 시 캐시경로 resolve까지만
  · F: USB HDD 마운트 assert + T0-11 미러 대상 경로 확인

[Step 1 — T0 환경 직렬(fail-fast 우선)]  🤖→🌐
  1. T0-00 driver+툴체인 preflight ── G-env(driver≥570 + bnb 백엔드 별도판정) ──
       통과 → 계속 / 미달 → 알람+정지(다운로드 前)
  2. T0-01 venv/ruff/preflight_drive  →  T0-10 예산 원장 · T0-11 미러 백업(독립 병렬)
  3. T0-02 torch cu128(+처리율 계측) [🌐 alarm]  → 4. T0-03 reqs-core(+cu128 assert)
  5. T0-05 bnb 4bit(+tolerance<5e-2) [🌐, 실패 시 🙋 피벗]
  6. T0-06 SDPA/accel → 7. T0-07 load smoke(+tokens/s) → 8. T0-08 QLoRA smoke(≥50step·≥10%↓) [🌐 alarm]
  9. T0-09 verify_env ALL PASS(+ledger+backup)

[Step 2 — A 정직화·코드픽스 병렬(5 worktree, T0와 독립 → 즉시 동시 시작)]  🤖
  그룹① TA-01 at-rest/KMS 삭제  ‖ TA-02 dead-infra 삭제  (각 diff 사람 승인 알람)
  그룹② TA-03a→TA-05a (rag.rs 공유 직렬) + TA-04a (독립)
  그룹③ TA-07 법령 fetcher(--smoke, 키 불요)  ‖ TA-09 합성기(mock)
  그룹④ TA-11a QLoRA config 2~4B + license_guard
  그룹⑤ TA-06T teacher 후보 grep(라이선스 표) — 최종 채택만 🙋
     → 각 그룹 cargo check/ruff/grep green 시 자동 commit + main FF

[Step 3 — 게이트/대기 도달 알람]
  · T0-08 성공 → TA-11c 미니 파인튜닝 → G0 착수 가능 알람
  · TA-03b/04b/05b(WDAC 테스트) → WSL2/CI 실행 허용 알람(🙋)
  · TA-06 법령키 · TA-06T teacher 라이선스 결정 알람(🙋)
  · 여기서부터 external(TA-08/10/11c) + user-gate 지배 구간 진입 고지
```

**첫 커밋 순서 요약**: T0-00 → T0-01 →(T0-10/11 병렬)→(정직화 TA-01/02 병렬)→ T0-02 → T0-03 → T0-05 → ... → T0-08 → TA-11c(G0 아티팩트) / TA-03a·04a·05a·07·09·11a·06T. **user-gate 4종(HF토큰·법령키·teacher 라이선스·WDAC테스트)만 알람**, 나머지는 무중단 ~6~9h.

---

## ⑧ 리스크·중단조건 (무중단 강제 중단 트리거)

| 중단조건 | 감지 | 자동 행동 |
|---|---|---|
| **F: USB HDD 분리** | preflight_drive assert 실패 | 즉시 정지 + 알람. **T0-11 미러 백업이 2차 사본 보유 → SPOF 완화**, 재연결까지 대기 |
| **GPU OOM** | smoke_qlora/학습 peak≥16GB | batch/seq_len/grad-checkpoint 조정 3회 → 실패 시 alarm+skip(클라우드GPU user-gate) |
| **bnb sm_120 미지원** | T0-05 forward 에러 or 상대오차≥5e-2 | **정지+user-gate 피벗**(torchao/HQQ/full-LoRA) — 임의 채택 금지. **SDPA 라우팅과 무관(별도 백엔드)** |
| **torch 클robber** | T0-03 후 `+cu128` assert 실패 | 즉시 실패(reqs-core 재설치) — unsloth/accel은 후속 격리 |
| **WDAC 테스트 차단** | nexusflow-server exe os error 4551 | claude-auto는 코드+cargo check까지 done, 실테스트는 WSL2/CI user-gate 이연(거짓 done 금지) |
| **게이트 미달** | 사전등록 임계 대비 판정 | ④표대로 자동 피벗 — 방향전환 |
| **정직성 이슈** | dead-infra 금지어 grep>0, citation exact-match 실패, 오염원 URL, teacher allowlist 위반 | **즉시 정지** — 정직화는 세일즈 前 절대선행 |
| **safety eval 미통과** | TB-07S 임계 미달 | **HF 공개 차단**(TB-08) — 역풍 모델 릴리스 방지 |
| **불량 commit 하류 붕괴** | 통합 빌드/테스트 실패(cargo check는 통과했던 commit) | ⑥ 롤백 절차(revert→main FF→재개→skip 알람) |
| **크로스레포 커밋 실패** | data/* gitignore로 manifest add 불가 | `!data/**/manifest.json` 예외 or manifests/ 이동 후 재시도 |
| **브랜치 파편화 미해소** | TC-01 4계층 통합 시 심볼 부재 | TC-00 화해 머지 완료까지 TC 정지(user-gate·**일단위**) |
| **API 예산 초과** | **전역 ledger 글로벌 상한**(per-segment 아님) | baseline 오픈모델 자동 대체 + 초과 시 정지 알람 |
| **external ETA 산출 불가** | perf.json 미계측 or rate-limit 미상 | '측정 후 산출'/'상한 미정' 정직 표기 — **허위 ETA 금지** |
| **런웨이 소진** | 0~6개월 무수익 구간 | hard_constraint — 버틸 자금 미확보 시 착수 보류(사용자 판단) |
| **TELEGRAM_DEAD** | 발송 2회 실패 | active ⚠ 마커 + 일시정지 |

---

### 종합 정직 판정
- **무중단으로 실제 닫히는 것(정직: ~6~9h)**: T0 환경(external 대기 포함) · A 정직화 · P1 코드픽스(cargo check) · 법령/합성/teacher 코드 스캐폴딩 · 측정 하네스·부품 크레이트·감사 템플릿·재현 컨테이너 **빌드**.
- **무중단이 아닌 것(정직 표기)**: **풀스케일 합성·릴리스급 GPU 학습·대용량 다운로드·폐쇄 채점**(external, 수시간~수일 ETA) / HF·법령키·**teacher 라이선스**·WDAC 테스트·**human 채점자 섭외**·릴리스 공개·브랜치 화해(일단위)·크리덴셜·**게이트 판정 승인**·**가격·파트너·계약**(user-gate). **대부분의 유명세·환금 트리거는 user-gate/external이며 도표(실선/점선)에서도 이를 숨기지 않는다.**
- **2차 검증에서 반영한 핵심 결함 10종**: ①G0가 base만 평가→**TA-11c 미니 파인튜닝을 G0 앞으로** ②teacher 라이선스 공백→**TA-06T 신설** ③smoke데이터로 릴리스 학습→**TB-00F 풀스케일 신설** ④human_agreement 위장→**TB-05H 채점 프로세스** ⑤safety 무게이트 공개→**TB-07S+TB-08 게이트** ⑥예산 per-segment만→**T0-10 전역 ledger** ⑦F: SPOF→**T0-11 미러 백업** ⑧롤백 부재→**⑥ revert 절차** ⑨제3자 재현 불능→**TB-13 이식 컨테이너** ⑩검정력 없는 ≥50→**TB-04p N 산정**. 추가: **모든 게이트 사전등록 수치 임계**, **이중 게이트 분리(G2-adopt/G3-approve/G3-close)**, **external 정직 ETA 모델**, **grep 프록시에 사람 검증 1회 병기**, **무중단 본체가 ~6~9h로 짧고 이후 게이트 지배임을 도표 차원에서 명시**.