# 리네이밍·마이그레이션 계획 (공개 전 필수)

> 최종 명칭 확정은 **오너 브랜딩 결정**이다. 이 문서는 충돌 사실·후보·마이그레이션 절차를 준비한다.
> 공개(HF 데이터셋/HRET PR/논문 초고)에 이름이 **박제되기 전**이 마이그레이션 비용 최저점이다.

## 1. 왜 리네이밍이 필요한가 (충돌 사실)

- **`FaithBench`** = Vectara, *FaithBench: A Diverse Hallucination Benchmark* (NAACL 2025) 선점.
  우리 `faithbench.py`/`K-FaithBench`는 **한국어 포트/파생물로 오독**되어 "own the ruler(자 소유)" 주장을 훼손.
- **`K-HALU`**(ICLR 2025), **`K-FinHallu`**(2026) 가 `K-*Hallu/HALU` 네이밍을 선점.
- 우리 슬롯(법령 closed-set + 결정적 char-span + judge-free + leak축)은 **방법론적으로 독립**인데,
  이름이 파생물로 읽히면 기술적 선점권과 무관하게 신뢰가 깎인다.

### 저장소 내 출현 (2026-07 조사)

| 문자열 | 파일 수 | 성격 |
|---|---:|---|
| `faithbench` | 33 | 대부분 코드(모듈/파일명·import·result 파일명) |
| `FaithBench` | 9 | 산문(strategy/실행계획) |
| `K-FaithBench` | 6 | **공개 브랜드**(strategy·실행계획·pilot·LICENSE-DATA) |
| `K-LeakBench` | 2 | **공개 브랜드**(strategy·실행계획) |

**공개에 박히는 핵심 지점:** `LICENSE-DATA`(데이터셋 명칭), `docs/public/citation-fingerprint.md`(공개 테크노트),
`README.md`, 그리고 향후 HF 데이터셋 카드.

## 2. 명명 기준

1. FaithBench / HALU / HALLU / RAGTruth 와 **명확히 구별**.
2. 핵심을 담을 것: **한국 법령(法令) + 결정적 char-span 인용 + judge 없음 + leak축**.
3. **소유 가능**(검색 시 충돌 없음, 도메인/HF org 가용).
4. 영어권(방법론 청중) + 한국어 양쪽에서 자연스러울 것.
5. **패밀리 + 축** 구조로 selection/citation(현 faithbench)과 leak(현 leakbench)을 한 브랜드로 통합.

## 3. 후보 (오너 택1 — 제안일 뿐 확정 아님)

패밀리 브랜드 + 축 접미사 구조를 권장한다. 예:

| 패밀리 후보 | 뉘앙스 | 축 이름 예시 |
|---|---|---|
| **LEXACT** | Lex(법) + exact(결정적 exact-match) — judge 없는 정확성 강조 | LEXACT-Cite / LEXACT-Leak |
| **JomunCite** (조문Cite) | 조문(article) 인용 — 한국 법령 특정성 명시 | JomunCite / JomunLeak |
| **K-StatuteCite** | 법령(statute) closed-set + 인용, 영어권 명료 | K-StatuteCite / K-StatuteLeak |
| **GroundLaw-KR** | grounding + 법령, GroundLM 워크숍과 어휘 정합 | GroundLaw-Cite / GroundLaw-Leak |
| **CharCite-KR** | char-span 인용의 방법론적 차별점을 이름에 | CharCite / CharLeak |

> 선정 시 **HF org·PyPI·GitHub·검색 충돌 확인**을 선행(자율 실행 가능). 최종 낙점은 오너.

## 4. 2단계 마이그레이션 절차

리스크를 낮추려 **공개 문자열 먼저, 코드 나중(alias)** 으로 나눈다.

### Stage 1 — 공개 문자열만 (저위험, 공개 전 필수)

대상: `K-FaithBench`/`K-LeakBench` 브랜드가 박히는 **공개 문서만**.

- `LICENSE-DATA` (데이터셋 명칭 line), `README.md`, `docs/public/citation-fingerprint.md`,
  `NOTICE`(있으면), 향후 HF 데이터셋 카드.
- `strategy.md`·`execution-plan-nonstop.md`는 내부 문서라 후순위(원하면 함께).
- 코드 파일명·모듈은 **건드리지 않는다**(Stage 2).
- 검증: 공개 문서에서 옛 브랜드 grep 0.

### Stage 2 — 코드 파일·모듈 (기계적, alias로 무파손)

대상: `scripts/eval/faithbench*.py`, `run_g0_faithbench.py`, import 13곳, result 파일명.

- **한 번에 rename하지 않는다.** 새 모듈명으로 옮기고, 옛 경로는 **thin shim**(re-export)으로 남겨
  runbook·CI·기존 result 파일 경로를 깨지 않는다.
- result JSON/transcript 파일명은 과거 산출물이므로 **rename 금지**(재현 링크 보존). 신규 산출물만 새 이름.
- 스코어러 버전 상수는 유지(리네이밍은 규칙 변경이 아니므로 golden 불변 — 값이 바뀌지 않는지 check_scorer_frozen으로 확인).
- 검증: `python scripts/smoke.py` 무파손 + import 전수 통과.

## 5. 하지 말 것

- **공개(G1) 전에 코드 대량 rename** — 이중작업. Stage 2는 이름 확정 + 공개 직전에.
- **오너 확정 없이 PyPI/HF org 선점** — 브랜딩 되돌리기 비용.
- result/transcript 파일명 rename — 재현 링크·golden 기준 깨짐.
- 스코어러 로직과 리네이밍을 같은 커밋에 — 리네이밍은 순수 문자열/경로 변경으로 격리.

## 6. 실행 순서 요약

1. (자율 가능) 후보별 HF/PyPI/GitHub/검색 충돌 확인 → 오너에게 리포트.
2. (오너) 최종 명칭 확정.
3. Stage 1 공개 문자열 치환 → grep 0 검증 → 커밋.
4. (공개 직전) Stage 2 코드 alias 마이그레이션 → smoke 무파손 → 커밋.
