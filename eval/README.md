# eval — 한국어 근거충실도 측정 하네스 (own the ruler)

이 프로젝트의 진짜 산출물은 모델이 아니라 **한국어 AI가 근거에 충실한가를 LLM-judge 없이
결정적으로 측정하는 벤치**다. 법령 closed-set 위에서 인용의 실존·정확·tight함을 참/거짓으로 채점한다.

## 측정 스위트

| 도구 | 측정 | 자기검증 |
|------|------|----------|
| `scripts/eval/citation_verify.py` | 인용 실존·substring(hallucinated/misquote/supported), 스코어러 v0.2 | `--demo` |
| `scripts/eval/faithbench.py` | K조문 중 **올바른 조문 선택** 인용(selection_exact) + leak 유형학(parametric/ungrounded) + gold-ablation 프로브, 스코어러 v0.3 | `--demo` |
| `scripts/eval/faithbench_partial.py` | **tight 부분인용**(span precision/recall/F1, 통째복사 페널티), 스코어러 v0.1 | `--demo` |
| `scripts/eval/score_predictions.py` | 오프라인 결정적 채점(모델 불요): `rescore`(공표결과 재현) / `predictions`(제3자 제출) | — |
| `scripts/eval/check_scorer_frozen.py` | 스코어러 동결 게이트 — 전체 aggregate를 golden과 byte-exact 비교(smoke/CI) | `(smoke)` |
| `scripts/eval/fingerprint_report.py` | result JSON → 결정적 fingerprint.json + markdown 표(공개물 수기 전사 대체) | `(smoke)` |
| `scripts/eval/faithbench_stats.py` | Wilson CI + 두-비율 + **paired exact McNemar + Holm 보정** | 단위검증 |
| `scripts/eval/run_meta.py` | 결과 provenance(git rev·모델·SHA256·seed·scorer 버전) | — |
| `scripts/train/run_g0_faithbench.py` / `run_g0_partial.py` | 소형 FT vs 대형 base 교차비교(GPU) + per-instance transcript + closed-book 프로브 | — |

## 경쟁 벤치 대비 차별점 (슬롯 선점)

2026-07 기준 한국어 신뢰성 벤치를 전수 확인한 결과, **"법령 closed-set + LLM judge 없는
결정적 char-span 채점 + 생성·인용 태스크 + 소형 FT vs 대형 base 통제 비교"** 슬롯은 비어 있다.

| 벤치/도구 | 도메인 | 채점 | 태스크 | 라이선스 |
|---|---|---|---|---|
| **이 프로젝트(faithbench)** | 법령 closed-set | **결정적 char-span(judge 없음)** | 생성+인용 | Apache/CC BY |
| K-HALU (ICLR 2025) | 뉴스·서적 | 다중정답 탐지 | 환각 탐지 | — |
| K-FinHallu (2026.5, KAIST+카뱅) | 금융 RAG | LLM judge 5기준 | 탐지 | CC BY-NC |
| KCL (EACL 2026) | 판례 | 루브릭 | 법적 추론 | — |
| KBL (lbox) | 법령/판례 | 지식 QA | 추론 | — |
| korean-law-mcp (2.2k★) | 법령 | 인용 존재검증 **도구**(벤치 없음) | — | MIT |

차별 4축: ①법령 closed-set ②judge 없는 결정적 채점 ③탐지가 아닌 생성+인용 ④소형-대형 통제 비교.

## G0 판정

**G0 = 분할(SPLIT)** — 관대 축(faithbench) 통과, 엄격 축(partial) 역전, 헤드라인 미확정.
정식 판정·마케팅 금지선·재판정 조건·벤치 동결 규칙은 **[`docs/env-verify/g0-verdict.md`](../docs/env-verify/g0-verdict.md)** 참조.
"소형이 대형을 이긴다"는 재판정(위생 수정 + Qwen3-4B + paired + 다법령) 전까지 마케팅 금지다.

2026-07-18의 5법령 30/30 curated-holdout seed는 기존 SPLIT보다 강한 신호를 냈지만, 아직 최종 G0는 아니다.
현재 seed 원본은 `eval/curated_law_seed.json`이며, tracked eval set은 `100` answerable + `100` partial로
확장되어 다음 holdout SFT/모델 재평가를 기다리는 상태다.

## 벤치 동결 규칙

faithbench **v0.3** / faithbench_partial **v0.1** / citation_verify **v0.2** 동결.
(v0.1→v0.2: gold-ablation + leak 유형학 additive. v0.2→v0.3: parametric 판정을 span단위
문맥제외로 교정. 기존 지표 selection_exact/leak_rate 등은 불변이다.) 버전 상수는 각 스코어러 코드에 있고,
`scripts/eval/check_scorer_frozen.py`가 전체 aggregate를 golden과 byte-exact 비교해 smoke/CI에서
동결을 **기계 강제**한다 — 규칙을 바꾸려면 버전 bump + golden 재생성이 강제된다. 다음 벤치 변형
(의미 채점·다중 gold 등)은 **다법령 수집 또는 HRET 기여 준비와 직접 연결될 때만** 착수한다
(g0-verdict.md §6). 지표 동결 후에만 모델 비교 주장 허용.

## 사용

```powershell
# 스코어러 자기검증(모델 불요)
python scripts/eval/citation_verify.py --demo
python scripts/eval/faithbench.py --demo

# 교차비교(GPU) — 새 실험은 --out으로 기준선 덮어쓰기 방지, transcript 자동 저장
python scripts/train/run_g0_faithbench.py --questions eval/questions.constitution.json \
  --k 5 --near --closed-book --out docs/env-verify/g0-faithbench-v02-result.json

# 유의성(paired McNemar 자동 — transcript 있으면)
python scripts/eval/faithbench_stats.py --result docs/env-verify/g0-faithbench-v02-result.json
```

### 질문셋 형식

```json
{ "헌법 제1조 ①": "대한민국의 국가 형태는 무엇인가?" }
```
```json
[ {"id": "헌법 제1조 ①", "question": "대한민국의 국가 형태는 무엇인가?"} ]
```

`--include-all-corpus`는 질문 없는 조항을 ID 기반 sanity 질문으로 채우는 smoke 옵션이다.
질문에 조항ID가 노출되므로 **정식 selection_exact 리포트에는 curated 질문셋만** 쓴다.

## 외부 벤치 정렬 (로드맵)

범용 품질 리더보드(Open Ko-LLM 등) 상위권 추격은 **전략상 폐기**(포화·반감기 극단).
대신 faithbench를 **HRET(haerae-evaluation-toolkit) 레지스트리 모듈로 기여**해 채택을
외부 인프라로 부트스트랩하고, K-HALU/K-FinHallu와 **항목 정렬**해 "통합 잣대"로 진입한다.
