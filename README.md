# nova-llm

한국어 법령 근거충실도 측정·검증 워크스페이스 (QLoRA 파인튜닝 + 결정적 citation scoring).
RTX 5070 Ti 16GB(Blackwell) 단일 GPU 기준.

## 현재 상태

- 공개 초점: 모델 성능 과장이 아니라 **한국어 법령 인용 행동의 결정적 측정**.
- G0 판정: **SPLIT**. selection 축에서는 FT 1.5B가 강하지만, tight span 축은 metric-dependent.
- 금지 서사: "소형이 대형을 이긴다"는 단정.
- 다음 병목: `LAW_API_KEY` 기반 다법령 closed-set 코퍼스 확장.

## 공개 산출물

- [`docs/public/citation-fingerprint.md`](docs/public/citation-fingerprint.md): 한국어 법령 인용 행동을
  선택·부분-span·거절·leak·암기 인출로 나눠 측정하는 공개용 테크노트 초안.
- [`docs/env-verify/G0-summary.md`](docs/env-verify/G0-summary.md): G0 실험의 단일 결론.
- [`docs/env-verify/g0-verdict.md`](docs/env-verify/g0-verdict.md): 공식 판정문. 현재 판정은
  **SPLIT**이며, "소형이 대형을 이긴다"는 단정은 금지한다.
- [`docs/data-pipeline.md`](docs/data-pipeline.md): 국가법령정보 OpenAPI 기반 다법령 코퍼스
  수집·정규화·질문셋 smoke 경로.
- [`docs/next-runbook.md`](docs/next-runbook.md): `LAW_API_KEY`를 받은 뒤 바로 이어갈 실행 순서.
- [`docs/hret-integration.md`](docs/hret-integration.md): HRET/HAE-RAE 생태계 기여 계획.

## 빠른 시작

```powershell
# 1) 가상환경
python -m venv .venv
.venv\Scripts\activate

# 2) PyTorch — Blackwell(sm_120)은 CUDA 12.8 빌드
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3) 검증된 핵심 의존성(Windows + Blackwell 무컴파일 경로)
pip install -r requirements-core.txt

# 4) 환경변수
copy .env.example .env    # HF_TOKEN 등 채우기

# 5) 파이프라인
python scripts/01_prepare_data.py
python scripts/02_train_sft.py --config configs/train_config.yaml
python scripts/03_eval.py --model checkpoints/sft-run1/lora_adapter
python scripts/04_merge_export.py --adapter checkpoints/sft-run1/lora_adapter
```

## 로컬 스모크

GPU/API 키 없이 공개 repo의 핵심 경로를 확인한다.

```powershell
python scripts/smoke.py
```

검증되는 경로:

- scorer demos
- lawService JSON/XML parser fixtures
- manifest/bulk fetch dry-run
- merged law corpus validation
- smoke answerable/partial/unanswerable question generation
- faithbench instance builder

## 기여

재현 결과, parser 버그, 질문셋 제안은 GitHub issue template을 사용한다.
기여 규칙은 [`CONTRIBUTING.md`](CONTRIBUTING.md)를 따른다.

## 구조

| 경로 | 내용 | git |
|------|------|-----|
| `scripts/` | 학습 파이프라인 코드 (01~04) | ✅ 추적 |
| `configs/` | 학습 설정(yaml) | ✅ 추적 |
| `eval/` | 평가 하네스/결과 | ✅ 코드만 |
| `data/` | 데이터셋 (raw/processed) | ❌ 무시 |
| `models/` | 베이스·산출 모델 | ❌ 무시 |
| `checkpoints/` | 학습 중간저장 | ❌ 무시 |
| `logs/` | 학습 로그 | ❌ 무시 |

## 첫 관문: Blackwell 환경 검증

RTX 50 시리즈(sm_120)는 최신 CUDA/라이브러리만 지원한다. 설치 후 아래가 통과해야 함:

```powershell
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
# 기대: NVIDIA GeForce RTX 5070 Ti True
```

`False`거나 sm_120 관련 에러가 나면 torch/bitsandbytes 버전을 최신으로 올려야 한다.
이 프로젝트의 현재 검증 경로는 `unsloth`/`flash-attn` 없이 SDPA attention을 사용한다.

## 주의

- 이 repo는 **F 드라이브(USB 외장 HDD)**. 학습 중 드라이브 연결을 유지할 것.
- 모델·데이터는 git에 커밋하지 않는다(`.gitignore`). 산출 모델은 HuggingFace로 배포.
- 자세한 규칙: [`CLAUDE.md`](CLAUDE.md)
