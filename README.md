# nova-llm

한국어 LLM 학습 워크스페이스 (QLoRA 파인튜닝 + 모델병합).
RTX 5070 Ti 16GB(Blackwell) 단일 GPU 기준.

## 빠른 시작

```powershell
# 1) 가상환경
python -m venv .venv
.venv\Scripts\activate

# 2) PyTorch — Blackwell(sm_120)은 CUDA 12.8 빌드
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3) 나머지 의존성
pip install -r requirements.txt

# 4) 환경변수
copy .env.example .env    # HF_TOKEN 등 채우기

# 5) 파이프라인
python scripts/01_prepare_data.py
python scripts/02_train_sft.py --config configs/train_config.yaml
python scripts/03_eval.py --model checkpoints/sft-run1/lora_adapter
python scripts/04_merge_export.py --adapter checkpoints/sft-run1/lora_adapter
```

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

`False`거나 sm_120 관련 에러가 나면 torch/bitsandbytes/unsloth 버전을 최신으로 올려야 한다.

## 주의

- 이 repo는 **F 드라이브(USB 외장 HDD)**. 학습 중 드라이브 연결을 유지할 것.
- 모델·데이터는 git에 커밋하지 않는다(`.gitignore`). 산출 모델은 HuggingFace로 배포.
- 자세한 규칙: [`CLAUDE.md`](CLAUDE.md)
