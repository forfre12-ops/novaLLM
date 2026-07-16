# nova-llm — 한국어 LLM 학습 프로젝트

> NovaXDB(`E:\NovaXDB`)와 **별도 repo**. 여기는 모델 학습 전용.
> NovaXDB 규칙(Rust/commitlint/rerere 등)은 **상속하지 않는다**.

## 프로젝트 목표

한국어에 강한 오픈 베이스 모델을 QLoRA 파인튜닝·모델병합으로 특화해
경쟁력 있는 한국어 LLM을 만들고 공개(HuggingFace/리더보드)하는 것.
전략 문서는 별도 작성(NovaXDB `docs/strategy/`).

## 소통 규칙

- 모든 소통·설명은 **한국어**. 코드·기술 용어는 영어 유지.
- 사용자는 비개발자 → git/학습 실행을 Claude가 자율 수행. 선택지 나열 대신 최선으로 진행.
- 파괴적 작업(데이터 삭제, 체크포인트 대량 삭제)만 1회 확인.

## 실행 환경 (중요)

- **GPU**: RTX 5070 Ti 16GB VRAM, **Blackwell(sm_120)**
- **CUDA**: Blackwell은 **CUDA 12.8+ 빌드 필수**. PyTorch 설치 시
  `pip install torch --index-url https://download.pytorch.org/whl/cu128`
- bitsandbytes / unsloth / flash-attn은 **Blackwell(sm_120) 지원되는 최신 버전** 필요.
  구버전은 sm_120 미지원 → 설치 후 첫 관문. `torch.cuda.is_available()` True 확인 필수.
- **16GB 상한**: QLoRA(4bit)로 7~14B가 현실적. full fine-tune은 1~3B만. 밑바닥 pre-training 불가.
- OS: Windows 11. WSL2로 학습 시 데이터는 WSL ext4 안에 둘 것 (F: `/mnt/f` 접근은 9p 오버헤드로 매우 느림).

## 저장소 규칙 (F 드라이브)

- 이 repo 위치: **F:\nova-llm** (USB 외장 HDD, 20TB). 학습 중 **드라이브 연결 유지 필수**
  (장시간 학습 도중 USB 분리 시 체크포인트 손상 위험).
- **대용량 파일 절대 git 커밋 금지**: `data/`, `models/`, `checkpoints/`, `logs/`,
  `*.safetensors`, `*.gguf`, `*.bin` → `.gitignore` 처리됨.
- 모델 산출물은 **HuggingFace Hub**로 배포. git엔 코드·설정만.

## Python 규칙

- Python 3.11+. 타입 힌트 사용. 포맷: `ruff format`.
- 의존성: `requirements.txt`. 가상환경 권장(`.venv`).
- 스크립트는 `scripts/` 번호 순서: 01 데이터 → 02 학습 → 03 평가 → 04 병합/내보내기.

## NovaXDB 연결 원칙

- 학습 코드를 NovaXDB repo에 넣지 않는다. **완성된 모델만** 추론 엔드포인트(Ollama/vLLM)로
  서빙하여 NovaXDB가 소비. (NovaXDB는 이미 Ollama 임베딩을 소비 중 → 학습 모델도 동일 방식.)

## 파이프라인

1. `01_prepare_data.py` — 원시 데이터 → chat jsonl (`data/processed/train.jsonl`)
2. `02_train_sft.py` — QLoRA SFT 학습 (`configs/train_config.yaml`)
3. `03_eval.py` — 생성 테스트 + 벤치마크(lm-eval: LogicKor/KMMLU/HAERAE)
4. `04_merge_export.py` — LoRA 병합 → GGUF 변환 → Ollama 등록

## git

- 가벼운 커밋. commitlint/hook 없음. 커밋 메시지는 한국어 자유 형식(간결하게).
- 실험 브랜치 자유. 대용량 산출물은 절대 커밋 금지(위 저장소 규칙).
