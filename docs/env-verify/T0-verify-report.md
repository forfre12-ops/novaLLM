# T0 환경 검증 리포트 — Blackwell QLoRA 실동작 확인

**일자**: 2026-07-17
**장비**: NVIDIA GeForce RTX 5070 Ti · 16GB VRAM · **Blackwell sm_120** · driver 610.47
**워크스페이스**: `F:\nova-llm` (venv: `.venv`, Python 3.11.9)

## 결과 요약 — 전 관문 PASS ✅

| 단락 | 검증 | 결과 |
|------|------|------|
| **T0-00** | preflight (driver/toolchain/disk) | ✅ driver 610.47≥570, sm_120, F: 17.8TB. 컴파일러(cl/nvcc) 없음 → SDPA 경로 |
| **T0-01** | venv + pip | ✅ `.venv` 생성, pip 26.1.2 (권한문제 없음) |
| **T0-02** | PyTorch cu128 sm_120 | ✅ **torch 2.11.0+cu128**, cuda available, capability (12,0), bf16 matmul 19.6 TFLOP/s |
| **T0-03** | core deps | ✅ transformers 5.14.1 · peft 0.19.1 · trl 1.8.0 · **bitsandbytes 0.49.2** · datasets 5.0.0 · numpy 2.4.6 |
| **T0-05** | bitsandbytes 4bit sm_120 | ✅ NF4 round-trip cosine **0.9958**, Linear4bit GPU forward finite&nonzero |
| **T0-06** | attention 백엔드 | ✅ **SDPA** (flash-attn/triton 무컴파일 미가용 → SDPA 폴백 채택) |
| **T0-07/08** | QLoRA 마이크로 스모크 | ✅ Qwen2.5-1.5B 4bit + LoRA(r=16, 4.36M/0.28%), loss **5.35→0.007 (99.9%↓)**, **peak VRAM 1.84 GB**, 2.85 it/s |

## 핵심 결론

1. **Blackwell(sm_120) 지원 확인** — 최대 불확실성이던 "RTX 50 시리즈에서 PyTorch/bitsandbytes가 도는가"가 **둘 다 통과**. prebuilt 휠만으로 무컴파일 동작.
2. **QLoRA 파이프라인 실동작** — 표준 transformers+peft+bitsandbytes 스택(unsloth 미사용)으로 4bit 로드→LoRA→학습→loss 하락 전 과정 확인.
3. **VRAM 여유 압도적** — 1.5B가 peak 1.84GB. 16GB에서 **7~14B QLoRA도 현실적**(계획의 상한 확인).

## 스택 참고 (재현용)

- torch는 별도: `pip install torch --index-url https://download.pytorch.org/whl/cu128`
- 나머지: `pip install -r requirements-core.txt`
- attention: `attn_implementation="sdpa"` (Windows 무컴파일)
- unsloth/flash-attn 미사용 (소스빌드 불가 환경)

## 실제 후보 베이스 벤치 (실측, seq 1024 · batch 2 · QLoRA r16)

| 모델 | 실파라미터 | load VRAM | **학습 peak VRAM** | 추론 tok/s | 16GB |
|------|-----------|-----------|--------------------|-----------|------|
| **Qwen2.5-7B-Instruct** (Apache) | 7.6B | 5.69 GB | **13.72 GB** | 16.8 | ✅ FIT (2.3GB 여유) |
| **Qwen3-4B** (Apache) | 4B | 3.40 GB | 9.37 GB | 10.7 | ✅ FIT (넉넉) |

**결론**: config 기본값 **Qwen2.5-7B가 seq 1024·batch 2에서 13.72GB로 16GB에 실제로 맞음** — 7B QLoRA 학습이 이 GPU에서 현실적. 더 긴 seq/큰 batch는 gradient checkpointing + batch 1로 확보 가능. 4B는 여유가 커 실험 반복에 유리.

참고:
- json `model-bench.json`의 `params_B`(4.35/2.21)는 4bit 양자화 텐서 numel 카운트 아티팩트 — 실제 파라미터는 7.6B/4B.
- 7B `load_s` 355s는 **F: USB HDD에서 ~15GB 읽기 + 4bit 양자화** 시간(1회성). 반복 학습 시 활성 모델을 C: NVMe로 스테이징하면 로드 가속 가능.
- 추론 tok/s가 낮은 편은 flash-attn 없이 SDPA + USB HDD 환경 특성 — 학습(QLoRA)엔 영향 미미.

## 다음 단계 (계획 참조)

T0 인프라 PASS → **G0 파일럿 학습**(TA-11c 미니 grounder) + **A 트랙 정직화·데이터**로 진행 가능.
단, 정직화(TA-01/02)·P1 코드픽스는 NovaXDB repo를 건드리므로 격리 worktree + 사용자 확인 후.
