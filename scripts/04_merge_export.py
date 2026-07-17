"""04_merge_export.py — LoRA 병합 + 배포용 내보내기

QLoRA 어댑터를 베이스에 병합해 단일 모델(16bit)로 저장한다.
그 다음 GGUF로 변환하면 Ollama/vLLM에서 서빙 → NovaXDB가 소비할 수 있다.

이 환경 주의(T0): unsloth 미사용. 표준 transformers+peft로 병합.
  - 병합은 full/half precision 로드 필요(4bit 병합 불가) → 7B는 bf16 ~15GB.
  - 16GB GPU에선 OOM 위험이 커 기본값은 **CPU 병합**(--device cuda로 강제 가능).
  - CPU 병합은 시스템 RAM ~15GB+ 필요(7B 기준).

GGUF 변환(Ollama용)은 llama.cpp 필요:
    python llama.cpp/convert_hf_to_gguf.py models/output/merged --outfile models/output/nova-ko.gguf
    # 그 후 Modelfile 작성 → ollama create nova-ko -f Modelfile

실행:
    python scripts/04_merge_export.py --adapter checkpoints/sft-run1/lora_adapter --out models/output/merged
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="LoRA 어댑터 경로")
    parser.add_argument("--out", default="models/output/merged", help="병합 모델 저장 경로")
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="병합 로드 위치(기본 cpu — 7B bf16는 16GB GPU에서 OOM 위험)",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_cfg = Path(args.adapter) / "adapter_config.json"
    if not adapter_cfg.exists():
        raise SystemExit(f"adapter_config.json 없음: {args.adapter} (LoRA 어댑터 경로가 맞나요?)")
    base_id = json.loads(adapter_cfg.read_text(encoding="utf-8"))["base_model_name_or_path"]

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda 지정했으나 CUDA 사용 불가.")

    print(f"베이스 로드(bf16, {args.device}): {base_id}")
    device_map = {"": 0} if args.device == "cuda" else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=torch.bfloat16, device_map=device_map, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)

    print(f"어댑터 부착 후 병합: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))

    print(f"병합 모델 저장 완료: {out}")
    print("다음 단계:")
    print("  1) GGUF 변환 (llama.cpp) → Ollama Modelfile 작성")
    print("  2) ollama create nova-ko -f Modelfile")
    print("  3) NovaXDB 추론 엔드포인트에 nova-ko 연결")


if __name__ == "__main__":
    main()
