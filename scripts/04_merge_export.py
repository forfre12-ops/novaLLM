"""04_merge_export.py — LoRA 병합 + 배포용 내보내기

QLoRA 어댑터를 베이스에 병합해 단일 모델로 저장한다.
그 다음 GGUF로 변환하면 Ollama/vLLM에서 서빙 → NovaXDB가 소비할 수 있다.

GGUF 변환(Ollama용)은 llama.cpp 필요:
    python llama.cpp/convert_hf_to_gguf.py models/output/merged --outfile models/output/nova-ko.gguf
    # 그 후 Modelfile 작성 → ollama create nova-ko -f Modelfile

실행:
    python scripts/04_merge_export.py --adapter checkpoints/sft-run1/lora_adapter --out models/output/merged
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="LoRA 어댑터 경로")
    parser.add_argument("--out", default="models/output/merged", help="병합 모델 저장 경로")
    args = parser.parse_args()

    from unsloth import FastLanguageModel

    # 어댑터 로드 (베이스 정보는 어댑터 config에 포함)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=2048,
        load_in_4bit=False,     # 병합은 full precision 로드
    )

    Path(args.out).mkdir(parents=True, exist_ok=True)
    # 16bit 병합 저장 (서빙/추가 변환용)
    model.save_pretrained_merged(args.out, tokenizer, save_method="merged_16bit")

    print(f"병합 모델 저장 완료: {args.out}")
    print("다음 단계:")
    print("  1) GGUF 변환 (llama.cpp) → Ollama Modelfile 작성")
    print("  2) ollama create nova-ko -f Modelfile")
    print("  3) NovaXDB 추론 엔드포인트에 nova-ko 연결")


if __name__ == "__main__":
    main()
