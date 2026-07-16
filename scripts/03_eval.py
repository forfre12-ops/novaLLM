"""03_eval.py — 간단 생성 테스트 + 한국어 벤치마크 안내

빠른 육안 확인용 생성 테스트. 정식 벤치마크는 lm-evaluation-harness 사용:
  - LogicKor, KMMLU, HAERAE-Bench, KoBEST
  - Open Ko-LLM Leaderboard 제출이 '유명세 트리거'의 핵심 경로

실행:
    python scripts/03_eval.py --model checkpoints/sft-run1/lora_adapter
"""
from __future__ import annotations

import argparse

PROMPTS = [
    "대한민국의 수도는 어디이고, 그 도시의 특징을 세 가지 설명해줘.",
    "다음 문장을 정중한 존댓말로 바꿔줘: '내일 회의 몇 시야?'",
    "간단한 파이썬 함수로 피보나치 수열의 n번째 값을 구현해줘.",
    "조선 세종대왕의 업적을 한 문단으로 요약해줘.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="LoRA 어댑터 또는 병합 모델 경로")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    for p in PROMPTS:
        messages = [{"role": "user", "content": p}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        out = model.generate(
            input_ids=inputs, max_new_tokens=args.max_new_tokens, temperature=0.7, do_sample=True
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print("=" * 60)
        print("Q:", p)
        print("A:", text)

    print("\n정식 벤치마크는 lm-eval 사용:")
    print("  lm_eval --model hf --model_args pretrained=<merged_model> --tasks kmmlu,haerae")


if __name__ == "__main__":
    main()
