"""T0-08 smoke_qlora — 작은 모델을 4bit 로드→LoRA 부착→몇 스텝 학습해
QLoRA 파이프라인이 Blackwell(sm_120)에서 실제로 도는지 검증.

표준 transformers + peft + bitsandbytes 스택(unsloth 미사용 — Windows/무컴파일 안전 경로).
성공조건: peak VRAM < 16GB AND loss 초기 대비 >=10% 하락(no-OOM).

실행:
    python scripts/env/smoke_qlora.py --model Qwen/Qwen2.5-1.5B-Instruct --steps 50
"""
from __future__ import annotations

import argparse
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print(f"loading {args.model} in 4bit (nf4) ...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_cfg, device_map={"": 0},
        attn_implementation="sdpa", torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    texts = [
        "대한민국의 수도는 서울이다.",
        "한국어 QLoRA 스모크 테스트입니다.",
        "근거에 충실한 답변을 생성한다.",
        "16GB GPU에서 파인튜닝이 동작한다.",
    ] * 8
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    input_ids = enc["input_ids"].to("cuda")
    attn = enc["attention_mask"].to("cuda")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    bs = 4
    t0 = time.time()
    for step in range(args.steps):
        i = (step * bs) % (len(texts) - bs)
        ids, am = input_ids[i:i + bs], attn[i:i + bs]
        out = model(input_ids=ids, attention_mask=am, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(out.loss.item())
        if step % 10 == 0:
            print(f"step {step}: loss {out.loss.item():.4f}")
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    drop = (early - late) / early * 100
    print(f"peak VRAM {peak:.2f} GB | {args.steps} steps {dt:.1f}s ({args.steps / dt:.2f} it/s)")
    print(f"loss {early:.4f} -> {late:.4f} ({drop:.1f}% drop)")

    passed = (peak < 16.0) and (drop >= 10.0)
    print("T0-08", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
