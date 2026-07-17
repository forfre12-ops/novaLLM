"""실제 후보 베이스 모델 VRAM/처리율 벤치.

각 모델을 4bit 로드 → load VRAM·추론 tokens/s → QLoRA 학습(peak VRAM·it/s·loss하락) 측정.
모델별 결과를 docs/env-verify/model-bench.json 에 누적(한 모델 실패해도 이전 결과 보존).

    python scripts/env/bench_models.py --models Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-4B --seq 1024 --steps 20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def bench(model_id: str, seq_len: int, steps: int, bs: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map={"": 0},
        attn_implementation="sdpa", dtype=torch.bfloat16,
    )
    load_s = time.time() - t0
    load_vram = torch.cuda.max_memory_allocated() / 1e9
    nparams = sum(p.numel() for p in model.parameters())

    # 추론 처리율
    model.eval()
    p = tok("한국어로 대한민국 헌법 제1조의 의미를 설명해줘.", return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(**p, max_new_tokens=8)  # warmup
        torch.cuda.synchronize()
        t1 = time.time()
        out = model.generate(**p, max_new_tokens=128, do_sample=False)
        torch.cuda.synchronize()
        gen_s = time.time() - t1
    gen_toks = out.shape[1] - p["input_ids"].shape[1]
    tok_s = gen_toks / gen_s if gen_s > 0 else 0

    # QLoRA 학습
    model = prepare_model_for_kbit_training(model)
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    text = "대한민국의 근거에 충실한 답변을 생성한다. 인용은 [n] 형식을 따른다. " * 40
    enc = tok([text] * bs, return_tensors="pt", padding="max_length",
              truncation=True, max_length=seq_len)
    ids = enc["input_ids"].to("cuda")
    am = enc["attention_mask"].to("cuda")
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=2e-4)
    losses = []
    t2 = time.time()
    for _ in range(steps):
        o = model(input_ids=ids, attention_mask=am, labels=ids)
        o.loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(o.loss.item())
    torch.cuda.synchronize()
    train_s = time.time() - t2
    train_vram = torch.cuda.max_memory_allocated() / 1e9

    return {
        "model": model_id, "params_B": round(nparams / 1e9, 2),
        "load_s": round(load_s, 1), "load_vram_GB": round(load_vram, 2),
        "infer_tok_s": round(tok_s, 1),
        "train_bs": bs, "train_seq": seq_len,
        "train_peak_GB": round(train_vram, 2), "train_it_s": round(steps / train_s, 2),
        "loss_drop_pct": round((losses[0] - losses[-1]) / losses[0] * 100, 1),
        "fits_16gb": train_vram < 16.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--bs", type=int, default=2)
    args = ap.parse_args()

    outp = Path("docs/env-verify/model-bench.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(outp.read_text(encoding="utf-8")) if outp.exists() else []

    for m in args.models:
        print(f"\n===== BENCH {m} (seq={args.seq}, bs={args.bs}, steps={args.steps}) =====")
        try:
            r = bench(m, args.seq, args.steps, args.bs)
            print(json.dumps(r, ensure_ascii=False, indent=2))
            results = [x for x in results if x.get("model") != m] + [r]
            outp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  → {'✅ 16GB FIT' if r['fits_16gb'] else '❌ OOM위험'}  saved")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {m}: {type(e).__name__}: {e}")

    print("\n===== 벤치 요약 =====")
    for r in results:
        print(f"  {r['model']:38s} {r['params_B']:>5}B | load {r['load_vram_GB']:>5}GB | "
              f"train {r['train_peak_GB']:>5}GB seq{r['train_seq']} | "
              f"{r['infer_tok_s']:>5} tok/s | {'FIT' if r['fits_16gb'] else 'OOM'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
