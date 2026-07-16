"""T0-05 verify_bnb — bitsandbytes 4bit가 Blackwell(sm_120) CUDA 커널에서 도는지 검증.

성공조건:
  (1) NF4 quantize→dequantize round-trip 이 유한하고 원본과 cosine>0.98 (NaN-only 통과 금지)
  (2) Linear4bit forward 가 GPU에서 에러 없이 유한한 출력
실패 시(에러/NaN)= user-gate 피벗(torchao/HQQ/full-LoRA).
"""
from __future__ import annotations


def main() -> int:
    import torch
    import bitsandbytes as bnb
    from bitsandbytes.functional import quantize_4bit, dequantize_4bit
    from bitsandbytes.nn import Linear4bit

    print("bitsandbytes:", bnb.__version__)
    dev = "cuda"
    torch.manual_seed(0)

    # (1) functional round-trip
    w = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
    try:
        qw, state = quantize_4bit(w, quant_type="nf4")
        w2 = dequantize_4bit(qw, state).to(torch.bfloat16)
    except Exception as e:
        print("T0-05 FAIL: quantize/dequantize 에러:", repr(e))
        return 1
    finite = torch.isfinite(w2).all().item()
    cos = torch.nn.functional.cosine_similarity(
        w.flatten().float(), w2.flatten().float(), dim=0
    ).item()
    rel = ((w - w2).abs().mean() / w.abs().mean()).item()
    print(f"round-trip: finite={finite}, cosine={cos:.4f}, mean_rel_err={rel:.4f}")

    # (2) Linear4bit forward
    try:
        lin = Linear4bit(512, 512, bias=False, compute_dtype=torch.bfloat16,
                         quant_type="nf4").to(dev)
        x = torch.randn(4, 512, device=dev, dtype=torch.bfloat16)
        y = lin(x)
    except Exception as e:
        print("T0-05 FAIL: Linear4bit forward 에러:", repr(e))
        return 1
    fwd_ok = torch.isfinite(y).all().item() and y.abs().sum().item() > 0
    print("Linear4bit forward: shape", tuple(y.shape), "finite&nonzero", fwd_ok)

    passed = finite and (cos > 0.98) and fwd_ok
    print("T0-05", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
