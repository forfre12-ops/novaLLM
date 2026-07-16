"""T0-02 verify_torch — PyTorch가 Blackwell(sm_120)에서 도는지 실증 + 처리율 계측.

성공조건: torch.cuda.is_available()=True, capability major>=12, bf16 matmul 동작.
부수: docs/env-verify/perf.json 에 처리율 기록(external ETA 모델 입력).
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def main() -> int:
    import torch

    print("torch:", torch.__version__, "| torch.version.cuda:", torch.version.cuda)
    ok = torch.cuda.is_available()
    print("cuda available:", ok)
    if not ok:
        print("T0-02 FAIL: CUDA 미인식")
        return 1

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print("device:", name, "| capability:", cap)

    dev = "cuda"
    n = 4096
    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    iters = 20
    t0 = time.time()
    for _ in range(iters):
        c = a @ b  # noqa: F841
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    tflops = (2 * n ** 3) / dt / 1e12
    print(f"bf16 matmul {n}x{n}: {dt * 1000:.2f} ms/iter, ~{tflops:.1f} TFLOP/s")

    outdir = Path("docs/env-verify")
    outdir.mkdir(parents=True, exist_ok=True)
    perf = {
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": name,
        "capability": list(cap), "bf16_matmul_ms": round(dt * 1000, 3),
        "tflops": round(tflops, 1),
    }
    (outdir / "perf.json").write_text(json.dumps(perf, indent=2, ensure_ascii=False))

    passed = ok and cap[0] >= 12
    print("T0-02", "PASS" if passed else "FAIL", "(sm_120 =", cap, ")")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
