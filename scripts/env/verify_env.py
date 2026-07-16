"""T0-09 verify_env — 환경 검증 오케스트레이터.

빠른 검증(torch/bnb)을 순차 실행하고 요약. QLoRA 스모크(T0-08)는 다운로드가 커서 별도 실행.

    python scripts/env/verify_env.py
"""
from __future__ import annotations

import subprocess
import sys

CHECKS = [
    ("T0-02 torch cu128 sm_120", "scripts/env/verify_torch.py"),
    ("T0-05 bitsandbytes 4bit", "scripts/env/verify_bnb.py"),
]


def main() -> int:
    py = sys.executable
    results = []
    for name, script in CHECKS:
        print(f"\n===== {name} =====")
        r = subprocess.run([py, script])
        results.append((name, r.returncode == 0))

    print("\n===== 환경 검증 요약 =====")
    all_ok = True
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        all_ok = all_ok and ok
    print("\n환경:", "ALL PASS" if all_ok else "일부 실패")
    print("QLoRA 스모크(T0-08)는 별도 실행: python scripts/env/smoke_qlora.py")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
