"""K-FaithBench 확장용 보수적 표본수 산정.

paired McNemar의 정확한 검정력은 모델별 같은 인스턴스 불일치쌍 구조가 있어야 한다.
다법령 확장 전에는 그 구조가 없으므로, 여기서는 unpaired two-proportion z-test 근사로
보수적인 최소 N을 잡는다. 실제 판정은 transcript 기반 paired McNemar/Holm을 정본으로 쓴다.

    python scripts/eval/power_analysis.py --base 0.39 --target 0.74
"""
from __future__ import annotations

import argparse
import math


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def inv_norm_cdf(p: float) -> float:
    """Acklam 근사. scipy 없이 CI에서 쓰기 위한 표준정규 분위수."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if phigh < p:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def required_n_per_group(p1: float, p2: float, *, alpha: float = 0.05, power: float = 0.8) -> int:
    """Equal-size two-proportion normal approximation."""
    if p1 == p2:
        return math.inf
    z_alpha = inv_norm_cdf(1 - alpha / 2)
    z_power = inv_norm_cdf(power)
    pbar = (p1 + p2) / 2
    num = (
        z_alpha * math.sqrt(2 * pbar * (1 - pbar))
        + z_power * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return math.ceil(num / ((p1 - p2) ** 2))


def achieved_power(p1: float, p2: float, n: int, *, alpha: float = 0.05) -> float:
    if n <= 0 or p1 == p2:
        return 0.0
    z_alpha = inv_norm_cdf(1 - alpha / 2)
    se_alt = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    effect = abs(p1 - p2) / se_alt
    return round(norm_cdf(effect - z_alpha), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=float, default=0.387, help="baseline proportion")
    ap.add_argument("--target", type=float, default=0.742, help="target/FT proportion")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--margins", default="0.05,0.10,0.15,0.20,0.30,0.35")
    args = ap.parse_args()

    print("K-FaithBench conservative sample-size estimate")
    print("method: equal-size unpaired two-proportion normal approximation")
    print("note: final G0 should use paired McNemar when transcripts exist\n")

    n = required_n_per_group(args.base, args.target, alpha=args.alpha, power=args.power)
    print(f"observed p_base={args.base:.3f}, p_target={args.target:.3f}, diff={args.target - args.base:+.3f}")
    print(f"required n per model for power={args.power:.2f}, alpha={args.alpha:.2f}: {n}\n")

    print("| target diff | target p | required n/model | power at n=93 |")
    print("|---:|---:|---:|---:|")
    for m in [float(x.strip()) for x in args.margins.split(",") if x.strip()]:
        p2 = min(0.99, args.base + m)
        rn = required_n_per_group(args.base, p2, alpha=args.alpha, power=args.power)
        pw = achieved_power(args.base, p2, 93, alpha=args.alpha)
        print(f"| {m:.2f} | {p2:.3f} | {rn} | {pw:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
