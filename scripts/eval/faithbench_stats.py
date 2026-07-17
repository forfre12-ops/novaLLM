"""faithbench 결과 유의성 분석 — Wilson CI + 두-비율 검정 (신뢰도 정직화).

전략 TB-04p(검정력·최소 N 산정)의 최소 버전. g0-faithbench-result.json의
selection_exact 카운트로 "소형 FT vs 대형 base" 차이의 95% 신뢰구간·p값을 계산해,
'우위 유지'라는 서술이 통계적으로 얼마나 강한지 정직하게 수치화한다.

주의: 두 모델은 **같은 인스턴스**에서 평가되므로 정확히는 paired(McNemar)가 더 강력하다.
현재 결과 json은 집계(count/n)만 담아 **unpaired 근사**만 가능하다. paired McNemar는
인스턴스별(모델×인스턴스 정오답) 저장 후 재실행이 필요하다(권장 후속).

    python scripts/eval/faithbench_stats.py --result docs/env-verify/g0-faithbench-result.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

Z = 1.959963984540054  # 95%


def _phi(x: float) -> float:
    """표준정규 CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilson(x: int, n: int) -> tuple[float, float, float]:
    """Wilson score 95% 구간 → (phat, lo, hi)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = x / n
    denom = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denom
    half = (Z / denom) * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return p, max(0.0, center - half), min(1.0, center + half)


def two_prop(x1: int, n1: int, x2: int, n2: int) -> dict:
    """unpaired 두-비율 비교: 차이 + Newcombe 95% CI + pooled z-검정 p값."""
    p1, l1, u1 = wilson(x1, n1)
    p2, l2, u2 = wilson(x2, n2)
    diff = p1 - p2
    # Newcombe(method 10) 차이 구간
    lo = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    # pooled z-검정
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) if 0 < p_pool < 1 else 0.0
    z = diff / se if se > 0 else 0.0
    pval = 2 * (1 - _phi(abs(z)))
    return {
        "p1": round(p1, 3), "ci1": (round(l1, 3), round(u1, 3)),
        "p2": round(p2, 3), "ci2": (round(l2, 3), round(u2, 3)),
        "diff": round(diff, 3), "diff_ci": (round(lo, 3), round(hi, 3)),
        "z": round(z, 3), "p_value": round(pval, 4),
        "sig_05": bool(pval < 0.05 and lo > 0),
    }


def counts(res: dict) -> dict:
    """result json → 모델별 (overall/seen/unseen)의 exact 카운트(정수)."""
    out = {}
    bysplit = res["by_split"]
    for m, sp in bysplit.items():
        seen_n, unseen_n = sp["seen"]["n"], sp["unseen"]["n"]
        seen_x = round(sp["seen"]["selection_exact"] * seen_n)
        unseen_x = round(sp["unseen"]["selection_exact"] * unseen_n)
        out[m] = {
            "overall": (seen_x + unseen_x, seen_n + unseen_n),
            "seen": (seen_x, seen_n),
            "unseen": (unseen_x, unseen_n),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="docs/env-verify/g0-faithbench-result.json")
    ap.add_argument("--metric", default="selection_exact", help="(현재 selection_exact 고정)")
    args = ap.parse_args()

    res = json.loads(Path(args.result).read_text(encoding="utf-8"))
    c = counts(res)
    ft = "ft_small_zeroshot"
    comparisons = [
        (ft, "base_large_fewshot", "소형 FT vs 대형 base(7B) — 전략 핵심 가정"),
        (ft, "base_small_fewshot", "소형 FT vs 소형 base(파인튜닝 효과)"),
    ]

    print(f"유의성 분석: {args.metric} (unpaired Wilson/Newcombe, α=0.05)")
    print("주의: 같은 인스턴스 비교 → paired McNemar가 더 강함. 이건 보수적 unpaired 근사.\n")

    for a, b, title in comparisons:
        if a not in c or b not in c:
            continue
        print(f"■ {title}")
        for split in ("overall", "unseen", "seen"):
            x1, n1 = c[a][split]
            x2, n2 = c[b][split]
            r = two_prop(x1, n1, x2, n2)
            verdict = "유의(diff CI>0)" if r["sig_05"] else "미유의/불확실"
            print(
                f"  [{split:<7} n={n1:>2}] FT {x1}/{n1}={r['p1']} {r['ci1']} vs "
                f"base {x2}/{n2}={r['p2']} {r['ci2']} | "
                f"Δ={r['diff']:+} CI{r['diff_ci']} p={r['p_value']} → {verdict}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
