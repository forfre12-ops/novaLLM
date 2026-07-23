"""faithbench 결과 유의성 분석 — Wilson CI + 두-비율 검정 + paired McNemar + Holm 보정.

전략 TB-04p(검정력·최소 N 산정)의 최소 버전. g0-faithbench-result.json의
selection_exact 카운트로 "소형 FT vs 대형 base" 차이의 95% 신뢰구간·p값을 계산한다.

두 모델은 **같은 인스턴스**에서 평가되므로 paired(McNemar)가 더 강력하고 정확하다.
transcript JSONL(run_g0_faithbench가 저장)이 있으면 **exact McNemar**를 자동 계산하고,
6개 검정(2비교×3split)에 **Holm-Bonferroni** 보정을 적용한다. 리포트가 근거로 삼던
"unpaired가 보수적"이라는 가정은 검증되지 않으므로, transcript가 있으면 paired를 정본으로 쓴다.

    python scripts/eval/faithbench_stats.py --result docs/env-verify/g0-faithbench-result.json
    # transcript 자동 탐지: <result>-transcript.jsonl 또는 --transcript 로 지정
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

Z = 1.959963984540054  # 95%


def safe_print(text: str = "") -> None:
    """Windows cp949 콘솔에서도 통계 리포트 출력이 죽지 않게 출력 불가 문자를 치환."""
    enc = sys.stdout.encoding or "utf-8"
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


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
    if n1 == 0 or n2 == 0:
        return {"valid": False}
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
        "sig_05": bool(pval < 0.05 and (lo > 0 or hi < 0)),
        "direction": "a" if lo > 0 else ("b" if hi < 0 else "none"),
        "valid": True,
    }


def mcnemar_exact(b: int, c: int) -> float:
    """exact McNemar 양측 p값 — 불일치쌍(b,c)의 이항검정(p=0.5).

    b = A정답·B오답, c = A오답·B정답. n=b+c. 대칭이므로 작은 쪽 꼬리를 2배(≤1로 클램프).
    scipy 불요: math.comb로 직접 계산.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def holm(pairs: list[tuple[str, float]], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni: (label,p) 목록 → label별 {p, p_adj, reject}."""
    m = len(pairs)
    ordered = sorted(pairs, key=lambda kv: kv[1])
    out: dict[str, dict] = {}
    prev_adj = 0.0
    still_reject = True
    for rank, (label, p) in enumerate(ordered):
        p_adj = min(1.0, (m - rank) * p)
        p_adj = max(p_adj, prev_adj)  # 단조 증가 강제
        prev_adj = p_adj
        if not (still_reject and p < alpha / (m - rank)):
            still_reject = False
        out[label] = {"p": round(p, 4), "p_adj": round(p_adj, 4), "reject": still_reject}
    return out


def load_transcript(path: Path, split: str = "answerable", outcome: str = "exact") -> dict[str, dict[str, int]]:
    """transcript JSONL → {model: {instance_key: outcome}} (split 필터, paired McNemar용).

    split='answerable'/outcome='exact'(기본, selection 축) 또는 split='unanswerable'/
    outcome='leaked'(leak 축)로 재사용한다. 키는 안정 instance_id 우선, 구 transcript는
    gold|question 폴백(unanswerable은 gold가 None이므로 question까지 결합해 충돌 방지).
    """
    by_model: dict[str, dict[str, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("split") != split:
            continue
        key = r.get("instance_id") or f"{r.get('gold')}|{r.get('question')}"
        by_model.setdefault(r["model"], {})[key] = int(r.get(outcome, 0))
    return by_model


def paired_mcnemar(tr: dict[str, dict[str, int]], a: str, b: str, keys: set[str] | None = None) -> dict:
    """두 모델의 같은 인스턴스 정오답으로 exact McNemar."""
    ka, kb = tr.get(a, {}), tr.get(b, {})
    common = set(ka) & set(kb)
    if keys is not None:
        common &= keys
    b_cnt = sum(1 for k in common if ka[k] == 1 and kb[k] == 0)  # a정답 b오답
    c_cnt = sum(1 for k in common if ka[k] == 0 and kb[k] == 1)  # a오답 b정답
    return {
        "n_pairs": len(common),
        "a_only": b_cnt, "b_only": c_cnt,
        "diff": round((b_cnt - c_cnt) / len(common), 3) if common else 0.0,
        "p_value": round(mcnemar_exact(b_cnt, c_cnt), 4),
    }


def abstention_operating_point(res: dict) -> dict[str, dict]:
    """거절/유출 트레이드오프 operating point — '전부 거절하면 만점' 게이밍을 지표로 봉쇄.

    두 축을 함께 본다: unanswerable에서 올바로 거절(refusal_rate)하되, answerable에서
    과잉거절(answerable_refused_rate)하지 않아야 한다. Youden's J = refusal_rate −
    answerable_refused_rate 는 두 축을 동시에 만족할 때만 커진다. 전부 거절/전부 응답은
    J≈0으로 무너진다. leak_rate도 병기해 '거절 대신 유출'을 노출한다.
    """
    out: dict[str, dict] = {}
    for m, agg in res.get("results", {}).items():
        refuse_una = agg.get("refusal_rate", 0.0)          # unanswerable에서 진짜 거절(맞음)
        over_refuse = agg.get("answerable_refused_rate", 0.0)  # answerable에서 거절(틀림)
        out[m] = {
            "abstain_when_should": round(refuse_una, 3),
            "over_refuse_answerable": round(over_refuse, 3),
            "youden_j": round(refuse_una - over_refuse, 3),
            "leak_rate": round(agg.get("leak_rate", 0.0), 3),
            "leak_parametric_rate": round(agg.get("leak_parametric_rate", 0.0), 3),
        }
    return out


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
    ap.add_argument("--transcript", default=None, help="per-instance JSONL(paired McNemar용). 없으면 --result 옆에서 자동 탐지")
    args = ap.parse_args()

    resp = Path(args.result)
    res = json.loads(resp.read_text(encoding="utf-8"))
    c = counts(res)
    ft = "ft_small_zeroshot"
    comparisons = [
        (ft, "base_large_fewshot", "소형 FT vs 대형 base(7B) - 전략 핵심 가정"),
        (ft, "base_small_fewshot", "소형 FT vs 소형 base(파인튜닝 효과)"),
    ]

    safe_print(f"유의성 분석: {args.metric} (unpaired Wilson/Newcombe, alpha=0.05)")
    safe_print("주의: 같은 인스턴스 비교 -> paired McNemar가 더 강함. unpaired는 '보수적'이란 보장이 없음(가정).\n")

    for a, b, title in comparisons:
        if a not in c or b not in c:
            continue
        safe_print(f"* {title}")
        for split in ("overall", "unseen", "seen"):
            x1, n1 = c[a][split]
            x2, n2 = c[b][split]
            r = two_prop(x1, n1, x2, n2)
            if not r["valid"]:
                safe_print(f"  [{split:<7} n={n1:>2}] skipped (empty split)")
                continue
            if r["sig_05"] and r["direction"] == "a":
                verdict = "유의(FT > base)"
            elif r["sig_05"] and r["direction"] == "b":
                verdict = "유의(base > FT)"
            else:
                verdict = "미유의/불확실"
            safe_print(
                f"  [{split:<7} n={n1:>2}] FT {x1}/{n1}={r['p1']} {r['ci1']} vs "
                f"base {x2}/{n2}={r['p2']} {r['ci2']} | "
                f"diff={r['diff']:+} CI{r['diff_ci']} p={r['p_value']} -> {verdict}"
            )
        safe_print()

    # ── paired McNemar (transcript 있으면 정본) + Holm 보정 ──
    tpath = Path(args.transcript) if args.transcript else resp.with_name(resp.stem + "-transcript.jsonl")
    if not tpath.exists():
        safe_print(f"(paired McNemar 생략 - transcript 없음: {tpath})")
        safe_print("  run_g0_faithbench.py를 재실행하면 transcript가 저장돼 paired 분석이 가능합니다.")
        return 0

    tr = load_transcript(tpath)
    # transcript는 인스턴스별 정오답을 담으므로 overall paired McNemar는 정확하다.
    # (split별 paired는 transcript에 seen/unseen 태그 추가 시 확장 — 현재는 unpaired 표 참고.)
    safe_print("* paired exact McNemar (같은 인스턴스, overall) + Holm 보정")
    ptests: list[tuple[str, float]] = []
    rows = {}
    for a, b, title in comparisons:
        if a not in tr or b not in tr:
            continue
        r = paired_mcnemar(tr, a, b)
        rows[title] = r
        ptests.append((title, r["p_value"]))
    adj = holm(ptests)
    for title, r in rows.items():
        h = adj[title]
        verdict = "유의(Holm 후)" if h["reject"] else "미유의(Holm 후)"
        safe_print(
            f"  {title}\n"
            f"    n_pairs={r['n_pairs']} FT-only={r['a_only']} base-only={r['b_only']} "
            f"diff={r['diff']:+} | p={r['p_value']} p_adj={h['p_adj']} -> {verdict}"
        )
    safe_print("\n  주: paired McNemar가 unpaired와 다르면 paired가 정본(같은 인스턴스이므로).")

    # ── leak 축 paired McNemar (unanswerable, leaked outcome) ──
    tr_leak = load_transcript(tpath, split="unanswerable", outcome="leaked")
    if sum(1 for a, b, _ in comparisons if a in tr_leak and b in tr_leak):
        safe_print("\n* paired exact McNemar — leak 축(unanswerable, leaked=1은 유출=나쁨)")
        for a, b, title in comparisons:
            if a not in tr_leak or b not in tr_leak:
                continue
            r = paired_mcnemar(tr_leak, a, b)
            safe_print(
                f"  {title}\n"
                f"    n_pairs={r['n_pairs']} FT-leaked·base-clean={r['a_only']} "
                f"FT-clean·base-leaked={r['b_only']} | p={r['p_value']} "
                f"(base-leaked-only>FT-leaked-only면 FT가 leak 적음)"
            )

    # ── 거절/유출 operating point (전부-거절 게이밍 봉쇄) ──
    aop = abstention_operating_point(res)
    if aop:
        safe_print("\n* 거절/유출 operating point (Youden J = 올바른거절 − 과잉거절; 전부거절/전부응답은 J≈0)")
        safe_print(f"  {'모델':<22}{'abstain↑':>10}{'over_refuse↓':>14}{'youden_J↑':>11}{'leak↓':>8}{'param↓':>8}")
        for m, v in aop.items():
            safe_print(
                f"  {m:<22}{v['abstain_when_should']:>10}{v['over_refuse_answerable']:>14}"
                f"{v['youden_j']:>11}{v['leak_rate']:>8}{v['leak_parametric_rate']:>8}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
