"""faithbench 부분-span 확장 — '조문 통째 복사' 관대함을 닫는 결정적 채점.

faithbench 본판의 약점: gold 답변이 조문 전체를 「」로 감싸므로, substring 검증이
'제공 근거를 통째 복사'만 해도 충족된다(faithfulness가 관대). 이 확장은 질문이 조문의
**특정 부분(clause)**을 겨냥하고, 채점이 인용 span과 gold_span의 **문자 단위 겹침
(precision/recall/F1)**을 잰다 → 통째 복사는 precision이 떨어져 페널티.

여전히 결정적(LLM-judge 없음): gold_span은 조문의 exact substring이고, 모델 인용의
문자 범위와 겹침만 계산한다. 컨텍스트엔 distractor 조문도 섞어 '올바른 조문 선택'도 요구.

    python scripts/eval/faithbench_partial.py --demo
    python scripts/eval/faithbench_partial.py --items eval/questions.partial.constitution.json --dump 2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from citation_verify import _norm, load_corpus, verify  # noqa: E402
from faithbench import _context_block, _pick_distractors  # noqa: E402

# 부분-span 스코어러 버전 — 채점 규칙 변경 시 bump. 현재 동결 v0.1.
PARTIAL_VERSION = "0.1"

PARTIAL_SYS = (
    "너는 제공된 근거 조항만 사용해 답한다. 여러 근거 중 질문에 해당하는 조항을 찾아, "
    "그 조항에서 **질문에 해당하는 부분만** 「원문 인용」[조항ID] 형식으로 인용한다. "
    "조문 전체를 통째로 인용하지 말고 질문과 관련된 부분만 인용하라. "
    "해당하는 근거가 없으면 '제공된 근거에서는 확인할 수 없습니다'라고 답한다."
)


def load_items(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for it in data:
        if not all(k in it for k in ("id", "question", "gold_span")):
            raise SystemExit(f"항목 형식 오류(id/question/gold_span 필요): {it!r}")
    return data


def validate_items(items: list[dict], corpus: dict[str, str]) -> None:
    """gold_span이 조문의 exact substring인지 확인(오염 데이터 원천 차단)."""
    for it in items:
        cid, span = _norm(it["id"]), _norm(it["gold_span"])
        if cid not in corpus:
            raise SystemExit(f"코퍼스에 없는 ID: {it['id']}")
        if span not in corpus[cid]:
            raise SystemExit(f"gold_span이 조문의 substring 아님: {it['id']}")


def _span_prf(quote: str, gold_span: str, article: str) -> tuple[float, float, float]:
    """인용문·gold_span을 조문 내 문자 범위로 찾아 precision/recall/F1 계산.

    통째 복사 → precision 하락(recall만 1.0)으로 페널티. 정확한 부분 인용 → 1.0.
    """
    a, q, g = _norm(article), _norm(quote), _norm(gold_span)
    qi, gi = a.find(q), a.find(g)
    if qi < 0 or gi < 0 or not q:
        return 0.0, 0.0, 0.0
    qs, qe, gs, ge = qi, qi + len(q), gi, gi + len(g)
    inter = max(0, min(qe, ge) - max(qs, gs))
    prec = inter / len(q) if len(q) else 0.0
    rec = inter / len(g) if len(g) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return round(prec, 3), round(rec, 3), round(f1, 3)


def build_partial_instances(corpus, items, k, near, seed) -> list[dict]:
    rng = random.Random(seed)
    insts = []
    for it in items:
        gid = it["id"]
        if gid not in corpus:
            continue
        distractors = _pick_distractors(corpus, gid, k - 1, near, rng)
        ctx = [(gid, corpus[gid])] + [(d, corpus[d]) for d in distractors]
        rng.shuffle(ctx)
        insts.append({
            "split": "partial",
            "gold": [gid],
            "gold_span": it["gold_span"],
            "question": it["question"],
            "context_ids": [c for c, _ in ctx],
            "messages": [
                {"role": "system", "content": PARTIAL_SYS},
                {"role": "user", "content": f"{_context_block(ctx)}\n\n질문: {it['question']}"},
            ],
        })
    return insts


def score_partial(inst: dict, answer: str, corpus: dict[str, str], f1_ok: float = 0.7) -> dict:
    rep = verify(answer, corpus)
    gid = _norm(inst["gold"][0])
    gold_span = inst["gold_span"]
    cited = {_norm(c["cited_id"]) for c in rep["citations"]}

    best = (0.0, 0.0, 0.0)
    for c in rep["citations"]:
        if _norm(c["cited_id"]) == gid and c["supported"]:
            prf = _span_prf(c["quote"], gold_span, corpus[gid])
            if prf[2] > best[2]:
                best = prf
    selected = int(gid in cited and any(
        _norm(c["cited_id"]) == gid and c["supported"] for c in rep["citations"]))
    return {
        "split": "partial",
        "gold": inst["gold"][0],
        "n_citations": rep["n_citations"],
        "selected_gold": selected,
        "distractor_cited": int(bool(cited - {gid})),
        "span_precision": best[0],
        "span_recall": best[1],
        "span_f1": best[2],
        "span_ok": int(best[2] >= f1_ok),
        "partial_exact": int(selected and best[2] >= f1_ok and not (cited - {gid})),
    }


def aggregate_partial(scored: list[dict]) -> dict:
    def mean(key: str) -> float:
        return round(sum(s[key] for s in scored) / len(scored), 3) if scored else 0.0

    return {
        "n": len(scored),
        "selected_gold": mean("selected_gold"),
        "span_precision": mean("span_precision"),
        "span_recall": mean("span_recall"),
        "span_f1": mean("span_f1"),
        "span_ok": mean("span_ok"),
        "partial_exact": mean("partial_exact"),
    }


def _run_demo(corpus: dict[str, str]) -> int:
    gid = "헌법 제10조"
    art = corpus[gid]
    gold_span = "모든 국민은 인간으로서의 존엄과 가치를 가지며, 행복을 추구할 권리를 가진다."
    other_span = "국가는 개인이 가지는 불가침의 기본적 인권을 확인하고 이를 보장할 의무를 진다."
    dist = "헌법 제3조"
    inst = {"split": "partial", "gold": [gid], "gold_span": gold_span, "context_ids": [gid, dist]}

    cases = [
        ("정확한 부분 인용", f"헌법은 「{gold_span}」[{gid}]라고 규정한다.",
         lambda s: s["span_f1"] == 1.0 and s["partial_exact"] == 1),
        ("조문 통째 복사(관대함 차단)", f"헌법은 「{art}」[{gid}]라고 규정한다.",
         lambda s: s["span_recall"] == 1.0 and s["span_precision"] < 1.0 and s["span_ok"] == 0),
        ("같은 조문의 다른 부분(오답 clause)", f"헌법은 「{other_span}」[{gid}]라고 규정한다.",
         lambda s: s["span_f1"] < 0.3 and s["partial_exact"] == 0),
        ("distractor 조문 인용", f"헌법은 「{corpus[dist]}」[{dist}]라고 규정한다.",
         lambda s: s["selected_gold"] == 0 and s["partial_exact"] == 0),
    ]
    all_ok = True
    for name, ans, check in cases:
        s = score_partial(inst, ans, corpus)
        ok = check(s)
        all_ok = all_ok and ok
        metrics = {k: s[k] for k in ("span_precision", "span_recall", "span_f1", "span_ok", "partial_exact")}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {json.dumps(metrics, ensure_ascii=False)}")
    print(f"\n부분-span 스코어러 자기검증: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/seed/constitution.json")
    ap.add_argument("--items", default="eval/questions.partial.constitution.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--near", action="store_true")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--dump", type=int, default=0)
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    print(f"corpus {len(corpus)}조문 (closed-set)")
    if args.demo:
        return _run_demo(corpus)

    items = load_items(args.items)
    validate_items(items, corpus)
    insts = build_partial_instances(corpus, items, args.k, args.near, args.seed)
    print(f"부분-span 인스턴스 {len(insts)}개 (k={args.k}, near={args.near})")
    for inst in insts[: args.dump]:
        print("\n" + "=" * 60)
        print(f"gold={inst['gold']} | gold_span=「{inst['gold_span']}」")
        print(inst["messages"][1]["content"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
