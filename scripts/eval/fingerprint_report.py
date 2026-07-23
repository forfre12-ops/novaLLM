"""fingerprint 카드 생성기 — result JSON을 결정적 fingerprint.json + markdown 표로.

공개물(citation-fingerprint.md)의 결과 표를 손으로 옮겨 적으면 수치 드리프트가 난다.
이 스크립트는 run_g0_faithbench/run_g0_partial의 result JSON에서 축별 지표 + provenance +
스코어러 버전을 뽑아 (a) fingerprint.json(제3자 모델의 표준 산출물 포맷) (b) 붙여넣을
markdown 표를 결정적으로 생성한다. rescore/score_predictions 결과에도 그대로 적용된다.

    python scripts/eval/fingerprint_report.py \
      --faithbench docs/env-verify/g0-faithbench-v02-result.json \
      --partial docs/env-verify/g0-partial-v02-result.json \
      --json-out fingerprint.json --md-out fingerprint.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from citation_verify import SCORER_VERSION  # noqa: E402
from faithbench import FAITHBENCH_VERSION  # noqa: E402
from faithbench_partial import PARTIAL_VERSION  # noqa: E402


def _r(x) -> float:
    return round(float(x), 3)


def build_fingerprint(fb: dict, partial: dict | None) -> dict:
    results = fb.get("results", {})
    by_split = fb.get("by_split", {})
    closed = fb.get("closed_book") or {}
    part_results = (partial or {}).get("results", {})

    models: dict[str, dict] = {}
    for m, agg in results.items():
        sel = {k: _r(agg[k]) for k in (
            "selection_exact", "gold_recall", "distractor_cite_rate", "faithfulness_mean",
            "refusal_rate", "answerable_no_citation_rate", "answerable_refused_rate") if k in agg}
        leak = {k: _r(agg[k]) for k in (
            "leak_rate", "leak_citation_rate", "leak_uncited_rate",
            "leak_parametric_rate", "leak_ungrounded_rate") if k in agg}
        entry: dict = {"selection": sel, "leak": leak}
        if m in closed:
            cb = _r(closed[m]["verbatim_recall_rate"])
            entry["closed_book"] = {
                "verbatim_recall_rate": cb,
                "grounding_gain": _r(agg.get("selection_exact", 0.0) - cb),
            }
        if m in by_split and "unseen" in by_split[m]:
            u = by_split[m]["unseen"]
            entry["unseen"] = {"n": u.get("n"), "selection_exact": _r(u.get("selection_exact", 0.0))}
        if m in part_results:
            entry["partial"] = {k: _r(part_results[m][k]) for k in (
                "selected_gold", "span_precision", "span_recall", "span_f1",
                "span_ok", "partial_exact") if k in part_results[m]}
        models[m] = entry

    meta = fb.get("meta", {})
    return {
        "scorer_versions": {
            "faithbench": FAITHBENCH_VERSION,
            "faithbench_partial": PARTIAL_VERSION,
            "citation_verify": SCORER_VERSION,
        },
        "config": {"k": fb.get("k"), "near": fb.get("near"),
                   "n_answerable": fb.get("n_answerable"), "n_unanswerable": fb.get("n_unanswerable")},
        "provenance": {k: meta.get(k) for k in (
            "corpus_path", "corpus_sha256", "questions_path", "questions_sha256",
            "adapter_path", "adapter_sha256", "git_rev", "seed", "models") if k in meta},
        "models": models,
    }


def _table(header: list[str], rows: list[list], align_right_from: int = 1) -> str:
    sep = ["---" if i < align_right_from else "---:" for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(f"{c:.3f}" if isinstance(c, float) else str(c) for c in r) + " |")
    return "\n".join(lines)


def build_markdown(fp: dict) -> str:
    v = fp["scorer_versions"]
    out = [f"<!-- 자동 생성: scripts/eval/fingerprint_report.py (faithbench v{v['faithbench']}, "
           f"partial v{v['faithbench_partial']}, citation_verify v{v['citation_verify']}). 수기 편집 금지. -->", ""]
    models = fp["models"]

    out.append("### Source Selection And Supported Citation\n")
    rows = [[m, e["selection"].get("selection_exact", 0.0), e["selection"].get("faithfulness_mean", 0.0),
             e["leak"].get("leak_rate", 0.0), e["selection"].get("answerable_no_citation_rate", 0.0)]
            for m, e in models.items()]
    out.append(_table(["Model", "selection_exact", "faithfulness", "leak_rate", "answerable_no_citation"], rows))

    if any("leak_parametric_rate" in e["leak"] for e in models.values()):
        out.append("\n### Leak Typology (unanswerable)\n")
        rows = [[m, e["leak"].get("leak_rate", 0.0), e["leak"].get("leak_citation_rate", 0.0),
                 e["leak"].get("leak_parametric_rate", 0.0), e["leak"].get("leak_ungrounded_rate", 0.0)]
                for m, e in models.items()]
        out.append(_table(["Model", "leak_rate", "leak_citation", "leak_parametric", "leak_ungrounded"], rows))

    if any("closed_book" in e for e in models.values()):
        out.append("\n### Closed-Book Memorization Check\n")
        rows = [[m, e["closed_book"]["verbatim_recall_rate"], e["selection"].get("selection_exact", 0.0),
                 e["closed_book"]["grounding_gain"]]
                for m, e in models.items() if "closed_book" in e]
        out.append(_table(["Model", "closed-book verbatim recall", "open selection_exact", "grounding gain"], rows))

    if any("partial" in e for e in models.values()):
        out.append("\n### Tight Span Citation\n")
        rows = [[m, e["partial"].get("partial_exact", 0.0), e["partial"].get("span_f1", 0.0),
                 e["partial"].get("span_precision", 0.0), e["partial"].get("span_recall", 0.0),
                 e["partial"].get("selected_gold", 0.0)]
                for m, e in models.items() if "partial" in e]
        out.append(_table(["Model", "partial_exact", "span_f1", "span_precision", "span_recall", "selected_gold"], rows))

    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faithbench", required=True)
    ap.add_argument("--partial", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--md-out", default=None)
    args = ap.parse_args()

    fb = json.loads(Path(args.faithbench).read_text(encoding="utf-8"))
    partial = json.loads(Path(args.partial).read_text(encoding="utf-8")) if args.partial else None
    fp = build_fingerprint(fb, partial)
    md = build_markdown(fp)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(fp, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"fingerprint.json: {args.json_out}")
    if args.md_out:
        Path(args.md_out).write_text(md, encoding="utf-8")
        print(f"fingerprint.md: {args.md_out}")
    if not args.json_out and not args.md_out:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
