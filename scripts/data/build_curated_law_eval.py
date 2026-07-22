"""Build curated law eval files from a data-driven seed spec.

The output files are tracked eval artifacts. Partial gold spans are extracted
from the normalized corpus when a spec row provides ``sentence_index``, so typos
cannot silently enter the partial-span set.

    python scripts/data/build_curated_law_eval.py --corpus data/processed/laws.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.citation_verify import _norm as _scorer_norm  # noqa: E402

DEFAULT_SPEC = Path("eval/curated_law_seed.json")

# Minimum contiguous article substring (normalized chars) that, if it appears in
# an answerable question, counts as a gold leak. Below this length the overlap is
# almost always a statute name or common legal phrase (false positive).
ANSWERABLE_LEAK_MIN_CHARS = 20


def _wsnorm(s: str) -> str:
    """스코어러(citation_verify._norm)와 동일 정규화 — 가드가 스코어러의 substring 동등성과
    일치해야 folded-variant(ㆍ vs ·, 전각) leak을 놓치지 않는다."""
    return _scorer_norm(s or "")


def load_articles(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("articles", data)


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise SystemExit("--spec must be a JSON object")
    for key in ("answerable", "partial", "unanswerable"):
        if key not in spec or not isinstance(spec[key], list):
            raise SystemExit(f"--spec missing list field: {key}")
    for key in ("answerable", "partial"):
        for idx, row in enumerate(spec[key]):
            if not isinstance(row, dict):
                raise SystemExit(f"{key} row {idx} must be an object")
    return spec


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=\.)\s+", text)
    return [p for p in parts if p]


def require_text(row: dict[str, Any], key: str, section: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{section} row needs non-empty string field: {key}")
    return value


def assert_ids_exist(
    articles: dict[str, str],
    answerable: list[dict[str, Any]],
    partial: list[dict[str, Any]],
) -> None:
    missing: list[str] = []
    missing += [require_text(row, "id", "answerable") for row in answerable if row.get("id") not in articles]
    missing += [require_text(row, "id", "partial") for row in partial if row.get("id") not in articles]
    if missing:
        raise SystemExit("missing curated IDs: " + ", ".join(sorted(set(missing))))


def build_questions(rows: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        cid = require_text(row, "id", "answerable")
        question = require_text(row, "question", "answerable")
        if cid in out:
            raise SystemExit(f"duplicate answerable id: {cid}")
        out[cid] = question
    return out


def resolve_gold_span(articles: dict[str, str], row: dict[str, Any]) -> str:
    cid = require_text(row, "id", "partial")
    if "gold_span" in row:
        span = require_text(row, "gold_span", "partial")
        if span not in articles[cid]:
            raise SystemExit(f"gold_span is not exact substring: {cid}")
        return span

    if "sentence_index" not in row:
        raise SystemExit(f"partial row needs sentence_index or gold_span: {cid}")
    sent_idx = row["sentence_index"]
    if not isinstance(sent_idx, int):
        raise SystemExit(f"partial sentence_index must be int: {cid}")
    spans = split_sentences(articles[cid])
    if sent_idx >= len(spans):
        raise SystemExit(f"sentence index out of range: {cid} idx={sent_idx} n={len(spans)}")
    span = spans[sent_idx]
    if span not in articles[cid]:
        raise SystemExit(f"span is not exact substring: {cid}")
    return span


def build_partial_items(articles: dict[str, str], rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    out = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        cid = require_text(row, "id", "partial")
        question = require_text(row, "question", "partial")
        key = (cid, question)
        if key in seen:
            raise SystemExit(f"duplicate partial row: {cid} / {question}")
        seen.add(key)
        out.append({"id": cid, "question": question, "gold_span": resolve_gold_span(articles, row)})
    return out


def build_unanswerable(rows: list[Any]) -> list[str]:
    out: list[str] = []
    for idx, row in enumerate(rows):
        if isinstance(row, str) and row.strip():
            out.append(row)
            continue
        if isinstance(row, dict):
            question = row.get("question")
            if isinstance(question, str) and question.strip():
                out.append(question)
                continue
        raise SystemExit(f"unanswerable row {idx} must be a string or {{question}} object")
    if len(out) != len(set(out)):
        raise SystemExit("duplicate unanswerable questions")
    return out


def _row_source(row: dict[str, Any]) -> str:
    """Provenance of a seed row. Untagged rows default to ``curated`` (strict)."""
    return str(row.get("source", "curated"))


def validate_curated_quality(
    articles: dict[str, str],
    answerable: list[dict[str, Any]],
    partial: list[dict[str, Any]],
) -> dict[str, Any]:
    """Enforce quality guards on hand-curated (``source == curated``) rows.

    The auto-templated subset (``source == auto``) is formulaic and quotes the
    gold prefix by construction; it is reported but not gated. The curated core
    is the trustworthy set backing any headline G0, so it must stay leak-free.
    Raises SystemExit on any curated violation (fail-fast for CI).
    """
    violations: list[str] = []

    # --- partial curated: gold-leak / whole-article / duplicate span ---
    seen_curated_span: set[tuple[str, str]] = set()
    auto_partial_prefix_leak = 0
    for row in partial:
        cid = row["id"]
        gold = resolve_gold_span(articles, row)
        gnorm = _wsnorm(gold)
        qnorm = _wsnorm(row["question"])
        src = _row_source(row)
        if src == "auto":
            if gnorm[:8] and gnorm[:8] in qnorm:
                auto_partial_prefix_leak += 1
            continue
        if gnorm and gnorm in qnorm:
            violations.append(f"partial gold-leak in question: {cid} / {row['question']!r}")
        sentences = split_sentences(articles[cid])
        if len(sentences) > 1 and gnorm == _wsnorm(articles[cid]):
            violations.append(f"partial whole-article gold on multi-sentence article: {cid}")
        key = (cid, gnorm)
        if key in seen_curated_span:
            violations.append(f"duplicate curated partial gold span: {cid}")
        seen_curated_span.add(key)

    # --- answerable curated: long article substring leaked into question ---
    for row in answerable:
        if _row_source(row) != "curated":
            continue
        cid = require_text(row, "id", "answerable")
        anorm = _wsnorm(articles[cid])
        qnorm = _wsnorm(row["question"])
        window = ANSWERABLE_LEAK_MIN_CHARS
        if any(anorm[i : i + window] in qnorm for i in range(0, max(1, len(anorm) - window + 1))):
            violations.append(f"answerable article-text leak in question: {cid} / {row['question']!r}")

    if violations:
        raise SystemExit(
            "curated-quality guard failed ("
            + str(len(violations))
            + "):\n  - "
            + "\n  - ".join(violations)
        )

    n_ans_curated = sum(1 for r in answerable if _row_source(r) == "curated")
    n_par_curated = sum(1 for r in partial if _row_source(r) == "curated")
    return {
        "answerable_curated": n_ans_curated,
        "answerable_auto": len(answerable) - n_ans_curated,
        "partial_curated": n_par_curated,
        "partial_auto": len(partial) - n_par_curated,
        "auto_partial_prefix_leak": auto_partial_prefix_leak,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/laws.json")
    ap.add_argument("--spec", default=str(DEFAULT_SPEC))
    ap.add_argument("--questions-out", default="eval/questions.laws.curated.json")
    ap.add_argument("--partial-out", default="eval/questions.partial.laws.curated.json")
    ap.add_argument("--unanswerable-out", default="eval/questions.unanswerable.laws.curated.json")
    args = ap.parse_args()

    articles = load_articles(Path(args.corpus))
    spec = load_spec(Path(args.spec))
    answerable = spec["answerable"]
    partial_rows = spec["partial"]
    assert_ids_exist(articles, answerable, partial_rows)
    prov = validate_curated_quality(articles, answerable, partial_rows)

    questions = build_questions(answerable)
    partial = build_partial_items(articles, partial_rows)
    unanswerable = build_unanswerable(spec["unanswerable"])

    Path(args.questions_out).write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(args.partial_out).write_text(
        json.dumps(partial, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(args.unanswerable_out).write_text(
        json.dumps(unanswerable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"loaded spec: {args.spec}")
    print(f"saved answerable: {args.questions_out} ({len(questions)})")
    print(f"saved partial: {args.partial_out} ({len(partial)})")
    print(f"saved unanswerable: {args.unanswerable_out} ({len(unanswerable)})")
    print(
        "provenance: "
        f"answerable curated {prov['answerable_curated']} / auto {prov['answerable_auto']}; "
        f"partial curated {prov['partial_curated']} / auto {prov['partial_auto']} "
        f"(auto partial gold-prefix hints: {prov['auto_partial_prefix_leak']})"
    )
    print(
        "genuine curated core (headline-eval eligible): "
        f"answerable {prov['answerable_curated']} / partial {prov['partial_curated']}"
    )
    target = spec.get("final_target")
    if isinstance(target, dict):
        print(
            "final target: "
            f"answerable {target.get('answerable', '?')} / partial {target.get('partial', '?')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
