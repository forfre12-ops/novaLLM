"""Generate deterministic SFT data from a normalized law corpus.

The output is standard chat JSONL:

    {"messages": [{"role": "...", "content": "..."}], ...}

No teacher model is used. Positive answers quote exact corpus substrings and are
validated with citation_verify before the file is written.

Example:

    python scripts/data/gen_law_sft.py --corpus data/processed/laws.json \
      --out data/processed/law_sft.jsonl --max-full 1200 --max-tight 1200
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.citation_verify import _norm, load_corpus, verify  # noqa: E402


SYSTEM_PROMPT = (
    "너는 제공된 근거 조항만 사용해 답한다. 여러 근거 중 질문에 해당하는 조항을 찾고, "
    "반드시 「원문」[조항ID] 형식으로 인용한다. 근거에 없으면 "
    "'제공된 근거에서는 확인할 수 없습니다.'라고 답한다."
)

REFUSAL = "제공된 근거에서는 확인할 수 없습니다. 근거가 없는 사항은 추측하지 않습니다."

UNANSWERABLE_QUESTIONS = [
    "특허법상 특허출원 심사청구는 어떻게 규정되어 있는가?",
    "주택임대차보호법상 대항력 발생 요건은 어떻게 규정되어 있는가?",
    "상가건물 임대차보호법상 계약갱신 요구권은 어떻게 규정되어 있는가?",
    "저작권법상 저작재산권 제한은 어떻게 규정되어 있는가?",
    "근로기준법상 연차 유급휴가 산정은 어떻게 규정되어 있는가?",
    "민사소송법상 상고 제기 절차는 어떻게 규정되어 있는가?",
    "국가공무원법상 공무원의 결격사유는 어떻게 규정되어 있는가?",
    "형사소송법상 구속영장 청구 절차는 어떻게 규정되어 있는가?",
    "도로교통법상 운전면허 취소 사유는 어떻게 규정되어 있는가?",
    "상법상 상행위와 상인 자격은 어떻게 규정되어 있는가?",
]

_TITLE_RE = re.compile(r"^제\d+조(?:의\d+)?\(([^)]+)\)")
_ARTICLE_RE = re.compile(r"(제\d+조(?:의\d+)?)")
_SENTENCE_RE = re.compile(r"(?<=\.)\s+")


def load_articles(path: str | Path) -> dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.get("articles", data).items()}


def load_excluded_ids(paths: list[str] | None) -> set[str]:
    if not paths:
        return set()
    excluded = set()
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            excluded.update(_norm(str(k)) for k in data)
            continue
        if isinstance(data, list):
            before = len(excluded)
            for row in data:
                if isinstance(row, dict) and "id" in row:
                    excluded.add(_norm(str(row["id"])))
                elif isinstance(row, str):
                    excluded.add(_norm(row))
                else:
                    raise SystemExit(f"unsupported exclude row: {row!r}")
            if len(excluded) >= before:
                continue
        raise SystemExit("--exclude-questions must be JSON object, list[str], or list[{id,...}]")
    return excluded


def article_key(cid: str) -> str:
    match = _ARTICLE_RE.search(cid)
    return match.group(1) if match else cid


def topic_from_text(text: str, cid: str) -> str:
    title = _TITLE_RE.search(text)
    if title:
        return title.group(1)
    return article_key(cid)


def split_spans(text: str, min_len: int = 16) -> list[str]:
    spans = [s.strip() for s in _SENTENCE_RE.split(text.strip())]
    return [s for s in spans if min_len <= len(s) < len(text)]


def context_block(items: list[tuple[str, str]]) -> str:
    lines = [f"{i + 1}) {cid}: {text}" for i, (cid, text) in enumerate(items)]
    return "[근거]\n" + "\n".join(lines)


def pick_distractors(corpus: dict[str, str], gold: str, k: int, rng: random.Random) -> list[str]:
    others = [cid for cid in corpus if cid != gold]
    siblings = [cid for cid in others if article_key(cid) == article_key(gold)]
    rest = [cid for cid in others if article_key(cid) != article_key(gold)]
    rng.shuffle(siblings)
    rng.shuffle(rest)
    return (siblings + rest)[:k]


def make_context(corpus: dict[str, str], gold: str | None, k: int, rng: random.Random) -> tuple[str, list[str]]:
    if gold is None:
        ids = list(corpus)
        rng.shuffle(ids)
        chosen = ids[: min(k, len(ids))]
    else:
        chosen = [gold] + pick_distractors(corpus, gold, max(0, k - 1), rng)
        rng.shuffle(chosen)
    return context_block([(cid, corpus[cid]) for cid in chosen]), chosen


def make_positive(
    corpus: dict[str, str],
    cid: str,
    question: str,
    quote: str,
    label: str,
    k: int,
    rng: random.Random,
) -> dict:
    ctx, ctx_ids = make_context(corpus, cid, k, rng)
    answer = f"「{quote}」[{cid}]라고 규정하고 있습니다."
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{ctx}\n\n질문: {question}"},
            {"role": "assistant", "content": answer},
        ],
        "label": label,
        "gold_citations": [cid],
        "gold_span": quote,
        "context_ids": ctx_ids,
    }


def make_refusal(corpus: dict[str, str], question: str, k: int, rng: random.Random) -> dict:
    ctx, ctx_ids = make_context(corpus, None, k, rng)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{ctx}\n\n질문: {question}"},
            {"role": "assistant", "content": REFUSAL},
        ],
        "label": "refusal",
        "gold_citations": [],
        "context_ids": ctx_ids,
    }


def validate_rows(rows: list[dict], corpus_path: str) -> int:
    corpus = load_corpus(corpus_path)
    bad = 0
    for row in rows:
        if row["label"] == "refusal":
            if "[" in row["messages"][-1]["content"] or "「" in row["messages"][-1]["content"]:
                bad += 1
            continue
        rep = verify(row["messages"][-1]["content"], corpus)
        if not rep["leak_free"]:
            bad += 1
            print(f"bad positive: {row['gold_citations']} faithfulness={rep['faithfulness']}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="data/processed/law_sft.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-full", type=int, default=0, help="0 means all articles")
    ap.add_argument("--max-tight", type=int, default=0, help="0 means all tight spans")
    ap.add_argument("--refusal-ratio", type=float, default=0.2)
    ap.add_argument(
        "--exclude-questions",
        action="append",
        help="Question JSON whose IDs are held out from positive/context SFT rows. Repeatable.",
    )
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    excluded = load_excluded_ids(args.exclude_questions)
    full_corpus = load_articles(args.corpus)
    corpus = {cid: text for cid, text in full_corpus.items() if _norm(cid) not in excluded}
    if not corpus:
        raise SystemExit("no training corpus rows left after exclusions")
    ids = [cid for cid in corpus if _norm(cid) not in excluded]
    rng.shuffle(ids)
    if args.max_full:
        ids = ids[: args.max_full]

    rows: list[dict] = []
    for cid in ids:
        topic = topic_from_text(corpus[cid], cid)
        question = f"법령상 {topic}에 관해 어떻게 규정되어 있는가?"
        rows.append(make_positive(corpus, cid, question, corpus[cid], "grounded_full", args.k, rng))

    tight_candidates: list[tuple[str, str]] = []
    for cid, text in corpus.items():
        if _norm(cid) in excluded:
            continue
        for span in split_spans(text):
            tight_candidates.append((cid, span))
    rng.shuffle(tight_candidates)
    if args.max_tight:
        tight_candidates = tight_candidates[: args.max_tight]

    for cid, span in tight_candidates:
        question = f"{article_key(cid)}의 해당 문장만 원문 그대로 인용해줘."
        rows.append(make_positive(corpus, cid, question, span, "grounded_tight", args.k, rng))

    n_positive = len(rows)
    if args.refusal_ratio <= 0:
        n_refusal = 0
    else:
        n_refusal = int(n_positive * args.refusal_ratio / max(0.0001, 1 - args.refusal_ratio))
    for i in range(n_refusal):
        rows.append(make_refusal(corpus, UNANSWERABLE_QUESTIONS[i % len(UNANSWERABLE_QUESTIONS)], args.k, rng))

    bad = validate_rows(rows, args.corpus)
    if bad:
        raise SystemExit(f"self-consistency failed: {bad} bad rows")

    rng.shuffle(rows)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_full = sum(1 for r in rows if r["label"] == "grounded_full")
    n_tight = sum(1 for r in rows if r["label"] == "grounded_tight")
    n_refusal = sum(1 for r in rows if r["label"] == "refusal")
    print(f"saved: {outp}")
    print(f"rows: full {n_full} + tight {n_tight} + refusal {n_refusal} = {len(rows)}")
    print(f"excluded positive IDs: {len(excluded)}")
    print("self-consistency: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
