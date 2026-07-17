"""다법령 faithbench smoke 질문셋 생성기.

이 스크립트는 사람이 검수한 정식 벤치가 아니다. 법령 OpenAPI 수집 직후 scorer/runner를
끝까지 태우기 위한 deterministic smoke 질문을 만든다. 정식 공개 지표에는 별도 curated
질문셋을 사용해야 한다.

    python scripts/data/gen_law_questions.py --corpus data/processed/laws.json \
      --out eval/questions.laws.smoke.json --partial-out eval/questions.partial.laws.smoke.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

_TITLE_RE = re.compile(r"^제\d+조(?:의\d+)?\(([^)]+)\)")
_SENT_RE = re.compile(r"(?<=다\.)\s+")


def load_articles(path: str | Path) -> dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("articles", data)


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in _SENT_RE.split(text.strip()) if len(p.strip()) >= 8]


def topic_from_text(text: str) -> str:
    title = _TITLE_RE.search(text)
    if title:
        return title.group(1)
    body = re.sub(r"^제\d+조(?:의\d+)?\([^)]+\)\s*", "", text).strip()
    subject = re.split(r"[은는이가]\s", body, maxsplit=1)[0].strip()
    subject = subject[:24].strip(" ,·")
    return subject or "해당 사항"


def make_question(text: str) -> str:
    topic = topic_from_text(text)
    return f"법령상 {topic}에 관해 어떻게 규정되어 있는가?"


def make_partial_question(span: str) -> str:
    topic = topic_from_text(span)
    return f"법령상 {topic}에 해당하는 부분은 어떻게 규정되어 있는가?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--partial-out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    articles = load_articles(args.corpus)
    items = list(articles.items())
    rng = random.Random(args.seed)
    rng.shuffle(items)
    if args.limit:
        items = items[: args.limit]

    questions = {cid: make_question(text) for cid, text in items}
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {outp} ({len(questions)} questions)")

    if args.partial_out:
        partial = []
        for cid, text in items:
            for span in split_sentences(text):
                if span != text:
                    partial.append({"id": cid, "question": make_partial_question(span), "gold_span": span})
        pout = Path(args.partial_out)
        pout.parent.mkdir(parents=True, exist_ok=True)
        pout.write_text(json.dumps(partial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved: {pout} ({len(partial)} partial questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
