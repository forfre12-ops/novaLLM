"""law-corpus-v0.1 구조 검증기.

실제 법령 응답은 모양이 들쭉날쭉할 수 있으므로, 정규화 결과가 scorer에 안전한지 별도 검증한다.

    python scripts/data/validate_law_corpus.py --corpus data/processed/laws.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REQUIRED_TOP = ("schema_version", "closed_set", "n_entries", "articles")
REQUIRED_ENTRY = ("id", "law_name", "article_number", "text")
SCHEMA_VERSION = "law-corpus-v0.1"


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def warn(msg: str) -> None:
    print(f"WARN: {msg}")


def validate(path: str | Path, *, min_entries: int = 1) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in REQUIRED_TOP:
        if key not in data:
            fail(f"missing top-level key `{key}`")
    if data["schema_version"] != SCHEMA_VERSION:
        fail(f"unsupported schema_version: {data['schema_version']}")
    if data["closed_set"] is not True:
        fail("closed_set must be true")
    articles = data["articles"]
    if not isinstance(articles, dict):
        fail("articles must be an object")
    if len(articles) < min_entries:
        fail(f"too few entries: {len(articles)} < {min_entries}")
    if data["n_entries"] != len(articles):
        fail(f"n_entries mismatch: {data['n_entries']} != {len(articles)}")

    entries = data.get("entries", [])
    if entries:
        ids: list[str] = []
        for idx, entry in enumerate(entries):
            for key in REQUIRED_ENTRY:
                if not entry.get(key):
                    fail(f"entry[{idx}] missing `{key}`")
            cid = entry["id"]
            text = entry["text"]
            ids.append(cid)
            if cid not in articles:
                fail(f"entry id not in articles: {cid}")
            if articles[cid] != text:
                fail(f"entry/articles text mismatch: {cid}")
        if len(set(ids)) != len(ids):
            fail("duplicate entry ids")
        if set(ids) != set(articles):
            warn("entries and articles id sets differ")

    bad_ids = [cid for cid in articles if not re.search(r"제\d+조", cid)]
    empty = [cid for cid, text in articles.items() if not str(text).strip()]
    if bad_ids:
        warn(f"ids without article number pattern: {len(bad_ids)}")
    if empty:
        fail(f"empty article text: {empty[:3]}")

    law_names = sorted({entry.get("law_name", "") for entry in entries if entry.get("law_name")})
    result = {
        "path": str(path),
        "n_entries": len(articles),
        "n_laws": len(law_names) if law_names else data.get("n_laws", 0),
        "laws": law_names[:20],
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--min-entries", type=int, default=1)
    args = ap.parse_args()
    result = validate(args.corpus, min_entries=args.min_entries)
    print("law corpus validation: PASS")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
