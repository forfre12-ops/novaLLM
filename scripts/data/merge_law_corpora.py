"""여러 law-corpus-v0.1 파일을 하나의 closed-set 코퍼스로 병합한다.

각 입력은 `scripts/data/law_corpus.py`가 만든 표준 코퍼스여야 한다. scorer 호환을 위해
top-level `articles` mapping을 유지하고, provenance는 `entries`와 `sources`에 보존한다.

    python scripts/data/merge_law_corpora.py --in data/processed/law_*.json --out data/processed/laws.json
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import date
from pathlib import Path

SCHEMA_VERSION = "law-corpus-v0.1"


def load(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(f"unsupported schema: {path} ({data.get('schema_version')})")
    return data


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        hits = [Path(p) for p in glob.glob(pattern)]
        paths.extend(hits or [Path(pattern)])
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        rp = path.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(path)
    return out


def merge(paths: list[Path]) -> dict:
    articles: dict[str, str] = {}
    entries: list[dict] = []
    sources: list[dict] = []
    duplicates: list[str] = []

    for path in paths:
        data = load(path)
        sources.append({
            "path": str(path),
            "source": data.get("source", ""),
            "source_url": data.get("source_url", ""),
            "raw_sha256": data.get("raw_sha256", ""),
            "n_entries": data.get("n_entries", len(data.get("articles", {}))),
        })
        for entry in data.get("entries", []):
            cid = entry["id"]
            text = entry["text"]
            if cid in articles and articles[cid] != text:
                duplicates.append(cid)
                continue
            articles[cid] = text
            entries.append(entry)

    if duplicates:
        dup = ", ".join(sorted(set(duplicates))[:10])
        raise SystemExit(f"conflicting duplicate ids: {dup}")

    law_names = sorted({e.get("law_name", "") for e in entries if e.get("law_name")})
    return {
        "schema_version": SCHEMA_VERSION,
        "source": f"merged law corpus ({len(law_names)} laws)",
        "provenance": "merged by scripts/data/merge_law_corpora.py",
        "authoritative_source": "국가법령정보 OpenAPI",
        "license": "법령 텍스트 = 저작권법 제7조 비보호",
        "closed_set": True,
        "snapshot_date": date.today().isoformat(),
        "n_laws": len(law_names),
        "laws": law_names,
        "n_entries": len(articles),
        "articles": articles,
        "entries": entries,
        "sources": sources,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = expand_inputs(args.inputs)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing input(s): {', '.join(missing)}")

    corpus = merge(paths)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {outp} ({corpus['n_laws']} laws, {corpus['n_entries']} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
