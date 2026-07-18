"""lawSearch 응답에서 bulk fetch manifest를 만든다.

검색 결과의 필드명은 API target/응답 버전에 따라 조금씩 달라질 수 있으므로, 법령명과
MST/ID 후보 필드를 보수적으로 탐색한다.

    python scripts/data/plan_law_fetch.py --in data/raw/law_search_*.json --out data/raw/law_manifest.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


NAME_KEYS = ("법령명한글", "법령명_한글", "법령명", "법령약칭명")
MST_KEYS = ("MST", "mst", "법령일련번호", "법령키")
ID_KEYS = ("법령ID", "ID", "id")


def norm(value: Any) -> str:
    return "" if value is None else str(value).strip()


def first(node: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = norm(node.get(key))
        if value:
            return value
    return ""


def walk_dicts(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk_dicts(value)
    elif isinstance(node, list):
        for item in node:
            yield from walk_dicts(item)


def extract_manifest(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    rows: list[dict[str, str]] = []
    for node in walk_dicts(data):
        law_name = first(node, NAME_KEYS)
        mst = first(node, MST_KEYS)
        law_id = first(node, ID_KEYS)
        if law_name and (mst or law_id):
            rows.append({
                "law_name": law_name,
                "mst": mst,
                "law_id": law_id,
                "effective_date": norm(node.get("시행일자") or node.get("시행일")),
                "source_search_file": str(path),
            })
    return rows


def expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        hits = [Path(p) for p in glob.glob(pattern)]
        paths.extend(hits or [Path(pattern)])
    return paths


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = (row["law_name"], row["mst"], row["law_id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def filter_rows(rows: list[dict[str, str]], *, law_ids: list[str], name_exact: list[str]) -> list[dict[str, str]]:
    law_id_set = {x.strip() for x in law_ids if x.strip()}
    name_set = {x.strip().replace(" ", "") for x in name_exact if x.strip()}
    if not law_id_set and not name_set:
        return rows
    out = []
    for row in rows:
        name_key = row["law_name"].replace(" ", "")
        if row.get("law_id") in law_id_set or name_key in name_set:
            out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--law-id", action="append", default=[], help="포함할 법령ID. 여러 번 지정 가능")
    ap.add_argument("--name-exact", action="append", default=[], help="공백 무시 exact 법령명 필터. 여러 번 지정 가능")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows: list[dict[str, str]] = []
    for path in expand(args.inputs):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
        rows.extend(extract_manifest(path))
    rows = dedupe(rows)
    rows = filter_rows(rows, law_ids=args.law_id, name_exact=args.name_exact)
    if args.limit:
        rows = rows[: args.limit]

    manifest = {
        "schema_version": "law-fetch-manifest-v0.1",
        "n": len(rows),
        "laws": rows,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {outp} ({len(rows)} laws)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
