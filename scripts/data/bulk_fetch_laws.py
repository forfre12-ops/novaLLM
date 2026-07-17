"""manifest 기반 국가법령정보 bulk fetch.

`plan_law_fetch.py`가 만든 manifest를 받아 lawService 원문과 표준 코퍼스를 저장한다.
키가 없을 때는 `--dry-run`으로 URL/출력 경로만 검증한다.

    python scripts/data/bulk_fetch_laws.py --manifest data/raw/law_manifest.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.fetch_law import build_service_url, fetch_json, write_json  # noqa: E402
from data.law_corpus import normalize_law_payload  # noqa: E402


def slug(text: str) -> str:
    text = re.sub(r"[^\w가-힣.-]+", "_", text, flags=re.UNICODE).strip("_")
    return text or "law"


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != "law-fetch-manifest-v0.1":
        raise SystemExit(f"unsupported manifest schema: {data.get('schema_version')}")
    return data.get("laws", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--raw-dir", default="data/raw/laws")
    ap.add_argument("--corpus-dir", default="data/processed/laws")
    ap.add_argument("--target", default="eflaw")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    oc = os.environ.get("LAW_API_KEY", "")
    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    if not args.dry_run and not oc:
        raise SystemExit("LAW_API_KEY 미설정 - --dry-run 또는 키 주입이 필요합니다.")

    raw_dir = Path(args.raw_dir)
    corpus_dir = Path(args.corpus_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(rows, 1):
        name = row["law_name"]
        token = slug(f"{idx:03d}_{name}")
        url = build_service_url(oc or "<LAW_API_KEY>", mst=row.get("mst", ""), law_id=row.get("law_id", ""), target=args.target)
        raw_out = raw_dir / f"{token}.json"
        corpus_out = corpus_dir / f"{token}.json"
        if args.dry_run:
            print(f"[dry-run] {name}: {url}")
            print(f"          raw={raw_out} corpus={corpus_out}")
            continue
        print(f"[{idx}/{len(rows)}] fetching {name}: {url}")
        data = fetch_json(url)
        write_json(raw_out, data)
        corpus = normalize_law_payload(data, source_url=url, raw_text=json.dumps(data, ensure_ascii=False))
        write_json(corpus_out, corpus)
        if args.sleep:
            time.sleep(args.sleep)
    print(f"done: {len(rows)} laws ({'dry-run' if args.dry_run else 'fetched'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
