"""Provenance 무결성 감사 — 기록된 raw_sha256이 디스크 raw 파일에서 재현되는지 검증한다.

각 법령 소스의 raw_sha256(원문 API 응답의 sha256)을, 저장된 raw 파일(data/raw/laws/<basename>)의
sha256과 대조한다. 일치하면 제3자가 provenance를 재검증할 수 있다("full provenance" A2 주장의 근거).

기본은 진단(exit 0, 리포트만). --strict면 불일치/누락에서 exit 1(clean 스냅샷 재수집 후 CI 게이트용).

    python scripts/data/verify_provenance_chain.py --corpus data/processed/laws.json
    python scripts/data/verify_provenance_chain.py --corpus data/processed/laws.json --strict

주의: 2026-07 스냅샷은 구버전 fetch(JSON 재직렬화 저장) 산물이라 raw_sha256이 디스크와
불일치한다(legacy). bulk_fetch_laws는 이제 원문을 verbatim 저장하므로, LAW_API_KEY로 재수집한
스냅샷부터 이 검증이 통과한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/laws.json")
    ap.add_argument("--raw-dir", default="data/raw/laws")
    ap.add_argument("--strict", action="store_true", help="불일치/누락 시 exit 1")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"verify_provenance_chain: SKIP (no corpus at {corpus_path})")
        return 0
    sources = json.loads(corpus_path.read_text(encoding="utf-8")).get("sources", [])
    if not sources:
        print("verify_provenance_chain: SKIP (corpus에 sources 없음 — 단일 코퍼스일 수 있음)")
        return 0

    raw_dir = Path(args.raw_dir)
    n_match = n_mismatch = n_missing = 0
    for s in sources:
        recorded = s.get("raw_sha256", "")
        base = Path(s.get("path", "")).name  # 처리본 basename == raw 파일 basename
        raw_file = raw_dir / base
        if not recorded:
            print(f"  [NO-HASH] {base}: raw_sha256 없음")
            n_missing += 1
            continue
        if not raw_file.exists():
            print(f"  [MISSING] {base}: raw 파일 없음 ({raw_file})")
            n_missing += 1
            continue
        actual = _sha256(raw_file)
        if actual == recorded:
            print(f"  [MATCH]    {base}")
            n_match += 1
        else:
            print(f"  [MISMATCH] {base}: recorded={recorded[:16]}… disk={actual[:16]}…")
            n_mismatch += 1

    print(f"\nprovenance: match {n_match} / mismatch {n_mismatch} / missing {n_missing}")
    if n_mismatch or n_missing:
        print("  → 불일치는 대개 구버전 fetch(재직렬화) legacy. LAW_API_KEY로 재수집 시 verbatim 저장되어 일치한다.")
    if args.strict and (n_mismatch or n_missing):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
