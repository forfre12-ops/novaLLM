"""TA-07 국가법령정보 OpenAPI fetcher (법제처 공동활용).

closed-set 코퍼스의 authoritative 소스. 실 수집(TA-08, bulk)은 OC 키가 필요(user-gate).
--smoke 는 키 없이 요청 URL 구성만 검증하고 seed 코퍼스로 폴백 — 키 없이 실 데이터를
가져오는 척하지 않는다(정직).

키 발급: https://open.law.go.kr 회원가입 후 OC(이메일 ID). 환경변수 LAW_API_KEY 로 주입.

    python scripts/data/fetch_law.py --smoke
    LAW_API_KEY=<oc> python scripts/data/fetch_law.py --query 헌법 --raw-out data/raw/law_search_헌법.json
    LAW_API_KEY=<oc> python scripts/data/fetch_law.py --mst 166520 --raw-out data/raw/law_service.json \
        --corpus-out data/processed/law_corpus.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.law_corpus import normalize_law_payload  # noqa: E402

SEARCH_URL = "https://www.law.go.kr/DRF/lawSearch.do"
SERVICE_URL = "https://www.law.go.kr/DRF/lawService.do"


def safe_print(text: str = "") -> None:
    enc = sys.stdout.encoding or "utf-8"
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


def build_search_url(oc: str, query: str) -> str:
    return SEARCH_URL + "?" + urlencode({
        "OC": oc, "target": "law", "type": "JSON", "query": query, "display": "20",
    })


def build_service_url(oc: str, *, mst: str = "", law_id: str = "", target: str = "eflaw") -> str:
    params = {"OC": oc, "target": target, "type": "JSON"}
    if mst:
        params["MST"] = mst
    elif law_id:
        params["ID"] = law_id
    return SERVICE_URL + "?" + urlencode(params)


def fetch_json(url: str) -> dict:
    import urllib.request
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def write_json(path: str | Path, data: dict) -> None:
    outp = Path(path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    safe_print(f"저장: {outp}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="헌법")
    ap.add_argument("--mst", help="lawService MST 값으로 본문 조회")
    ap.add_argument("--law-id", help="lawService ID 값으로 본문 조회")
    ap.add_argument("--target", default="eflaw", help="lawService target (기본: eflaw)")
    ap.add_argument("--raw-out", default="data/raw/law.json")
    ap.add_argument("--corpus-out", help="lawService 응답을 표준 closed-set corpus로 정규화해 저장")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    oc = os.environ.get("LAW_API_KEY", "")

    if args.smoke:
        sample_oc = oc or "<LAW_API_KEY>"
        safe_print("요청 URL 구성 검증(smoke):")
        safe_print("  search : " + build_search_url(sample_oc, args.query))
        safe_print("  service: " + build_service_url(sample_oc, mst=args.mst or "<MST>", law_id=args.law_id or ""))
        seed = Path("data/seed/constitution.json")
        if seed.exists():
            n = len(json.loads(seed.read_text(encoding="utf-8"))["articles"])
            safe_print(f"  seed 코퍼스 폴백: {seed} ({n}개 조문)")
        if not oc:
            safe_print("  WARN LAW_API_KEY 미설정 -> 실 수집(TA-08)은 user-gate. URL 구성만 검증 완료.")
        safe_print("T0/TA-07 smoke PASS")
        return 0

    if not oc:
        safe_print("LAW_API_KEY 미설정 - 실 수집 불가(user-gate). open.law.go.kr 에서 OC 키 발급 후 재실행.")
        return 2

    if args.mst or args.law_id:
        url = build_service_url(oc, mst=args.mst or "", law_id=args.law_id or "", target=args.target)
        safe_print("fetching service: " + url)
        data = fetch_json(url)
        write_json(args.raw_out, data)
        if args.corpus_out:
            corpus = normalize_law_payload(data, source_url=url, raw_text=json.dumps(data, ensure_ascii=False))
            write_json(args.corpus_out, corpus)
        return 0

    url = build_search_url(oc, args.query)
    safe_print("fetching: " + url)
    data = fetch_json(url)
    write_json(args.raw_out, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
