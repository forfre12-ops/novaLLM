"""TA-07 국가법령정보 OpenAPI fetcher (법제처 공동활용).

closed-set 코퍼스의 authoritative 소스. 실 수집(TA-08, bulk)은 OC 키가 필요(user-gate).
--smoke 는 키 없이 요청 URL 구성만 검증하고 seed 코퍼스로 폴백 — 키 없이 실 데이터를
가져오는 척하지 않는다(정직).

키 발급: https://open.law.go.kr 회원가입 후 OC(이메일 ID). 환경변수 LAW_API_KEY 로 주입.

    python scripts/data/fetch_law.py --smoke
    LAW_API_KEY=<oc> python scripts/data/fetch_law.py --query 헌법 --out data/raw/law_헌법.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlencode

SEARCH_URL = "https://www.law.go.kr/DRF/lawSearch.do"
SERVICE_URL = "https://www.law.go.kr/DRF/lawService.do"


def build_search_url(oc: str, query: str) -> str:
    return SEARCH_URL + "?" + urlencode({
        "OC": oc, "target": "law", "type": "JSON", "query": query, "display": "20",
    })


def build_service_url(oc: str, mst: str) -> str:
    return SERVICE_URL + "?" + urlencode({"OC": oc, "target": "law", "type": "JSON", "MST": mst})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="헌법")
    ap.add_argument("--out", default="data/raw/law.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    oc = os.environ.get("LAW_API_KEY", "")

    if args.smoke:
        sample_oc = oc or "<LAW_API_KEY>"
        print("요청 URL 구성 검증(smoke):")
        print("  search :", build_search_url(sample_oc, args.query))
        print("  service:", build_service_url(sample_oc, "<MST>"))
        seed = Path("data/seed/constitution.json")
        if seed.exists():
            n = len(json.loads(seed.read_text(encoding="utf-8"))["articles"])
            print(f"  seed 코퍼스 폴백: {seed} ({n}개 조문)")
        if not oc:
            print("  ⚠ LAW_API_KEY 미설정 → 실 수집(TA-08)은 user-gate. URL 구성만 검증 완료.")
        print("T0/TA-07 smoke PASS")
        return 0

    if not oc:
        print("LAW_API_KEY 미설정 — 실 수집 불가(user-gate). open.law.go.kr 에서 OC 키 발급 후 재실행.")
        return 2

    # 실 수집 경로 (키 있을 때만)
    import urllib.request
    url = build_search_url(oc, args.query)
    print("fetching:", url)
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
        data = json.loads(resp.read().decode("utf-8"))
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
