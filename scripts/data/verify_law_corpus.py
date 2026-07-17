"""law_corpus parser fixture 검증.

pytest 없이도 CI/Windows에서 바로 돌릴 수 있는 smoke test다.

    python scripts/data/verify_law_corpus.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.law_corpus import load_payload, normalize_law_payload  # noqa: E402


FIXTURES = [
    Path("tests/fixtures/law_service_sample.json"),
    Path("tests/fixtures/law_service_sample.xml"),
]


def check(path: Path) -> bool:
    payload, raw = load_payload(path)
    corpus = normalize_law_payload(payload, source_url=f"fixture://{path.name}", raw_text=raw)
    articles = corpus["articles"]
    expected = {
        "예시법 제1조": "제1조(목적) 이 법은 예시를 목적으로 한다.",
        "예시법 제2조 ①": "인용문은 원문 그대로 옮긴다.",
        "예시법 제2조 ②": "근거가 없으면 답변하지 아니한다. 1. 제공된 근거가 없는 경우",
    }
    ok = corpus["schema_version"] == "law-corpus-v0.1" and articles == expected
    print(f"{path}: {'PASS' if ok else 'FAIL'} ({corpus['n_entries']} entries)")
    if not ok:
        print("expected:", expected)
        print("actual  :", articles)
    return ok


def main() -> int:
    results = [check(path) for path in FIXTURES]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
