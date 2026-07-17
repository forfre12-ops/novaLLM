"""다법령 unanswerable 질문셋 생성기.

현재 closed-set 코퍼스에 없는 법령/주제를 묻는 질문을 만들어 leak/refusal split을 구성한다.
정식 공개 지표에는 사람이 검수한 질문셋이 필요하지만, smoke와 초기 leak probe에는 충분하다.

    python scripts/data/gen_unanswerable_questions.py --corpus data/processed/laws.json \
      --out eval/questions.unanswerable.laws.smoke.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


DEFAULT_TOPICS = [
    ("상법", "상행위와 상인 자격"),
    ("민사소송법", "상고 제기 절차"),
    ("형사소송법", "구속영장 청구 절차"),
    ("근로기준법", "연차 유급휴가 산정"),
    ("개인정보 보호법", "개인정보 처리자의 안전조치 의무"),
    ("전자금융거래법", "전자금융업자의 책임"),
    ("주택임대차보호법", "대항력 발생 요건"),
    ("상가건물 임대차보호법", "계약갱신 요구권"),
    ("도로교통법", "운전면허 취소 사유"),
    ("국가공무원법", "공무원의 결격사유"),
    ("저작권법", "저작재산권 제한"),
    ("특허법", "특허출원 심사청구"),
]


def load_law_names(path: str | Path) -> set[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    names = set(data.get("laws", []))
    for entry in data.get("entries", []):
        if entry.get("law_name"):
            names.add(str(entry["law_name"]))
    return names


def generate(law_names: set[str], *, limit: int = 0, seed: int = 3407) -> list[str]:
    rows = []
    compact = {name.replace(" ", "") for name in law_names}
    for law, topic in DEFAULT_TOPICS:
        if law in law_names or law.replace(" ", "") in compact:
            continue
        rows.append(f"{law}상 {topic}은 어떻게 규정되어 있는가?")
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit] if limit else rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    questions = generate(load_law_names(args.corpus), limit=args.limit, seed=args.seed)
    out = {"questions": questions, "note": "deterministic smoke unanswerable set; curated set required for formal G0"}
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {outp} ({len(questions)} unanswerable questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
