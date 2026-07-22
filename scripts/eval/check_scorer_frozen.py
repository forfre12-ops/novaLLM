"""스코어러 동결 게이트 — 채점 규칙이 몰래 바뀌지 않았음을 기계적으로 강제한다.

tracked한 v02 transcript(모델별 원문답변 보존)를 현재 faithbench 스코어러로 재채점해
**전체 aggregate(모든 지표, leak 유형학 포함)**를 golden과 byte-exact 비교한다. 하나라도
어긋나면 실패 → 규칙을 바꾸려면 FAITHBENCH_VERSION을 bump하고 --write-golden으로 재생성해야
한다. '동결' 선언(eval/README·g0-verdict §6)을 문서 약속에서 CI 강제로 승격한다.

    python scripts/eval/check_scorer_frozen.py              # 검증(CI)
    python scripts/eval/check_scorer_frozen.py --write-golden  # 규칙 변경 후 golden 재생성
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from citation_verify import SCORER_VERSION, load_corpus  # noqa: E402
from faithbench import FAITHBENCH_VERSION, aggregate, score_answer  # noqa: E402

TRANSCRIPT = ROOT / "docs/env-verify/g0-faithbench-v02-result-transcript.jsonl"
CORPUS = ROOT / "data/seed/constitution.json"
GOLDEN = ROOT / "tests/fixtures/scorer_frozen_golden.json"


def _compute() -> dict:
    corpus = load_corpus(str(CORPUS))
    rows = [json.loads(l) for l in TRANSCRIPT.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        gold = r.get("gold")
        inst = {"split": r["split"], "gold": [gold] if gold else [], "context_ids": r.get("context_ids", [])}
        by_model.setdefault(r["model"], []).append(score_answer(inst, r.get("answer", ""), corpus))
    return {
        "faithbench_version": FAITHBENCH_VERSION,
        "citation_verify_version": SCORER_VERSION,
        "source_transcript": TRANSCRIPT.relative_to(ROOT).as_posix(),
        "aggregate": {m: aggregate(scored) for m, scored in sorted(by_model.items())},
    }


def _canon(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-golden", action="store_true", help="현재 스코어러 출력으로 golden 재생성")
    args = ap.parse_args()

    current = _compute()

    if args.write_golden:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(_canon(current) + "\n", encoding="utf-8")
        print(f"golden 재생성: {GOLDEN.relative_to(ROOT).as_posix()} "
              f"(faithbench v{FAITHBENCH_VERSION}, citation_verify v{SCORER_VERSION})")
        return 0

    if not GOLDEN.exists():
        print(f"golden 없음: {GOLDEN} — 먼저 --write-golden 실행")
        return 1
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))

    problems = []
    if golden.get("faithbench_version") != FAITHBENCH_VERSION:
        problems.append(
            f"FAITHBENCH_VERSION 불일치: golden={golden.get('faithbench_version')} "
            f"code={FAITHBENCH_VERSION} — 규칙이 바뀌었는데 golden 미갱신이거나 그 반대"
        )
    if _canon(current["aggregate"]) != _canon(golden.get("aggregate", {})):
        # 어느 모델·지표가 어긋났는지 짚어준다.
        for m, agg in current["aggregate"].items():
            gold_agg = golden.get("aggregate", {}).get(m, {})
            diffs = [k for k in agg if agg.get(k) != gold_agg.get(k)]
            if diffs:
                problems.append(f"aggregate 드리프트 [{m}]: " + ", ".join(
                    f"{k} {gold_agg.get(k)}→{agg.get(k)}" for k in diffs))

    if problems:
        print("check_scorer_frozen: FAIL")
        for p in problems:
            print("  - " + p)
        print("  규칙을 의도적으로 바꿨다면 FAITHBENCH_VERSION bump 후 --write-golden.")
        return 1
    print(f"check_scorer_frozen: PASS (faithbench v{FAITHBENCH_VERSION}, citation_verify v{SCORER_VERSION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
