"""Governance guard for the curated law eval + its holdout SFT.

Two deterministic checks, both corpus-gated so they no-op in a checkout without
the (gitignored) generated corpus (e.g. GitHub CI):

1. drift: rebuild the tracked curated eval files from the seed and assert they
   are byte-identical. Catches hand-editing of the tracked eval away from the
   seed (the eval files must always be regenerable from ``curated_law_seed.json``).
2. holdout leak: if a curated-holdout SFT file exists, assert that no curated
   eval ID appears in any training row's ``gold_citations`` or ``context_ids``.
   This is what keeps the eval a true holdout.

    python scripts/data/verify_curated_holdout.py
    python scripts/data/verify_curated_holdout.py --sft data/processed/law_sft_curated_holdout.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# The importable ``eval`` package lives under scripts/ (scripts/eval); the repo
# root is one level above that and is where relative data/eval paths resolve.
SCRIPTS = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
from eval.citation_verify import _norm  # noqa: E402

PY = sys.executable


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO / path)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_drift(corpus: Path, spec: Path, tracked: dict[str, Path]) -> list[str]:
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="nova-curated-") as tmp:
        t = Path(tmp)
        out = {name: t / f"{name}.json" for name in tracked}
        subprocess.run(
            [
                PY, str(REPO / "scripts/data/build_curated_law_eval.py"),
                "--corpus", str(corpus), "--spec", str(spec),
                "--questions-out", str(out["answerable"]),
                "--partial-out", str(out["partial"]),
                "--unanswerable-out", str(out["unanswerable"]),
            ],
            cwd=REPO, check=True,
        )
        for name, path in tracked.items():
            if not path.exists():
                problems.append(f"tracked eval missing: {path}")
            elif _read(out[name]) != _read(path):
                problems.append(f"tracked eval drifted from seed: {path} (rerun build_curated_law_eval.py)")
    return problems


def curated_ids(tracked: dict[str, Path]) -> set[str]:
    ids: set[str] = set()
    ans = json.loads(_read(tracked["answerable"]))
    ids.update(_norm(str(k)) for k in ans)
    par = json.loads(_read(tracked["partial"]))
    ids.update(_norm(str(row["id"])) for row in par)
    return ids


def check_holdout_leak(sft: Path, ids: set[str]) -> list[str]:
    problems: list[str] = []
    gold_hits = ctx_hits = 0
    for line in _read(sft).splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        for cid in row.get("gold_citations", []) or []:
            if _norm(str(cid)) in ids:
                gold_hits += 1
        for cid in row.get("context_ids", []) or []:
            if _norm(str(cid)) in ids:
                ctx_hits += 1
    if gold_hits or ctx_hits:
        problems.append(
            f"holdout leak in {sft}: curated eval IDs appear as gold={gold_hits}, context={ctx_hits} (must be 0)"
        )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/laws.json")
    ap.add_argument("--spec", default="eval/curated_law_seed.json")
    ap.add_argument("--answerable", default="eval/questions.laws.curated.json")
    ap.add_argument("--partial", default="eval/questions.partial.laws.curated.json")
    ap.add_argument("--unanswerable", default="eval/questions.unanswerable.laws.curated.json")
    ap.add_argument("--sft", default="data/processed/law_sft_curated_holdout.jsonl")
    args = ap.parse_args()

    corpus = _resolve(args.corpus)
    if not corpus.exists():
        print(f"verify_curated_holdout: SKIP (no corpus at {corpus}; nothing to check in this checkout)")
        return 0

    tracked = {
        "answerable": _resolve(args.answerable),
        "partial": _resolve(args.partial),
        "unanswerable": _resolve(args.unanswerable),
    }
    problems = check_drift(corpus, _resolve(args.spec), tracked)

    sft = _resolve(args.sft)
    if sft.exists():
        problems += check_holdout_leak(sft, curated_ids(tracked))
    else:
        print(f"verify_curated_holdout: holdout SFT not present ({sft}); skipping leak scan")

    if problems:
        print("verify_curated_holdout: FAIL")
        for p in problems:
            print("  - " + p)
        return 1
    print("verify_curated_holdout: PASS (eval regenerable from seed; holdout leak-free)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
