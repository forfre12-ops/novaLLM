"""로컬 통합 smoke runner.

GitHub Actions와 같은 핵심 검증을 로컬에서 한 번에 돌린다. GPU, API key, 네트워크를 요구하지 않는다.

    python scripts/smoke.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-eval-demos", action="store_true")
    args = ap.parse_args()

    run([PY, "-m", "compileall", "-q", "scripts"])
    if not args.skip_eval_demos:
        run([PY, "scripts/eval/citation_verify.py", "--demo"])
        run([PY, "scripts/eval/faithbench.py", "--demo"])
        run([PY, "scripts/eval/faithbench_partial.py", "--demo"])
    run([PY, "scripts/eval/power_analysis.py", "--base", "0.387", "--target", "0.742"])
    run([PY, "scripts/data/fetch_law.py", "--smoke"])
    run([PY, "scripts/data/fetch_law.py", "--smoke", "--response-type", "XML"])
    run([PY, "scripts/data/law_corpus.py", "--demo"])
    run([PY, "scripts/data/verify_law_corpus.py"])

    with tempfile.TemporaryDirectory(prefix="nova-law-") as tmp:
        t = Path(tmp)
        manifest = t / "manifest.json"
        law = t / "law.json"
        laws = t / "laws.json"
        questions = t / "questions.json"
        partial = t / "questions.partial.json"
        unanswerable = t / "questions.unanswerable.json"
        law_sft = t / "law_sft.jsonl"
        run([PY, "scripts/data/plan_law_fetch.py", "--in", "tests/fixtures/law_search_sample.json", "--out", str(manifest)])
        run([PY, "scripts/data/bulk_fetch_laws.py", "--manifest", str(manifest), "--dry-run"])
        run([PY, "scripts/data/bulk_fetch_laws.py", "--manifest", str(manifest), "--response-type", "XML", "--dry-run"])
        run([PY, "scripts/data/law_corpus.py", "--in", "tests/fixtures/law_service_sample.json", "--out", str(law)])
        run([PY, "scripts/data/validate_law_corpus.py", "--corpus", str(law)])
        run([PY, "scripts/data/merge_law_corpora.py", "--in", str(law), "--out", str(laws)])
        run([PY, "scripts/data/validate_law_corpus.py", "--corpus", str(laws)])
        run([PY, "scripts/data/gen_law_questions.py", "--corpus", str(laws), "--out", str(questions), "--partial-out", str(partial)])
        run([PY, "scripts/data/gen_unanswerable_questions.py", "--corpus", str(laws), "--out", str(unanswerable)])
        run([PY, "scripts/eval/faithbench.py", "--corpus", str(laws), "--questions", str(questions),
             "--unanswerable-file", str(unanswerable), "--dump", "1"])
        run([PY, "scripts/eval/faithbench_partial.py", "--corpus", str(laws), "--items", str(partial), "--dump", "1"])
        run([PY, "scripts/data/gen_law_sft.py", "--corpus", str(laws), "--out", str(law_sft),
             "--max-full", "3", "--max-tight", "3", "--refusal-ratio", "0.25"])

    print("smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
