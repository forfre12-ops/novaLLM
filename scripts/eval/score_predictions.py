"""오프라인 결정적 채점 CLI — 모델/GPU 없이 답변을 채점한다.

두 모드:

1. rescore — 기존 transcript(모델별 원문답변 보존)를 재채점해 aggregate를 재도출한다.
   공표된 결과가 스코어러로 재현되는지 누구나 CPU로 수초 만에 검증하는 경로.

       python scripts/eval/score_predictions.py rescore \
         --transcript docs/env-verify/g0-faithbench-v02-result-transcript.jsonl \
         --corpus data/seed/constitution.json \
         --expect docs/env-verify/g0-faithbench-v02-result.json

2. predictions — 제3자 제출 답변을 채점한다. 벤치가 배포한 instances.jsonl(faithbench --out,
   instance_id 포함)과 제출자의 predictions.jsonl([{instance_id, answer}])을 instance_id로
   조인해 채점 → result JSON + transcript. 모델을 우리 하드웨어에서 돌릴 필요가 없다.

       python scripts/eval/score_predictions.py predictions \
         --instances eval/instances.laws.curated.jsonl \
         --predictions my_answers.jsonl --model my-model \
         --corpus data/processed/laws.json --out result.json

채점은 faithbench의 결정적 score_answer/aggregate를 그대로 쓴다(LLM-judge 없음).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from citation_verify import load_corpus  # noqa: E402
from faithbench import aggregate, score_answer  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{ln} JSON 파싱 실패: {e}") from e
    return rows


def _inst_from_row(row: dict) -> dict:
    """transcript/instances 행에서 채점에 필요한 최소 inst 재구성.

    score_answer는 inst['split']·inst['gold'](list)·inst['context_ids']를 읽는다.
    context_ids는 unanswerable leak 유형학(parametric vs ungrounded)에서 '제공된 근거'를
    판별하는 데 필수 — 빠지면 문맥 내 조문 복사를 parametric으로 오분류한다.
    """
    gold = row.get("gold")
    if isinstance(gold, list):
        gold_list = [g for g in gold if g]
    elif gold:
        gold_list = [gold]
    else:
        gold_list = []
    return {"split": row["split"], "gold": gold_list, "context_ids": row.get("context_ids", [])}


def _score_rows(rows: list[dict], answers: list[str], corpus: dict[str, str]) -> dict:
    scored = [score_answer(_inst_from_row(r), a, corpus) for r, a in zip(rows, answers)]
    return aggregate(scored)


def rescore(args) -> int:
    corpus = load_corpus(args.corpus)
    rows = _read_jsonl(Path(args.transcript))
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r.get("model", "model"), []).append(r)

    results = {
        m: _score_rows(rs, [r.get("answer", "") for r in rs], corpus)
        for m, rs in by_model.items()
    }
    for m, agg in results.items():
        print(f"  {m}: selection_exact={agg['selection_exact']} leak_rate={agg['leak_rate']} "
              f"n_ans={agg['n_answerable']} n_una={agg['n_unanswerable']}")

    ok = True
    if args.expect:
        expected = json.loads(Path(args.expect).read_text(encoding="utf-8")).get("results", {})
        keys = ["selection_exact", "gold_recall", "distractor_cite_rate", "faithfulness_mean",
                "leak_rate", "leak_citation_rate", "leak_uncited_rate",
                "leak_parametric_rate", "leak_ungrounded_rate",
                "refusal_rate", "answerable_no_citation_rate"]
        for m, agg in results.items():
            exp = expected.get(m)
            if exp is None:
                print(f"  [WARN] expect에 모델 없음: {m}")
                continue
            diffs = [k for k in keys if k in exp and agg.get(k) != exp.get(k)]
            status = "MATCH" if not diffs else "MISMATCH " + ",".join(diffs)
            print(f"  [{'OK' if not diffs else 'FAIL'}] {m}: {status}")
            ok = ok and not diffs
        print(f"\nrescore vs expect: {'PASS' if ok else 'FAIL'}")

    if args.out:
        _write_result(Path(args.out), results, corpus_path=args.corpus, source=args.transcript)
    return 0 if ok else 1


def predictions(args) -> int:
    corpus = load_corpus(args.corpus)
    insts = _read_jsonl(Path(args.instances))
    if any("instance_id" not in i for i in insts):
        raise SystemExit("instances 파일에 instance_id가 없습니다. faithbench --out으로 재생성하세요.")
    preds = _read_jsonl(Path(args.predictions))
    pred_map: dict[str, str] = {}
    for p in preds:
        if "instance_id" not in p or "answer" not in p:
            raise SystemExit(f"predictions 행 형식 오류(필수: instance_id, answer): {p!r}")
        pred_map[str(p["instance_id"])] = str(p["answer"])

    missing = [i["instance_id"] for i in insts if i["instance_id"] not in pred_map]
    if missing:
        raise SystemExit(
            f"제출이 불완전합니다. 미응답 인스턴스 {len(missing)}개 "
            f"(예: {missing[:3]}). 모든 instance_id에 답변이 필요합니다."
        )
    extra = [k for k in pred_map if k not in {i["instance_id"] for i in insts}]
    if extra:
        print(f"  [WARN] instances에 없는 예측 {len(extra)}개는 무시합니다.")

    answers = [pred_map[i["instance_id"]] for i in insts]
    agg = _score_rows(insts, answers, corpus)
    results = {args.model: agg}
    print(f"  {args.model}: selection_exact={agg['selection_exact']} leak_rate={agg['leak_rate']} "
          f"n_ans={agg['n_answerable']} n_una={agg['n_unanswerable']}")

    out = Path(args.out) if args.out else None
    if out:
        _write_result(out, results, corpus_path=args.corpus, source=args.instances)
        # transcript(제출 감사용): instance_id + 원문답변 + 정오답
        tpath = out.with_name(out.stem + "-transcript.jsonl")
        with tpath.open("w", encoding="utf-8", newline="\n") as tf:
            for i in insts:
                s = score_answer(_inst_from_row(i), pred_map[i["instance_id"]], corpus)
                tf.write(json.dumps({
                    "model": args.model, "instance_id": i["instance_id"],
                    "gold": i["gold"][0] if i.get("gold") else None,
                    "question": i.get("question"), "context_ids": i.get("context_ids"),
                    "answer": pred_map[i["instance_id"]], **s,
                }, ensure_ascii=False) + "\n")
        print(f"  -> transcript: {tpath}")
    return 0


def _write_result(out: Path, results: dict, corpus_path: str, source: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ans = max((a["n_answerable"] for a in results.values()), default=0)
    n_una = max((a["n_unanswerable"] for a in results.values()), default=0)
    out.write_text(json.dumps({
        "scorer": "score_predictions",
        "corpus_path": corpus_path,
        "source": source,
        "n_answerable": n_ans,
        "n_unanswerable": n_una,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> result: {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="오프라인 결정적 채점(모델 불요)")
    sub = ap.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("rescore", help="기존 transcript 재채점(공표 결과 재현 검증)")
    r.add_argument("--transcript", required=True)
    r.add_argument("--corpus", default="data/processed/laws.json")
    r.add_argument("--expect", help="비교할 기존 result JSON(있으면 aggregate 일치 검증)")
    r.add_argument("--out", help="재도출 result JSON 저장 경로")
    r.set_defaults(func=rescore)

    p = sub.add_parser("predictions", help="제3자 제출 답변 채점")
    p.add_argument("--instances", required=True, help="faithbench --out 인스턴스(instance_id 포함)")
    p.add_argument("--predictions", required=True, help="[{instance_id, answer}] JSONL")
    p.add_argument("--model", default="submission", help="제출 모델명(리포트 표기)")
    p.add_argument("--corpus", default="data/processed/laws.json")
    p.add_argument("--out", help="result JSON 저장 경로(옆에 -transcript.jsonl도 생성)")
    p.set_defaults(func=predictions)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
