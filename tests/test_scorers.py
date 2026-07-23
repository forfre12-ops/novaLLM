"""결정적 스코어러·도구 단위 테스트 (pytest 불요 — 스크립트로도 실행).

    python tests/test_scorers.py     # 전부 실행, 실패 시 exit 1
    pytest tests/test_scorers.py     # pytest가 있으면 이것도 동작

smoke.py가 스크립트 모드로 호출한다. 네트워크·GPU·API 키 불요(constitution.json은 tracked).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from citation_verify import load_corpus, verify  # noqa: E402
from faithbench import (  # noqa: E402
    _parametric_verbatim,
    aggregate,
    build_instances,
    instance_id,
    is_refusal,
    score_answer,
)
from faithbench_partial import aggregate_partial, score_partial  # noqa: E402
from faithbench_stats import abstention_operating_point, mcnemar_exact, wilson  # noqa: E402
from score_predictions import _inst_from_row, _score_rows  # noqa: E402

CORPUS = load_corpus(str(ROOT / "data/seed/constitution.json"))
GID, DIST, OOB = "헌법 제1조 ①", "헌법 제3조", "헌법 제9조"


def _ans(gold, ctx):
    return {"split": "answerable", "gold": [gold], "context_ids": ctx}


def _una(ctx):
    return {"split": "unanswerable", "gold": [], "context_ids": ctx}


# ── citation_verify ──
def test_verify_supported():
    rep = verify(f"「{CORPUS[GID]}」[{GID}]", CORPUS)
    assert rep["n_citations"] == 1 and rep["citations"][0]["supported"]


def test_verify_hallucinated_id():
    rep = verify("「대한민국은 사회주의 국가이다」[헌법 제99조]", CORPUS)
    assert rep["faithfulness"] < 1.0


def test_verify_misquote_not_substring():
    rep = verify(f"「존재하지 않는 문장이다」[{GID}]", CORPUS)
    assert rep["citations"][0]["supported"] == 0


# ── faithbench score_answer (실패모드) ──
def test_answer_perfect():
    s = score_answer(_ans(GID, [GID, DIST]), f"「{CORPUS[GID]}」[{GID}]", CORPUS)
    assert s["exact"] == 1 and s["gold_recall"] == 1 and s["distractor_cited"] == 0


def test_answer_distractor():
    s = score_answer(_ans(GID, [GID, DIST]), f"「{CORPUS[DIST]}」[{DIST}]", CORPUS)
    assert s["exact"] == 0 and s["distractor_cited"] == 1


def test_answer_refusal():
    s = score_answer(_una([DIST, GID]), "제공된 근거에서는 확인할 수 없습니다.", CORPUS)
    assert s["clean_refusal"] == 1 and s["leaked"] == 0


def test_leak_citation():
    s = score_answer(_una([DIST, GID]), f"「{CORPUS[DIST]}」[{DIST}]", CORPUS)
    assert s["leaked"] == 1 and s["leaked_citation"] == 1


def test_leak_ungrounded():
    s = score_answer(_una([DIST, GID]), "대통령의 임기는 5년이며 중임할 수 없습니다.", CORPUS)
    assert s["leaked_ungrounded"] == 1 and s["leaked_parametric"] == 0


def test_leak_parametric():
    # 문맥 밖 조문(제9조)의 원문을 무인용으로 재현 → parametric
    s = score_answer(_una([DIST, GID]), CORPUS[OOB], CORPUS)
    assert s["leaked_parametric"] == 1 and s["leaked_ungrounded"] == 0


def test_parametric_helper_excludes_in_context():
    # 문맥 안 조문(30자↑)을 그대로 옮기면 parametric 아님(제공근거 이동).
    # 12조③은 문맥 밖 16조와 영장 clause span을 공유 → span단위 제외가 없으면 오분류(버그).
    a12 = next(x for x in CORPUS if "제12조" in x and "③" in x)
    assert _parametric_verbatim(CORPUS[a12], CORPUS, [a12, GID]) is False
    # 문맥 밖 조문 원문 재현은 parametric
    assert _parametric_verbatim(CORPUS[OOB], CORPUS, [DIST, GID]) is True


def test_leak_incontext_copy_not_parametric():
    # 회귀: 문맥 내 조문을 무인용 복사 → score_answer가 ungrounded로(공유 span에도 불구).
    a12 = next(x for x in CORPUS if "제12조" in x and "③" in x)
    s = score_answer(_una([a12, GID]), CORPUS[a12], CORPUS)
    assert s["leaked_ungrounded"] == 1 and s["leaked_parametric"] == 0


def test_is_refusal():
    assert is_refusal("제공된 근거에서는 확인할 수 없습니다.")
    assert not is_refusal(f"「{CORPUS[GID]}」[{GID}]")


# ── build_instances: instance_id / gold-ablation ──
def test_instance_id_deterministic():
    assert instance_id("answerable", [GID], "q?", ["a", "b"]) == \
        instance_id("answerable", [GID], "q?", ["a", "b"])
    assert instance_id("answerable", [GID], "q?", ["a", "b"]) != \
        instance_id("answerable", [GID], "q?", ["b", "a"])


def test_build_deterministic_and_unique():
    a = build_instances(CORPUS, 5, True, 3407)
    b = build_instances(CORPUS, 5, True, 3407)
    ids_a = [i["instance_id"] for i in a]
    assert ids_a == [i["instance_id"] for i in b]  # 결정적
    assert len(set(ids_a)) == len(ids_a)  # 유일


def test_gold_ablation_gold_absent_and_invariant():
    base = build_instances(CORPUS, 5, True, 3407)
    abl = build_instances(CORPUS, 5, True, 3407, gold_ablation=True)
    # 기존 인스턴스 불변(앞부분 동일)
    assert [i["instance_id"] for i in base] == [i["instance_id"] for i in abl[: len(base)]]
    ga = [i for i in abl if i.get("probe_type") == "gold_ablation"]
    n_ans = sum(1 for i in abl if i["split"] == "answerable")
    assert len(ga) == n_ans
    assert all(i["source_gold"] not in i["context_ids"] for i in ga)  # gold 절대 부재


# ── faithbench_partial ──
def test_partial_exact_and_wholecopy():
    gid = "헌법 제10조"
    gold_span = "모든 국민은 인간으로서의 존엄과 가치를 가지며, 행복을 추구할 권리를 가진다."
    inst = {"split": "partial", "gold": [gid], "gold_span": gold_span}
    s_exact = score_partial(inst, f"「{gold_span}」[{gid}]", CORPUS)
    s_whole = score_partial(inst, f"「{CORPUS[gid]}」[{gid}]", CORPUS)
    assert s_exact["partial_exact"] == 1 and s_exact["span_f1"] == 1.0
    assert s_whole["span_recall"] == 1.0 and s_whole["span_precision"] < 1.0 and s_whole["span_ok"] == 0


def test_aggregate_partial_shape():
    agg = aggregate_partial([{"selected_gold": 1, "span_precision": 1.0, "span_recall": 1.0,
                              "span_f1": 1.0, "span_ok": 1, "partial_exact": 1}])
    assert agg["n"] == 1 and agg["partial_exact"] == 1.0


# ── score_predictions core ──
def test_score_predictions_rescore_core():
    rows = [
        {"model": "m", "split": "answerable", "gold": GID, "answer": f"「{CORPUS[GID]}」[{GID}]"},
        {"model": "m", "split": "unanswerable", "gold": None, "answer": "제공된 근거에서는 확인할 수 없습니다."},
    ]
    agg = _score_rows(rows, [r["answer"] for r in rows], CORPUS)
    assert agg["selection_exact"] == 1.0 and agg["refusal_rate"] == 1.0


def test_inst_from_row_gold_list():
    assert _inst_from_row({"split": "answerable", "gold": GID})["gold"] == [GID]
    assert _inst_from_row({"split": "unanswerable", "gold": None})["gold"] == []


# ── faithbench_stats ──
def test_abstention_youden():
    res = {"results": {
        "refuse_all": {"refusal_rate": 1.0, "answerable_refused_rate": 1.0, "leak_rate": 0.0},
        "ideal": {"refusal_rate": 1.0, "answerable_refused_rate": 0.0, "leak_rate": 0.0},
    }}
    a = abstention_operating_point(res)
    assert a["refuse_all"]["youden_j"] == 0.0 and a["ideal"]["youden_j"] == 1.0


def test_load_transcript_two_axes():
    from faithbench_stats import load_transcript
    tp = ROOT / "docs/env-verify/g0-faithbench-v02-result-transcript.jsonl"
    sel = load_transcript(tp, "answerable", "exact")
    leak = load_transcript(tp, "unanswerable", "leaked")
    assert len(sel) == 3 and len(leak) == 3  # 3 models 각 축
    assert all(v in (0, 1) for m in leak.values() for v in m.values())


def test_mcnemar_and_wilson():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) < 0.01  # 한쪽으로 완전 쏠리면 유의
    p, lo, hi = wilson(5, 10)
    assert p == 0.5 and lo < 0.5 < hi


def _run() -> int:
    tests = {k: v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)}
    failed = []
    for name, fn in tests.items():
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as e:  # noqa: BLE001
            failed.append((name, repr(e)))
            print(f"  [FAIL] {name}: {e!r}")
    print(f"\ntest_scorers: {len(tests) - len(failed)}/{len(tests)} PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
