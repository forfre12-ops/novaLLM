"""K-FaithBench 프로토타입 — distractor 포함 다조문 근거충실 벤치 (결정적 채점).

기존 G0 파일럿의 한계를 보완한다:
  - 근거를 1조문만 제공 → '복사'만 하면 faithfulness 1.0, 정답조항 선택이 자명.
  - unanswerable 셋이 포화(모두 거절 1.0)라 leak 차원이 변별 안 됨.

이 벤치는 각 인스턴스에 K개 조문(gold + distractor)을 섞어 제공하고, 질문은
**조항ID를 명시하지 않는 내용 기반** → 모델이 여러 근거 중 올바른 조문을 '선택'해
인용해야 정답. distractor에 같은 조(條)의 인접 항을 섞어(--near) 난이도를 높인다.

채점은 결정적(LLM-judge 없음): citation_verify(실존·substring)로 각 인용을 검증 +
cited-ID 집합을 gold 집합과 비교(정밀도/재현율). 두 split:
  · answerable  : gold가 컨텍스트에 포함 → gold만 정확 인용해야 exact 정답.
  · unanswerable: gold 부재(distractor만) → 거절해야 정답, 인용하면 leak.

    python scripts/eval/faithbench.py --demo             # 스코어러 자기검증(모델 불요)
    python scripts/eval/faithbench.py --dump 3           # 인스턴스 3개 미리보기
    python scripts/eval/faithbench.py --questions eval/questions.json --out eval/instances.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from citation_verify import _norm, load_corpus, verify  # noqa: E402

SYS = (
    "너는 제공된 근거 조항만 사용해 답한다. 여러 근거 중 질문에 해당하는 조항을 찾아 "
    "반드시 「원문 인용」[조항ID] 형식으로 인용한다. 해당하는 근거가 없으면 "
    "'제공된 근거에서는 확인할 수 없습니다'라고 답한다."
)

# 거절 의도 결정적 감지 — 인용 존재 여부와 별개로 '답변 거부'를 식별한다.
# 이 패턴에 걸리지 않으면서 인용도 없는 답변은 '무인용 실질답변'(uncited_answer)으로,
# unanswerable에서는 파라메트릭 지식 유출(leak) 후보다.
_REFUSAL_RE = re.compile(
    r"(확인할\s*수\s*없|답변하지\s*않|답할\s*수\s*없|근거가\s*없|해당(하는|되는)?\s*(근거|조항|내용)[이가]?\s*없"
    r"|제공된\s*근거에는|알\s*수\s*없)"
)


def is_refusal(answer: str) -> bool:
    """답변이 명시적 거절인지 결정적으로 판정."""
    return bool(_REFUSAL_RE.search(answer))

# 내용 기반 질문 → gold 조항ID. 질문에 조항ID를 노출하지 않아 '선택'을 강제한다.
# (코퍼스에 실재하는 ID만 런타임에 사용 — build 시 필터링)
QUESTIONS: dict[str, str] = {
    "헌법 제1조 ①": "대한민국의 국가 형태(정체)는 무엇인가?",
    "헌법 제1조 ②": "대한민국의 주권은 누구에게 있는가?",
    "헌법 제3조": "대한민국의 영토 범위는 어떻게 규정되는가?",
    "헌법 제5조 ②": "국군의 사명과 정치적 중립성은 어떻게 규정되는가?",
    "헌법 제8조 ①": "정당 설립의 자유와 복수정당제 보장은 어떻게 규정되는가?",
    "헌법 제9조": "국가의 전통문화 계승·민족문화 창달 노력 의무는 어떻게 규정되는가?",
    "헌법 제10조": "인간으로서의 존엄과 가치, 행복추구권은 어떻게 규정되는가?",
    "헌법 제11조 ①": "법 앞의 평등과 차별금지 원칙은 어떻게 규정되는가?",
    "헌법 제12조 ①": "신체의 자유와 적법절차 원칙은 어떻게 규정되는가?",
    "헌법 제14조": "거주·이전의 자유는 어떻게 규정되는가?",
    "헌법 제15조": "직업선택의 자유는 어떻게 규정되는가?",
    "헌법 제17조": "사생활의 비밀과 자유는 어떻게 규정되는가?",
    "헌법 제19조": "양심의 자유는 어떻게 규정되는가?",
    "헌법 제21조 ①": "언론·출판의 자유와 집회·결사의 자유는 어떻게 규정되는가?",
    "헌법 제22조 ①": "학문과 예술의 자유는 어떻게 규정되는가?",
    "헌법 제23조 ①": "재산권 보장과 그 내용·한계는 어떻게 규정되는가?",
    "헌법 제24조": "선거권은 어떻게 규정되는가?",
    "헌법 제27조 ④": "형사피고인의 무죄추정 원칙은 어떻게 규정되는가?",
    "헌법 제31조 ①": "능력에 따라 균등하게 교육을 받을 권리는 어떻게 규정되는가?",
    "헌법 제31조 ③": "의무교육의 무상 원칙은 어떻게 규정되는가?",
    "헌법 제32조 ①": "근로의 권리와 최저임금제는 어떻게 규정되는가?",
    "헌법 제33조 ①": "근로자의 단결권·단체교섭권·단체행동권은 어떻게 규정되는가?",
    "헌법 제34조 ①": "인간다운 생활을 할 권리는 어떻게 규정되는가?",
    "헌법 제35조 ①": "건강하고 쾌적한 환경에서 생활할 권리(환경권)는 어떻게 규정되는가?",
    "헌법 제36조 ①": "혼인과 가족생활의 존엄·양성평등 기초는 어떻게 규정되는가?",
    "헌법 제38조": "납세의 의무는 어떻게 규정되는가?",
    "헌법 제39조 ①": "국방의 의무는 어떻게 규정되는가?",
}

# 코퍼스(제1~39조) 밖 주제 — unanswerable split용(어떤 조문으로도 답할 수 없음)
UNANSWERABLE = [
    "대통령의 임기는 몇 년인가?",
    "국회의원의 정수는 최소 몇 명인가?",
    "헌법개정은 어떤 절차로 이루어지는가?",
    "지방자치단체의 종류는 무엇으로 정하는가?",
    "대한민국 경제질서의 기본은 무엇인가?",
    "감사원의 직무는 무엇인가?",
    "대법원의 구성은 어떻게 되는가?",
    "헌법재판소가 관장하는 사항은 무엇인가?",
]


def load_questions(path: str | None) -> dict[str, str]:
    """외부 질문셋 로드. JSON object 또는 [{"id","question"}] 모두 허용."""
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    if isinstance(data, list):
        questions: dict[str, str] = {}
        for row in data:
            if not isinstance(row, dict) or "id" not in row or "question" not in row:
                raise SystemExit(f"질문셋 항목 형식 오류: {row!r}")
            questions[str(row["id"])] = str(row["question"])
        return questions
    raise SystemExit("--questions 는 JSON object 또는 list 여야 합니다.")


def load_unanswerable(path: str | None) -> list[str] | None:
    """외부 unanswerable 질문셋 로드. JSON list 또는 {"questions": [...]} 허용."""
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("questions")
    if not isinstance(data, list) or not all(isinstance(q, str) for q in data):
        raise SystemExit("--unanswerable-file 은 문자열 배열이어야 합니다.")
    return data


def _article_num(cid: str) -> str:
    """'헌법 제31조 ①' → '제31조' (같은 조의 인접 항 판별용)."""
    m = re.search(r"(제\d+조)", cid)
    return m.group(1) if m else cid


def _pick_distractors(
    corpus: dict[str, str],
    gold: str,
    k: int,
    near: bool,
    rng: random.Random,
) -> list[str]:
    others = [c for c in corpus if c != gold]
    if near:
        g = _article_num(gold)
        siblings = [c for c in others if _article_num(c) == g]
        rest = [c for c in others if _article_num(c) != g]
        rng.shuffle(siblings)
        rng.shuffle(rest)
        pool = siblings + rest  # 같은 조의 인접 항을 우선(하드 distractor)
    else:
        pool = others[:]
        rng.shuffle(pool)
    return pool[:k]


def _context_block(items: list[tuple[str, str]]) -> str:
    lines = [f"{i + 1}) {cid}: {text}" for i, (cid, text) in enumerate(items)]
    return "[근거]\n" + "\n".join(lines)


def build_instances(
    corpus: dict[str, str],
    k: int,
    near: bool,
    seed: int,
    questions: dict[str, str] | None = None,
    unanswerable: list[str] | None = None,
    include_all_corpus: bool = False,
) -> list[dict]:
    """answerable(내용질문+distractor) + unanswerable(distractor만) 인스턴스 생성."""
    rng = random.Random(seed)
    insts: list[dict] = []
    question_bank = dict(QUESTIONS)
    if questions:
        question_bank.update(questions)
    if include_all_corpus:
        for cid in corpus:
            question_bank.setdefault(cid, f"{cid}의 핵심 내용을 원문으로 인용해줘.")

    for gold_id, q in question_bank.items():
        if gold_id not in corpus:
            continue
        distractors = _pick_distractors(corpus, gold_id, k - 1, near, rng)
        ctx = [(gold_id, corpus[gold_id])] + [(d, corpus[d]) for d in distractors]
        rng.shuffle(ctx)
        question_source = (
            "curated"
            if gold_id in QUESTIONS or (questions and gold_id in questions)
            else "id_fallback"
        )
        insts.append({
            "split": "answerable",
            "gold": [gold_id],
            "question": q,
            "question_source": question_source,
            "context_ids": [c for c, _ in ctx],
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": f"{_context_block(ctx)}\n\n질문: {q}"},
            ],
        })
    all_ids = list(corpus.keys())
    for q in (unanswerable if unanswerable is not None else UNANSWERABLE):
        ctx_ids = rng.sample(all_ids, min(k, len(all_ids)))
        ctx = [(c, corpus[c]) for c in ctx_ids]
        insts.append({
            "split": "unanswerable",
            "gold": [],
            "question": q,
            "context_ids": ctx_ids,
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": f"{_context_block(ctx)}\n\n질문: {q}"},
            ],
        })
    return insts


def write_jsonl(path: str, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_answer(inst: dict, answer: str, corpus: dict[str, str]) -> dict:
    """한 인스턴스의 모델 답변을 결정적으로 채점.

    실패모드를 분해해 '형식/거절 오류'와 '선택 오류'를 구분한다:
      · no_citation   — 인용 형식이 아예 없음(형식 미준수 또는 거절)
      · refused       — 명시적 거절 문자열(is_refusal)
      · uncited_answer— 거절도 인용도 아닌 실질 답변(파라메트릭 유출 후보)
    """
    rep = verify(answer, corpus)
    cited = {_norm(c["cited_id"]) for c in rep["citations"]}
    supported = {_norm(c["cited_id"]) for c in rep["citations"] if c["supported"]}
    gold = {_norm(g) for g in inst["gold"]}
    refusal = is_refusal(answer)
    no_cite = rep["n_citations"] == 0

    out = {
        "split": inst["split"],
        "n_citations": rep["n_citations"],
        "faithfulness": rep["faithfulness"] if rep["n_citations"] else 0.0,
        "refused": int(refusal),
        "no_citation": int(no_cite),
        "uncited_answer": int(no_cite and not refusal),  # 무인용 실질답변
    }
    if inst["split"] == "answerable":
        faithful = rep["n_citations"] > 0 and rep["faithfulness"] == 1.0
        out["gold_recall"] = int(bool(gold & supported))        # gold를 지지된 형태로 인용
        out["distractor_cited"] = int(bool(cited - gold))       # gold 아닌 것을 인용(오답/환각)
        out["faithful"] = int(faithful)
        out["exact"] = int(faithful and cited == gold)          # gold만, 전부 지지
    else:
        # leak 재정의: '인용 유출' + '무인용 실질답변(파라메트릭 인출)' 둘 다 유출로 본다.
        # (구 정의는 인용이 없으면 무조건 거절로 집계 → 암기 기반 무인용 답변을 놓쳤다.)
        out["leaked_citation"] = int(rep["n_citations"] > 0)    # 근거 없는데 인용 → leak
        out["leaked"] = int(rep["n_citations"] > 0 or out["uncited_answer"])
        out["clean_refusal"] = int(refusal and no_cite)          # 진짜 거절만
    return out


def aggregate(scored: list[dict]) -> dict:
    ans = [s for s in scored if s["split"] == "answerable"]
    una = [s for s in scored if s["split"] == "unanswerable"]

    def mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    report = {
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
        "faithfulness_mean": mean([s["faithfulness"] for s in ans]),
        "gold_recall": mean([s["gold_recall"] for s in ans]),
        "distractor_cite_rate": mean([s["distractor_cited"] for s in ans]),
        "selection_exact": mean([s["exact"] for s in ans]),   # 핵심 지표: gold만 정확 인용
        # answerable 실패모드 분해 — '형식/거절 오류'와 '선택 오류'를 분리
        "answerable_no_citation_rate": mean([s["no_citation"] for s in ans]),
        "answerable_refused_rate": mean([s["refused"] for s in ans]),
        # unanswerable — 진짜 거절 vs 유출(인용유출 + 무인용 실질답변)
        "refusal_rate": mean([s["clean_refusal"] for s in una]),
        "leak_rate": mean([s["leaked"] for s in una]),
        "leak_citation_rate": mean([s["leaked_citation"] for s in una]),
        "leak_uncited_rate": mean([s["uncited_answer"] for s in una]),
    }
    return report


# ── --demo: 모델 없이 스코어러 자기검증 ──
def _run_demo(corpus: dict[str, str]) -> int:
    gold_id, gold_txt = "헌법 제1조 ①", corpus["헌법 제1조 ①"]
    dist_id, dist_txt = "헌법 제3조", corpus["헌법 제3조"]
    ans_inst = {"split": "answerable", "gold": [gold_id], "context_ids": [gold_id, dist_id]}
    una_inst = {"split": "unanswerable", "gold": [], "context_ids": [dist_id, gold_id]}

    cases = [
        ("perfect(gold만 충실)", ans_inst,
         f"헌법은 「{gold_txt}」[{gold_id}]라고 규정한다.",
         lambda s: s["exact"] == 1 and s["gold_recall"] == 1 and s["distractor_cited"] == 0),
        ("distractor 인용(오답 선택)", ans_inst,
         f"헌법은 「{dist_txt}」[{dist_id}]라고 규정한다.",
         lambda s: s["exact"] == 0 and s["gold_recall"] == 0 and s["distractor_cited"] == 1),
        ("gold+distractor 혼합", ans_inst,
         f"「{gold_txt}」[{gold_id}] 그리고 「{dist_txt}」[{dist_id}].",
         lambda s: s["exact"] == 0 and s["gold_recall"] == 1 and s["distractor_cited"] == 1),
        ("환각 ID", ans_inst,
         "헌법은 「대한민국은 사회주의 국가이다」[헌법 제99조]라고 규정한다.",
         lambda s: s["faithfulness"] < 1.0 and s["exact"] == 0),
        ("unanswerable 거절", una_inst,
         "제공된 근거에서는 확인할 수 없습니다.",
         lambda s: s["refused"] == 1 and s["leaked"] == 0),
        ("unanswerable leak(distractor 인용)", una_inst,
         f"헌법은 「{dist_txt}」[{dist_id}]라고 규정한다.",
         lambda s: s["refused"] == 0 and s["leaked"] == 1 and s["leaked_citation"] == 1),
        ("unanswerable leak(무인용 실질답변=암기 인출)", una_inst,
         "대통령의 임기는 5년이며 중임할 수 없습니다.",
         lambda s: s["leaked"] == 1 and s["uncited_answer"] == 1 and s["clean_refusal"] == 0),
    ]
    all_ok = True
    for name, inst, ans, check in cases:
        s = score_answer(inst, ans, corpus)
        ok = check(s)
        all_ok = all_ok and ok
        metrics = json.dumps({k: v for k, v in s.items() if k != "split"}, ensure_ascii=False)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {metrics}")
    print(f"\n스코어러 자기검증: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/seed/constitution.json")
    ap.add_argument("--k", type=int, default=5, help="컨텍스트 조문 수(gold 1 + distractor k-1)")
    ap.add_argument("--near", action="store_true", help="같은 조의 인접 항을 하드 distractor로 우선")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--questions", help="추가/대체 질문셋 JSON(object 또는 list[{id,question}])")
    ap.add_argument("--unanswerable-file", help="추가 unanswerable 질문 JSON(list 또는 {questions})")
    ap.add_argument(
        "--include-all-corpus",
        action="store_true",
        help="질문 없는 모든 코퍼스 항목도 ID 기반 sanity 질문으로 포함(정식 선택평가용 아님)",
    )
    ap.add_argument("--out", help="생성된 벤치 인스턴스를 JSONL로 저장")
    ap.add_argument("--demo", action="store_true", help="모델 없이 스코어러 자기검증")
    ap.add_argument("--dump", type=int, default=0, help="인스턴스 N개 미리보기")
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    print(f"corpus {len(corpus)}조문 (closed-set)")

    if args.demo:
        return _run_demo(corpus)

    questions = load_questions(args.questions)
    unanswerable = load_unanswerable(args.unanswerable_file)
    insts = build_instances(
        corpus,
        args.k,
        args.near,
        args.seed,
        questions=questions,
        unanswerable=unanswerable,
        include_all_corpus=args.include_all_corpus,
    )
    n_ans = sum(1 for i in insts if i["split"] == "answerable")
    n_una = sum(1 for i in insts if i["split"] == "unanswerable")
    print(
        f"인스턴스: answerable {n_ans} + unanswerable {n_una} = {len(insts)} "
        f"(k={args.k}, near={args.near})"
    )
    if args.out:
        write_jsonl(args.out, insts)
        print(f"저장: {args.out}")

    for inst in insts[: args.dump]:
        print("\n" + "=" * 60)
        print(f"[{inst['split']}] gold={inst['gold']}")
        print(inst["messages"][1]["content"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
