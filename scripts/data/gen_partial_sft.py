"""부분-span SFT 데이터 생성 — precision(질문 해당 부분만 인용)을 가르친다.

g0-pilot 어댑터의 결함: gold 답변이 조문 전체를 「」로 감싸 '통째 복사'를 학습 →
부분-span 평가에서 과잉 인용(precision 낮음). 이 생성기는 gold를 **조문 전체가 아니라
문장 단위 tight span**으로 만들어 그 결함을 고친다.

방식(결정적, teacher 불요):
  - 각 조문을 문장('...다.') 단위로 분리.
  - 다문장 조문: 각 문장 Si마다 '{앞부분 힌트}로 시작하는 규정을 정확히 인용' 질문 +
    gold=「Si」[ID] (tight). → 조문 전체가 아니라 해당 문장만 인용하는 법을 학습.
  - 단문장 조문: 일반 질문 + gold=「문장」[ID].
  - refusal: 코퍼스 밖 질문 → 거절.
  - **평가셋 조항(eval/questions.partial.constitution.json)은 held-out으로 제외.**
  - 생성 즉시 citation_verify로 self-consistency 검증(오염 0).

    python scripts/data/gen_partial_sft.py --out data/processed/partial_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.citation_verify import _norm, load_corpus, verify  # noqa: E402

SYS = (
    "너는 제공된 근거 조항만 사용해 답한다. 여러 근거 중 질문에 해당하는 조항을 찾아, "
    "그 조항에서 **질문에 해당하는 부분만** 「원문 인용」[조항ID] 형식으로 인용한다. "
    "조문 전체를 통째로 인용하지 말고 질문과 관련된 부분만 인용하라. "
    "해당하는 근거가 없으면 '제공된 근거에서는 확인할 수 없습니다'라고 답한다."
)

NEGATIVES = [
    "대통령의 임기는 몇 년인가?",
    "국회의원의 정수는 몇 명 이상인가?",
    "헌법개정은 어떤 절차로 이루어지는가?",
    "지방자치단체의 종류는 무엇으로 정하는가?",
    "대한민국의 경제질서는 무엇을 기본으로 하는가?",
]
REFUSAL = "제공된 근거 조문에서는 해당 내용을 확인할 수 없습니다. 근거가 없는 사항은 답변하지 않습니다."


def split_spans(text: str, min_len: int = 8) -> list[str]:
    """조문을 절(clause) 단위로 분리 — 문장 종결('...다.') + 쉼표 절 경계.

    각 조각은 조문의 exact substring(쉼표는 제거되나 앞 조각은 여전히 substring). tight
    부분인용 학습용 세밀도(eval의 clause-level과 정합). 너무 짧은 조각(<min_len)은 제외.
    """
    parts = re.split(r"(?<=다\.)\s+|,\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= min_len]


def user_msg(cid: str, article: str, question: str) -> dict:
    return {"role": "user", "content": f"[근거]\n{cid}: {article}\n\n질문: {question}"}


def make_example(cid: str, article: str, question: str, gold_span: str) -> dict:
    answer = f"헌법은 「{gold_span}」[{cid}]라고 규정하고 있습니다."
    return {
        "messages": [
            {"role": "system", "content": SYS},
            user_msg(cid, article, question),
            {"role": "assistant", "content": answer},
        ],
        "label": "partial",
        "gold_citations": [cid],
        "gold_span": gold_span,
    }


def make_refusal(article_cid: str, article: str, question: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYS},
            user_msg(article_cid, article, question),
            {"role": "assistant", "content": REFUSAL},
        ],
        "label": "refusal",
        "gold_citations": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/seed/constitution.json")
    ap.add_argument("--eval-items", default="eval/questions.partial.constitution.json")
    ap.add_argument("--out", default="data/processed/partial_sft.jsonl")
    ap.add_argument("--hint-chars", type=int, default=8)
    args = ap.parse_args()

    corpus_raw = json.loads(Path(args.corpus).read_text(encoding="utf-8"))["articles"]
    corpus = load_corpus(args.corpus)
    # 평가셋 조항 held-out 제외
    eval_ids = {_norm(it["id"]) for it in json.loads(Path(args.eval_items).read_text(encoding="utf-8"))}
    print(f"held-out(평가셋) 조항 {len(eval_ids)}개 제외")

    rows: list[dict] = []
    multi = 0
    import random

    rng = random.Random(3407)
    for cid, article in corpus_raw.items():
        if _norm(cid) in eval_ids:
            continue
        spans = split_spans(_norm(article))
        if len(spans) <= 1:
            gold = spans[0] if spans else _norm(article)
            rows.append(make_example(cid, article, "이 조항이 규정하는 내용을 원문 그대로 인용해줘.", gold))
        else:
            multi += 1
            for si in spans:
                hint = si[: args.hint_chars]
                q = f"이 조항에서 '{hint}'로 시작하는 규정을 원문 그대로 정확히 인용해줘."
                rows.append(make_example(cid, article, q, si))
        # 조문마다 refusal 1개(코퍼스 밖 질문)
        rows.append(make_refusal(cid, article, rng.choice(NEGATIVES)))

    # self-consistency: 모든 partial gold가 citation_verify 지지 + tight(조문보다 짧음) 확인
    bad = 0
    loose = 0
    for r in rows:
        if r["label"] != "partial":
            continue
        ans = r["messages"][-1]["content"]
        rep = verify(ans, corpus)
        if not rep["leak_free"]:
            bad += 1
            print(f"  ✗ 오염: {r['gold_citations']} f={rep['faithfulness']}")
        cid = _norm(r["gold_citations"][0])
        if _norm(r["gold_span"]) == corpus[cid]:
            loose += 1  # gold_span == 조문 전체(단문장) — tight 아님(정보용)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_partial = sum(1 for r in rows if r["label"] == "partial")
    n_ref = sum(1 for r in rows if r["label"] == "refusal")
    print(f"생성: partial {n_partial}(다문장 조문 {multi}) + refusal {n_ref} = {len(rows)}건 → {outp}")
    print(f"  tight span(조문보다 짧음): {n_partial - loose}/{n_partial} | 단문장(=전체): {loose}")
    print(f"self-consistency(citation_verify): {'PASS (오염 0)' if bad == 0 else f'FAIL ({bad}건)'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
