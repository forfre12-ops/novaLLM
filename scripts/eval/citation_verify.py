"""폐쇄집합(closed-set) 인용 검증 — 한국어 grounding faithfulness/leak 측정 프리미티브.

전략의 핵심 자산("측정을 소유하라"). LLM-judge 없이 **기계적으로 참/거짓** 판정하는
'객관 채점 우선지대'. 답변 내 인용은 다음 형식:

    「인용문」[법령ID]     예)  「대한민국은 민주공화국이다」[헌법 제1조 ①]

각 인용에 대해 결정적으로 검증:
  (1) 법령ID가 closed-set 코퍼스에 존재하는가 → 없으면 hallucinated_citation (환각/leak)
  (2) 인용문이 해당 조문의 exact substring(공백 정규화) 인가 → 아니면 misquote (오귀속)

faithfulness_score = 지지된 인용 수 / 전체 인용 수.
closed-set 이므로 false-negative는 코퍼스 범위 내에서 원천 0(정직: 코퍼스 밖은 판정 불가로 분리).

사용:
    python scripts/eval/citation_verify.py --demo
    python scripts/eval/citation_verify.py --corpus data/seed/constitution.json --answer-file <ans.txt>
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

CITE_RE = re.compile(r"「(.+?)」\s*\[(.+?)\]")


def _norm(s: str) -> str:
    """공백/유니코드 정규화 — exact-match 비교용."""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_corpus(path: str) -> dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    arts = data.get("articles", data)
    return {_norm(k): _norm(v) for k, v in arts.items()}


def verify(answer: str, corpus: dict[str, str]) -> dict:
    """답변의 모든 인용을 검증하고 판정 리포트를 반환."""
    cites = CITE_RE.findall(answer)
    results = []
    for quote, cid in cites:
        q, c = _norm(quote), _norm(cid)
        if c not in corpus:
            verdict = "hallucinated_citation"  # 존재하지 않는 법령ID
            supported = False
        elif q in corpus[c]:
            verdict = "supported"  # 조문의 exact substring
            supported = True
        else:
            verdict = "misquote"  # ID는 있으나 인용문이 조문에 없음
            supported = False
        results.append({"quote": quote, "cited_id": cid, "verdict": verdict, "supported": supported})

    total = len(results)
    n_sup = sum(1 for r in results if r["supported"])
    return {
        "n_citations": total,
        "n_supported": n_sup,
        "n_hallucinated": sum(1 for r in results if r["verdict"] == "hallucinated_citation"),
        "n_misquote": sum(1 for r in results if r["verdict"] == "misquote"),
        "faithfulness": round(n_sup / total, 4) if total else None,
        "leak_free": total > 0 and n_sup == total,  # 전 인용이 지지됨
        "citations": results,
    }


DEMO_GOOD = (
    "질문: 대한민국의 국가 형태와 주권의 소재는?\n"
    "답변: 우리 헌법은 「대한민국은 민주공화국이다」[헌법 제1조 ①]라고 규정하며, "
    "나아가 「대한민국의 주권은 국민에게 있고, 모든 권력은 국민으로부터 나온다」[헌법 제1조 ②]고 명시한다."
)
DEMO_BAD = (
    "질문: 대한민국의 국가 형태와 국방의 의무는?\n"
    "답변: 헌법은 「대한민국은 민주공화국이다」[헌법 제1조 ①]라고 하며, "
    "「모든 국민은 국방의 의무를 진다」[헌법 제5조 ②]고 규정하고, "  # 존재하지 않는 조문 → 환각인용
    "또한 「대한민국은 자유민주주의 국가이다」[헌법 제1조 ①]라고 선언한다."  # 조문에 없는 문구 → misquote
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/seed/constitution.json")
    ap.add_argument("--answer-file")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    print(f"corpus: {len(corpus)}개 조문 (closed-set)")

    if args.demo:
        for name, ans in [("GOOD(전부 지지)", DEMO_GOOD), ("BAD(환각+오귀속)", DEMO_BAD)]:
            rep = verify(ans, corpus)
            print(f"\n===== {name} =====")
            print(f"  faithfulness={rep['faithfulness']}  leak_free={rep['leak_free']}  "
                  f"(supported {rep['n_supported']}/{rep['n_citations']}, "
                  f"환각 {rep['n_hallucinated']}, 오귀속 {rep['n_misquote']})")
            for c in rep["citations"]:
                mark = "✓" if c["supported"] else "✗"
                print(f"    {mark} [{c['cited_id']}] {c['verdict']}: 「{c['quote'][:30]}…」")
        # 자기검증: GOOD은 1.0, BAD는 <1.0 이어야 함
        good = verify(DEMO_GOOD, corpus)
        bad = verify(DEMO_BAD, corpus)
        ok = good["faithfulness"] == 1.0 and good["leak_free"] and (bad["faithfulness"] < 1.0) and (not bad["leak_free"])
        print(f"\n측정 프리미티브 자기검증: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    if args.answer_file:
        ans = Path(args.answer_file).read_text(encoding="utf-8")
        print(json.dumps(verify(ans, corpus), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
