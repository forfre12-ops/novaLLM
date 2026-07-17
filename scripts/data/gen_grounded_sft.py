"""TA-09 근거충실 SFT 데이터 생성기 (closed-set 코퍼스 기반).

전략 핵심: 인용 형식 「인용문」[법령ID]을 각인시키는 학습 데이터를 만든다.
  - positive: 코퍼스 조문을 verbatim 인용 + 정확한 [ID] → citation_verify 100% 통과 보장
  - negative(거절): 코퍼스 밖 질문 → 근거 없음 거절(인용 없음) → 환각/과답 억제 학습

생성 즉시 citation_verify로 self-consistency 검증(생성기가 스스로 오염 데이터를 못 내게).
기본은 결정적(deterministic) 생성 — teacher 모델 없이 신뢰가능. --use-model 로 로컬 모델
질문 다양화(옵션)만 추가(답변/인용은 항상 결정적으로 코퍼스에서 생성 → 오염 불가).

    python scripts/data/gen_grounded_sft.py --corpus data/seed/constitution.json \
        --out data/processed/grounded_sft_demo.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.citation_verify import verify, load_corpus  # noqa: E402

# 각 조문에 대한 자연스러운 질문(데모용 hand-crafted). 확장은 --use-model.
QUESTIONS = {
    "헌법 제1조 ①": "대한민국의 국가 형태는 무엇인가?",
    "헌법 제1조 ②": "대한민국에서 주권은 누구에게 있는가?",
    "헌법 제10조": "헌법이 보장하는 인간의 존엄과 행복추구권은 어떻게 규정되어 있는가?",
    "헌법 제11조 ①": "법 앞의 평등과 차별 금지는 헌법에 어떻게 명시되어 있는가?",
    "헌법 제31조 ①": "교육을 받을 권리는 헌법에 어떻게 규정되어 있는가?",
    "헌법 제34조 ①": "인간다운 생활을 할 권리는 헌법 어디에 규정되어 있는가?",
}

# 코퍼스 밖 질문(거절 학습용)
NEGATIVES = [
    "대한민국의 수도는 어디인가?",
    "국회의원의 임기는 몇 년인가?",
    "대통령의 임기는 몇 년인가?",
    "형사피고인은 어떤 권리를 가지는가?",
]
REFUSAL = "제공된 근거 조문에서는 해당 내용을 확인할 수 없습니다. 근거가 없는 사항은 답변하지 않습니다."


def make_positive(cid: str, text: str, question: str) -> dict:
    answer = f"헌법은 「{text}」[{cid}]라고 규정하고 있습니다."
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "label": "grounded",
        "gold_citations": [cid],
    }


def make_negative(question: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": REFUSAL},
        ],
        "label": "refusal",
        "gold_citations": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/seed/constitution.json")
    ap.add_argument("--out", default="data/processed/grounded_sft_demo.jsonl")
    args = ap.parse_args()

    corpus_raw = json.loads(Path(args.corpus).read_text(encoding="utf-8"))["articles"]
    corpus = load_corpus(args.corpus)

    rows = []
    for cid, text in corpus_raw.items():
        q = QUESTIONS.get(cid, f"{cid}의 내용은 무엇인가?")
        rows.append(make_positive(cid, text, q))
    for q in NEGATIVES:
        rows.append(make_negative(q))

    # self-consistency 검증: 모든 grounded 답변이 citation_verify 100% 통과해야 함
    bad = 0
    for r in rows:
        if r["label"] != "grounded":
            continue
        ans = r["messages"][-1]["content"]
        rep = verify(ans, corpus)
        if not rep["leak_free"]:
            bad += 1
            print(f"  ✗ 오염 생성 감지: {r['gold_citations']} faithfulness={rep['faithfulness']}")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pos = sum(1 for r in rows if r["label"] == "grounded")
    n_neg = sum(1 for r in rows if r["label"] == "refusal")
    print(f"생성: grounded {n_pos} + refusal {n_neg} = {len(rows)}건 → {outp}")
    print(f"self-consistency(citation_verify): {'PASS (오염 0)' if bad == 0 else f'FAIL ({bad}건 오염)'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
