"""부분-span SFT 데이터 생성 v2 — precision(요구된 만큼만 인용)을 가르친다.

v1의 확인된 결함(recipe 감사, 전부 코드 대조 CONFIRMED)을 고친다:
  1. **스퓨리어스 상관** — v1은 의미 질문("~는 어떻게 규정되는가?")이 refusal에만 등장해
     '의미 질문→거절'을 학습, 평가셋(전부 의미 질문)에서 과잉거절 위험. → v2는 positive와
     refusal을 **같은 의미-질문 스타일**로 만들어(faith 질문셋 재활용) 상관을 절단한다.
  2. **학습 K=1 vs 평가 K=5** — v1은 컨텍스트에 gold 1조문만 넣어 '선택' 기술 미학습.
     → v2는 faithbench의 near-distractor 컨텍스트(K조문)를 재사용한다.
  3. **hint 리크** — v1은 질문에 gold span 첫 8자를 노출(표층 복사 휴리스틱). → v2는 hint-echo를
     폐기하고 **위치 기반 질문(N번째 문장)**으로 대체 + 리크가드(질문이 gold의 6자+ 연속
     substring이면 거부).
  4. **통짜 gold 57%** — v2는 granularity를 명시 혼합: **full(전문 요구)** vs **tight(한 문장
     요구)** 을 섞어 '요구된 만큼만 인용'을 가르친다(precision 캘리브레이션).

정직한 한계(teacher/DPO로 이연): 위치 기반 tight 질문은 '어느 절인지'를 위치로 가르치지만,
평가셋이 요구하는 **의미 질문→해당 절 매핑**은 결정적으로 생성할 수 없다(teacher 필요).
따라서 v2는 confound 제거 + 공정 비교 기반 확보가 목적이고, 의미-질문 precision의 직접 교정은
**span_f1 보상 DPO(teacher-free)** 또는 로컬 teacher 의미질문 합성(v3)이 담당한다.

    python scripts/data/gen_partial_sft.py --out data/processed/partial_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.citation_verify import _norm, load_corpus, verify  # noqa: E402
from eval.faithbench import _context_block, _pick_distractors  # noqa: E402

SYS = (
    "너는 제공된 근거 조항만 사용해 답한다. 여러 근거 중 질문에 해당하는 조항을 찾아, "
    "그 조항에서 **질문이 요구하는 범위만큼만** 「원문 인용」[조항ID] 형식으로 인용한다. "
    "한 부분만 물으면 그 부분만, 전문을 물으면 전문을 인용하라. "
    "해당하는 근거가 없으면 '제공된 근거에서는 확인할 수 없습니다'라고 답한다."
)

REFUSAL = "제공된 근거 조문에서는 해당 내용을 확인할 수 없습니다. 근거가 없는 사항은 답변하지 않습니다."
_ORD = ["첫", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉", "열"]


def split_spans(text: str, min_len: int = 8) -> list[str]:
    """조문을 문장 종결('...다.') 단위로 분리 — 각 조각은 조문의 exact substring."""
    parts = re.split(r"(?<=다\.)\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= min_len]


def _leaks(question: str, gold_span: str, n: int = 6) -> bool:
    """질문이 gold_span의 연속 n자 이상을 그대로 포함하면 리크(표층 복사 학습 유발)."""
    g = _norm(gold_span)
    q = _norm(question)
    return any(g[i : i + n] in q for i in range(len(g) - n + 1))


def _ctx(cid: str, corpus: dict[str, str], pool: list[str], k: int, rng: random.Random) -> tuple[str, list[str]]:
    """gold + near-distractor K조문 컨텍스트(평가 분포와 정합)."""
    sub = {c: corpus[c] for c in pool}
    sub[cid] = corpus[cid]
    dist = _pick_distractors(sub, cid, k - 1, near=True, rng=rng)
    items = [(cid, corpus[cid])] + [(d, corpus[d]) for d in dist]
    rng.shuffle(items)
    return _context_block(items), [c for c, _ in items]


def make_example(cid, corpus, pool, k, question, gold_span, label, rng) -> dict:
    ctx, ctx_ids = _ctx(cid, corpus, pool, k, rng)
    answer = f"「{gold_span}」[{cid}]라고 규정하고 있습니다."
    return {
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": f"{ctx}\n\n질문: {question}"},
            {"role": "assistant", "content": answer},
        ],
        "label": label,
        "gold_citations": [cid],
        "gold_span": gold_span,
        "context_ids": ctx_ids,
    }


def make_refusal(corpus, pool, k, question, rng) -> dict:
    """컨텍스트에 정답 조문이 **없는** 상태에서 의미 질문 → 거절.

    positive와 같은 의미-질문 스타일을 쓰되 gold를 컨텍스트에서 제외해, 스타일이 아니라
    '근거 관련성'으로 거절을 학습하게 한다(v1 스퓨리어스 상관 제거).
    """
    anchor = rng.choice(pool)
    ctx, ctx_ids = _ctx(anchor, corpus, [p for p in pool if p != anchor], k, rng)
    return {
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": f"{ctx}\n\n질문: {question}"},
            {"role": "assistant", "content": REFUSAL},
        ],
        "label": "refusal",
        "gold_citations": [],
        "context_ids": ctx_ids,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/seed/constitution.json")
    ap.add_argument("--eval-items", default="eval/questions.partial.constitution.json")
    ap.add_argument("--faith-questions", default="eval/questions.constitution.json",
                    help="의미-질문 풀(positive/refusal 스타일 공유용)")
    ap.add_argument("--out", default="data/processed/partial_sft.jsonl")
    ap.add_argument("--k", type=int, default=5, help="컨텍스트 조문 수(평가와 정합)")
    ap.add_argument("--refusal-ratio", type=float, default=0.22)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    eval_ids = {_norm(it["id"]) for it in json.loads(Path(args.eval_items).read_text(encoding="utf-8"))}
    faith_q = {_norm(k): v for k, v in json.loads(Path(args.faith_questions).read_text(encoding="utf-8")).items()}
    print(f"held-out(partial 평가셋) 조항 {len(eval_ids)}개 제외 | 의미-질문 풀 {len(faith_q)}개")

    # 학습 대상 = held-out 아닌 조항. distractor 풀도 held-out 제외(순수 unseen 유지).
    train_ids = [c for c in corpus if c not in eval_ids]
    rng = random.Random(args.seed)

    rows: list[dict] = []
    n_full = n_tight = n_multi_clause = 0
    guard_dropped = 0

    for cid in train_ids:
        pool = [c for c in train_ids if c != cid]
        spans = split_spans(corpus[cid])

        # (a) full 인용 positive — faith 의미 질문(있으면) 또는 전문 요구. gold = 조문 전체.
        q_full = faith_q.get(cid, "이 조항의 내용을 전문 그대로 인용해줘.")
        rows.append(make_example(cid, corpus, pool, args.k, q_full, corpus[cid], "full", rng))
        n_full += 1

        # (b) tight 인용 positive — 다문장 조문의 각 문장을 위치 기반 질문으로. 리크가드 통과만.
        if len(spans) > 1:
            n_multi_clause += 1
            for idx, si in enumerate(spans):
                ord_word = _ORD[idx] if idx < len(_ORD) else f"{idx + 1}"
                q_tight = f"이 조항의 {ord_word} 번째 문장에 해당하는 규정을 원문 그대로 인용해줘."
                if _leaks(q_tight, si):
                    guard_dropped += 1
                    continue
                rows.append(make_example(cid, corpus, pool, args.k, q_tight, si, "tight", rng))
                n_tight += 1

    # (c) refusal — 의미 질문(positive와 같은 스타일)을 gold 부재 컨텍스트에 붙임. 목표 비율까지.
    n_pos = len(rows)
    target_ref = int(n_pos * args.refusal_ratio / (1 - args.refusal_ratio))
    faith_pool = list(faith_q.items())
    for _ in range(target_ref):
        _, q = rng.choice(faith_pool)
        rows.append(make_refusal(corpus, train_ids, args.k, q, rng))
    n_ref = target_ref

    # self-consistency: 모든 positive gold가 citation_verify 지지 + tight가 실제로 조문보다 짧은지
    bad = loose = tight_ok = 0
    for r in rows:
        if r["label"] == "refusal":
            continue
        rep = verify(r["messages"][-1]["content"], corpus)
        if not rep["leak_free"]:
            bad += 1
            print(f"  ✗ 오염: {r['gold_citations']} f={rep['faithfulness']}")
        cid = _norm(r["gold_citations"][0])
        if _norm(r["gold_span"]) == corpus[cid]:
            loose += 1
        elif r["label"] == "tight":
            tight_ok += 1

    rng.shuffle(rows)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"생성: full {n_full} + tight {n_tight}(다문장 조문 {n_multi_clause}) + refusal {n_ref} "
        f"= {len(rows)}건 → {outp}"
    )
    print(f"  granularity: full(전문 gold) {n_full} | tight(부분 gold, 조문보다 짧음) {tight_ok} "
          f"| loose(단문장=전체) {loose - n_full if loose > n_full else 0}")
    print(f"  리크가드 제거 질문: {guard_dropped} | 컨텍스트 K={args.k}(near-distractor)")
    print(f"  refusal 비율: {n_ref}/{len(rows)} = {n_ref / len(rows):.2f} (목표 {args.refusal_ratio})")
    print(f"self-consistency(citation_verify): {'PASS (오염 0)' if bad == 0 else f'FAIL ({bad}건)'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
