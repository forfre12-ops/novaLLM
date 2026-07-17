"""부분-span G0 — tight 부분인용에서 소형 FT vs 대형 base 교차비교.

faithbench_partial 하네스로, '조문 통째 복사'가 페널티를 받는 char-level span P/R/F1
기준에서 각 모델을 채점한다. 근거충실 인용의 더 엄격한 형태(질문에 해당하는 부분만
정확히 인용)에서도 소형 FT가 대형 base를 앞서는지 본다.

공정성: base엔 few-shot(부분 인용 형식 예시 + 거절), FT는 zero-shot. 결정적 채점.

    python scripts/train/run_g0_partial.py --k 5 --near
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.citation_verify import load_corpus  # noqa: E402
from eval.faithbench_partial import (  # noqa: E402
    aggregate_partial,
    build_partial_instances,
    load_items,
    score_partial,
    validate_items,
)
from train.run_g0_faithbench import free, gen, load_base  # noqa: E402


def few_shot_msgs(corpus: dict[str, str]) -> list[dict]:
    """base용 few-shot: 조문 전체가 아니라 '질문에 해당하는 부분만' 인용하는 예시."""
    g = "헌법 제8조 ①"  # "정당의 설립은 자유이며, 복수정당제는 보장된다." (부분-span 평가셋과 무관)
    d1, d1t = "헌법 제7조 ①", corpus["헌법 제7조 ①"]
    d2, d2t = "헌법 제18조", corpus["헌법 제18조"]
    ctx1 = f"[근거]\n1) {d1}: {d1t}\n2) {g}: {corpus[g]}\n3) {d2}: {d2t}"
    ctx2 = f"[근거]\n1) {d1}: {d1t}\n2) {d2}: {d2t}"
    return [
        {"role": "user", "content": f"{ctx1}\n\n질문: 복수정당제 보장은 어떻게 규정되는가?"},
        {"role": "assistant", "content": "헌법은 「복수정당제는 보장된다.」[헌법 제8조 ①]라고 규정하고 있습니다."},
        {"role": "user", "content": f"{ctx2}\n\n질문: 대통령의 임기는 몇 년인가?"},
        {"role": "assistant", "content": "제공된 근거에서는 확인할 수 없습니다."},
    ]


def eval_model(model, tok, insts, corpus, fs) -> dict:
    scored = []
    for inst in insts:
        system, user = inst["messages"][0], inst["messages"][1]
        msgs = [system] + (fs if fs else []) + [user]
        ans = gen(model, tok, msgs)
        scored.append(score_partial(inst, ans, corpus))
    return aggregate_partial(scored)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--large", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="checkpoints/g0-pilot/lora_adapter")
    ap.add_argument("--corpus", default="data/seed/constitution.json")
    ap.add_argument("--items", default="eval/questions.partial.constitution.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--near", action="store_true")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    items = load_items(args.items)
    validate_items(items, corpus)
    insts = build_partial_instances(corpus, items, args.k, args.near, args.seed)
    fs = few_shot_msgs(corpus)
    print(f"부분-span: {len(insts)}문항 (k={args.k}, near={args.near})")

    results = {}
    print("\n[1/3] base small (few-shot) ...")
    m, t = load_base(args.small)
    results["base_small_fewshot"] = eval_model(m, t, insts, corpus, fs)
    print("  ", results["base_small_fewshot"])

    print("\n[2/3] FT small (zero-shot) ...")
    from peft import PeftModel

    ft = PeftModel.from_pretrained(m, args.adapter)
    ft.eval()
    results["ft_small_zeroshot"] = eval_model(ft, t, insts, corpus, None)
    print("  ", results["ft_small_zeroshot"])
    free(ft)
    free(m)

    print("\n[3/3] base large (few-shot) - USB HDD load can be slow ...")
    m2, t2 = load_base(args.large)
    results["base_large_fewshot"] = eval_model(m2, t2, insts, corpus, fs)
    print("  ", results["base_large_fewshot"])
    free(m2)

    outp = Path("docs/env-verify/g0-partial-result.json")
    outp.write_text(
        json.dumps({"k": args.k, "near": args.near, "n": len(insts), "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== 부분-span G0 — tight 인용 교차비교 =====")
    cols = ("partial_exact", "span_f1", "span_precision", "span_recall", "selected_gold")
    print(f"  {'모델':<22}" + "".join(f"{c[:13]:>15}" for c in cols))
    for name, r in results.items():
        print(f"  {name:<22}" + "".join(f"{r[c]:>15}" for c in cols))

    ft_r, bl = results["ft_small_zeroshot"], results["base_large_fewshot"]
    win = ft_r["partial_exact"] >= bl["partial_exact"] and ft_r["span_f1"] >= bl["span_f1"]
    print("\n  판정:", "소형 FT ≥ 대형 base (tight 부분인용, 가정 지지)" if win else "대형 base 우위")
    print("  주의: n=14 소표본·단일 법령. span 채점은 통째복사 페널티엔 강하나 의미적 적절성은 미측정.")
    print("  → 저장:", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
