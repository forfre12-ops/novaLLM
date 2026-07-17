"""진짜 G0 — faithbench(distractor 포함) 교차모델 비교.

run_g0_compare의 강화판. 근거를 1조문이 아니라 K조문(gold + distractor)으로 제공하고
질문은 조항ID를 노출하지 않아, 모델이 올바른 조문을 '선택'해 인용해야 정답이다.
이로써 파일럿의 자명한 target_cited·포화된 거절율 문제를 해소하고, 전략 핵심 가정
("소형 FT가 대형 base를 근거충실도에서 이긴다")을 더 어려운 태스크에서 검증한다.

공정성: base 모델엔 few-shot(선택+인용 형식 예시 2 + 거절 1), FT는 zero-shot.
채점은 결정적(citation_verify 기반, LLM-judge 없음).

    python scripts/train/run_g0_faithbench.py --k 5 --near
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.citation_verify import load_corpus  # noqa: E402
from eval.faithbench import aggregate, build_instances, score_answer  # noqa: E402


def few_shot_msgs(corpus: dict[str, str]) -> list[dict]:
    """base 모델용 few-shot: 다조문 컨텍스트에서 정답 선택+인용, 그리고 거절 예시.

    eval의 gold(QUESTIONS)와 겹치지 않는 조문으로 구성.
    """
    a1, a1t = "헌법 제2조 ②", corpus["헌법 제2조 ②"]
    d1, d1t = "헌법 제7조 ①", corpus["헌법 제7조 ①"]
    d2, d2t = "헌법 제18조", corpus["헌법 제18조"]
    ctx1 = f"[근거]\n1) {d1}: {d1t}\n2) {a1}: {a1t}\n3) {d2}: {d2t}"
    ctx2 = f"[근거]\n1) {d1}: {d1t}\n2) {d2}: {d2t}"
    return [
        {"role": "user", "content": f"{ctx1}\n\n질문: 재외국민 보호에 대한 국가의 의무는 어떻게 규정되는가?"},
        {"role": "assistant", "content": f"헌법은 「{a1t}」[{a1}]라고 규정하고 있습니다."},
        {"role": "user", "content": f"{ctx2}\n\n질문: 대통령의 임기는 몇 년인가?"},
        {"role": "assistant", "content": "제공된 근거에서는 확인할 수 없습니다."},
    ]


def gen(model, tok, messages: list[dict]) -> str:
    import torch

    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def eval_model(model, tok, insts, corpus, fs) -> dict:
    scored = []
    for inst in insts:
        system, user = inst["messages"][0], inst["messages"][1]
        msgs = [system] + (fs if fs else []) + [user]
        ans = gen(model, tok, msgs)
        scored.append(score_answer(inst, ans, corpus))
    return aggregate(scored)


def load_base(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map={"": 0},
        attn_implementation="sdpa", dtype=torch.bfloat16,
    )
    model.eval()
    return model, tok


def free(model) -> None:
    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--large", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="checkpoints/g0-pilot/lora_adapter")
    ap.add_argument("--corpus", default="data/seed/constitution.json")
    ap.add_argument("--k", type=int, default=5, help="컨텍스트 조문 수(gold 1 + distractor k-1)")
    ap.add_argument("--near", action="store_true", help="같은 조의 인접 항을 하드 distractor로 우선")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    insts = build_instances(corpus, args.k, args.near, args.seed)
    n_ans = sum(1 for i in insts if i["split"] == "answerable")
    n_una = sum(1 for i in insts if i["split"] == "unanswerable")
    fs = few_shot_msgs(corpus)
    print(f"faithbench: answerable {n_ans} + unanswerable {n_una} (k={args.k}, near={args.near})")

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

    print("\n[3/3] base large (few-shot) — USB HDD 로드 느림 ...")
    m2, t2 = load_base(args.large)
    results["base_large_fewshot"] = eval_model(m2, t2, insts, corpus, fs)
    print("  ", results["base_large_fewshot"])
    free(m2)

    outp = Path("docs/env-verify/g0-faithbench-result.json")
    outp.write_text(
        json.dumps({"k": args.k, "near": args.near, "n_answerable": n_ans, "n_unanswerable": n_una,
                    "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== 진짜 G0 — faithbench 교차비교 =====")
    cols = ("selection_exact", "gold_recall", "distractor_cite_rate", "faithfulness_mean", "leak_rate")
    print(f"  {'모델':<22}" + "".join(f"{c[:12]:>14}" for c in cols))
    for name, r in results.items():
        print(f"  {name:<22}" + "".join(f"{r[c]:>14}" for c in cols))

    ft_r, bl = results["ft_small_zeroshot"], results["base_large_fewshot"]
    win = ft_r["selection_exact"] >= bl["selection_exact"] and ft_r["leak_rate"] <= bl["leak_rate"]
    print("\n  판정:", "소형 FT ≥ 대형 base (선택적 근거충실, 가정 지지)" if win else "대형 base 우위 (가정 미지지)")
    print(
        "  주의: n 소표본·단일 법령 프로토타입. 강한 결론엔 검정력 산정 N·다법령·human anchor 필요."
    )
    print("  → 저장:", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
