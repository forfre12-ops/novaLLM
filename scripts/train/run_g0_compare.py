"""G0-lite 교차모델 비교 — "소형 파인튜닝 vs 대형 base"의 근거충실도.

전략 핵심 가정 검증: 파인튜닝된 소형(1.5B)이 대형 base(7B)를 근거충실 인용에서 이기는가?
공정성: base 모델엔 few-shot(형식 예시 3개=grounded 2 + refusal 1) 제공. FT는 zero-shot.

held-out 조항으로 두 종류 평가:
  · 답변가능: 근거=정답조항 → 충실 인용(faithfulness, target_cited)
  · 답변불가: 근거=정답조항이나 질문은 코퍼스 밖 주제 → 거절해야 정답(refusal_rate; 인용하면 환각)

    python scripts/train/run_g0_compare.py
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.citation_verify import verify, load_corpus  # noqa: E402

SYS = ("너는 제공된 근거 조항만 사용해 답한다. 반드시 「원문 인용」[조항ID] 형식으로 근거를 "
       "인용한다. 근거에 없으면 '제공된 근거에서는 확인할 수 없습니다'라고 답한다.")
EVAL_Q = "위 근거 조항의 핵심 내용을 원문을 인용하여 설명해줘."
OOC_TOPICS = ["대통령의 임기", "국회의원의 정수", "헌법개정 절차", "지방자치단체의 종류"]


def user_msg(cid, text, q):
    return {"role": "user", "content": f"[근거]\n{cid}: {text}\n\n질문: {q}"}


def build(corpus, rng):
    ids = list(corpus.keys())
    rng.shuffle(ids)
    n_eval = max(12, len(ids) // 5)
    eval_ids, train_ids = ids[:n_eval], ids[n_eval:]
    # few-shot 예시(train 조항)
    fs = []
    for cid in train_ids[:2]:
        fs.append(user_msg(cid, corpus[cid], EVAL_Q))
        fs.append({"role": "assistant", "content": f"헌법은 「{corpus[cid]}」[{cid}]라고 규정하고 있습니다."})
    ref_id = train_ids[2]
    fs.append(user_msg(ref_id, corpus[ref_id],
                       f"이 근거에 '{OOC_TOPICS[0]}'에 관한 내용이 있으면 인용하고 없으면 없다고 답하라."))
    fs.append({"role": "assistant", "content": "제공된 근거에서는 해당 내용을 확인할 수 없습니다."})
    # 평가셋
    answerable = [(cid, user_msg(cid, corpus[cid], EVAL_Q)) for cid in eval_ids]
    unanswerable = [(cid, user_msg(cid, corpus[cid],
                    f"이 근거에 '{rng.choice(OOC_TOPICS)}'에 관한 내용이 있으면 인용하고 없으면 없다고 답하라."))
                    for cid in eval_ids]
    return fs, answerable, unanswerable, eval_ids


def gen(model, tok, messages):
    import torch
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def is_refusal(ans, rep):
    return rep["n_citations"] == 0 and ("확인할 수 없" in ans or "없습니다" in ans)


def eval_model(model, tok, fs, answerable, unanswerable, corpus, few_shot):
    faiths, tgt = [], 0
    for cid, um in answerable:
        msgs = [{"role": "system", "content": SYS}] + (fs if few_shot else []) + [um]
        ans = gen(model, tok, msgs)
        rep = verify(ans, corpus)
        faiths.append(rep["faithfulness"] if rep["n_citations"] else 0.0)
        if any(c["cited_id"].strip() == cid and c["supported"] for c in rep["citations"]):
            tgt += 1
    refused = 0
    for cid, um in unanswerable:
        msgs = [{"role": "system", "content": SYS}] + (fs if few_shot else []) + [um]
        ans = gen(model, tok, msgs)
        rep = verify(ans, corpus)
        if is_refusal(ans, rep):
            refused += 1
    na = len(answerable)
    return {
        "ans_faithfulness": round(sum(faiths) / na, 3),
        "ans_target_cited": round(tgt / na, 3),
        "unans_refusal_rate": round(refused / len(unanswerable), 3),
    }


def load_base(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map={"": 0},
        attn_implementation="sdpa", dtype=torch.bfloat16)
    model.eval()
    return model, tok


def free(model):
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
    args = ap.parse_args()

    rng = random.Random(3407)
    corpus = load_corpus(args.corpus)
    fs, answerable, unanswerable, eval_ids = build(corpus, rng)
    print(f"corpus {len(corpus)} | eval {len(eval_ids)}조항 (답변가능+답변불가 각 {len(eval_ids)})")

    results = {}

    # base 1.5B (few-shot)
    print("\n[1/3] base 1.5B (few-shot) ...")
    m, t = load_base(args.small)
    results["base_1.5B_fewshot"] = eval_model(m, t, fs, answerable, unanswerable, corpus, True)
    print("  ", results["base_1.5B_fewshot"])

    # FT 1.5B (zero-shot) — 같은 base에 어댑터
    print("\n[2/3] 파인튜닝 1.5B (zero-shot) ...")
    from peft import PeftModel
    ft = PeftModel.from_pretrained(m, args.adapter)
    ft.eval()
    results["ft_1.5B_zeroshot"] = eval_model(ft, t, fs, answerable, unanswerable, corpus, False)
    print("  ", results["ft_1.5B_zeroshot"])
    free(ft)
    free(m)

    # base 7B (few-shot)
    print("\n[3/3] base 7B (few-shot) — USB HDD 로드 느림 ...")
    m2, t2 = load_base(args.large)
    results["base_7B_fewshot"] = eval_model(m2, t2, fs, answerable, unanswerable, corpus, True)
    print("  ", results["base_7B_fewshot"])
    free(m2)

    outp = Path("docs/env-verify/g0-compare-result.json")
    outp.write_text(json.dumps({"eval_n": len(eval_ids), "results": results}, ensure_ascii=False, indent=2),
                    encoding="utf-8")

    print("\n===== G0-lite 교차비교 (held-out 조항) =====")
    hdr = ("모델", "충실도", "정답인용", "거절율")
    print(f"  {hdr[0]:<22}{hdr[1]:>9}{hdr[2]:>9}{hdr[3]:>9}")
    for name, r in results.items():
        print(f"  {name:<22}{r['ans_faithfulness']:>9}{r['ans_target_cited']:>9}{r['unans_refusal_rate']:>9}")

    ft = results["ft_1.5B_zeroshot"]
    b7 = results["base_7B_fewshot"]
    win = (ft["ans_faithfulness"] >= b7["ans_faithfulness"] and ft["unans_refusal_rate"] >= b7["unans_refusal_rate"])
    print("\n  판정:", "소형 FT ≥ 대형 base (가정 지지)" if win else "대형 base 우위 (가정 미지지)")
    print("  → 저장:", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
