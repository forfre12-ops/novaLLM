"""G0-pre 파일럿 — 학습→측정 전체 루프 (단일 모델 로드).

가설: QLoRA 파인튜닝이 '근거(조문) 주어졌을 때 「원문 인용」[조항ID] 형식으로 충실히 인용'하는
기술을 가르치는가? held-out 조문(학습 미포함)에서 base vs 파인튜닝을 citation_verify로 비교.

절차: base 로드 → held-out eval(before) → QLoRA 학습 → eval(after) → before/after 리포트.
정직: 헌법 단일 법령·소규모라 '형식·근거충실 학습' 실증 수준(진짜 '소형 vs 대형' 신호는 더 다양한
코퍼스 필요). 그래도 학습 전후 인용 충실도 변화를 이 GPU에서 직접 측정한다.

    python scripts/train/run_g0_pilot.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.citation_verify import verify, load_corpus  # noqa: E402

SYS = ("너는 제공된 근거 조항만 사용해 답한다. 반드시 「원문 인용」[조항ID] 형식으로 근거를 "
       "인용한다. 근거에 없으면 '제공된 근거에서는 확인할 수 없습니다'라고 답한다.")

GROUNDED_Q = [
    "위 근거 조항의 핵심 내용을 원문을 인용하여 설명해줘.",
    "이 근거가 규정하는 바를 원문 인용과 함께 알려줘.",
    "위 조항은 무엇을 규정하는가? 근거를 인용해 답하라.",
]
EVAL_Q = "위 근거 조항의 핵심 내용을 원문을 인용하여 설명해줘."
OOC_TOPICS = ["대통령의 임기", "국회의원의 정수", "헌법개정 절차", "지방자치단체의 종류"]


def make_msgs(ctx_id: str, ctx_text: str, question: str, answer: str | None):
    msgs = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": f"[근거]\n{ctx_id}: {ctx_text}\n\n질문: {question}"},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def build_dataset(corpus: dict[str, str], rng: random.Random):
    ids = list(corpus.keys())
    rng.shuffle(ids)
    n_eval = max(12, len(ids) // 5)
    eval_ids, train_ids = ids[:n_eval], ids[n_eval:]

    train = []
    for cid in train_ids:
        text = corpus[cid]
        ans = f"헌법은 「{text}」[{cid}]라고 규정하고 있습니다."
        for q in GROUNDED_Q:
            train.append(make_msgs(cid, text, q, ans))
        # refusal: 근거 조항은 주되 코퍼스 밖 주제를 물음 → 거절
        topic = rng.choice(OOC_TOPICS)
        train.append(make_msgs(cid, text,
                               f"이 근거에 '{topic}'에 관한 내용이 있으면 인용해 설명하고, 없으면 없다고 답하라.",
                               "제공된 근거에서는 해당 내용을 확인할 수 없습니다."))
    rng.shuffle(train)
    eval_set = [(cid, make_msgs(cid, corpus[cid], EVAL_Q, None)) for cid in eval_ids]
    return train, eval_set, train_ids, eval_ids


def evaluate(model, tok, eval_set, corpus) -> dict:
    import torch
    model.eval()
    n = len(eval_set)
    faiths, fmt, target_hit = [], 0, 0
    for cid, msgs in eval_set:
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=160, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        rep = verify(gen, corpus)
        if rep["n_citations"] > 0:
            fmt += 1
            faiths.append(rep["faithfulness"])
        else:
            faiths.append(0.0)
        if any(c["cited_id"].strip() == cid and c["supported"] for c in rep["citations"]):
            target_hit += 1
    return {
        "n": n,
        "format_rate": round(fmt / n, 3),
        "faithfulness_mean": round(sum(faiths) / n, 3),
        "target_cited_rate": round(target_hit / n, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--corpus", default="data/seed/constitution.json")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    rng = random.Random(args.seed)
    corpus = load_corpus(args.corpus)
    train, eval_set, train_ids, eval_ids = build_dataset(corpus, rng)
    print(f"corpus {len(corpus)} | train {len(train)}건 | eval(held-out) {len(eval_set)}조항")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("loading base model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0},
        attn_implementation="sdpa", dtype=torch.bfloat16)

    # ── BEFORE (base) ──
    print("\n[BEFORE] base 모델 held-out 평가 ...")
    before = evaluate(model, tok, eval_set, corpus)
    print("  before:", before)

    # ── QLoRA 학습 ──
    model = prepare_model_for_kbit_training(model)
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

    def encode(msgs):
        prompt_text = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False)
        full_text = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
        pids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        fids = tok(full_text, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(pids) + fids[len(pids):]
        return fids, labels

    print(f"\n[TRAIN] QLoRA {args.epochs}에폭 x {len(train)}건 ...")
    torch.cuda.reset_peak_memory_stats()
    t0, accum, step = time.time(), 8, 0
    for ep in range(args.epochs):
        rng.shuffle(train)
        ep_loss = []
        for i, msgs in enumerate(train):
            fids, labels = encode(msgs)
            ii = torch.tensor([fids], device="cuda")
            ll = torch.tensor([labels], device="cuda")
            out = model(input_ids=ii, labels=ll)
            (out.loss / accum).backward()
            ep_loss.append(out.loss.item())
            if (i + 1) % accum == 0:
                opt.step(); opt.zero_grad(); step += 1
        print(f"  epoch {ep+1}: mean loss {sum(ep_loss)/len(ep_loss):.4f}")
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  학습 {time.time()-t0:.1f}s, peak VRAM {peak:.2f} GB")

    # ── AFTER (fine-tuned) ──
    print("\n[AFTER] 파인튜닝 모델 held-out 평가 ...")
    after = evaluate(model, tok, eval_set, corpus)
    print("  after:", after)

    # ── 리포트 ──
    result = {
        "model": args.model, "corpus_n": len(corpus),
        "train_n": len(train), "eval_n": len(eval_set),
        "eval_ids": eval_ids, "epochs": args.epochs,
        "before": before, "after": after,
        "delta": {k: round(after[k] - before[k], 3) for k in ("format_rate", "faithfulness_mean", "target_cited_rate")},
        "train_peak_GB": round(peak, 2),
    }
    outp = Path("docs/env-verify/g0-pilot-result.json")
    outp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    adapter = Path("checkpoints/g0-pilot/lora_adapter")
    adapter.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter))

    print("\n===== G0-pre 파일럿 결과 (held-out 조항) =====")
    print(f"  {'지표':<20}{'BEFORE':>10}{'AFTER':>10}{'Δ':>10}")
    for k in ("format_rate", "faithfulness_mean", "target_cited_rate"):
        print(f"  {k:<20}{before[k]:>10}{after[k]:>10}{result['delta'][k]:>+10}")
    print(f"\n  → 결과 저장: {outp}")
    print(f"  → 어댑터 저장: {adapter}")
    improved = result["delta"]["faithfulness_mean"] > 0 and result["delta"]["target_cited_rate"] >= 0
    print("  판정:", "PASS (파인튜닝이 근거충실 인용을 향상)" if improved else "무개선/역행 — 데이터·에폭 재검토")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
