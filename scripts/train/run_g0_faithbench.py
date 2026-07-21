"""진짜 G0 — faithbench(distractor 포함) 교차모델 비교.

run_g0_compare의 강화판. 근거를 1조문이 아니라 K조문(gold + distractor)으로 제공하고
질문은 조항ID를 노출하지 않아, 모델이 올바른 조문을 '선택'해 인용해야 정답이다.
이로써 파일럿의 자명한 target_cited·포화된 거절율 문제를 해소하고, 전략 핵심 가정
("소형 FT가 대형 base를 근거충실도에서 이긴다")을 더 어려운 태스크에서 검증한다.

공정성: base 모델엔 few-shot(**허구 예시법**으로 선택+인용 형식 시연 1 + 거절 1 —
평가 코퍼스/정답을 절대 노출하지 않는다), FT는 zero-shot.
채점은 결정적(citation_verify 기반, LLM-judge 없음).

    python scripts/train/run_g0_faithbench.py \
        --questions eval/questions.constitution.json --k 5 --near

per-instance transcript(모델별 원문 답변+정오답)를 함께 저장해 paired McNemar·실패모드
감사를 사후 가능하게 한다(methodology 감사 반영).
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.citation_verify import SCORER_VERSION, load_corpus  # noqa: E402
from eval.faithbench import (  # noqa: E402
    aggregate,
    build_instances,
    load_questions,
    load_unanswerable,
    score_answer,
)
from eval.run_meta import run_metadata  # noqa: E402

# pilot 학습/held-out 분할의 시드는 run_g0_pilot과 **고정 일치**해야 한다.
# args.seed(인스턴스 셔플용)를 쓰면 --seed 변경 시 seen/unseen이 조용히 오태깅된다.
PILOT_SPLIT_SEED = 3407


def pilot_split(corpus: dict[str, str], seed: int = PILOT_SPLIT_SEED) -> tuple[set[str], set[str]]:
    """run_g0_pilot의 학습/held-out 분할을 재현 → seen/unseen 편향 분리용."""
    ids = list(corpus.keys())
    random.Random(seed).shuffle(ids)
    n_eval = max(12, len(ids) // 5)
    return set(ids[n_eval:]), set(ids[:n_eval])  # (train=seen, eval=unseen)


def subset_exact(per: list[dict], keep_golds: set[str]) -> dict:
    """answerable 중 gold가 keep_golds에 속하는 것만 집계(seen/unseen 분리)."""
    sub = [s for s in per if s["split"] == "answerable" and s["gold"] in keep_golds]

    def mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    return {
        "n": len(sub),
        "selection_exact": mean([s["exact"] for s in sub]),
        "gold_recall": mean([s["gold_recall"] for s in sub]),
        "distractor_cite_rate": mean([s["distractor_cited"] for s in sub]),
    }


# few-shot은 **평가 코퍼스(헌법)와 완전히 무관한 허구 예시법**으로만 형식을 시연한다.
# (구버전은 헌법 제2조 ②·"대통령 임기"를 예시로 써서 평가 문항 2개의 정답을 그대로
#  노출했다 — methodology 감사에서 확인된 벤치 위생 결함. 허구법이면 노출 0.)
_FS_LAW = {
    "예시법 제1조": "이 법은 인용 형식의 예시를 목적으로 한다.",
    "예시법 제2조 ①": "인용은 제시된 근거 안에서만 하여야 한다.",
    "예시법 제3조": "인용문은 근거의 원문을 그대로 옮겨야 한다.",
    "예시법 제9조": "이 법의 시행일은 공포한 날로 한다.",
}


def few_shot_msgs(corpus: dict[str, str]) -> list[dict]:
    """base 모델용 few-shot: 허구 예시법에서 정답 선택+인용 1건 + 거절 1건.

    실제 평가 코퍼스/질문/정답을 일절 포함하지 않아 답 노출이 원천 불가능하다.
    """
    a, at = "예시법 제3조", _FS_LAW["예시법 제3조"]
    d1, d1t = "예시법 제1조", _FS_LAW["예시법 제1조"]
    d2, d2t = "예시법 제2조 ①", _FS_LAW["예시법 제2조 ①"]
    d3, d3t = "예시법 제9조", _FS_LAW["예시법 제9조"]
    ctx1 = f"[근거]\n1) {d1}: {d1t}\n2) {a}: {at}\n3) {d2}: {d2t}"
    ctx2 = f"[근거]\n1) {d1}: {d1t}\n2) {d3}: {d3t}"
    return [
        {"role": "user", "content": f"{ctx1}\n\n질문: 인용문은 어떻게 옮겨야 하는가?"},
        {"role": "assistant", "content": f"「{at}」[{a}]라고 규정하고 있습니다."},
        {"role": "user", "content": f"{ctx2}\n\n질문: 이 법의 벌칙 조항은 무엇인가?"},
        {"role": "assistant", "content": "제공된 근거에서는 확인할 수 없습니다."},
    ]


def apply_chat(tok, messages, **kwargs):
    """apply_chat_template with thinking mode disabled.

    Qwen3 emits a ``<think>`` block by default; with ``max_new_tokens=160`` that
    budget is spent on reasoning and the citation answer is truncated, silently
    invalidating the eval. ``enable_thinking`` is ignored by templates that do
    not support it (e.g. Qwen2.5), so existing 1.5B results are unchanged.
    """
    try:
        return tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except (TypeError, ValueError):
        return tok.apply_chat_template(messages, **kwargs)


def gen(model, tok, messages: list[dict]) -> str:
    import torch

    enc = apply_chat(
        tok, messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def eval_model(model, tok, insts, corpus, fs):
    """전체 aggregate + 인스턴스별 점수(gold 태깅·원문 답변 포함) 반환."""
    per = []
    for inst in insts:
        system, user = inst["messages"][0], inst["messages"][1]
        msgs = [system] + (fs if fs else []) + [user]
        ans = gen(model, tok, msgs)
        s = score_answer(inst, ans, corpus)
        s["gold"] = inst["gold"][0] if inst["gold"] else None
        s["question"] = inst["question"]
        s["context_ids"] = inst["context_ids"]
        s["answer"] = ans  # paired McNemar·실패모드 감사용 원문 보존
        per.append(s)
    return aggregate(per), per


def closed_book_probe(model, tok, insts, corpus) -> dict:
    """암기 프로브 — 근거를 제공하지 않고 질문만 던져 verbatim 인출률을 잰다.

    base가 헌법을 암기했다면 여기서 gold 조문 원문이 답변에 그대로 나온다. open-book
    selection_exact와의 차이가 'grounding gain'(제공 근거를 실제로 읽는 정도)이다.
    (redteam 감사: '헌법 암기 confound'를 데이터로 선제 방어.)
    """
    sys_free = "질문에 대한민국 헌법 조문을 근거로 답하라. 해당 조문의 원문을 그대로 인용하라."
    recalled = 0
    n = 0
    for inst in insts:
        if inst["split"] != "answerable":
            continue
        n += 1
        gold_id = inst["gold"][0]
        gold_txt = corpus[gold_id]
        msgs = [
            {"role": "system", "content": sys_free},
            {"role": "user", "content": f"질문: {inst['question']}"},
        ]
        ans = gen(model, tok, msgs)
        # verbatim 인출: gold 조문의 앞 24자(정규화)가 답변에 substring으로 등장하면 암기로 간주
        from eval.citation_verify import _norm

        probe = _norm(gold_txt)[:24]
        if probe and probe in _norm(ans):
            recalled += 1
    return {"n": n, "verbatim_recall_rate": round(recalled / n, 3) if n else 0.0}


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
    ap.add_argument("--questions", help="추가/대체 질문셋 JSON(object 또는 list[{id,question}])")
    ap.add_argument("--unanswerable-file", help="추가 unanswerable 질문 JSON(list 또는 {questions})")
    ap.add_argument(
        "--include-all-corpus",
        action="store_true",
        help="질문 없는 모든 코퍼스 항목도 ID 기반 sanity 질문으로 포함(정식 리포트용 아님)",
    )
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument(
        "--out",
        default="docs/env-verify/g0-faithbench-result.json",
        help="결과 JSON 경로. 재실행이 기존 기준선을 덮지 않도록 새 실험은 새 경로 지정 권장.",
    )
    ap.add_argument(
        "--transcript",
        default=None,
        help="per-instance 원문답변+정오답 JSONL 경로. 기본은 --out 옆에 *-transcript.jsonl",
    )
    ap.add_argument(
        "--closed-book",
        action="store_true",
        help="근거 미제공 암기 프로브 추가 실행(grounding gain 측정)",
    )
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    insts = build_instances(
        corpus,
        args.k,
        args.near,
        args.seed,
        questions=load_questions(args.questions),
        unanswerable=load_unanswerable(args.unanswerable_file),
        include_all_corpus=args.include_all_corpus,
    )
    n_ans = sum(1 for i in insts if i["split"] == "answerable")
    n_una = sum(1 for i in insts if i["split"] == "unanswerable")
    fs = few_shot_msgs(corpus)
    print(f"faithbench: answerable {n_ans} + unanswerable {n_una} (k={args.k}, near={args.near})")

    seen, unseen = pilot_split(corpus, args.seed)
    n_unseen = sum(1 for i in insts if i["split"] == "answerable" and i["gold"][0] in unseen)
    print(f"seen/unseen 분리(pilot split): answerable {n_ans} 중 unseen(미학습) {n_unseen}")

    results, per_model, closed_book = {}, {}, {}

    print("\n[1/3] base small (few-shot) ...")
    m, t = load_base(args.small)
    results["base_small_fewshot"], per_model["base_small_fewshot"] = eval_model(
        m, t, insts, corpus, fs
    )
    if args.closed_book:
        closed_book["base_small_fewshot"] = closed_book_probe(m, t, insts, corpus)
    print("  ", results["base_small_fewshot"])

    print("\n[2/3] FT small (zero-shot) ...")
    from peft import PeftModel

    ft = PeftModel.from_pretrained(m, args.adapter)
    ft.eval()
    results["ft_small_zeroshot"], per_model["ft_small_zeroshot"] = eval_model(
        ft, t, insts, corpus, None
    )
    if args.closed_book:
        closed_book["ft_small_zeroshot"] = closed_book_probe(ft, t, insts, corpus)
    print("  ", results["ft_small_zeroshot"])
    free(ft)
    free(m)

    print("\n[3/3] base large (few-shot) - USB HDD load can be slow ...")
    m2, t2 = load_base(args.large)
    results["base_large_fewshot"], per_model["base_large_fewshot"] = eval_model(
        m2, t2, insts, corpus, fs
    )
    if args.closed_book:
        closed_book["base_large_fewshot"] = closed_book_probe(m2, t2, insts, corpus)
    print("  ", results["base_large_fewshot"])
    free(m2)

    # seen/unseen 분리 집계(FT의 학습 친숙도 편향 제거)
    by_split = {
        name: {"seen": subset_exact(per, seen), "unseen": subset_exact(per, unseen)}
        for name, per in per_model.items()
    }

    meta = run_metadata(
        models={
            "base_small": args.small,
            "base_large": args.large,
            "ft_small": f"{args.small}+{args.adapter}",
        },
        corpus_path=args.corpus,
        questions_path=args.questions,
        adapter_path=args.adapter,
        seed=args.seed,
        scorer_version=SCORER_VERSION,
        extra={
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pilot_split_seed": PILOT_SPLIT_SEED,
        },
    )

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(
        json.dumps({
            "meta": meta,
            "k": args.k,
            "near": args.near,
            "n_answerable": n_ans,
            "n_unanswerable": n_una,
            "questions": args.questions,
            "unanswerable_file": args.unanswerable_file,
            "include_all_corpus": args.include_all_corpus,
            "n_unseen_answerable": n_unseen,
            "results": results,
            "by_split": by_split,
            "closed_book": closed_book if closed_book else None,
        }, ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    # per-instance transcript — paired McNemar·실패모드 감사용(모델×인스턴스 원문+정오답)
    tpath = Path(args.transcript) if args.transcript else outp.with_name(outp.stem + "-transcript.jsonl")
    with tpath.open("w", encoding="utf-8", newline="\n") as tf:
        for name, per in per_model.items():
            for row in per:
                tf.write(json.dumps({"model": name, **row}, ensure_ascii=False) + "\n")
    print("  -> transcript:", tpath)

    print("\n===== 진짜 G0 - faithbench 교차비교 (전체) =====")
    cols = (
        "selection_exact",
        "gold_recall",
        "distractor_cite_rate",
        "faithfulness_mean",
        "leak_rate",
    )
    print(f"  {'모델':<22}" + "".join(f"{c[:12]:>14}" for c in cols))
    for name, r in results.items():
        print(f"  {name:<22}" + "".join(f"{r[c]:>14}" for c in cols))

    print(f"\n===== unseen(미학습 {n_unseen}조항)만 - 친숙도 편향 제거 =====")
    print(f"  {'모델':<22}{'selection_ex':>14}{'gold_recall':>14}{'distractor_c':>14}")
    for name, sp in by_split.items():
        u = sp["unseen"]
        print(
            f"  {name:<22}{u['selection_exact']:>14}"
            f"{u['gold_recall']:>14}{u['distractor_cite_rate']:>14}"
        )

    # NOTE: G0 = SPLIT. selection_exact is only the lenient axis; the tight-span
    # precision axis (run_g0_partial) can favor the large base. Do NOT emit a
    # single-axis "small beats large" verdict here — it violates the g0-verdict
    # marketing ban. Report the axes descriptively and leave adjudication to the
    # combined selection + span_precision + leak criteria.
    ft_r, bl = results["ft_small_zeroshot"], results["base_large_fewshot"]
    ftu, blu = by_split["ft_small_zeroshot"]["unseen"], by_split["base_large_fewshot"]["unseen"]
    print("\n  대조(selection축, 단정 아님 — span precision·leak과 병기해야 판정):")
    print(
        f"    전체  selection_exact  FT={ft_r['selection_exact']:>6}  base={bl['selection_exact']:>6}"
        f"  | leak FT={ft_r['leak_rate']:>6} base={bl['leak_rate']:>6}"
    )
    print(
        f"    unseen selection_exact FT={ftu['selection_exact']:>6}  base={blu['selection_exact']:>6}"
    )
    print("    → tight-span precision은 run_g0_partial.py에서 확인. headline 'small beats large' 금지.")
    if closed_book:
        print("\n===== 암기 프로브(closed-book) — grounding gain =====")
        for name, cb in closed_book.items():
            oa = results[name]["selection_exact"]
            print(
                f"  {name:<22} verbatim_recall={cb['verbatim_recall_rate']:>6} "
                f"| open selection_exact={oa:>6} | gain={round(oa - cb['verbatim_recall_rate'], 3):>+6}"
            )
    print("  주의: n 소표본(특히 unseen)·단일 법령 프로토타입. 강한 결론엔 검정력 N·다법령·human anchor 필요.")
    print("  주의: selection_exact는 인용 형식 순응도를 포함한다 — answerable_no_citation_rate와 함께 해석할 것.")
    print("  -> 저장:", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
