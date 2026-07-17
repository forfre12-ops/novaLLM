"""03_eval.py — 생성 육안 테스트 + 근거충실 인용 결정적 평가

이 환경 주의(T0): unsloth/datasets 미사용. 표준 transformers+peft+bitsandbytes(sdpa)로
LoRA 어댑터(또는 병합 모델)를 로드한다.

두 가지 평가:
  1) (기본) 한국어 프롬프트 육안 생성 테스트 — 빠른 품질 확인.
  2) --citation — 이 프로젝트의 '진짜 평가': closed-set 코퍼스(헌법)를 근거로 제공하고
     「원문 인용」[조항ID] 형식의 근거충실도를 scripts/eval/citation_verify로 결정적 채점
     (LLM-judge 없음). format_rate / faithfulness / target_cited 리포트.

정식 벤치마크(LogicKor/KMMLU/HAERAE)는 lm-evaluation-harness 별도 사용.

실행:
    python scripts/03_eval.py --model checkpoints/sft-run1/lora_adapter
    python scripts/03_eval.py --model checkpoints/g0-pilot/lora_adapter --citation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval.citation_verify import load_corpus, verify  # noqa: E402

PROMPTS = [
    "대한민국의 수도는 어디이고, 그 도시의 특징을 세 가지 설명해줘.",
    "다음 문장을 정중한 존댓말로 바꿔줘: '내일 회의 몇 시야?'",
    "간단한 파이썬 함수로 피보나치 수열의 n번째 값을 구현해줘.",
    "조선 세종대왕의 업적을 한 문단으로 요약해줘.",
]

CITE_SYS = (
    "너는 제공된 근거 조항만 사용해 답한다. 반드시 「원문 인용」[조항ID] 형식으로 근거를 "
    "인용한다. 근거에 없으면 '제공된 근거에서는 확인할 수 없습니다'라고 답한다."
)
CITE_Q = "위 근거 조항의 핵심 내용을 원문을 인용하여 설명해줘."


def load_model(model_path: str):
    """LoRA 어댑터 경로면 베이스+어댑터, 아니면 병합/일반 모델로 로드(둘 다 4bit+sdpa)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    adapter_cfg = Path(model_path) / "adapter_config.json"
    if adapter_cfg.exists():
        from peft import PeftModel

        base_id = json.loads(adapter_cfg.read_text(encoding="utf-8"))["base_model_name_or_path"]
        print(f"LoRA 어댑터 로드: base={base_id} + adapter={model_path}")
        tok = AutoTokenizer.from_pretrained(model_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_id, quantization_config=bnb, device_map={"": 0},
            attn_implementation="sdpa", dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        print(f"병합/일반 모델 로드: {model_path}")
        tok = AutoTokenizer.from_pretrained(model_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb, device_map={"": 0},
            attn_implementation="sdpa", dtype=torch.bfloat16,
        )
    model.eval()
    return model, tok


def generate(model, tok, messages: list[dict], max_new_tokens: int, greedy: bool) -> str:
    import torch

    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
            do_sample=not greedy, temperature=None if greedy else 0.7,
        )
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def run_generation(model, tok, max_new_tokens: int) -> None:
    for p in PROMPTS:
        text = generate(model, tok, [{"role": "user", "content": p}], max_new_tokens, greedy=False)
        print("=" * 60)
        print("Q:", p)
        print("A:", text)
    print("\n정식 벤치마크는 lm-eval 사용:")
    print("  lm_eval --model hf --model_args pretrained=<merged_model> --tasks kmmlu,haerae")


def run_citation(model, tok, corpus_path: str, n: int, max_new_tokens: int) -> None:
    corpus = load_corpus(corpus_path)
    ids = list(corpus.keys())
    if n > 0:
        ids = ids[:n]
    print(f"근거충실 인용 평가: {len(ids)}조항 (corpus {len(corpus)}, closed-set, 결정적 채점)")

    fmt, faiths, target_hit = 0, [], 0
    for cid in ids:
        msgs = [
            {"role": "system", "content": CITE_SYS},
            {"role": "user", "content": f"[근거]\n{cid}: {corpus[cid]}\n\n질문: {CITE_Q}"},
        ]
        gen = generate(model, tok, msgs, max_new_tokens, greedy=True)
        rep = verify(gen, corpus)
        if rep["n_citations"] > 0:
            fmt += 1
            faiths.append(rep["faithfulness"])
        else:
            faiths.append(0.0)
        if any(c["cited_id"].strip() == cid and c["supported"] for c in rep["citations"]):
            target_hit += 1

    n_eval = len(ids)
    result = {
        "n": n_eval,
        "format_rate": round(fmt / n_eval, 3),
        "faithfulness_mean": round(sum(faiths) / n_eval, 3),
        "target_cited_rate": round(target_hit / n_eval, 3),
    }
    print("\n===== 근거충실 인용 평가 =====")
    for k in ("format_rate", "faithfulness_mean", "target_cited_rate"):
        print(f"  {k:<20}{result[k]:>8}")
    print(
        "\n  주의: gold 형식이 조문 전체를 「」로 감싸므로 이 지표는 '근거 verbatim 복사+형식'을 "
        "측정한다(부분 span 인용·distractor·paraphrase는 미측정). 강한 신호는 다조문 컨텍스트 필요."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="LoRA 어댑터 또는 병합 모델 경로")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--citation", action="store_true", help="근거충실 인용 결정적 평가 실행")
    parser.add_argument("--corpus", default="data/seed/constitution.json")
    parser.add_argument("--n", type=int, default=0, help="citation 평가 조항 수(0=전체)")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 사용 불가. cu128 torch 설치를 확인하세요.")

    model, tok = load_model(args.model)
    if args.citation:
        run_citation(model, tok, args.corpus, args.n, min(args.max_new_tokens, 160))
    else:
        run_generation(model, tok, args.max_new_tokens)


if __name__ == "__main__":
    main()
