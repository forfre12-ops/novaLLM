"""02_train_sft.py — QLoRA SFT 학습 (transformers + peft + bitsandbytes, 수동 루프)

RTX 5070 Ti 16GB(Blackwell/sm_120)에서 실증된 스택으로 베이스 모델을 QLoRA
파인튜닝한다. 설정은 configs/train_config.yaml.

이 환경 주의 (T0 검증 결과 반영):
  - unsloth / flash-attn: 무컴파일 Windows에서 미가용 → 사용 안 함(attn=sdpa).
  - datasets / pyarrow: 네이티브 DLL이 소프트웨어 제한 정책(WDAC)으로 차단됨
    → `from datasets import ...` 및 transformers `Trainer` / trl `SFTTrainer` 불가.
  - 따라서 jsonl은 표준 json으로 읽고, run_g0_pilot.py에서 실증된 수동 QLoRA
    학습 루프(grad accumulation + label 마스킹 + cosine 스케줄)를 일반화해 쓴다.

데이터 포맷: 한 줄당 {"messages": [{"role","content"}, ...]} (마지막이 assistant).
  프롬프트(assistant 이전)는 -100으로 마스킹하고 assistant 응답만 학습한다.

실행:
    python scripts/02_train_sft.py --config configs/train_config.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import yaml

# LoRA 부착 대상. 기본은 attention 4-module(T0 벤치에서 7B가 16GB에 FIT 검증된 조합).
# 품질을 위해 MLP(gate/up/down)까지 넣으려면 config의 target_modules로 확장(VRAM↑).
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln} JSON 파싱 실패: {e}") from e
    return rows


def encode_example(tok, messages: list[dict], max_len: int):
    """chat 메시지를 토큰화하고 프롬프트 구간을 -100으로 마스킹.

    프롬프트(마지막 assistant 직전까지) + generation prompt = 마스킹,
    마지막 assistant 응답 토큰만 학습 대상. 단일턴/멀티턴 모두 마지막 턴만 학습.
    """
    prompt_text = tok.apply_chat_template(
        messages[:-1], add_generation_prompt=True, tokenize=False
    )
    full_text = tok.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )
    pids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    fids = tok(full_text, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(pids) + fids[len(pids):]
    return fids[:max_len], labels[:max_len]


def collate(batch: list[tuple[list[int], list[int]]], pad_id: int):
    """가변 길이 배치를 오른쪽 패딩. labels는 -100으로 패딩(손실 제외)."""
    import torch

    maxlen = max(len(ids) for ids, _ in batch)
    input_ids, attn, labels = [], [], []
    for ids, lab in batch:
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attn.append([1] * len(ids) + [0] * pad)
        labels.append(lab + [-100] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def save_checkpoint(model, out_dir: Path, kept: list[Path], limit: int) -> None:
    """LoRA 어댑터 중간 체크포인트 저장 + save_total_limit 초과분 정리."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    kept.append(out_dir)
    while len(kept) > limit:
        old = kept.pop(0)
        shutil.rmtree(old, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        get_cosine_schedule_with_warmup,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA 사용 불가. Blackwell은 cu128 빌드 필요: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )

    max_seq = int(cfg["max_seq_length"])
    batch_size = int(cfg["batch_size"])
    accum = int(cfg["grad_accum"])
    epochs = int(cfg["epochs"])
    lr = float(cfg["learning_rate"])
    seed = int(cfg["seed"])
    weight_decay = float(cfg.get("weight_decay", 0.01))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.03))
    logging_steps = int(cfg.get("logging_steps", 10))
    save_steps = int(cfg.get("save_steps", 50))
    save_total_limit = int(cfg.get("save_total_limit", 3))
    grad_ckpt = bool(cfg.get("gradient_checkpointing", True))
    target_modules = cfg.get("target_modules") or DEFAULT_TARGET_MODULES

    # ── 1) 데이터 로드 (datasets 미사용) ──
    rows = read_jsonl(cfg["train_file"])
    rows = [r for r in rows if r.get("messages") and r["messages"][-1].get("role") == "assistant"]
    if not rows:
        raise SystemExit(
            f"학습 데이터 0건: {cfg['train_file']} (messages 배열이 assistant로 끝나야 함). "
            "먼저 scripts/data/gen_grounded_sft.py 등으로 데이터를 생성하세요."
        )
    print(f"학습 데이터 {len(rows)}건 로드 → {cfg['train_file']}")

    # ── 2) 베이스 모델 4bit 로드 (QLoRA) ──
    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        raise SystemExit(
            f"토크나이저에 chat_template이 없습니다: {cfg['base_model']}. "
            "chat 템플릿이 내장된 instruct 모델을 쓰세요(예: Qwen2.5-*-Instruct)."
        )

    print(f"베이스 모델 4bit 로드: {cfg['base_model']} (attn=sdpa) ...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb,
        device_map={"": 0},
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ── 3) LoRA 어댑터 부착 ──
    lcfg = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        target_modules=list(target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    # ── 4) 토큰화(+라벨 마스킹) ──
    encoded = [encode_example(tok, r["messages"], max_seq) for r in rows]

    # ── 5) 옵티마이저 + cosine 스케줄 ──
    rng = random.Random(seed)
    torch.manual_seed(seed)
    micro_per_epoch = math.ceil(len(encoded) / batch_size)
    total_micro = micro_per_epoch * epochs
    total_optim_steps = max(1, total_micro // accum)
    warmup_steps = int(warmup_ratio * total_optim_steps)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_optim_steps)

    # ── 6) 학습 루프 ──
    out_root = Path(cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    kept_ckpts: list[Path] = []
    model.train()
    torch.cuda.reset_peak_memory_stats()
    print(
        f"[TRAIN] {epochs}에폭 × {len(rows)}건 | 유효배치 {batch_size}×{accum}={batch_size * accum} | "
        f"optim step {total_optim_steps} (warmup {warmup_steps}) | grad_ckpt={grad_ckpt}"
    )
    t0 = time.time()
    micro = 0
    ostep = 0
    win_loss: list[float] = []
    for ep in range(epochs):
        rng.shuffle(encoded)
        for start in range(0, len(encoded), batch_size):
            ids, am, labels = collate(encoded[start:start + batch_size], tok.pad_token_id)
            ids, am, labels = ids.to("cuda"), am.to("cuda"), labels.to("cuda")
            out = model(input_ids=ids, attention_mask=am, labels=labels)
            (out.loss / accum).backward()
            win_loss.append(out.loss.item())
            micro += 1
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                sched.step()
                opt.zero_grad()
                ostep += 1
                if ostep % logging_steps == 0:
                    avg = sum(win_loss) / len(win_loss)
                    win_loss = []
                    print(f"  step {ostep}/{total_optim_steps} | loss {avg:.4f} | lr {sched.get_last_lr()[0]:.2e}")
                if save_steps > 0 and ostep % save_steps == 0:
                    save_checkpoint(model, out_root / f"checkpoint-{ostep}", kept_ckpts, save_total_limit)
                    print(f"  체크포인트 저장: {out_root / f'checkpoint-{ostep}'}")
        print(f"  epoch {ep + 1}/{epochs} 완료")
    # 남은 그래디언트 flush
    if micro % accum != 0:
        opt.step()
        opt.zero_grad()

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"학습 완료: {time.time() - t0:.1f}s | peak VRAM {peak:.2f} GB")

    # ── 7) 최종 LoRA 어댑터 저장 ──
    final = out_root / "lora_adapter"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final))
    tok.save_pretrained(str(final))
    print(f"LoRA 어댑터 저장 완료: {final}")


if __name__ == "__main__":
    main()
