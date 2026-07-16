"""02_train_sft.py — QLoRA SFT 학습 (unsloth, 단일 GPU 16GB)

한국어 instruction 데이터로 베이스 모델을 QLoRA 파인튜닝한다.
RTX 5070 Ti 16GB(Blackwell) 기준. 설정은 configs/train_config.yaml.

주의: unsloth/trl API는 버전에 따라 바뀐다. 이 스크립트는 시작 템플릿이며,
설치된 버전에서 에러가 나면 unsloth 공식 예제의 최신 시그니처에 맞춰 조정할 것.

실행:
    python scripts/02_train_sft.py --config configs/train_config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # unsloth는 transformers보다 먼저 import 해야 최적화 패치가 적용된다.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    # 1) 베이스 모델 4bit 로드 (QLoRA — 16GB에 필수)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,             # 자동 (Blackwell은 bf16)
        load_in_4bit=cfg.get("load_in_4bit", True),
    )

    # 2) LoRA 어댑터 부착
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",   # VRAM 절약
        random_state=cfg["seed"],
    )

    # 3) 채팅 템플릿 + 데이터 포맷
    tokenizer = get_chat_template(tokenizer, chat_template=cfg["chat_template"])

    def format_chat(examples: dict) -> dict:
        texts = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
            for c in examples["messages"]
        ]
        return {"text": texts}

    dataset = load_dataset("json", data_files=cfg["train_file"], split="train")
    dataset = dataset.map(format_chat, batched=True)

    # 4) 학습
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        args=SFTConfig(
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            warmup_ratio=0.03,
            num_train_epochs=cfg["epochs"],
            learning_rate=float(cfg["learning_rate"]),
            bf16=True,
            logging_steps=10,
            optim="adamw_8bit",          # VRAM 절약
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=cfg["seed"],
            output_dir=cfg["output_dir"],
            report_to=cfg.get("report_to", "none"),
        ),
    )
    trainer.train()

    # 5) LoRA 어댑터 저장
    out = Path(cfg["output_dir"]) / "lora_adapter"
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"LoRA 어댑터 저장 완료: {out}")


if __name__ == "__main__":
    main()
