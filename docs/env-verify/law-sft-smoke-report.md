# Law SFT Data Smoke Report

실행일: 2026-07-18

## Summary

5개 법령 코퍼스(`data/processed/laws.json`)에서 teacher 없이 deterministic SFT JSONL을 생성했다.
모든 positive 답변은 `citation_verify`로 exact substring 인용 검증을 통과해야만 파일에 저장된다.

이 산출물은 정식 릴리스 학습셋이 아니라, 다법령 grounder 재학습을 시작하기 위한 smoke 학습셋이다.

## Command

```powershell
python scripts/data/gen_law_sft.py `
  --corpus data/processed/laws.json `
  --out data/processed/law_sft.jsonl `
  --max-full 1200 `
  --max-tight 1200 `
  --refusal-ratio 0.2
```

## Counts

| label | rows |
|---|---:|
| `grounded_full` | 1200 |
| `grounded_tight` | 1200 |
| `refusal` | 600 |
| total | 3000 |

All rows use `k=5` context items.

## Validation

```powershell
python scripts/data/gen_law_sft.py --corpus data/processed/laws.json --out data/processed/law_sft.jsonl --max-full 1200 --max-tight 1200 --refusal-ratio 0.2
python scripts/smoke.py --skip-eval-demos
```

Both passed locally. The generated file is ignored by git because it is a data artifact.

## Training Entry Point

```powershell
python scripts/02_train_sft.py --config configs/train_law_sft.yaml
```

The first training run should be treated as a smoke adapter, not a release model.

## Holdout Variant

To avoid training directly on the same answerable IDs used by `eval/laws_runner.smoke.json`, generate a holdout
variant:

```powershell
python scripts/data/gen_law_sft.py `
  --corpus data/processed/laws.json `
  --out data/processed/law_sft_holdout.jsonl `
  --max-full 1200 `
  --max-tight 1200 `
  --refusal-ratio 0.2 `
  --exclude-questions eval/laws_runner.smoke.json
```

This excluded `30` positive IDs and produced:

| label | rows |
|---|---:|
| `grounded_full` | 1200 |
| `grounded_tight` | 1200 |
| `refusal` | 600 |
| total | 3000 |

Training:

```powershell
python scripts/02_train_sft.py --config configs/train_law_holdout.yaml
```

The training script dropped `46` examples whose assistant labels were fully truncated at `max_seq_length=1536`,
then trained `2954` rows for two epochs. Peak VRAM was `5.72 GB`.

