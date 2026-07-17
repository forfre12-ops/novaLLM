# Contributing

`nova-llm` is currently a prototype for deterministic Korean legal citation evaluation.
The project is more interested in reproducible measurement than in model-win claims.

## What Contributions Help Most

- Parser fixes for real `open.law.go.kr` responses
- New deterministic scorer tests
- Reproducible model result transcripts
- Curated Korean legal questions with exact source IDs and spans
- Documentation fixes that make claims more precise

## What Not To Submit

- Claims that a small model generally beats a large model
- LLM-judge-only scores without deterministic evidence
- Large model weights or raw private data
- Non-commercial or license-unclear training data

## Local Smoke

```powershell
python scripts/smoke.py
```

This should pass without GPU, API keys, or network access.

## Reporting A Model Result

Please include:

- model name and version
- command used
- corpus and question files
- result JSON
- transcript JSONL if available
- git commit hash

For paired comparisons, transcript-based McNemar is preferred over unpaired tests.

## Data Contributions

Curated question items should avoid article-ID hints in the question text.

Faithbench question format:

```json
{
  "법령명 제1조 ①": "내용 기반 질문"
}
```

Partial-span question format:

```json
[
  {
    "id": "법령명 제1조 ①",
    "question": "특정 부분을 묻는 질문",
    "gold_span": "해당 조문 안에 실제로 존재하는 exact substring"
  }
]
```

The `gold_span` must be an exact substring after scorer normalization.
