# Law Runner Smoke Report

실행일: 2026-07-18

## Summary

`LAW_API_KEY`로 수집한 5개 법령 코퍼스를 `run_g0_faithbench.py`에 연결해 end-to-end GPU smoke를 실행했다.
이 실행은 정식 G0 판정이 아니라, 다법령 closed-set 데이터가 모델 러너와 결정적 채점기를 통과하는지 확인하기 위한
호환성 검증이다.

핵심 결론:

- 코퍼스/질문/러너/전사 저장 경로는 정상 동작했다.
- 기존 `g0-pilot` 1.5B FT 어댑터는 헌법 파일럿에 맞춰진 상태라, 다법령 smoke에서는 과잉거절 및 무인용 응답으로 무너졌다.
- 7B base는 selection_exact에서 가장 높았지만, unanswerable leak_rate가 높아 공개 주장에는 부적합하다.
- 따라서 다음 공개 주장은 모델 승패가 아니라 "한국어 법령 인용 행동을 측정하는 프로파일러"여야 한다.

## Inputs

- corpus: `data/processed/laws.json`
- corpus SHA256: `c25df3c753bfa177058cb9a8656e89aa0331a9c8ead8cf5a05c2b940a3808f30`
- questions: `eval/laws_runner.smoke.json`
- questions SHA256: `6f99a31dcda473657759176810737f0f2eece6495ef1036a00276b5f20033730`
- answerable: `30`
- unanswerable: `10`
- k: `5`
- scorer version: `0.2`
- git rev: `7d23a716ae42cd2980fd64e94c369fe7ac07aa82`

## Results

| model | selection_exact | gold_recall | distractor_cite_rate | answerable_no_citation_rate | answerable_refused_rate | refusal_rate | leak_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_small_fewshot` | 0.100 | 0.167 | 0.467 | 0.200 | 0.000 | 0.500 | 0.500 |
| `ft_small_zeroshot` | 0.000 | 0.000 | 0.000 | 0.933 | 0.467 | 1.000 | 0.000 |
| `base_large_fewshot` | 0.367 | 0.367 | 0.033 | 0.433 | 0.033 | 0.600 | 0.400 |

All 30 answerable items are unseen relative to the constitution pilot split, so the `seen` split is empty and skipped by
`faithbench_stats.py`.

## Statistical Check

```powershell
python scripts/eval/faithbench_stats.py --result docs/env-verify/law-runner-smoke-result.json
```

Paired exact McNemar over answerable instances:

- `ft_small_zeroshot` vs `base_large_fewshot`: `n_pairs=30`, FT-only `0`, base-only `11`, `p=0.001`, Holm-adjusted `p=0.002`.
- `ft_small_zeroshot` vs `base_small_fewshot`: `n_pairs=30`, FT-only `0`, base-only `3`, `p=0.25`, Holm-adjusted `p=0.25`.

This confirms the old FT adapter should not be presented as a multi-law winner.

## Interpretation

The result is useful because it exposes a real failure mode before publication: a format-trained or narrow-domain adapter can appear
strong on the original pilot but collapse when the closed set expands. That strengthens the benchmark story, because the tool is
catching the project's own model weakness rather than rubber-stamping it.

Do not use this run as a benchmark leaderboard. The smoke questions are generated heuristically and include weak prompts such as
deleted provisions or awkward article headings. Their purpose is pipeline verification, not a defensible public score.

## Next Actions

1. Generate a multi-law SFT dataset from the 5-law corpus with positive, refusal, and distractor-heavy examples.
2. Train a fresh 1.5B smoke adapter only after the generated records pass citation-substring validation.
3. Promote a small curated question set for formal G0; keep generated smoke sets out of public claims.
4. Ship the first public artifact as a citation fingerprint/profiler report, not a "small beats large" claim.

