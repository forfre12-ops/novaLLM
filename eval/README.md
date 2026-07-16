# eval — 평가 하네스

한국어 LLM 평가 코드/설정을 여기에 둔다. (결과 산출물은 git 무시 대상)

## 벤치마크

| 벤치마크 | 측정 | 용도 |
|----------|------|------|
| **LogicKor** | 한국어 추론/멀티턴 | 종합 품질 |
| **KMMLU** | 한국어 지식(45개 과목) | 지식 능력 |
| **HAERAE-Bench** | 한국 문화/역사/언어 | 한국 특화 |
| **KoBEST** | 한국어 이해 | 기초 NLU |
| **Ko-Arena / LMSYS** | 사람 선호 | 실사용 품질 |

## 유명세 경로

- **Open Ko-LLM Leaderboard** 제출 → 상위권이면 화제성 (Upstage SOLAR 사례)
- 단, 리더보드 오버핏/데이터 오염 주의 — 실사용 품질과 괴리되면 역효과

## lm-evaluation-harness 사용 예

```bash
pip install lm-eval
lm_eval --model hf \
  --model_args pretrained=models/output/merged \
  --tasks kmmlu,haerae \
  --device cuda:0 --batch_size auto
```
