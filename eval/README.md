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

## K-FaithBench 확장

현재 프로토타입은 헌법 seed 코퍼스의 curated 질문을 기본으로 쓴다. 다법령 코퍼스로 키울 때는
질문셋을 별도 JSON으로 두고 `--questions`로 주입한다.

헌법 seed 코퍼스 전체용 curated 질문셋은 `eval/questions.constitution.json`에 있다.

```powershell
python scripts/eval/faithbench.py --questions eval/questions.constitution.json --near --out eval/instances.jsonl
```

모델 교차비교도 같은 질문셋으로 돌릴 수 있다.

```powershell
python scripts/train/run_g0_faithbench.py --questions eval/questions.constitution.json --k 5 --near
```

질문셋 형식은 아래 둘 중 하나다.

```json
{
  "헌법 제1조 ①": "대한민국의 국가 형태는 무엇인가?"
}
```

```json
[
  {"id": "헌법 제1조 ①", "question": "대한민국의 국가 형태는 무엇인가?"}
]
```

`--include-all-corpus`는 질문이 없는 모든 조항을 ID 기반 sanity 질문으로 채우는 옵션이다. 표본 수를
빠르게 늘리는 smoke에는 유용하지만, 질문에 조항ID가 노출되므로 정식 selection_exact 리포트에는
curated 질문셋을 우선한다.
