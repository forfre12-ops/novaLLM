# Power Report

정식 다법령 G0 전에 필요한 표본수의 하한을 잡기 위한 보수적 계산이다.

## Method

- 계산: equal-size unpaired two-proportion normal approximation
- 목적: paired transcript가 없을 때 다법령 확장 N의 하한을 잡는 planning 도구
- 정식 판정: 결과 transcript가 생기면 paired exact McNemar + Holm 보정을 정본으로 사용

## Current v0.2 Observed Gap

`g0-faithbench-v02-result.json` 기준:

- base 7B selection_exact: `0.387`
- FT 1.5B selection_exact: `0.742`
- observed diff: `+0.355`

이 큰 관측 차이만 검증한다면 필요한 N은 작다. 하지만 정식 G0는 더 작은 margin, 다법령 corpus,
tight span/leak 복합 지표를 포함해야 하므로 아래 margin table을 planning 기준으로 본다.

```powershell
python scripts/eval/power_analysis.py --base 0.387 --target 0.742
```

## Planning Table

| target diff | target p | required n/model | power at n=93 |
|---:|---:|---:|---:|
| 0.05 | 0.437 | 1520 | 0.103 |
| 0.10 | 0.487 | 386 | 0.282 |
| 0.15 | 0.537 | 173 | 0.546 |
| 0.20 | 0.587 | 97 | 0.795 |
| 0.30 | 0.687 | 43 | 0.990 |
| 0.35 | 0.737 | 31 | 0.999 |

## Implication

- 헌법 93문항은 큰 차이(`~0.30+`)를 보기에는 충분하지만, 작은 차이(`0.05~0.10`)를 검증하기에는 부족하다.
- 다법령 curated 질문셋의 1차 목표는 최소 `N≈200` answerable 이상으로 잡는 것이 안전하다.
- `partial`과 `leak`까지 복합 primary로 묶을 경우 effective N이 줄 수 있으므로, 실제 작성 목표는
  answerable `300~500`이 더 현실적이다.
