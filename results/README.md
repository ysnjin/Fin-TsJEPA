# 실험 결과

모든 수치는 845종목(data/cache_sp1500), 1990-2025 구간, 학습 인코더는 seed 7/42/99 평균입니다. `python scripts/run_backtest.py`로 이 폴더의 결과를 그대로 다시 만들 수 있습니다.

| 파일 | 내용 |
|---|---|
| `backtest_summary.json` | 전체 성과 지표, seed별 지표, 모든 검정 결과 |
| `backtest_daily.npz` | 각 전략의 일별 수익률 시계열 |
| `backtest_log.txt` | 백테스트 실행 로그 |
| `pretrain_ema_log.txt` | EMA 사전학습 로그 (학습률 탐색, seed별 검증 지표) |

## 백테스트 규약

- M: 12-1 momentum 상위 10% 동일가중, 21거래일 리밸런스, 거래비용 10bp
- AI 점수: 인코더 표현 → ridge probe로 21일 후 초과수익 예측 (expanding window, embargo 42거래일)
- BLEND: z(momentum) + λ·z(AI 점수) 상위 10%, λ ∈ {0, 0.25, 0.5, 1}은 과거 성과로 walk-forward 선택
- RAW: 인코더 없이 22개 지표를 그대로 ridge probe에 넣음
- 검정: 월간 수익 차이의 Newey-West t-검정, ARR 차이 95% 신뢰구간은 stationary bootstrap (5,000회)

## 1. 성과

| 전략 | ARR | Sharpe | MDD | 변동성 |
|---|---|---|---|---|
| M (momentum 단독) | 27.31% | 1.11 | -61.6% | 24.6% |
| BLEND-RAW (인코더 없음) | 38.92% | 1.37 | -57.3% | 26.5% |
| BLEND-ts-random | 33.14% | 1.25 | -60.8% | 25.6% |
| BLEND-ts-sigreg-trained | 34.18% | 1.27 | -57.9% | 25.7% |
| **BLEND-ts-ema-trained (제안)** | **36.35%** | **1.33** | **-57.3%** | **25.8%** |
| BLEND-orig-random | 38.21% | 1.38 | -58.2% | 26.0% |
| BLEND-orig-sigreg-trained (팀 원본 모델) | 36.85% | 1.34 | -58.1% | 25.9% |
| BLEND-orig-ema-trained | 37.78% | 1.36 | -60.4% | 26.1% |

seed별 ARR

| 전략 | seed 7 | seed 42 | seed 99 |
|---|---|---|---|
| BLEND-ts-random | 32.99% | 32.29% | 33.87% |
| BLEND-ts-sigreg-trained | 36.12% | 33.49% | 32.73% |
| BLEND-ts-ema-trained | 36.17% | 37.60% | 35.01% |
| BLEND-orig-random | 38.05% | 40.04% | 36.32% |
| BLEND-orig-sigreg-trained | 38.12% | 37.42% | 34.81% |
| BLEND-orig-ema-trained | 38.36% | 38.06% | 36.70% |

## 2. momentum 대비

모든 BLEND 조합이 M을 유의하게 이깁니다.

| 전략 vs M | ARR 차이 | 95% CI | t | p |
|---|---|---|---|---|
| BLEND-RAW | +11.64%p | [7.51, 16.21] | 5.08 | 3.8e-07 |
| BLEND-ts-random | +5.84%p | [3.03, 8.83] | 4.01 | 6.2e-05 |
| BLEND-ts-sigreg-trained | +6.88%p | [4.00, 9.91] | 4.98 | 6.3e-07 |
| BLEND-ts-ema-trained | +9.05%p | [5.72, 12.62] | 5.34 | 9.2e-08 |
| BLEND-orig-random | +10.92%p | [7.19, 15.05] | 5.40 | 6.6e-08 |
| BLEND-orig-sigreg-trained | +9.54%p | [6.07, 13.48] | 5.09 | 3.6e-07 |
| BLEND-orig-ema-trained | +10.50%p | [6.80, 14.52] | 5.39 | 7.2e-08 |

## 3. 학습된 인코더 vs 무작위 인코더 (핵심)

| | SIGReg 학습 | EMA 학습 |
|---|---|---|
| 원본 인코더 | -1.38%p, CI [-2.90, 0.08], p=0.092 (구분 불가) | -0.42%p, CI [-1.87, 1.05], p=0.66 (구분 불가) |
| TS 인코더 | +1.04%p, CI [-0.73, 2.75], p=0.247 (구분 불가) | **+3.21%p, CI [1.33, 5.08], p=0.001 (유의)** |

학습이 무작위보다 유의하게 나은 조합은 TS + EMA 하나뿐입니다.

## 4. 방식 간 비교, RAW 대비

| 비교 | ARR 차이 | 95% CI | p |
|---|---|---|---|
| TS + EMA vs 팀 원본 모델 (orig + SIGReg) | -0.50%p | [-2.35, 1.32] | 0.590 |
| 원본 + EMA vs 팀 원본 모델 (orig + SIGReg) | +0.96%p | [-0.37, 2.32] | 0.117 |
| TS + EMA vs RAW | -2.59%p | [-5.55, 0.17] | 0.059 |
| TS + SIGReg vs RAW | -4.76%p | [-7.95, -1.85] | 0.003 |
| 원본 + EMA vs RAW | -1.14%p | [-3.61, 1.28] | 0.313 |
| 원본 + SIGReg vs RAW | -2.10%p | [-4.80, 0.47] | 0.074 |

- TS + EMA는 팀 원본 모델과 통계적으로 구별되지 않습니다.
- EMA로 바꾸는 것만으로 원본 인코더가 유의하게 나아지지는 않습니다.
- 인코더를 쓰는 어떤 조합도 RAW를 유의하게 이기지 못했습니다.

## 5. 사전학습 지표

검증 구간 2024-2025의 다음날 표현 예측 오차(val_pred), seed 3개 평균.

| | SIGReg | EMA |
|---|---|---|
| 원본 인코더 | 0.1075 | 0.0297 (게으른 예측 대비 0.850배) |
| TS 인코더 | 0.0926 (원본보다 13.9% 낮음) | 0.0277 (게으른 예측 대비 0.398배) |

- SIGReg와 EMA는 표현 공간의 크기가 달라 val_pred 값을 서로 직접 비교하면 안 됩니다. EMA 쪽은 "내일 표현 = 오늘 표현" 예측 오차 대비 비율로 보는 것이 맞습니다.
- EMA 학습 인코더의 유효 차원(eff_rank)은 원본 5.6~6.6, TS 5.3~7.2로 팀 학습 점검 기준(유효 차원 > 8)에 못 미칩니다. SIGReg처럼 차원을 고르게 쓰도록 강제하는 항이 없어, 표현이 64차원 중 일부에 몰려 있다는 뜻입니다. 세부 값은 `pretrain_ema_log.txt`.
