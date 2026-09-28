# TS-JEPA Proposal

FinJEPA의 인코더를 시계열을 고려한 구조(causal 1D-CNN, 최근 9일 참조)로 바꾸고, collapse 방지를 SIGReg 대신 EMA target encoder로 학습한 제안 모델입니다. 이 폴더 하나만으로 데이터 준비, 학습, 백테스트를 모두 실행할 수 있고, 실험 결과도 `results/`에 들어 있습니다.

## 무엇이 바뀌었나

| 항목 | 팀 원본 FinJEPA | 이 제안 |
|---|---|---|
| 인코더 | `PriceEncoder`: 하루치 22개 지표를 독립적으로 인코딩 | `TSPriceEncoder`: 오늘 + 직전 8일을 causal conv로 보고 인코딩 |
| collapse 방지 | SIGReg + detach (단일 인코더) | EMA target encoder (m=0.996), SIGReg 없음 |
| predictor, 입력 22개 지표, 30일→10일 창, 백테스트 규약 | 그대로 | 그대로 |

## 구성

```
TS_JEPA_Proposal/
├─ data/cache_sp1500/   845종목 일별 OHLCV (팀 저장소의 data/cache_sp1500과 같은 파일, 약 235MB)
├─ checkpoints/         학습 완료된 인코더 가중치 (seed 7/42/99)
│   ├─ {ts,orig}_ema_trained_s*.pt      EMA target encoder로 학습
│   └─ {ts,orig}_sigreg_trained_s*.pt   팀 원본 방식(SIGReg + detach)으로 학습
├─ results/             실험 결과 (아래 "결과" 참고)
├─ tsjepa/
│   ├─ encoder_ts.py    TSPriceEncoder (causal 1D-CNN, 수용 범위 9일)   ← 새로 추가
│   ├─ ema.py           FinJEPA_EMA (context/target encoder, EMA 갱신)   ← 새로 추가
│   ├─ finjepa_orig.py  팀 원본 PriceEncoder, TransformerPredictor (수정 없이 옮김)
│   ├─ features.py      22개 기술지표
│   ├─ data.py          데이터 읽기 (역분할 등으로 남은 가격 점프 보정 포함)
│   ├─ grid.py          리밸런스 시점별 지표 창, 연도 인과 정규화, 인코딩, 블렌드
│   ├─ engine.py        momentum / 블렌드 백테스트 (21거래일 리밸런스, 상위 10%, 10bp)
│   ├─ probe.py         expanding-window ridge probe (embargo 42거래일)
│   └─ stats_tests.py   Newey-West 검정, bootstrap 신뢰구간
└─ scripts/
    ├─ run_backtest.py  M / RAW / 무작위 / SIGReg 학습 / EMA 학습 인코더 비교
    ├─ build_panel.py   학습용 패널 생성
    └─ train_ema.py     EMA 사전학습 (GPU 권장)
```

## 1. 결과 재현 (학습 불필요, CPU 약 12분)

```bash
pip install -r requirements.txt
python scripts/run_backtest.py
```

포함된 가중치로 모든 비교를 다시 계산해 `results/backtest_summary.json`, `results/backtest_daily.npz`에 저장합니다. 일부만 보려면 `--arch ts`, `--method ema`처럼 지정합니다.

## 2. 직접 다시 학습하기 (GPU 권장)

```bash
python scripts/build_panel.py                            # -> panels/panel_full845.npz (약 450MB)
python scripts/train_ema.py --arch ts                    # -> checkpoints_new/ts_ema_trained_s{7,42,99}.pt
python scripts/train_ema.py --arch orig                  # 원본 인코더를 EMA로 학습 (비교용)
python scripts/run_backtest.py --method ema --ckpt-dir checkpoints_new --out-dir results_new
```

- 기본 학습률은 TS 1e-5, 원본 5e-5입니다. 제공된 체크포인트도 이 값으로 학습했습니다.
- `--scan`을 주면 학습률 5개(5e-4 ~ 1e-5)를 먼저 비교하고 가장 좋은 값으로 학습합니다.
- Colab에서는 이 저장소를 clone한 뒤 같은 명령을 그대로 실행하면 됩니다. A100 기준 seed당 약 4분입니다.
- SIGReg 체크포인트는 팀 FinJEPA 학습 코드에서 인코더만 `TSPriceEncoder`로 바꿔 학습한 것입니다. 이 저장소에는 EMA 학습 코드만 들어 있습니다.

## 결과

845종목, 1990-2025, seed 3개 평균. 자세한 내용은 [`results/README.md`](results/README.md).

| | 무작위 | SIGReg 학습 | EMA 학습 |
|---|---|---|---|
| 원본 인코더 | 38.21% | 36.85% (팀 원본 모델) | 37.78% |
| TS 인코더 | 33.14% | 34.18% | **36.35%** |

M(momentum 단독) 27.31%, BLEND-RAW(인코더 없음) 38.92%. 모든 BLEND 조합이 M을 유의하게 이깁니다 (p < 0.001).

- 학습 vs 무작위가 유의한 조합은 **TS + EMA 하나뿐**입니다 (+3.21%p, p=0.001). 나머지 세 조합은 구분되지 않습니다.
- TS + EMA는 팀 원본 모델과 통계적으로 같은 수준이고 (p=0.59), BLEND-RAW에는 근소하게 못 미칩니다 (p=0.059).

## 팀 FinJEPA 코드에 직접 넣어 쓰려면

검증에는 필요 없고, 팀의 학습·백테스트 코드에서 TS-JEPA를 쓰고 싶을 때만 해당합니다.

| 위치 | 바꿀 내용 |
|---|---|
| 모델 생성 | 인코더를 `TSPriceEncoder`로, collapse 방지를 SIGReg에서 EMA target encoder로 교체 (`tsjepa/ema.py` 참고) |
| 백테스트 인코딩 | 하루치 `(N, 22)` 대신 그날로 끝나는 9일 창 `(N, 9, 22)`을 넣고 마지막 날 출력 `z[:, -1, :]`을 사용 (`tsjepa/grid.py` 참고) |
