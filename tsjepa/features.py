"""일별 기술지표 22개 (SMA/EMA 비율, ROC, Stochastic %K/%D, MACD, RSI)."""
import numpy as np
import pandas as pd

SMA_WINDOWS = [3, 5, 13, 21]
EMA_WINDOWS = [3, 5, 13, 21]
ROC_WINDOWS = [13, 21]
STOCH_WINDOWS = [7, 14, 21]
RSI_WINDOWS = [9, 14, 21]

FEATURE_COLS22 = (
    [f"sma{n}" for n in SMA_WINDOWS]
    + [f"ema{n}" for n in EMA_WINDOWS]
    + [f"roc{n}" for n in ROC_WINDOWS]
    + [f"stoch_k{n}" for n in STOCH_WINDOWS]
    + [f"stoch_d{n}" for n in STOCH_WINDOWS]
    + ["macd", "macd_signal", "macd_hist"]
    + [f"rsi{n}" for n in RSI_WINDOWS]
)
assert len(FEATURE_COLS22) == 22


def _rsi(close, n):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def compute_22_features(df):
    close, high, low = df["close"], df["high"], df["low"]
    out = {}
    for n in SMA_WINDOWS:
        out[f"sma{n}"] = close / close.rolling(n).mean() - 1.0
    for n in EMA_WINDOWS:
        out[f"ema{n}"] = close / close.ewm(span=n, adjust=False).mean() - 1.0
    for n in ROC_WINDOWS:
        out[f"roc{n}"] = close / close.shift(n) - 1.0
    stoch_k = {}
    for n in STOCH_WINDOWS:
        hh, ll = high.rolling(n).max(), low.rolling(n).min()
        k = (close - ll) / (hh - ll + 1e-12) * 100.0
        stoch_k[n] = k
        out[f"stoch_k{n}"] = k
    for n in STOCH_WINDOWS:
        out[f"stoch_d{n}"] = stoch_k[n].rolling(3).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = (ema12 - ema26) / close
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_line - macd_signal
    for n in RSI_WINDOWS:
        out[f"rsi{n}"] = _rsi(close, n)
    feat = pd.DataFrame(out, index=df.index)[FEATURE_COLS22]
    return feat.replace([np.inf, -np.inf], np.nan)
