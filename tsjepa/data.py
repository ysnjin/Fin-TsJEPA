"""data/cache_sp1500 (845종목 일별 OHLCV parquet, 1985-2025)을 읽는다."""
import numpy as np
import pandas as pd

from .config import CACHE_DIR

JUMP_HI, JUMP_LO = 3.0, -0.9


def list_tickers():
    return sorted(p.name[:-len(".parquet")] for p in CACHE_DIR.glob("*.parquet"))


def load_ticker_raw(ticker):
    df = pd.read_parquet(CACHE_DIR / f"{ticker}.parquet")
    return df[~df.index.duplicated(keep="first")].sort_index()


def _clean_split_artifacts(df):
    """일부 종목(예: CHRD)은 역분할/회생 때문에 하루 수백~수만 % 점프가 조정되지 않은 채
    남아 있다. 점프 이전 가격 전체에 점프 비율을 곱해 수동으로 보정한다."""
    close = df["close"].values.astype(float)
    if len(close) < 2:
        return df
    price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    vals = {c: df[c].values.astype(float) for c in price_cols}
    for i in range(1, len(close)):
        c0, c1 = close[i - 1], close[i]
        if not (np.isfinite(c0) and np.isfinite(c1)) or c0 == 0:
            continue
        r = c1 / c0 - 1.0
        if r > JUMP_HI or r < JUMP_LO:
            factor = c1 / c0
            for c in price_cols:
                vals[c][:i] *= factor
            close[:i] *= factor
    out = df.copy()
    for c in price_cols:
        out[c] = vals[c]
    return out


def load_ticker_ohlcv(ticker):
    return _clean_split_artifacts(load_ticker_raw(ticker))


def build_close_panel(tickers, start, end, min_rows=500):
    closes, kept, dropped = {}, [], []
    for tk in tickers:
        s = load_ticker_ohlcv(tk)["close"]
        s = s[(s.index >= start) & (s.index <= end)]
        if len(s) < min_rows:
            dropped.append(tk)
            continue
        closes[tk] = s
        kept.append(tk)
    return pd.DataFrame(closes).sort_index(), kept, dropped
