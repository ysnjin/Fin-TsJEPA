"""Backtest engine: 12-1 momentum, top-decile selector, z-blend, metrics.

Re-implemented from the team FinJEPA repository's 30.Strategy/momentum.py and hybrids.py
(same conventions: 21-day rebalance, top 10% equal weight, 10bps cost).
"""
import numpy as np
import pandas as pd

TC = 0.001          # 10bps per turnover
REBAL = 21          # 21-trading-day rebalance
DECILE = 0.10       # top-decile
WARMUP = 252        # momentum lookback warmup


def momentum_scores(px, skip=21, lookback=252):
    """12-1 momentum: t-252 ~ t-21 return."""
    return px.shift(skip) / px.shift(lookback) - 1.0


def backtest(px, select_fn, tc=TC, rebal=REBAL, warmup=WARMUP):
    """Monthly-rebalanced long-only backtest.

    select_fn(t, valid) -> weight vector (N,) or None (keep previous).
      t: integer rebalance-day index, valid: bool (N,) valid-name mask.
    Returns dict(daily, turnover, dates).
    """
    rets = px.pct_change().values
    T, N = rets.shape
    w = np.zeros(N)
    daily, turns = [], []
    for t in range(warmup, T - 1):
        turnover = 0.0
        if (t - warmup) % rebal == 0:
            valid = np.isfinite(rets[t])
            neww = select_fn(t, valid)
            if neww is not None:
                turnover = 0.5 * np.abs(neww - w).sum()
                w = neww
        g = float(np.nansum(w * rets[t + 1]))
        daily.append(g - turnover * tc)
        turns.append(turnover)
    return dict(daily=np.array(daily), turnover=np.array(turns),
                dates=px.index[warmup + 1:])


def make_momentum_selector(scores, decile=DECILE):
    """Momentum top-decile equal-weight selector."""
    S = scores.values

    def select(t, valid):
        s = S[t]
        m = np.isfinite(s) & valid
        if m.sum() < 50:
            return None
        k = max(1, int(m.sum() * decile))
        sv = np.where(m, s, -np.inf)
        order = np.argsort(sv)[::-1]
        pick = order[:k]
        w = np.zeros(len(s))
        w[pick] = 1.0 / len(pick)
        return w
    return select


def zscore(x, mask):
    v = x[mask]
    mu, sd = np.nanmean(v), np.nanstd(v) + 1e-12
    return (x - mu) / sd


def make_blend_selector(mom_scores, ai_scores_by_grid, lam_series,
                         decile=DECILE, warmup=WARMUP, rebal=REBAL):
    """z(momentum) + lam * z(ai) top-decile selector.

    Grid g corresponds to rebalance day t = warmup + g*rebal (same grid
    convention used to build ai_scores_by_grid / lam_series), so this must
    only be called at t's satisfying (t-warmup) % rebal == 0 (i.e. as the
    select_fn passed to engine.backtest, which calls it exactly there).
    ai_scores_by_grid: dict grid g -> (N,) ai score array (NaN allowed).
    lam_series: array (G+1,) walk-forward-selected lambda per grid.
    """
    S = mom_scores.values

    def select(t, valid):
        g = (t - warmup) // rebal
        lam = lam_series[min(g, len(lam_series) - 1)]
        s = S[t]
        m = np.isfinite(s) & valid
        if m.sum() < 50:
            return None
        k = max(1, int(m.sum() * decile))
        mz = zscore(np.where(m, s, np.nan), m)
        a = ai_scores_by_grid.get(g)
        if a is None or lam == 0.0:
            blend = np.where(m, mz, -np.inf)
        else:
            pm = m & np.isfinite(a)
            az = zscore(np.where(pm, a, np.nan), pm)
            az = np.where(np.isfinite(az), az, 0.0)
            blend = np.where(m, mz + lam * az, -np.inf)
        order = np.argsort(blend)[::-1]
        pick = order[:k]
        w = np.zeros(len(s))
        w[pick] = 1.0 / len(pick)
        return w
    return select


# ---- metrics -----------------------------------------------------------
def all_metrics(daily):
    d = np.asarray(daily, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return dict(ARR=np.nan, SR=np.nan, MDD=np.nan, VOL=np.nan)
    arr = float((np.prod(1 + d) ** (252 / len(d)) - 1) * 100)
    vol = float(d.std(ddof=1) * np.sqrt(252) * 100)
    sr = float(d.mean() / (d.std(ddof=1) + 1e-12) * np.sqrt(252))
    curve = np.cumprod(1 + d)
    peak = np.maximum.accumulate(curve)
    mdd = float((curve / peak - 1).min() * 100)
    return dict(ARR=arr, SR=sr, MDD=mdd, VOL=vol)


def yearly_arr(daily, dates, min_days=30):
    s = pd.Series(daily, index=dates).dropna()
    out = {}
    for y, grp in s.groupby(s.index.year):
        if len(grp) < min_days:
            continue
        out[int(y)] = float((np.prod(1 + grp.values) ** (252 / len(grp)) - 1) * 100)
    return out


def walk_forward_lambda(mom_scores, ai_scores_by_grid, close_grid, valid_grid,
                         grid_dates, lam_grid=(0.0, 0.25, 0.5, 1.0),
                         min_hist=24, decile=DECILE, tc=TC, default_lam=0.25):
    """Pick lambda per rebalance grid using only past pseudo-returns
    (expanding-window Sharpe of each lambda's candidate portfolio), exactly
    as in the team repository's A12.WideUni/run_wide.py walk-forward procedure.
    """
    G = len(grid_dates) - 1
    S_v = mom_scores
    fwd = np.full_like(close_grid, np.nan)
    fwd[:-1] = close_grid[1:] / close_grid[:-1] - 1.0

    picks = {}
    vret = {}
    for lam in lam_grid:
        pk = []
        v = np.full(G + 1, np.nan)
        prev = None
        for g in range(G + 1):
            s = S_v[g]
            m = np.isfinite(s) & valid_grid[g]
            if m.sum() < 50:
                pk.append(None)
                prev = None
                continue
            k = max(1, int(m.sum() * decile))
            a = ai_scores_by_grid.get(g)
            if lam == 0.0 or a is None or np.isfinite(a[m]).sum() < 30:
                b = np.where(m, s, -np.inf)
            else:
                mz = zscore(np.where(m, s, np.nan), m)
                pm = m & np.isfinite(a)
                az = zscore(np.where(pm, a, np.nan), pm)
                az = np.where(np.isfinite(az), az, 0.0)
                b = np.where(m, mz + lam * az, -np.inf)
            pick = np.argsort(b)[::-1][:k]
            pk.append(pick)
            if g < G:
                f = fwd[g][pick]
                r = float(np.nanmean(f)) if np.isfinite(f).any() else 0.0
                ov = (len(set(pick) & set(prev)) / max(len(pick), 1)
                      if prev is not None else 0.0)
                v[g] = r - tc * (1.0 - ov)
            prev = pick
        picks[lam] = pk
        vret[lam] = v

    def sharpe(v):
        v = np.asarray(v)
        v = v[np.isfinite(v)]
        if len(v) < 8:
            return -np.inf
        return float(v.mean() / (v.std() + 1e-9))

    lam_series = []
    cur = default_lam
    for g in range(G + 1):
        if g >= min_hist:
            best_s, best_lam = -np.inf, cur
            for lam in lam_grid:
                sh = sharpe(vret[lam][:g])
                if sh > best_s:
                    best_s, best_lam = sh, lam
            cur = best_lam
        lam_series.append(cur)
    return np.array(lam_series), picks
