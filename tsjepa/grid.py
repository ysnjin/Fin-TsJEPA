"""리밸런스 시점(21거래일 간격)별 22개 지표 창 구성, 연도 인과 정규화, 인코딩, 블렌드 실행."""
import numpy as np
import torch

from . import engine, probe
from .data import load_ticker_ohlcv
from .features import compute_22_features

WINDOW = 9
LAM_GRID = (0.0, 0.25, 0.5, 1.0)
MIN_HIST_GRIDS = 24


def build_grid_windows(tickers, px, window=WINDOW, log=print):
    gi = np.arange(engine.WARMUP, len(px.index), engine.REBAL)
    grid_dates = px.index[gi]
    N, G1 = len(tickers), len(gi)
    grid_win = np.full((G1, N, window, 22), np.nan, dtype=np.float32)

    year_sum, year_sumsq, year_count = {}, {}, {}
    for j, tk in enumerate(tickers):
        feat = compute_22_features(load_ticker_ohlcv(tk))
        feat_hist = feat[feat.index <= px.index[-1]]
        for y, grp in feat_hist.groupby(feat_hist.index.year):
            v = grp.values
            ok = np.isfinite(v)
            year_sum[y] = year_sum.get(y, 0.0) + np.where(ok, v, 0.0).sum(axis=0)
            year_sumsq[y] = year_sumsq.get(y, 0.0) + np.where(ok, v * v, 0.0).sum(axis=0)
            year_count[y] = year_count.get(y, 0.0) + ok.sum(axis=0)

        pos = feat.index.searchsorted(grid_dates, side="right") - 1
        vals = feat.values
        for gpos, p in enumerate(pos):
            if p - window + 1 < 0:
                continue
            grid_win[gpos, j] = vals[p - window + 1:p + 1].astype(np.float32)
        if (j + 1) % 100 == 0:
            log(f"  22-feature calc: {j + 1}/{N} tickers")

    # 연도 y의 정규화 통계는 y년 이전 데이터로만 계산한다(미래 정보 차단).
    causal_mean, causal_std = {}, {}
    csum, csumsq, ccount = np.zeros(22), np.zeros(22), np.zeros(22)
    for y in sorted(year_sum):
        if ccount.min() > 0:
            m = csum / ccount
            causal_mean[y] = m
            causal_std[y] = np.sqrt(np.maximum(csumsq / ccount - m * m, 1e-8))
        csum, csumsq, ccount = csum + year_sum[y], csumsq + year_sumsq[y], ccount + year_count[y]

    grid_years = np.array([d.year for d in grid_dates])
    norm = np.full_like(grid_win, np.nan)
    for g in range(G1):
        y = int(grid_years[g])
        if y in causal_mean:
            norm[g] = (grid_win[g] - causal_mean[y]) / causal_std[y]
    return dict(norm_win=norm, grid_dates=grid_dates, gi=gi)


@torch.no_grad()
def encode_grid(encoder, norm_win, windowed):
    """windowed=True: TS 인코더에 (N, 9, 22) 창을 넣고 마지막 날 출력 사용.
    windowed=False: 원본 인코더에 마지막 날 (N, 22)만 넣음."""
    G1, N = norm_win.shape[:2]
    Z = np.full((G1, N, 64), np.nan, dtype=np.float32)
    for g in range(G1):
        x = norm_win[g] if windowed else norm_win[g, :, -1, :]
        if not np.isfinite(x).any():
            continue
        axes = (1, 2) if windowed else 1
        ok = np.isfinite(x).all(axis=axes)
        xc = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
        z = encoder(torch.from_numpy(xc))
        z = (z[:, -1, :] if windowed else z).numpy()
        z[~ok] = np.nan
        Z[g] = z
    return Z


def run_blend(Z, px, fd, mom_full):
    """z(momentum) + λ·z(ridge probe score), λ는 과거 성과로 walk-forward 선택."""
    close_grid = px.values[fd["gi"]]
    valid_grid = np.isfinite(close_grid)
    mom_grid = mom_full.values[fd["gi"]]
    ai_scores = probe.build_probe_scores(Z, close_grid)
    lam_series, _ = engine.walk_forward_lambda(
        mom_grid, ai_scores, close_grid, valid_grid, fd["grid_dates"],
        lam_grid=LAM_GRID, min_hist=MIN_HIST_GRIDS)
    bt = engine.backtest(px, engine.make_blend_selector(mom_full, ai_scores, lam_series))
    return bt["daily"]
