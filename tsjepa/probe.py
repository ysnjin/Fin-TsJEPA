"""Expanding-window ridge probe: frozen latent z -> forward 21d
cross-sectional demeaned return. Embargo = 2 grid steps = 42 trading days.

Re-implemented from the team FinJEPA repository's A10.CrossSec/ic_expanded.py
(incremental Gram-matrix accumulation for speed).
"""
import numpy as np

RIDGE_ALPHA = 10.0
EMBARGO_STEPS = 2
MIN_TRAIN_ROWS = 2000


def build_probe_scores(Z, close_grid, embargo=EMBARGO_STEPS,
                        alpha=RIDGE_ALPHA, min_train_rows=MIN_TRAIN_ROWS):
    """Z: (G1,N,D) grid-level latents (NaN allowed). close_grid: (G1,N).
    Returns dict grid g -> (N,) probe score (only for trainable grids)."""
    G1, N, D = Z.shape
    fwd = np.full_like(close_grid, np.nan)
    fwd[:-1] = close_grid[1:] / close_grid[:-1] - 1.0
    fwd_dm = fwd - np.nanmean(fwd, axis=1, keepdims=True)

    XtX = np.zeros((D, D))
    Xty = np.zeros(D)
    sx = np.zeros(D)
    sy = 0.0
    n_ok = 0
    done = 0
    out = {}
    for g in range(G1):
        hi = g - embargo + 1
        while done < max(hi, 0):
            Zs, ys = Z[done], fwd_dm[done]
            ok = np.isfinite(Zs).all(axis=1) & np.isfinite(ys)
            if ok.any():
                Xs = Zs[ok].astype(np.float64)
                yv = ys[ok]
                XtX += Xs.T @ Xs
                Xty += Xs.T @ yv
                sx += Xs.sum(axis=0)
                sy += yv.sum()
                n_ok += int(ok.sum())
            done += 1
        if hi < 8 or n_ok < min_train_rows:
            continue
        mu_x, mu_y = sx / n_ok, sy / n_ok
        A0 = XtX - n_ok * np.outer(mu_x, mu_x)
        b0 = Xty - n_ok * mu_x * mu_y
        w = np.linalg.solve(A0 + alpha * np.eye(D), b0)
        b = mu_y - mu_x @ w
        zg = Z[g]
        fin = np.isfinite(zg).all(axis=1)
        s = zg @ w + b
        s[~fin] = np.nan
        out[g] = s.astype(np.float32)
    return out
