"""Newey-West t-test on monthly return differences + bootstrap CI.

Formula follows the team FinJEPA repository's A10.CrossSec/cs_lib.py::nw_t
(Bartlett-kernel Newey-West on the mean).
"""
import numpy as np
from scipy import stats as sps


def monthly_sums(x, block=21):
    n = len(x) // block
    return np.array([x[i * block:(i + 1) * block].sum() for i in range(n)])


def newey_west_t(x, lags=None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    mu = x.mean()
    u = x - mu
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    s = float(u @ u)
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        s += 2.0 * w * float(u[L:] @ u[:-L])
    se = np.sqrt(s / n) / np.sqrt(n)
    t = mu / se
    p = float(2 * (1 - sps.norm.cdf(abs(t))))
    return dict(mean=float(mu), se=float(se), t=float(t), p=p, n=int(n),
                lags=int(lags))


def stationary_bootstrap_idx(n, rng, mean_block=21):
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=np.int64)
    t = rng.integers(0, n)
    for i in range(n):
        idx[i] = t
        if rng.random() < p:
            t = rng.integers(0, n)
        else:
            t = (t + 1) % n
    return idx


def arr_from_daily(d):
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return np.nan
    return float((np.prod(1 + d) ** (252 / len(d)) - 1) * 100)


def bootstrap_diff_ci(ra, rb, rng, n_boot=5000):
    n = min(len(ra), len(rb))
    ra, rb = ra[:n], rb[:n]
    d_arr = np.empty(n_boot)
    for b in range(n_boot):
        ii = stationary_bootstrap_idx(n, rng)
        d_arr[b] = arr_from_daily(ra[ii]) - arr_from_daily(rb[ii])
    return dict(d_arr_mean=float(np.nanmean(d_arr)),
                d_arr_ci=[float(np.nanpercentile(d_arr, 2.5)),
                          float(np.nanpercentile(d_arr, 97.5))])


def paired_test(daily_a, daily_b, rng=None, n_boot=5000):
    """Full comparison: monthly NW t-test + bootstrap ARR-diff CI."""
    n = min(len(daily_a), len(daily_b))
    da, db = np.asarray(daily_a[:n]), np.asarray(daily_b[:n])
    m = np.isfinite(da) & np.isfinite(db)
    da, db = da[m], db[m]
    dm = monthly_sums(da) - monthly_sums(db)
    nw = newey_west_t(dm)
    if rng is None:
        rng = np.random.default_rng(0)
    bs = bootstrap_diff_ci(da, db, rng, n_boot=n_boot)
    return dict(nw_t=nw["t"], p=nw["p"], n_months=nw["n"], lags=nw["lags"],
                mean_monthly_diff_bp=nw["mean"] * 1e4,
                d_arr_mean=bs["d_arr_mean"], d_arr_ci95=bs["d_arr_ci"],
                arr_a=arr_from_daily(da), arr_b=arr_from_daily(db))
