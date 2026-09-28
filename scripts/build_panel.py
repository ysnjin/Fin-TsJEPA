"""학습용 패널 생성: (T일, N종목, 22지표) float32 -> panels/panel_full845.npz

기본값은 제공된 체크포인트를 학습할 때와 동일한 설정(845종목 전체, 1990-2025).
그때와 같게 하려고 분할 보정 없이 원본 parquet을 그대로 쓴다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsjepa.config import EVAL_END, EVAL_START, PANEL_PATH  # noqa: E402
from tsjepa.data import list_tickers, load_ticker_raw  # noqa: E402
from tsjepa.features import compute_22_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=EVAL_START)
    ap.add_argument("--end", default=EVAL_END)
    ap.add_argument("--out", default=str(PANEL_PATH))
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N종목만 (빠른 점검용)")
    args = ap.parse_args()

    tickers = list_tickers()[:args.limit]
    feats = {}
    for i, tk in enumerate(tickers):
        df = load_ticker_raw(tk)
        df = df[(df.index >= args.start) & (df.index <= args.end)]
        if len(df) >= 300:
            feats[tk] = compute_22_features(df)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(tickers)} tickers", flush=True)

    tickers = sorted(feats)
    master = None
    for tk in tickers:
        master = feats[tk].index if master is None else master.union(feats[tk].index)
    master = master.sort_values()

    X = np.full((len(master), len(tickers), 22), np.nan, dtype=np.float32)
    for j, tk in enumerate(tickers):
        X[:, j, :] = feats[tk].reindex(master).values.astype(np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, dates=np.array([str(d.date()) for d in master]),
                        tickers=np.array(tickers))
    print(f"panel: {X.shape[0]} days x {X.shape[1]} tickers x 22, "
          f"{master[0].date()}~{master[-1].date()} -> {out}")


if __name__ == "__main__":
    main()
