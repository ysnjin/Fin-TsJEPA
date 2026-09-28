"""momentum 블렌드 백테스트 (845종목, 1990-2025, CPU로 충분).

비교 대상 (학습 인코더는 seed 7/42/99 평균)
- M: 12-1 momentum 상위 10%
- BLEND-RAW: 인코더 없이 22개 지표를 그대로 ridge probe에 넣음
- BLEND-{arch}-random: 학습하지 않은 무작위 인코더
- BLEND-{arch}-sigreg-trained: 팀 원본 방식(SIGReg + detach)으로 학습한 인코더
- BLEND-{arch}-ema-trained: EMA target encoder로 학습한 인코더
arch: ts(TSPriceEncoder), orig(팀 원본 PriceEncoder). orig-sigreg-trained가 팀 원본 모델이다.
검정: 각 arm vs M, 학습 vs 무작위, 학습 vs RAW, 방식 간 비교 (월간 수익 차이 Newey-West)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsjepa import engine, stats_tests  # noqa: E402
from tsjepa.config import CKPT_DIR, EVAL_END, EVAL_START, RESULTS_DIR, SEEDS  # noqa: E402
from tsjepa.data import build_close_panel, list_tickers  # noqa: E402
from tsjepa.encoder_ts import TSPriceEncoder  # noqa: E402
from tsjepa.finjepa_orig import PriceEncoder  # noqa: E402
from tsjepa.grid import build_grid_windows, encode_grid, run_blend  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_encoder(cls, seed, ckpt=None):
    torch.manual_seed(seed)
    model = cls()
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["ts", "orig", "both"], default="both")
    ap.add_argument("--method", choices=["ema", "sigreg", "both"], default="both")
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--start", default=EVAL_START)
    ap.add_argument("--end", default=EVAL_END)
    args = ap.parse_args()
    ckpt_dir, out_dir = Path(args.ckpt_dir), Path(args.out_dir)
    archs = ["ts", "orig"] if args.arch == "both" else [args.arch]
    methods = ["ema", "sigreg"] if args.method == "both" else [args.method]
    cls = {"ts": TSPriceEncoder, "orig": PriceEncoder}

    px, kept, _ = build_close_panel(list_tickers(), args.start, args.end, min_rows=100)
    log(f"panel: {px.shape[0]} days x {px.shape[1]} tickers")
    mom_full = engine.momentum_scores(px)
    daily = {"M": engine.backtest(px, engine.make_momentum_selector(mom_full))["daily"]}

    log("building 22-feature grid ...")
    fd = build_grid_windows(kept, px, log=log)

    daily["BLEND-RAW"] = run_blend(fd["norm_win"][:, :, -1, :].copy(), px, fd, mom_full)

    per_seed = {}
    for tag in archs:
        windowed = tag == "ts"
        for kind in ["random"] + [f"{m}-trained" for m in methods]:
            name = f"BLEND-{tag}-{kind}"
            runs = []
            for seed in args.seeds:
                ckpt = None if kind == "random" else ckpt_dir / f"{tag}_{kind.replace('-', '_')}_s{seed}.pt"
                enc = make_encoder(cls[tag], seed, ckpt)
                runs.append(run_blend(encode_grid(enc, fd["norm_win"], windowed), px, fd, mom_full))
                log(f"  {name} s{seed}: ARR {engine.all_metrics(runs[-1])['ARR']:.2f}%")
            daily[name] = np.mean(runs, axis=0)
            per_seed[name] = {int(s): engine.all_metrics(r) for s, r in zip(args.seeds, runs)}

    results = {"metrics": {k: engine.all_metrics(v) for k, v in daily.items()},
               "per_seed": per_seed, "tests": {}}
    log("=== 성과 ===")
    for k, m in results["metrics"].items():
        log(f"{k:<28} ARR {m['ARR']:6.2f}%  SR {m['SR']:.2f}  MDD {m['MDD']:.1f}%")

    pairs = [(k, "M") for k in daily if k != "M"]
    for tag in archs:
        for m in methods:
            pairs += [(f"BLEND-{tag}-{m}-trained", f"BLEND-{tag}-random"),
                      (f"BLEND-{tag}-{m}-trained", "BLEND-RAW")]
    if len(methods) == 2 and "orig" in archs:
        pairs.append(("BLEND-orig-ema-trained", "BLEND-orig-sigreg-trained"))
        if "ts" in archs:
            pairs.append(("BLEND-ts-ema-trained", "BLEND-orig-sigreg-trained"))
    log("=== Newey-West 검정 ===")
    for a, b in pairs:
        t = stats_tests.paired_test(daily[a], daily[b])
        results["tests"][f"{a} vs {b}"] = t
        log(f"{a} vs {b}: t={t['nw_t']:+.3f} p={t['p']:.4g} "
            f"dARR={t['d_arr_mean']:+.2f}pp CI[{t['d_arr_ci95'][0]:.2f}, {t['d_arr_ci95'][1]:.2f}]")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "backtest_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    np.savez(out_dir / "backtest_daily.npz", **daily)
    log(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
