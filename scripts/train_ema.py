"""EMA target encoder로 JEPA 사전학습 (GPU 권장, Colab A100 기준 seed당 약 4분).

기본: TS 인코더, 학습률 1e-5, seed 7/42/99 -> checkpoints_new/ts_ema_trained_s{seed}.pt
--arch orig 는 팀 원본 PriceEncoder로 같은 조건 학습(학습률 5e-5).
--scan 을 주면 5개 학습률(4 epoch, 20만 창)을 먼저 비교한 뒤 가장 좋은 값으로 학습한다.
저장되는 것은 context_encoder 가중치만(백테스트에 필요한 부분).
"""
import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsjepa.config import PANEL_PATH, REPO_ROOT, SEEDS  # noqa: E402
from tsjepa.ema import FinJEPA_EMA, collapse_metrics, identity_baseline_mse  # noqa: E402
from tsjepa.encoder_ts import TSPriceEncoder  # noqa: E402
from tsjepa.finjepa_orig import PriceEncoder  # noqa: E402

TRAIN_END, VAL_START = "2023-12-31", "2024-01-01"
HP = dict(latent_dim=64, predictor_layers=2, n_heads=4, ff_dim=64, dropout=0.1,
          ctx_len=30, tgt_len=10, weight_decay=1e-5, batch_size=256, epochs=8,
          horizons=(1,), window_cap=600_000, ema_m=0.996)
DEFAULT_LR = {"ts": 1e-5, "orig": 5e-5}
SCAN_LRS = [5e-4, 2e-4, 1e-4, 5e-5, 1e-5]
SCAN_EPOCHS, SCAN_CAP, SCAN_SEED = 4, 200_000, 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PanelData:
    def __init__(self, path, ctx_len, tgt_len, device):
        z = np.load(path)
        self.X = torch.from_numpy(z["X"]).to(device)
        self.dates = z["dates"].astype("datetime64[D]")
        self.device = device
        self.ctx = ctx_len
        win = ctx_len + tgt_len
        finite = torch.isfinite(self.X).all(dim=2)
        c = finite.float().cumsum(dim=0)
        full = torch.zeros_like(finite)
        full[win - 1:] = (c[win - 1:] - torch.cat(
            [torch.zeros(1, finite.shape[1], device=device), c[:-win]], dim=0)) == win
        self.valid_win = full.bool()
        self.offsets = torch.arange(win - 1, -1, -1, device=device)

    def window_index(self, start=None, end=None):
        m = self.valid_win.clone()
        if start is not None:
            m[self.dates < np.datetime64(start)] = False
        if end is not None:
            m[self.dates > np.datetime64(end)] = False
        t_idx, j_idx = torch.nonzero(m, as_tuple=True)
        return t_idx.int(), j_idx.int()

    def gather(self, t_idx, j_idx):
        tt = t_idx.long().unsqueeze(1) - self.offsets.unsqueeze(0)
        return self.X[tt, j_idx.long().unsqueeze(1), :]

    def fit_scaler(self, start, end):
        m = (self.dates >= np.datetime64(start)) & (self.dates <= np.datetime64(end))
        flat = self.X[torch.from_numpy(m).to(self.device)].reshape(-1, self.X.shape[-1])
        flat = flat[torch.isfinite(flat).all(dim=1)]
        return flat.mean(dim=0), flat.std(dim=0) + 1e-8


@torch.no_grad()
def evaluate(model, panel, t_idx, j_idx, mean, std, bs):
    model.eval()
    if len(t_idx) > 100_000:
        g = torch.Generator(device="cpu").manual_seed(0)
        sel = torch.randperm(len(t_idx), generator=g)[:100_000]
        t_idx, j_idx = t_idx[sel.to(t_idx.device)], j_idx[sel.to(j_idx.device)]
    val_pred = ident = 0.0
    cm_sum, nb = None, 0
    for i in range(0, len(t_idx), bs):
        w = (panel.gather(t_idx[i:i + bs], j_idx[i:i + bs]) - mean) / std
        ctx, tgt = w[:, :panel.ctx], w[:, panel.ctx:]
        out = model(ctx, tgt, HP["horizons"])
        val_pred += float(out["loss_pred"])
        ident += identity_baseline_mse(model, ctx, tgt, HP["horizons"])
        cm = collapse_metrics(out["z_all"])
        cm_sum = cm if cm_sum is None else {k: cm_sum[k] + cm[k] for k in cm}
        nb += 1
    nb = max(1, nb)
    return val_pred / nb, ident / nb, {k: v / nb for k, v in cm_sum.items()}


def train_one(name, encoder_cls, peak_lr, seed, state, out_path=None, epochs=None, cap=None):
    epochs = epochs or HP["epochs"]
    set_seed(seed)
    model = FinJEPA_EMA(n_features=22, latent_dim=HP["latent_dim"],
                        predictor_layers=HP["predictor_layers"], n_heads=HP["n_heads"],
                        ff_dim=HP["ff_dim"], dropout=HP["dropout"],
                        encoder_cls=encoder_cls, ema_m=HP["ema_m"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=HP["weight_decay"])
    panel, tr_t, tr_j = state["panel"], state["tr_t"], state["tr_j"]
    mean, std, bs = state["mean"], state["std"], HP["batch_size"]
    cap = min(cap or HP["window_cap"], len(tr_t))
    steps_per_epoch = max(1, cap // bs)
    total = epochs * steps_per_epoch
    warmup = max(1, int(0.1 * total))

    def lr_at(step):
        if step < warmup:
            return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    step, t0 = 0, time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr_t), device=DEVICE)[:cap]
        tr_loss = 0.0
        for bi in range(steps_per_epoch):
            idx = perm[bi * bs:(bi + 1) * bs]
            w = (panel.gather(tr_t[idx], tr_j[idx]) - mean) / std
            for pg in opt.param_groups:
                pg["lr"] = peak_lr * lr_at(step)
            out = model(w[:, :panel.ctx], w[:, panel.ctx:], HP["horizons"])
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.update_target()
            step += 1
            tr_loss += float(out["loss"].detach())
        print(f"  [{name} s{seed} lr={peak_lr:.0e}] epoch {ep + 1}/{epochs} "
              f"train_loss={tr_loss / steps_per_epoch:.4f} ({time.time() - t0:.0f}s)", flush=True)

    val_pred, ident, cm = evaluate(model, panel, state["va_t"], state["va_j"], mean, std, bs)
    msg = (f"  -> val_pred={val_pred:.4f} identity={ident:.4f} "
           f"ratio={val_pred / ident:.3f} std_z={cm['std_z']:.3f} eff_rank={cm['eff_rank']:.1f}")
    if out_path:
        torch.save(model.context_encoder.state_dict(), out_path)
        msg += f" saved -> {out_path}"
    print(msg, flush=True)
    return val_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(PANEL_PATH))
    ap.add_argument("--arch", choices=["ts", "orig", "both"], default="ts")
    ap.add_argument("--lr", type=float, default=None, help="미지정 시 ts=1e-5, orig=5e-5")
    ap.add_argument("--scan", action="store_true", help="학습률 탐색 후 최적값으로 학습")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--epochs", type=int, default=HP["epochs"])
    ap.add_argument("--window-cap", type=int, default=HP["window_cap"])
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "checkpoints_new"))
    args = ap.parse_args()

    print("device:", DEVICE)
    panel = PanelData(args.panel, HP["ctx_len"], HP["tgt_len"], DEVICE)
    train_start = str(panel.dates[0])
    tr_t, tr_j = panel.window_index(train_start, TRAIN_END)
    va_t, va_j = panel.window_index(VAL_START, str(panel.dates[-1]))
    mean, std = panel.fit_scaler(train_start, TRAIN_END)
    state = dict(panel=panel, tr_t=tr_t, tr_j=tr_j, va_t=va_t, va_j=va_j, mean=mean, std=std)
    print(f"train windows: {len(tr_t)}, val windows: {len(va_t)}", flush=True)

    archs = ["orig", "ts"] if args.arch == "both" else [args.arch]
    enc = {"ts": TSPriceEncoder, "orig": PriceEncoder}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tag in archs:
        lr = args.lr or DEFAULT_LR[tag]
        if args.scan:
            print(f"\n=== LR scan: {tag} ===", flush=True)
            res = {l: train_one(tag, enc[tag], l, SCAN_SEED, state,
                                epochs=SCAN_EPOCHS, cap=SCAN_CAP) for l in SCAN_LRS}
            lr = min(res, key=res.get)
            print(f"{tag}: " + "  ".join(f"{l:.0e}={v:.4f}" for l, v in res.items())
                  + f"  -> best={lr:.0e}", flush=True)
        print(f"\n=== train: {tag} (lr={lr:.0e}) ===", flush=True)
        for seed in args.seeds:
            train_one(tag, enc[tag], lr, seed, state,
                      out_path=out_dir / f"{tag}_ema_trained_s{seed}.pt",
                      epochs=args.epochs, cap=args.window_cap)


if __name__ == "__main__":
    main()
