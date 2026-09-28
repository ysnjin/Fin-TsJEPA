"""FinJEPA_EMA: collapse 방지를 SIGReg + detach 대신 EMA target encoder로 바꾼 JEPA.

- context_encoder: 역전파로 학습
- target_encoder: 초기엔 context의 복사본, 이후 매 스텝 EMA(m=0.996)로만 갱신, gradient 없음
- SIGReg 항 없음. 손실은 예측 오차(MSE) 하나
- predictor는 팀 원본 TransformerPredictor(finjepa_orig.py)를 그대로 사용
"""
import copy

import torch
import torch.nn as nn

from .encoder_ts import TSPriceEncoder
from .finjepa_orig import TransformerPredictor


class FinJEPA_EMA(nn.Module):
    def __init__(self, n_features=22, latent_dim=64, predictor_layers=6,
                 n_heads=4, ff_dim=256, dropout=0.1, max_seq_len=256,
                 encoder_cls=TSPriceEncoder, ema_m=0.996):
        super().__init__()
        self.context_encoder = encoder_cls(n_features, latent_dim)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = TransformerPredictor(
            latent_dim, predictor_layers, n_heads, ff_dim, dropout, max_seq_len)
        self.latent_dim = latent_dim
        self.ema_m = ema_m

    def encode(self, x):
        return self.context_encoder(x)

    @torch.no_grad()
    def update_target(self):
        m = self.ema_m
        for pt, pc in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            pt.data.mul_(m).add_(pc.data, alpha=1 - m)

    def forward(self, ctx_x, tgt_x, horizons=(1,)):
        Tc, Tt = ctx_x.shape[1], tgt_x.shape[1]
        x_all = torch.cat([ctx_x, tgt_x], dim=1)
        z_ctx = self.context_encoder(x_all)
        with torch.no_grad():
            z_tgt = self.target_encoder(x_all)
        pred = self.predictor(z_ctx[:, :-1, :])
        target = z_tgt[:, 1:, :]
        multi = (len(horizons) > 1) or (max(horizons) > 1)
        lo, hi = (Tc - 1, Tc + Tt - 1) if multi else (Tc - 1, Tc)
        p, t = pred[:, lo:hi, :], target[:, lo:hi, :]
        loss_pred = ((p - t) ** 2).mean()
        return dict(loss=loss_pred, loss_pred=loss_pred.detach(), z_all=z_ctx.detach())


@torch.no_grad()
def identity_baseline_mse(model, ctx_x, tgt_x, horizons=(1,)):
    """'내일 표현 = 오늘 표현' 예측의 오차. target_encoder 공간에서 계산해
    loss_pred와 같은 기준으로 비교한다."""
    Tc, Tt = ctx_x.shape[1], tgt_x.shape[1]
    z_all = model.target_encoder(torch.cat([ctx_x, tgt_x], dim=1))
    pred, target = z_all[:, :-1, :], z_all[:, 1:, :]
    multi = (len(horizons) > 1) or (max(horizons) > 1)
    lo, hi = (Tc - 1, Tc + Tt - 1) if multi else (Tc - 1, Tc)
    return float(((pred[:, lo:hi, :] - target[:, lo:hi, :]) ** 2).mean())


@torch.no_grad()
def collapse_metrics(z):
    if z.dim() > 2:
        z = z.reshape(-1, z.shape[-1])
    z = z.float()
    std_z = float(z.std(dim=0).mean())
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(1, zc.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
    p = eig / eig.sum()
    eff_rank = float(torch.exp(-(p * p.log()).sum()))
    return dict(std_z=std_z, eff_rank=eff_rank)
