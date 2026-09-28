"""팀 FinJEPA 원본 모듈 (20.JEPA/encoder.py의 PriceEncoder, 20.JEPA/predictor.py의
TransformerPredictor). 비교와 학습에 그대로 쓰기 위해 수정 없이 옮겨 왔다."""
import torch
import torch.nn as nn


class PriceEncoder(nn.Module):
    def __init__(self, n_features=22, latent_dim=64):
        super().__init__()
        d = latent_dim
        h = 2 * d
        self.input_proj = nn.Linear(n_features, d)
        self.encoder = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, d),
        )
        self.projector = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, d),
        )
        self.latent_dim = d

    def forward(self, x):
        h = self.input_proj(x)
        h = self.encoder(h)
        z = self.projector(h)
        return z


class TransformerPredictor(nn.Module):
    def __init__(self, latent_dim=64, n_layers=6, n_heads=4, ff_dim=256,
                 dropout=0.1, max_seq_len=256):
        super().__init__()
        d = latent_dim
        self.pos = nn.Parameter(torch.zeros(1, max_seq_len, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, d),
        )
        self.latent_dim = d
        self.max_seq_len = max_seq_len

    def _causal_mask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, z_seq):
        B, T, D = z_seq.shape
        h = z_seq + self.pos[:, :T, :]
        mask = self._causal_mask(T, z_seq.device)
        h = self.encoder(h, mask=mask)
        return self.head(h)
