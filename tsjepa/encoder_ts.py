"""TSPriceEncoder: 원본 PriceEncoder(하루치 22개 지표를 독립적으로 인코딩)를
causal 1D-CNN으로 바꾼 인코더. 오늘의 표현을 오늘 + 직전 8일(총 9일)을 보고 만든다.

- kernel_size=5 causal conv 2층, 채널 22 -> 32 -> 64, 사이에 GELU
  -> receptive field = 2*(5-1)+1 = 9일
- 왼쪽 패딩만 사용: 위치 t의 출력은 t 이전 입력에만 의존(미래 정보 누수 없음)
- projector head는 원본 PriceEncoder와 동일
"""
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.left_pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x):
        x = nn.functional.pad(x, (self.left_pad, 0))
        return self.conv(x)


class TSPriceEncoder(nn.Module):
    """입력 (..., T, n_features) -> 출력 (..., T, latent_dim).

    원본과 달리 시간축이 반드시 필요하다. 하루치 (N, F) 대신
    그날로 끝나는 (N, 9, F) 창을 넣고 마지막 위치 z[:, -1, :]를 쓴다."""

    def __init__(self, n_features=22, latent_dim=64, kernel_size=5, mid_channels=32):
        super().__init__()
        d = latent_dim
        self.conv1 = CausalConv1d(n_features, mid_channels, kernel_size)
        self.conv2 = CausalConv1d(mid_channels, d, kernel_size)
        self.act = nn.GELU()
        self.projector = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, d),
        )
        self.latent_dim = d
        self.kernel_size = kernel_size
        self.receptive_field = 2 * (kernel_size - 1) + 1

    def forward(self, x):
        if x.dim() < 3:
            raise ValueError(
                f"TSPriceEncoder expects (B, T, n_features), got {tuple(x.shape)}. "
                "A single-day (N, n_features) input would be misread as one "
                "N-day sequence; pass a (N, T_window, n_features) window instead.")
        *lead, T, F = x.shape
        h = x.reshape(-1, T, F).transpose(1, 2)
        h = self.act(self.conv1(h))
        h = self.conv2(h)
        h = h.transpose(1, 2)
        z = self.projector(h)
        return z.reshape(*lead, T, self.latent_dim)
