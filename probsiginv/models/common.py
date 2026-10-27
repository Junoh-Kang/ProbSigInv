"""Shared building blocks for the 1D U-Net diffusion model.

References
----------
[1] Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
    https://arxiv.org/abs/2006.11239
[2] Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models",
    ICML 2021.  https://arxiv.org/abs/2102.09672
[3] Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
    https://arxiv.org/abs/1706.03762  (sinusoidal positional embeddings)
[4] Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer",
    AAAI 2018.  https://arxiv.org/abs/1709.07871
[5] Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image
    Segmentation", MICCAI 2015.  https://arxiv.org/abs/1505.04597
[6] Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis",
    NeurIPS 2021.  https://arxiv.org/abs/2105.05233  (architecture details)
[7] Kong et al., "DiffWave: A Versatile Diffusion Model for Audio Synthesis",
    ICLR 2021.  https://arxiv.org/abs/2009.09761  (1-D diffusion backbone)
[8] Tashiro et al., "CSDI: Conditional Score-based Diffusion Models for
    Probabilistic Time Series Imputation", NeurIPS 2021.
    https://arxiv.org/abs/2107.03502
"""

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps.

    Maps integer timestep t -> R^dim using the sin/cos scheme of [3],
    adapted for diffusion models following [1, Sec. 3.2].
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimestepEmbedding(nn.Module):
    """Timestep -> embedding vector.  SinusoidalPosEmb -> MLP [1, 6]."""

    def __init__(self, emb_dim: int = 128):
        super().__init__()
        half = emb_dim // 2
        self.sinusoidal = SinusoidalPosEmb(half)
        self.mlp = nn.Sequential(
            nn.Linear(half, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


class ConditioningEmbedding(nn.Module):
    """Conditioning vector (e.g. log-signature) -> embedding vector."""

    def __init__(self, cond_dim: int = 8, emb_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.mlp(c)


class ResBlock1d(nn.Module):
    """Residual block with FiLM conditioning [4].

    GroupNorm -> SiLU -> Conv1d(in, out)
    -> FiLM(emb -> scale, shift)
    -> GroupNorm -> scale * norm(h) + shift -> SiLU -> Conv1d(out, out)
    + residual skip connection.
    """

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int = 128):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.film = nn.Linear(emb_dim, 2 * out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        # FiLM: affine modulation from conditioning embedding
        scale, shift = self.film(emb).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample1d(nn.Module):
    """Strided convolution downsample (factor 2) [5, 6]."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """Transposed convolution upsample (factor 2) [5, 6]."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SelfAttention1d(nn.Module):
    """Multi-head self-attention over the length dimension [3, 6]."""

    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).permute(0, 2, 1)  # (B, L, C)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1)
