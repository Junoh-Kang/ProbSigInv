"""Conditional 1D U-Net for noise prediction in DDPM.

A lightweight 4-level encoder-decoder with skip connections, FiLM-conditioned
residual blocks, and bottleneck self-attention.  Designed for 1-D time-series
generation (seq_len ~ 1000).

References
----------
[1] Ronneberger et al., "U-Net", MICCAI 2015.
    https://arxiv.org/abs/1505.04597
[2] Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
    https://arxiv.org/abs/2006.11239
[3] Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis",
    NeurIPS 2021.  https://arxiv.org/abs/2105.05233
[4] Kong et al., "DiffWave", ICLR 2021.  https://arxiv.org/abs/2009.09761
[5] Yuan & Qiao, "Diffusion-TS: Interpretable Diffusion for General Time
    Series Generation", ICLR 2024.  https://arxiv.org/abs/2403.01742
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    TimestepEmbedding,
    ConditioningEmbedding,
    ResBlock1d,
    Downsample1d,
    Upsample1d,
    SelfAttention1d,
)


class UNet1D(nn.Module):
    """Conditional 1D U-Net for diffusion noise prediction.

    Parameters
    ----------
    seq_len : int
        Length of input time series (e.g. 1001).
    cond_dim : int
        Dimension of the conditioning vector (e.g. 8 for depth-4 logsig).
    path_dim : int
        Number of path channels in channel-first inputs.
    channels : list[int]
        Channel widths at each encoder level.
    num_res_blocks : int
        Number of residual blocks per encoder/decoder level.
    emb_dim : int
        Dimension of the shared timestep + conditioning embedding.
    attn_heads : int
        Number of attention heads in the bottleneck self-attention.
    """

    def __init__(
        self,
        seq_len: int = 1001,
        cond_dim: int = 8,
        path_dim: int = 1,
        channels: list[int] | None = None,
        num_res_blocks: int = 2,
        emb_dim: int = 128,
        attn_heads: int = 4,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 128]
        if int(path_dim) < 1:
            raise ValueError(f"path_dim must be positive, got {path_dim}")

        self.seq_len = seq_len
        self.path_dim = int(path_dim)
        self.num_levels = len(channels)

        # Pad input to the nearest multiple of 2^num_levels so that each
        # downsample halves an even-length tensor (avoids size mismatches in
        # the skip connections).
        divisor = 2 ** self.num_levels
        self.pad_to = math.ceil(seq_len / divisor) * divisor

        # --- Conditioning ---
        self.time_emb = TimestepEmbedding(emb_dim)
        self.cond_emb = ConditioningEmbedding(cond_dim, emb_dim) if cond_dim > 0 else None

        # --- Encoder ---
        self.init_conv = nn.Conv1d(self.path_dim, channels[0], kernel_size=7, padding=3)

        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        in_ch = channels[0]
        for out_ch in channels:
            level_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                level_blocks.append(
                    ResBlock1d(in_ch if j == 0 else out_ch, out_ch, emb_dim)
                )
            in_ch = out_ch
            self.enc_blocks.append(level_blocks)
            self.downsamples.append(Downsample1d(out_ch))

        # --- Bottleneck ---
        bot_ch = channels[-1]
        self.bot_res1 = ResBlock1d(bot_ch, bot_ch, emb_dim)
        self.bot_attn = SelfAttention1d(bot_ch, attn_heads)
        self.bot_res2 = ResBlock1d(bot_ch, bot_ch, emb_dim)

        # --- Decoder ---
        self.upsamples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        dec_channels = list(reversed(channels))  # mirror encoder
        prev_ch = bot_ch
        for i, out_ch in enumerate(dec_channels):
            skip_ch = out_ch  # encoder skip has the same width
            self.upsamples.append(Upsample1d(prev_ch))
            concat_ch = prev_ch + skip_ch
            level_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                level_blocks.append(
                    ResBlock1d(concat_ch if j == 0 else out_ch, out_ch, emb_dim)
                )
            self.dec_blocks.append(level_blocks)
            prev_ch = out_ch

        # --- Output ---
        self.out_norm = nn.GroupNorm(min(8, channels[0]), channels[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv1d(channels[0], self.path_dim, kernel_size=7, padding=3)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise from noisy input x_t conditioned on (t, c).

        Parameters
        ----------
        x_t : (B, seq_len) or (B, path_dim, seq_len) noisy time series
        t   : (B,) integer diffusion timesteps
        c   : (B, cond_dim) conditioning vectors (log-signatures)

        Returns
        -------
        noise_pred : same layout as x_t
        """
        input_was_scalar = x_t.ndim == 2
        if input_was_scalar:
            if self.path_dim != 1:
                raise ValueError(f"2-D scalar input is only valid for path_dim=1, got path_dim={self.path_dim}")
            x_t = x_t.unsqueeze(1)
        elif x_t.ndim != 3:
            raise ValueError(f"x_t must have shape (B,T) or (B,D,T), got {tuple(x_t.shape)}")
        if int(x_t.shape[1]) != self.path_dim:
            raise ValueError(f"x_t path_dim mismatch: expected {self.path_dim}, got {int(x_t.shape[1])}")
        if int(x_t.shape[-1]) != self.seq_len:
            raise ValueError(f"x_t seq_len mismatch: expected {self.seq_len}, got {int(x_t.shape[-1])}")

        # Shared conditioning embedding
        emb = self.time_emb(t)
        if self.cond_emb is not None:
            emb = emb + self.cond_emb(c)

        # Reshape and zero-pad to pad_to length
        x = x_t
        if self.pad_to > self.seq_len:
            x = F.pad(x, (0, self.pad_to - self.seq_len))

        x = self.init_conv(x)

        # Encoder — collect skip connections
        skips = []
        for i in range(self.num_levels):
            for block in self.enc_blocks[i]:
                x = block(x, emb)
            skips.append(x)
            x = self.downsamples[i](x)

        # Bottleneck
        x = self.bot_res1(x, emb)
        x = self.bot_attn(x)
        x = self.bot_res2(x, emb)

        # Decoder — consume skip connections (LIFO)
        for i in range(self.num_levels):
            x = self.upsamples[i](x)
            x = torch.cat([x, skips.pop()], dim=1)
            for block in self.dec_blocks[i]:
                x = block(x, emb)

        # Output projection and slice back to original length
        x = self.out_act(self.out_norm(x))
        x = self.out_conv(x)
        x = x[:, :, : self.seq_len]
        return x[:, 0, :] if input_was_scalar else x
