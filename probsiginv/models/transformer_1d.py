"""Patch-based Diffusion Transformer (DiT) for 1-D time-series noise prediction.

Splits the input sequence into non-overlapping patches, projects each to
d_model, processes with N Transformer blocks conditioned via AdaLN-Zero,
then un-patches back to the original sequence length.

References
----------
[1] Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023.
    https://arxiv.org/abs/2212.09748  (DiT, AdaLN-Zero conditioning)
[2] Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
    https://arxiv.org/abs/2006.11239
[3] Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
    https://arxiv.org/abs/1706.03762
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import TimestepEmbedding, ConditioningEmbedding


class TransformerBlock(nn.Module):
    """Single DiT block: AdaLN-Zero → MHA → AdaLN-Zero → FFN.

    Parameters
    ----------
    d_model  : token dimension
    n_heads  : number of attention heads
    emb_dim  : conditioning embedding dimension (from timestep + logsig)
    """

    def __init__(self, d_model: int, n_heads: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        # AdaLN-Zero: 6 scalars per token dim (scale1, shift1, gate1, scale2, shift2, gate2)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 6 * d_model),
        )
        # Zero-init output projection so blocks start as identity
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : (B, n_patches, d_model)
        emb : (B, emb_dim)
        """
        s1, b1, g1, s2, b2, g2 = self.adaLN(emb).chunk(6, dim=-1)  # each (B, d_model)
        s1 = s1.unsqueeze(1); b1 = b1.unsqueeze(1); g1 = g1.unsqueeze(1)
        s2 = s2.unsqueeze(1); b2 = b2.unsqueeze(1); g2 = g2.unsqueeze(1)

        # Attention
        h = (1 + s1) * self.norm1(x) + b1
        h, _ = self.attn(h, h, h)
        x = x + g1 * h

        # FFN
        h = (1 + s2) * self.norm2(x) + b2
        h = self.ffn(h)
        x = x + g2 * h

        return x


class DiT1D(nn.Module):
    """Patch-based Diffusion Transformer for 1-D time-series noise prediction.

    Parameters
    ----------
    seq_len    : length of the input time series (e.g. 1001)
    cond_dim   : conditioning vector dimension (log-signature length)
    path_dim   : number of path channels in channel-first inputs
    patch_size : number of time steps per patch
    d_model    : token (patch embedding) dimension
    n_heads    : number of attention heads
    n_layers   : number of Transformer blocks
    emb_dim    : shared timestep + conditioning embedding dimension
    """

    def __init__(
        self,
        seq_len: int = 1001,
        cond_dim: int = 8,
        path_dim: int = 1,
        patch_size: int = 8,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        emb_dim: int = 128,
        final_deblock: bool = False,
    ):
        super().__init__()
        if int(path_dim) < 1:
            raise ValueError(f"path_dim must be positive, got {path_dim}")
        self.seq_len    = seq_len
        self.path_dim   = int(path_dim)
        self.patch_size = patch_size

        # Pad to nearest multiple of patch_size
        self.pad_to    = math.ceil(seq_len / patch_size) * patch_size
        self.n_patches = self.pad_to // patch_size

        # Conditioning embeddings (reuse from common.py)
        self.time_emb = TimestepEmbedding(emb_dim)
        self.cond_emb = ConditioningEmbedding(cond_dim, emb_dim) if cond_dim > 0 else None

        # Patch embedding: flatten all path channels within each time patch.
        self.patch_embed = nn.Linear(self.path_dim * patch_size, d_model)

        # Learnable positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, emb_dim)
            for _ in range(n_layers)
        ])

        # Output: LayerNorm + linear unpatchify
        self.out_norm  = nn.LayerNorm(d_model, elementwise_affine=False)
        self.out_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * d_model))
        self.out_proj  = nn.Linear(d_model, self.path_dim * patch_size)

        nn.init.zeros_(self.out_adaLN[-1].weight)
        nn.init.zeros_(self.out_adaLN[-1].bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Optional patch-boundary deblocking head used only for DiT regression.
        # Zero init preserves the original mapping at construction time.
        if final_deblock:
            self.out_deblock = nn.Conv1d(
                self.path_dim,
                self.path_dim,
                kernel_size=4 * patch_size + 1,
                padding=0,
                groups=self.path_dim,
                bias=False,
            )
            nn.init.zeros_(self.out_deblock.weight)
        else:
            self.out_deblock = None

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Predict noise from noisy input x_t conditioned on (t, c).

        Parameters
        ----------
        x_t : (B, seq_len) or (B, path_dim, seq_len) noisy time series
        t   : (B,)          integer diffusion timesteps
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

        B = x_t.shape[0]

        # Shared conditioning embedding
        emb = self.time_emb(t)
        if self.cond_emb is not None:
            emb = emb + self.cond_emb(c)

        # Patchify: zero-pad then reshape to (B, n_patches, path_dim * patch_size)
        if self.pad_to > self.seq_len:
            x = torch.cat([x_t, x_t.new_zeros(B, self.path_dim, self.pad_to - self.seq_len)], dim=-1)
        else:
            x = x_t
        x = x.reshape(B, self.path_dim, self.n_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, self.n_patches, self.path_dim * self.patch_size)

        # Patch embedding + positional embedding
        x = self.patch_embed(x) + self.pos_emb  # (B, n_patches, d_model)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, emb)

        # Output projection with AdaLN
        scale, shift = self.out_adaLN(emb).chunk(2, dim=-1)  # (B, d_model)
        x = (1 + scale.unsqueeze(1)) * self.out_norm(x) + shift.unsqueeze(1)
        x = self.out_proj(x)  # (B, n_patches, patch_size)

        # Unpatchify and trim to original seq_len
        x = x.reshape(B, self.n_patches, self.path_dim, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, self.path_dim, self.pad_to)
        x = x[:, :, :self.seq_len]

        # Trim before deblocking so padded zeros cannot influence valid steps.
        if self.out_deblock is not None:
            p = 2 * self.patch_size
            x_pad = F.pad(x, (p, p), mode="replicate")
            x = x + self.out_deblock(x_pad)

        return x[:, 0, :] if input_was_scalar else x
