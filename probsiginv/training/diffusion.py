"""Gaussian Diffusion (DDPM) with cosine noise schedule.

Simplified noise-prediction DDPM following [1] with the cosine schedule of [2].
Supports conditioning via the model interface ``model(x_t, t, c)``.

References
----------
[1] Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
    https://arxiv.org/abs/2006.11239
[2] Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models",
    ICML 2021.  https://arxiv.org/abs/2102.09672
[3] Yuan & Qiao, "Diffusion-TS", ICLR 2024.
    https://arxiv.org/abs/2403.01742  (simplified time-series diffusion)
"""

import math

import torch
import torch.nn.functional as F


def _batch_view(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return values.view(-1, *([1] * (reference.ndim - 1)))


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule from [2, Eq. 17].

    Returns
    -------
    betas : (T,) tensor of beta_t values in (0, 0.999].
    """
    steps = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
    alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(0, 0.999).float()


class GaussianDiffusion:
    """DDPM forward/reverse process with cosine schedule."""

    name = "ddpm"

    def __init__(self, T: int = 1000, device: torch.device | str = "cpu",
                 clip_x0: tuple[float, float] | None = None):
        self.T = T
        self.clip_x0 = clip_x0
        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt().to(device)
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt().to(device)

        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)
        self.posterior_mean_coef1 = (
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)
        ).to(device)
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod)
        ).to(device)

    def to(self, device: torch.device | str) -> "GaussianDiffusion":
        """Move all pre-computed tensors to *device*."""
        for attr in [
            "betas",
            "alphas_cumprod",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
            "posterior_mean_coef1",
            "posterior_mean_coef2",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t ~ q(x_t | x_0) = N(sqrt(abar_t) x_0, (1-abar_t) I)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_a = _batch_view(self.sqrt_alphas_cumprod[t], x_0)
        sqrt_1ma = _batch_view(self.sqrt_one_minus_alphas_cumprod[t], x_0)
        return sqrt_a * x_0 + sqrt_1ma * noise, noise

    @torch.no_grad()
    def p_sample(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        t_int: int,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Single reverse step: x_t -> x_{t-1}."""
        B = x_t.shape[0]
        t = torch.full((B,), t_int, device=x_t.device, dtype=torch.long)
        noise_pred = model(x_t, t, c)

        x_0_pred = (
            x_t - self.sqrt_one_minus_alphas_cumprod[t_int] * noise_pred
        ) / self.sqrt_alphas_cumprod[t_int]
        if self.clip_x0 is not None:
            x_0_pred = x_0_pred.clamp(*self.clip_x0)

        mean = (
            self.posterior_mean_coef1[t_int] * x_0_pred
            + self.posterior_mean_coef2[t_int] * x_t
        )

        if t_int == 0:
            return mean
        noise = torch.randn_like(x_t)
        return mean + self.posterior_variance[t_int].sqrt() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: torch.nn.Module,
        shape: tuple[int, ...],
        c: torch.Tensor,
        sampling_steps: int | None = None,
    ) -> torch.Tensor:
        """Full reverse chain: sample x_0 from pure noise."""
        x = torch.randn(shape, device=c.device)
        if sampling_steps is None or sampling_steps >= self.T:
            timesteps = list(reversed(range(self.T)))
        else:
            timesteps = torch.linspace(self.T - 1, 0, sampling_steps).long().tolist()
        for t in timesteps:
            x = self.p_sample(model, x, int(t), c)
        return x

    @torch.no_grad()
    def ddim_step(self, model, x_t, t_int, t_prev_int, c, eta=0.0):
        """Single DDIM reverse step: x_t -> x_{t_prev}."""
        B = x_t.shape[0]
        t = torch.full((B,), t_int, device=x_t.device, dtype=torch.long)
        eps = model(x_t, t, c)

        a_t = self.alphas_cumprod[t_int]
        a_prev = self.alphas_cumprod[t_prev_int] if t_prev_int >= 0 else torch.ones(1, device=x_t.device)

        x_0_pred = (x_t - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        if self.clip_x0 is not None:
            x_0_pred = x_0_pred.clamp(*self.clip_x0)
        sigma = eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).clamp(min=0).sqrt()
        direction = (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        noise = torch.randn_like(x_t) if (eta > 0 and t_prev_int >= 0) else torch.zeros_like(x_t)
        return a_prev.sqrt() * x_0_pred + direction + sigma * noise

    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, c, sampling_steps=50, eta=0.0):
        """Full DDIM reverse chain: sample x_0 from pure noise."""
        x = torch.randn(shape, device=c.device)
        if sampling_steps is None or sampling_steps >= self.T:
            timesteps = list(range(self.T - 1, -1, -1))
        else:
            timesteps = torch.linspace(self.T - 1, 0, sampling_steps).long().tolist()
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            x = self.ddim_step(model, x, int(t), int(t_prev), c, eta=eta)
        return x

    @torch.no_grad()
    def sample_loop(self, model, shape, c, sampling_steps=50, **kwargs):
        """Backend-agnostic sampling alias used by callers."""
        eta = kwargs.get("eta", 0.0)
        return self.ddim_sample_loop(model, shape, c, sampling_steps=sampling_steps, eta=eta)

    def training_loss(
        self,
        model: torch.nn.Module,
        x_0: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Simple noise-prediction MSE loss."""
        B = x_0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x_0.device)
        x_t, noise = self.q_sample(x_0, t)
        noise_pred = model(x_t, t, c)
        return F.mse_loss(noise_pred, noise)
