"""Analytic LS oracle reconstruction error for log-GBM."""

from __future__ import annotations


def _uniform_moments(a: float, b: float) -> dict[str, float]:
    """Return the low-order moments for ``Uniform(a, b)``."""
    e_x2 = (a**2 + a * b + b**2) / 3.0
    if abs(b - a) < 1e-15:
        e_x4 = a**4
    else:
        e_x4 = (b**5 - a**5) / (5.0 * (b - a))
    var_x = (b - a) ** 2 / 12.0
    return {
        "E_x2": e_x2,
        "Var_x": var_x,
        "Var_x2": e_x4 - e_x2**2,
    }


def gbm_analytic_mse(
    depths: list[int],
    mu: tuple[float, float],
    sigma: tuple[float, float],
    T_horizon: float = 1.0,
    distribution: str = "uniform",
) -> dict[int, float]:
    """Return analytic GBM pairwise path-MSE lower bounds by depth."""
    if distribution != "uniform":
        raise NotImplementedError(f"distribution={distribution!r} not supported")

    a_mu, b_mu = mu
    a_sig, b_sig = sigma
    T = float(T_horizon)

    sig_mom = _uniform_moments(a_sig, b_sig)
    e_sigma2 = sig_mom["E_x2"]
    var_sigma2 = sig_mom["Var_x2"]
    var_mu = (b_mu - a_mu) ** 2 / 12.0
    var_drift = var_mu + var_sigma2 / 4.0

    results: dict[int, float] = {}
    for depth in depths:
        if depth == 0:
            theory = e_sigma2 * T + (2.0 * T**2 / 3.0) * var_drift
        else:
            theory = e_sigma2 * depth * T / ((2 * depth - 1) * (2 * depth + 1))
        results[int(depth)] = float(theory)
    return results
