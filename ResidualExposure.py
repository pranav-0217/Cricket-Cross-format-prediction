"""Exposure-weighted residual covariance for the joint outcome models."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def exposure_matrix(df, meta, outcomes=("y1", "y2")) -> np.ndarray:
    """Per-row exposure for each outcome, ordered to match the outcome axis."""
    spec = meta.get("exposure")
    if spec is None:
        raise KeyError(
            "metadata has no 'exposure' block -- rebuild with PrepareData.py"
        )
    columns = [spec["columns"][o] for o in outcomes]
    missing = [c for c in columns if c not in df]
    if missing:
        raise KeyError(f"exposure columns absent from the data: {missing}")
    w = df[columns].to_numpy(dtype=float)
    if not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("exposure values must be finite and strictly positive")
    return w

def residual_likelihood(mu, y: np.ndarray, w: np.ndarray | None, name: str = "y_obs"):
    """Register the joint residual likelihood; returns the per-row scales."""
    sigma_real = pm.HalfNormal("sigma_real", 1.0, dims=("outcome",))

    if w is None:
        scale = pt.broadcast_to(sigma_real, y.shape)
    else:
        W = pm.Data("exposure", w, dims=("obs_id", "outcome"))
        sigma_sampling = pm.HalfNormal("sigma_sampling", 1.0, dims=("outcome",))
        scale = pt.sqrt(sigma_sampling**2 / W + sigma_real**2)

    # (rho + 1) / 2 ~ Beta(2, 2)  <=>  p(rho) proportional to 1 - rho^2  <=>  LKJ(eta=2), n=2
    rho_unit = pm.Beta("eps_corr_unit", alpha=2.0, beta=2.0)
    rho = pm.Deterministic("eps_corr", 2.0 * rho_unit - 1.0)

    s1, s2 = scale[:, 0], scale[:, 1]
    mu1, mu2 = mu[:, 0], mu[:, 1]

    pm.Normal(f"{name}_1", mu=mu1, sigma=s1, observed=y[:, 0], dims=("obs_id",))
    # Conditional mean/sd of outcome 2 given outcome 1 -- the Cholesky written out.
    pm.Normal(
        f"{name}_2",
        mu=mu2 + rho * (s2 / s1) * (y[:, 0] - mu1),
        sigma=s2 * pt.sqrt(1.0 - rho**2),
        observed=y[:, 1],
        dims=("obs_id",),
    )
    return scale

def residual_corr(posterior) -> float:
    """Posterior-mean residual correlation, whichever parameterisation was used."""
    if "eps_corr" in posterior:
        value = posterior["eps_corr"].mean(dim=("chain", "draw")).values
        return float(value) if value.ndim == 0 else float(value[0, 1])
    return float(posterior["chol_eps_corr"].mean(dim=("chain", "draw")).values[0, 1])

def summary_vars(weighted: bool) -> list[str]:
    """Residual parameter names to include in the posterior summary."""
    if weighted:
        return ["sigma_sampling", "sigma_real", "eps_corr"]
    return ["sigma_real", "eps_corr"]
