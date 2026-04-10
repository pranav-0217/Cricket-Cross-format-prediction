"""Fit multivariate mixed-effects batting model in PyMC.

Model (professor's design):
    y_ij = X_ij * beta + Z_ij * psi_i + eps_ij

    y   = [log(avg), log(sr)]              (2 outcomes per player-format row)
    X   = [format_odi, format_t20]         (fixed effects; Test = baseline)
    Z   = player indicator (one-hot)       (random effects design matrix)
    psi ~ MVN(0, Sigma_u)                  (player random effects; used for ranking)
    eps ~ MVN(0, Sigma_eps)                (residual noise)

After fitting, psi[i] captures player i's latent ability ABOVE the format-adjusted
mean.  Ranking by ||psi[i]|| (or the avg component) gives cross-format player rankings.

Usage:
    python fit_joint_batting_pymc.py [--tune 1000] [--draws 1000] [--chains 2]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import platformdirs

# Redirect ArviZ cache to workspace-local path (avoids restricted AppData on Windows)
_LOCAL_CACHE = (Path(".cache") / "arviz").resolve()
_LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
platformdirs.user_cache_dir = lambda *a, **kw: str(_LOCAL_CACHE)

import arviz as az
import pymc as pm

DATA_PATH = Path("Data") / "batting_joint.csv"
META_PATH = Path("Data") / "batting_joint_meta.json"
OUTPUT_DIR = Path("outputs") / "mixed_effects"
TRACE_PATH = OUTPUT_DIR / "joint_batting_trace.nc"
SUMMARY_PATH = OUTPUT_DIR / "joint_batting_pymc_summary.txt"
PARAMS_MEAN_PATH = OUTPUT_DIR / "joint_batting_params_mean.csv"
PLAYER_EFFECTS_PATH = OUTPUT_DIR / "player_random_effects.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Joint batting mixed-effects model (PyMC)")
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--cores", type=int, default=1)
    p.add_argument("--target_accept", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def extract_player_effects(idata: az.InferenceData, meta: dict) -> pd.DataFrame:
    """Extract posterior mean of psi (player random effects) and rank players."""
    # psi shape: (chain, draw, player, outcome)
    psi_mean = idata.posterior["u"].mean(dim=("chain", "draw")).values  # (n_players, 2)

    idx_to_name = meta["player_idx_to_name"]
    n_players = psi_mean.shape[0]

    rows = []
    for i in range(n_players):
        name = idx_to_name.get(str(i), f"player_{i}")
        psi_avg = float(psi_mean[i, 0])   # effect on log(avg)
        psi_sr = float(psi_mean[i, 1])    # effect on log(sr)
        # Combined latent ability: average of both random effects
        combined = (psi_avg + psi_sr) / 2.0
        rows.append({
            "player_idx": i,
            "player_name": name,
            "psi_log_avg": psi_avg,
            "psi_log_sr": psi_sr,
            "psi_combined": combined,
        })

    df = pd.DataFrame(rows)
    # Rank: higher psi_combined = better player (higher avg & sr above format-adjusted mean)
    df["rank"] = df["psi_combined"].rank(ascending=False).astype(int)
    df = df.sort_values("rank").reset_index(drop=True)
    return df


def build_params_summary(idata: az.InferenceData) -> pd.DataFrame:
    post = idata.posterior
    alpha_mean = post["alpha"].mean(dim=("chain", "draw")).values       # (2,)
    beta_mean = post["beta"].mean(dim=("chain", "draw")).values         # (2, 2)
    corr_u = float(post["chol_u_corr"].mean(dim=("chain", "draw")).values[0, 1])
    corr_eps = float(post["chol_eps_corr"].mean(dim=("chain", "draw")).values[0, 1])

    params = {
        "alpha_log_avg (Test baseline)": float(alpha_mean[0]),
        "alpha_log_sr  (Test baseline)": float(alpha_mean[1]),
        "beta_odi_log_avg":  float(beta_mean[0, 0]),
        "beta_t20_log_avg":  float(beta_mean[0, 1]),
        "beta_odi_log_sr":   float(beta_mean[1, 0]),
        "beta_t20_log_sr":   float(beta_mean[1, 1]),
        "player_corr(avg,sr)": corr_u,
        "residual_corr(avg,sr)": corr_eps,
    }
    return pd.DataFrame([params])


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    # Fixed-effects design matrix X: [format_odi, format_t20]
    X = df[["format_odi", "format_t20"]].to_numpy(dtype=float)
    # Outcomes: y = [log(avg), log(sr)]
    y = df[["y1", "y2"]].to_numpy(dtype=float)
    # Random effects grouping: player index (Z = one-hot indicator)
    player_idx = df["player_idx"].to_numpy(dtype=int)
    n_players = int(df["player_idx"].max()) + 1

    print(f"Data: {len(df)} rows, {n_players} players, {y.shape[1]} outcomes")

    coords = {
        "obs_id":    np.arange(y.shape[0]),
        "outcome":   ["log_avg", "log_sr"],
        "predictor": ["format_odi", "format_t20"],
        "player":    np.arange(n_players),
    }

    with pm.Model(coords=coords) as model:
        X_data      = pm.Data("X",          X,          dims=("obs_id", "predictor"))
        player_data = pm.Data("player_idx", player_idx, dims=("obs_id",))

        # Fixed effects: intercept (alpha) + format slopes (beta)
        alpha = pm.Normal("alpha", mu=0.0, sigma=2.5, dims=("outcome",))
        beta  = pm.Normal("beta",  mu=0.0, sigma=1.5, dims=("outcome", "predictor"))

        # Player random effects psi ~ MVN(0, Sigma_u)
        # LKJCholeskyCov gives us the Cholesky factor of Sigma_u
        chol_u, corr_u, sd_u = pm.LKJCholeskyCov(
            "chol_u", n=2, eta=2.0,
            sd_dist=pm.HalfNormal.dist(1.0),
            compute_corr=True,
        )
        # u[player, outcome] = psi in the professor's notation
        u = pm.MvNormal("u", mu=np.zeros(2), chol=chol_u, dims=("player", "outcome"))

        # Residual covariance Sigma_eps
        chol_eps, corr_eps, sd_eps = pm.LKJCholeskyCov(
            "chol_eps", n=2, eta=2.0,
            sd_dist=pm.HalfNormal.dist(1.0),
            compute_corr=True,
        )

        # Mean: mu_ij = alpha + X_ij * beta^T + u[player_i]
        mu = alpha + pm.math.dot(X_data, beta.T) + u[player_data]

        # Likelihood: y_ij ~ MVN(mu_ij, Sigma_eps)
        pm.MvNormal("y_obs", mu=mu, chol=chol_eps, observed=y, dims=("obs_id", "outcome"))

        sample_kwargs = dict(
            tune=args.tune,
            draws=args.draws,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=args.seed,
            return_inferencedata=True,
            progressbar=True,
        )
        try:
            idata = pm.sample(**sample_kwargs)
        except PermissionError:
            print("Multiprocessing blocked; retrying with cores=1.")
            sample_kwargs["cores"] = 1
            idata = pm.sample(**sample_kwargs)

    # Save posterior trace
    az.to_netcdf(idata, TRACE_PATH)
    print(f"Trace saved: {TRACE_PATH}")

    # Save parameter summary
    summary = az.summary(
        idata,
        var_names=["alpha", "beta", "chol_u_stds", "chol_u_corr",
                   "chol_eps_stds", "chol_eps_corr"],
        round_to=4,
    )
    SUMMARY_PATH.write_text(summary.to_string(), encoding="utf-8")
    print(f"Summary saved: {SUMMARY_PATH}")

    # Save population-level parameter means
    params_df = build_params_summary(idata)
    params_df.to_csv(PARAMS_MEAN_PATH, index=False)
    print(f"Params mean saved: {PARAMS_MEAN_PATH}")

    # Extract player random effects (psi) and rank players
    player_df = extract_player_effects(idata, meta)
    player_df.to_csv(PLAYER_EFFECTS_PATH, index=False)
    print(f"Player rankings saved: {PLAYER_EFFECTS_PATH}")
    print("\nTop 20 players by latent batting ability:")
    print(player_df[["rank", "player_name", "psi_log_avg", "psi_log_sr", "psi_combined"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
