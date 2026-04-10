"""Fit enriched joint batting model — X includes debut_year_scaled.

Professor's model: y_ij = X_ij * beta + Z_ij * psi_i + eps_ij
Enriched X = [format_odi, format_t20, debut_year_scaled, debut_year_missing]

This adds the "age" covariate the professor's spec required (X = [format, position, handed, age]).
debut_year_scaled is a proxy for career era / age-at-entry.
is_lefthanded and position_order will plug in automatically once scraping is available.

Usage:
    python fit_joint_batting_pymc_enriched.py [--tune 1000] [--draws 1000] [--chains 2]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import platformdirs

_LOCAL_CACHE = (Path(".cache") / "arviz").resolve()
_LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
platformdirs.user_cache_dir = lambda *a, **kw: str(_LOCAL_CACHE)

import arviz as az
import pymc as pm

DATA_PATH        = Path("Data") / "batting_joint_enriched.csv"
META_PATH        = Path("Data") / "batting_joint_meta.json"
OUTPUT_DIR       = Path("outputs") / "mixed_effects"
TRACE_PATH       = OUTPUT_DIR / "joint_batting_enriched_trace.nc"
SUMMARY_PATH     = OUTPUT_DIR / "joint_batting_enriched_pymc_summary.txt"
PARAMS_MEAN_PATH = OUTPUT_DIR / "joint_batting_enriched_params_mean.csv"
PLAYER_PATH      = OUTPUT_DIR / "batting_enriched_player_rankings.csv"

PREDICTORS = ["format_odi", "format_t20", "debut_year_scaled", "debut_year_missing",
              "is_lefthanded", "position_order", "age_at_debut_scaled"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tune",          type=int,   default=1000)
    p.add_argument("--draws",         type=int,   default=1000)
    p.add_argument("--chains",        type=int,   default=2)
    p.add_argument("--cores",         type=int,   default=1)
    p.add_argument("--target_accept", type=float, default=0.9)
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    # Fill any remaining NaN in predictors (debut_year_scaled already mean-imputed)
    for col in PREDICTORS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    X          = df[PREDICTORS].to_numpy(dtype=float)
    y          = df[["y1", "y2"]].to_numpy(dtype=float)
    player_idx = df["player_idx"].to_numpy(dtype=int)
    n_players  = int(df["player_idx"].max()) + 1
    n_pred     = X.shape[1]

    print(f"Data: {len(df)} rows, {n_players} players, {n_pred} predictors")
    print(f"Predictors: {PREDICTORS}")

    coords = {
        "obs_id":    np.arange(y.shape[0]),
        "outcome":   ["log_avg", "log_sr"],
        "predictor": PREDICTORS,
        "player":    np.arange(n_players),
    }

    with pm.Model(coords=coords) as model:
        X_data      = pm.Data("X",          X,          dims=("obs_id", "predictor"))
        player_data = pm.Data("player_idx", player_idx, dims=("obs_id",))

        alpha = pm.Normal("alpha", mu=0.0, sigma=2.5,  dims=("outcome",))
        beta  = pm.Normal("beta",  mu=0.0, sigma=1.5,  dims=("outcome", "predictor"))

        chol_u, corr_u, sd_u = pm.LKJCholeskyCov(
            "chol_u", n=2, eta=2.0,
            sd_dist=pm.HalfNormal.dist(1.0),
            compute_corr=True,
        )
        u = pm.MvNormal("u", mu=np.zeros(2), chol=chol_u, dims=("player", "outcome"))

        chol_eps, corr_eps, sd_eps = pm.LKJCholeskyCov(
            "chol_eps", n=2, eta=2.0,
            sd_dist=pm.HalfNormal.dist(1.0),
            compute_corr=True,
        )

        mu = alpha + pm.math.dot(X_data, beta.T) + u[player_data]
        pm.MvNormal("y_obs", mu=mu, chol=chol_eps, observed=y, dims=("obs_id", "outcome"))

        sample_kwargs = dict(
            tune=args.tune, draws=args.draws, chains=args.chains, cores=args.cores,
            target_accept=args.target_accept, random_seed=args.seed,
            return_inferencedata=True, progressbar=True,
        )
        idata = pm.sample(**sample_kwargs)

    az.to_netcdf(idata, TRACE_PATH)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = az.summary(
        idata,
        var_names=["alpha", "beta", "chol_u_stds", "chol_u_corr",
                   "chol_eps_stds", "chol_eps_corr"],
        round_to=4,
    )
    SUMMARY_PATH.write_text(summary.to_string(), encoding="utf-8")

    # ── Parameter means ───────────────────────────────────────────────────────
    post = idata.posterior
    alpha_mean = post["alpha"].mean(dim=("chain", "draw")).values
    beta_mean  = post["beta"].mean(dim=("chain", "draw")).values
    corr_u     = float(post["chol_u_corr"].mean(dim=("chain", "draw")).values[0, 1])
    corr_eps   = float(post["chol_eps_corr"].mean(dim=("chain", "draw")).values[0, 1])

    params = {"alpha_log_avg": float(alpha_mean[0]), "alpha_log_sr": float(alpha_mean[1])}
    for k, pred in enumerate(PREDICTORS):
        params[f"beta_{pred}_log_avg"] = float(beta_mean[0, k])
        params[f"beta_{pred}_log_sr"]  = float(beta_mean[1, k])
    params["player_corr(avg,sr)"]   = corr_u
    params["residual_corr(avg,sr)"] = corr_eps

    pd.DataFrame([params]).to_csv(PARAMS_MEAN_PATH, index=False)

    # ── Player rankings ───────────────────────────────────────────────────────
    idx_to_name = meta["player_idx_to_name"]
    u_mean = post["u"].mean(dim=("chain", "draw")).values
    rows = []
    for i in range(u_mean.shape[0]):
        psi_avg = float(u_mean[i, 0])
        psi_sr  = float(u_mean[i, 1])
        rows.append({
            "player_name":  idx_to_name.get(str(i), f"player_{i}"),
            "psi_log_avg":  psi_avg,
            "psi_log_sr":   psi_sr,
            "psi_combined": (2.0 * psi_avg + 1.0 * psi_sr) / 3.0,
        })
    psi_df = pd.DataFrame(rows)
    psi_df["rank"] = psi_df["psi_combined"].rank(ascending=False).astype(int)
    psi_df = psi_df.sort_values("rank").reset_index(drop=True)
    psi_df.to_csv(PLAYER_PATH, index=False)

    print(f"\nTrace:        {TRACE_PATH}")
    print(f"Summary:      {SUMMARY_PATH}")
    print(f"Params mean:  {PARAMS_MEAN_PATH}")
    print(f"Rankings:     {PLAYER_PATH}")
    print("\nTop 20 players (enriched model):")
    print(psi_df[["rank","player_name","psi_log_avg","psi_log_sr","psi_combined"]].head(20).to_string(index=False))

    # ── Beta interpretation ───────────────────────────────────────────────────
    print("\n=== Beta interpretation (enriched X effects) ===")
    for k, pred in enumerate(PREDICTORS):
        b_avg = float(beta_mean[0, k])
        b_sr  = float(beta_mean[1, k])
        print(f"  {pred:<25}  beta_avg={b_avg:+.4f} (x{np.exp(b_avg):.3f})  "
              f"beta_sr={b_sr:+.4f} (x{np.exp(b_sr):.3f})")


if __name__ == "__main__":
    main()
