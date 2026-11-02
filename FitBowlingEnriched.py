"""Fit the current complete-covariate joint bowling model."""

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

import ResidualExposure as rx

DATA_PATH        = Path("Data/new/processed/bowling_joint_enriched.csv")
META_PATH        = Path("Data/new/processed/bowling_joint_enriched_meta.json")
OUTPUT_DIR       = Path("outputs") / "mixed_effects"
TRACE_PATH       = OUTPUT_DIR / "joint_bowling_enriched_trace.nc"
SUMMARY_PATH     = OUTPUT_DIR / "joint_bowling_enriched_pymc_summary.txt"
PARAMS_MEAN_PATH = OUTPUT_DIR / "joint_bowling_enriched_params_mean.csv"
PLAYER_PATH      = OUTPUT_DIR / "bowling_enriched_player_rankings.csv"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tune",          type=int,   default=1000)
    p.add_argument("--draws",         type=int,   default=1000)
    p.add_argument("--chains",        type=int,   default=2)
    p.add_argument("--cores",         type=int,   default=1)
    p.add_argument("--target_accept", type=float, default=0.9)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument(
        "--no-exposure-weighting",
        action="store_true",
        help="Use one shared residual covariance instead of the two-component, "
             "exposure-weighted form. For comparison against the weighted fit.",
    )
    p.add_argument(
        "--exclude-players",
        type=Path,
        default=None,
        help="File of player_id values (one per line) to drop before fitting. "
             "Used for leakage-free K-fold evaluation: the held-out fold must "
             "not influence the fixed effects, variance components or any "
             "training player's psi.",
    )
    p.add_argument(
        "--out-suffix",
        type=str,
        default="",
        help="Suffix appended to every output filename, so fold fits do not "
             "overwrite the production outputs.",
    )
    return p.parse_args()

def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sfx = args.out_suffix
    trace_path  = TRACE_PATH.with_name(TRACE_PATH.stem + sfx + TRACE_PATH.suffix)
    summary_path = SUMMARY_PATH.with_name(SUMMARY_PATH.stem + sfx + SUMMARY_PATH.suffix)
    params_path = PARAMS_MEAN_PATH.with_name(PARAMS_MEAN_PATH.stem + sfx + PARAMS_MEAN_PATH.suffix)
    player_path = PLAYER_PATH.with_name(PLAYER_PATH.stem + sfx + PLAYER_PATH.suffix)

    df   = pd.read_csv(DATA_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    predictors = meta["predictors"]

    if args.exclude_players is not None:
        drop = {int(line) for line in args.exclude_players.read_text().split() if line.strip()}
        before = len(df)
        df = df[~df["player_id"].isin(drop)].copy()
        # player_idx must stay contiguous from zero after dropping held-out players.
        ids = sorted(df["player_id"].unique())
        remap = {pid: i for i, pid in enumerate(ids)}
        df["player_idx"] = df["player_id"].map(remap).astype(int)
        print(f"Holdout: dropped {len(drop)} players, {before - len(df)} rows; "
              f"fitting on {len(df)} rows from {len(ids)} players")

    absent = [column for column in predictors if column not in df]
    if absent:
        raise ValueError(f"Current bowling data lacks predictors: {absent}")
    if df[predictors].isna().any().any():
        raise ValueError("Current bowling predictors contain missing values")

    X          = df[predictors].to_numpy(dtype=float)
    y          = df[["y1", "y2"]].to_numpy(dtype=float)
    player_idx = df["player_idx"].to_numpy(dtype=int)
    n_players  = int(df["player_idx"].max()) + 1

    print(f"Data: {len(df)} rows, {n_players} players, {X.shape[1]} predictors")
    print(f"Predictors: {predictors}")

    coords = {
        "obs_id":    np.arange(y.shape[0]),
        "outcome":   ["log_econ", "log_avg"],
        "predictor": predictors,
        "player":    np.arange(n_players),
    }

    weighted = not args.no_exposure_weighting
    exposure_w = rx.exposure_matrix(df, meta) if weighted else None
    print("Residual: " + ("exposure-weighted (two-component)" if weighted
                          else "single shared covariance"))

    with pm.Model(coords=coords) as model:
        X_data      = pm.Data("X",          X,          dims=("obs_id", "predictor"))
        player_data = pm.Data("player_idx", player_idx, dims=("obs_id",))

        alpha = pm.Normal("alpha", mu=0.0, sigma=2.5, dims=("outcome",))
        beta  = pm.Normal("beta",  mu=0.0, sigma=1.5, dims=("outcome", "predictor"))

        chol_u, corr_u, sd_u = pm.LKJCholeskyCov(
            "chol_u", n=2, eta=2.0,
            sd_dist=pm.HalfNormal.dist(1.0),
            compute_corr=True,
        )
        u = pm.MvNormal("u", mu=np.zeros(2), chol=chol_u, dims=("player", "outcome"))

        mu = alpha + pm.math.dot(X_data, beta.T) + u[player_data]

        # Residual variance per outcome: sigma_sampling^2 / exposure + sigma_real^2.
        rx.residual_likelihood(mu, y, exposure_w)

        idata = pm.sample(
            tune=args.tune, draws=args.draws, chains=args.chains, cores=args.cores,
            target_accept=args.target_accept, random_seed=args.seed,
            return_inferencedata=True, progressbar=True,
        )

    idata.to_netcdf(trace_path)

    summary = az.summary(
        idata,
        var_names=["alpha", "beta", "chol_u_stds", "chol_u_corr",
                   *rx.summary_vars(weighted),],
        round_to=4,
    )
    summary_path.write_text(summary.to_string(), encoding="utf-8")

    post       = idata.posterior
    alpha_mean = post["alpha"].mean(dim=("chain", "draw")).values
    beta_mean  = post["beta"].mean(dim=("chain", "draw")).values
    corr_u     = float(post["chol_u_corr"].mean(dim=("chain", "draw")).values[0, 1])
    corr_eps   = rx.residual_corr(post)

    params = {"alpha_log_econ": float(alpha_mean[0]), "alpha_log_avg": float(alpha_mean[1])}
    for k, pred in enumerate(predictors):
        params[f"beta_{pred}_log_econ"] = float(beta_mean[0, k])
        params[f"beta_{pred}_log_avg"]  = float(beta_mean[1, k])
    params["player_corr(econ,avg)"]   = corr_u
    params["residual_corr(econ,avg)"] = corr_eps
    for var in ("chol_u_stds", "sigma_sampling", "sigma_real"):
        if var in post:
            means = post[var].mean(dim=("chain", "draw")).values
            params[f"{var}_log_econ"] = float(means[0])
            params[f"{var}_log_avg"]  = float(means[1])
    pd.DataFrame([params]).to_csv(params_path, index=False)

    # Built from the fitted rows so it stays correct after a holdout re-index.
    local = df[["player_idx", "player_id", "player_name"]].drop_duplicates("player_idx")
    idx_to_id = {str(r.player_idx): r.player_id for r in local.itertuples()}
    idx_to_name = {str(r.player_idx): r.player_name for r in local.itertuples()}
    u_mean = post["u"].mean(dim=("chain", "draw")).values
    rows = []
    for i in range(u_mean.shape[0]):
        psi_econ = float(u_mean[i, 0])
        psi_avg  = float(u_mean[i, 1])
        rows.append({
            "player_id":    idx_to_id.get(str(i), ""),
            "player_name":  idx_to_name.get(str(i), f"player_{i}"),
            "psi_log_econ": psi_econ,
            "psi_log_avg":  psi_avg,
            "psi_combined": -(2.0 * psi_avg + 1.0 * psi_econ) / 3.0,
        })
    psi_df = pd.DataFrame(rows)
    psi_df["rank"] = psi_df["psi_combined"].rank(ascending=False).astype(int)
    psi_df = psi_df.sort_values("rank").reset_index(drop=True)
    psi_df.to_csv(player_path, index=False)

    print(f"\nTrace:        {trace_path}")
    print(f"Summary:      {summary_path}")
    print(f"Params mean:  {PARAMS_MEAN_PATH}")
    print(f"Rankings:     {player_path}")
    print("\nTop 20 bowlers (enriched model):")
    print(psi_df[["rank", "player_name", "psi_log_econ", "psi_log_avg", "psi_combined"]]
          .head(20).to_string(index=False))

    print("\n=== Beta interpretation (enriched X effects) ===")
    for k, pred in enumerate(predictors):
        b_econ = float(beta_mean[0, k])
        b_avg  = float(beta_mean[1, k])
        print(f"  {pred:<25}  beta_econ={b_econ:+.4f} (x{np.exp(b_econ):.3f})  "
              f"beta_avg={b_avg:+.4f} (x{np.exp(b_avg):.3f})")

if __name__ == "__main__":
    main()
