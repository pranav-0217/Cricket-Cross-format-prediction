"""Fit multivariate mixed-effects batting model in PyMC."""

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

import ResidualExposure as rx

DATA_PATH = Path("Data/new/processed/batting_joint.csv")
META_PATH = Path("Data/new/processed/batting_joint_enriched_meta.json")
OUTPUT_DIR = Path("outputs") / "mixed_effects"
TRACE_PATH = OUTPUT_DIR / "joint_batting_trace.nc"
SUMMARY_PATH = OUTPUT_DIR / "joint_batting_pymc_summary.txt"
PARAMS_MEAN_PATH = OUTPUT_DIR / "joint_batting_params_mean.csv"
PLAYER_EFFECTS_PATH = OUTPUT_DIR / "batting_player_rankings.csv"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Joint batting mixed-effects model (PyMC)")
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--cores", type=int, default=1)
    p.add_argument("--target_accept", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-exposure-weighting",
        action="store_true",
        help="Use one shared residual covariance instead of the two-component, "
             "exposure-weighted form. For comparison against the weighted fit.",
    )
    return p.parse_args()

def extract_player_effects(idata: az.InferenceData, meta: dict) -> pd.DataFrame:
    """Extract posterior mean of psi (player random effects) and rank players."""
    psi_mean = idata.posterior["u"].mean(dim=("chain", "draw")).values

    idx_to_name = meta["player_idx_to_name"]
    idx_to_id = meta["player_idx_to_id"]
    n_players = psi_mean.shape[0]

    rows = []
    for i in range(n_players):
        name = idx_to_name.get(str(i), f"player_{i}")
        psi_avg = float(psi_mean[i, 0])
        psi_sr = float(psi_mean[i, 1])
        combined = (psi_avg + psi_sr) / 2.0
        rows.append({
            "player_idx": i,
            "player_id": idx_to_id.get(str(i), ""),
            "player_name": name,
            "psi_log_avg": psi_avg,
            "psi_log_sr": psi_sr,
            "psi_combined": combined,
        })

    df = pd.DataFrame(rows)
    df["rank"] = df["psi_combined"].rank(ascending=False).astype(int)
    df = df.sort_values("rank").reset_index(drop=True)
    return df

def build_params_summary(idata: az.InferenceData) -> pd.DataFrame:
    post = idata.posterior
    alpha_mean = post["alpha"].mean(dim=("chain", "draw")).values
    beta_mean = post["beta"].mean(dim=("chain", "draw")).values
    corr_u = float(post["chol_u_corr"].mean(dim=("chain", "draw")).values[0, 1])
    corr_eps = rx.residual_corr(post)

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

    X = df[["format_odi", "format_t20"]].to_numpy(dtype=float)
    # Outcomes: y = [log(avg), log(sr)]
    y = df[["y1", "y2"]].to_numpy(dtype=float)
    player_idx = df["player_idx"].to_numpy(dtype=int)
    n_players = int(df["player_idx"].max()) + 1

    print(f"Data: {len(df)} rows, {n_players} players, {y.shape[1]} outcomes")

    coords = {
        "obs_id":    np.arange(y.shape[0]),
        "outcome":   ["log_avg", "log_sr"],
        "predictor": ["format_odi", "format_t20"],
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

        # Player random effects psi ~ MVN(0, Sigma_u)
        chol_u, corr_u, sd_u = pm.LKJCholeskyCov(
            "chol_u", n=2, eta=2.0,
            sd_dist=pm.HalfNormal.dist(1.0),
            compute_corr=True,
        )
        u = pm.MvNormal("u", mu=np.zeros(2), chol=chol_u, dims=("player", "outcome"))

        mu = alpha + pm.math.dot(X_data, beta.T) + u[player_data]

        # Residual variance per outcome: sigma_sampling^2 / exposure + sigma_real^2.
        rx.residual_likelihood(mu, y, exposure_w)

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

    idata.to_netcdf(TRACE_PATH)
    print(f"Trace saved: {TRACE_PATH}")

    summary = az.summary(
        idata,
        var_names=["alpha", "beta", "chol_u_stds", "chol_u_corr",
                   *rx.summary_vars(weighted),],
        round_to=4,
    )
    SUMMARY_PATH.write_text(summary.to_string(), encoding="utf-8")
    print(f"Summary saved: {SUMMARY_PATH}")

    params_df = build_params_summary(idata)
    params_df.to_csv(PARAMS_MEAN_PATH, index=False)
    print(f"Params mean saved: {PARAMS_MEAN_PATH}")

    player_df = extract_player_effects(idata, meta)
    player_df.to_csv(PLAYER_EFFECTS_PATH, index=False)
    print(f"Player rankings saved: {PLAYER_EFFECTS_PATH}")
    print("\nTop 20 players by latent batting ability:")
    print(player_df[["rank", "player_name", "psi_log_avg", "psi_log_sr", "psi_combined"]]
          .head(20).to_string(index=False))

if __name__ == "__main__":
    main()
