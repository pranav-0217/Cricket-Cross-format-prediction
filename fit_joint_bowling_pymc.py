import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import platformdirs


# Force ArviZ cache/warning files into workspace-local storage to avoid
# restricted AppData paths on this machine.
LOCAL_ARVIZ_CACHE = (Path(".cache") / "arviz").resolve()
LOCAL_ARVIZ_CACHE.mkdir(parents=True, exist_ok=True)


def _workspace_cache_dir(*_args, **_kwargs) -> str:
    return str(LOCAL_ARVIZ_CACHE)


platformdirs.user_cache_dir = _workspace_cache_dir

import arviz as az
import pymc as pm


DATA_PATH = Path("Data") / "bowling_joint.csv"
META_PATH = Path("Data") / "bowling_joint_meta.json"
OUTPUT_DIR = Path("outputs") / "mixed_effects"
TRACE_PATH = OUTPUT_DIR / "joint_bowling_trace.nc"
SUMMARY_PATH = OUTPUT_DIR / "joint_bowling_pymc_summary.txt"
PARAMS_MEAN_PATH = OUTPUT_DIR / "joint_bowling_params_mean.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit joint multivariate mixed-effects bowling model in PyMC."
    )
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--target_accept", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_params_mean(idata: az.InferenceData) -> pd.DataFrame:
    posterior = idata.posterior

    alpha_mean = posterior["alpha"].mean(dim=("chain", "draw")).values
    beta_mean = posterior["beta"].mean(dim=("chain", "draw")).values
    corr_u_mean = float(posterior["chol_u_corr"].mean(dim=("chain", "draw")).values[0, 1])
    corr_eps_mean = float(posterior["chol_eps_corr"].mean(dim=("chain", "draw")).values[0, 1])

    params = {
        "alpha_y1_log_econ": float(alpha_mean[0]),
        "alpha_y2_log_avg": float(alpha_mean[1]),
        "beta_odi_y1_log_econ": float(beta_mean[0, 0]),
        "beta_t20_y1_log_econ": float(beta_mean[0, 1]),
        "beta_odi_y2_log_avg": float(beta_mean[1, 0]),
        "beta_t20_y2_log_avg": float(beta_mean[1, 1]),
        "corr_u_player_level_y1_y2": corr_u_mean,
        "corr_eps_residual_y1_y2": corr_eps_mean,
    }
    return pd.DataFrame([params])


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    _meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    X = df[["format_odi", "format_t20"]].to_numpy(dtype=float)
    y = df[["y1", "y2"]].to_numpy(dtype=float)
    player_idx = df["player_idx"].to_numpy(dtype=int)
    n_players = int(df["player_idx"].max()) + 1

    coords = {
        "obs_id": np.arange(y.shape[0]),
        "outcome": ["y1_log_econ", "y2_log_avg"],
        "predictor": ["format_odi", "format_t20"],
        "player": np.arange(n_players),
    }

    with pm.Model(coords=coords) as model:
        X_data = pm.Data("X", X, dims=("obs_id", "predictor"))
        player_data = pm.Data("player_idx", player_idx, dims=("obs_id",))

        alpha = pm.Normal("alpha", mu=0.0, sigma=2.5, dims=("outcome",))
        beta = pm.Normal("beta", mu=0.0, sigma=1.5, dims=("outcome", "predictor"))

        chol_u, corr_u, sd_u = pm.LKJCholeskyCov(
            "chol_u",
            n=2,
            eta=2.0,
            sd_dist=pm.HalfNormal.dist(1.0),
            compute_corr=True,
        )
        u = pm.MvNormal(
            "u",
            mu=np.zeros(2),
            chol=chol_u,
            dims=("player", "outcome"),
        )

        chol_eps, corr_eps, sd_eps = pm.LKJCholeskyCov(
            "chol_eps",
            n=2,
            eta=2.0,
            sd_dist=pm.HalfNormal.dist(1.0),
            compute_corr=True,
        )

        mu = alpha + pm.math.dot(X_data, beta.T) + u[player_data]

        pm.MvNormal(
            "y_obs",
            mu=mu,
            chol=chol_eps,
            observed=y,
            dims=("obs_id", "outcome"),
        )

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
            print("Multiprocessing blocked on this system; retrying with cores=1.")
            sample_kwargs["cores"] = 1
            idata = pm.sample(**sample_kwargs)

    az.to_netcdf(idata, TRACE_PATH)

    # ── Extract player random effects (psi) and rankings ─────────────────────
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    idx_to_name = meta["player_idx_to_name"]

    u_mean = idata.posterior["u"].mean(dim=("chain", "draw")).values  # (n_players, 2)
    rows = []
    for i in range(u_mean.shape[0]):
        name = idx_to_name.get(str(i), f"player_{i}")
        psi_econ = float(u_mean[i, 0])
        psi_avg  = float(u_mean[i, 1])
        # Lower econ and avg = better bowler, so negate for ranking
        # Mirrors batting: avg gets Test weight (2.0), econ gets T20 weight (1.0)
        combined = -(2.0 * psi_avg + 1.0 * psi_econ) / 3.0
        rows.append({
            "player_name":    name,
            "psi_log_econ":   psi_econ,
            "psi_log_avg":    psi_avg,
            "psi_combined":   combined,
            "econ_multiplier": float(np.exp(psi_econ)),
            "avg_multiplier":  float(np.exp(psi_avg)),
        })

    psi_df = pd.DataFrame(rows)
    psi_df["rank"] = psi_df["psi_combined"].rank(ascending=False).astype(int)
    psi_df = psi_df.sort_values("rank").reset_index(drop=True)

    player_effects_path = OUTPUT_DIR / "bowling_player_rankings.csv"
    psi_df.to_csv(player_effects_path, index=False)
    print(f"Bowling player rankings saved: {player_effects_path}")
    print("\nTop 20 bowlers by latent bowling ability:")
    print(psi_df[["rank","player_name","psi_log_econ","psi_log_avg","psi_combined"]].head(20).to_string(index=False))

    summary = az.summary(
        idata,
        var_names=[
            "alpha",
            "beta",
            "chol_u_stds",
            "chol_u_corr",
            "chol_eps_stds",
            "chol_eps_corr",
        ],
        round_to=4,
    )
    SUMMARY_PATH.write_text(summary.to_string(), encoding="utf-8")

    params_mean_df = build_params_mean(idata)
    params_mean_df.to_csv(PARAMS_MEAN_PATH, index=False)

    print("done")
    print(f"trace: {TRACE_PATH}")
    print(f"summary: {SUMMARY_PATH}")
    print(f"params mean: {PARAMS_MEAN_PATH}")


if __name__ == "__main__":
    main()
