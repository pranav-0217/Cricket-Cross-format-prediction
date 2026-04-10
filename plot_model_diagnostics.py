"""Model diagnostics for joint batting and bowling PyMC models.

Produces:
  1. Convergence diagnostics: Rhat bar chart, ESS bar chart, trace plots
  2. Posterior predictive checks: observed vs posterior-predicted distributions
  3. Parameter posterior plots (credible intervals)

Outputs saved to: outputs/diagnostics/
"""

from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import platformdirs

# Redirect ArviZ cache
_LOCAL_CACHE = (Path(".cache") / "arviz").resolve()
_LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
platformdirs.user_cache_dir = lambda *a, **kw: str(_LOCAL_CACHE)

OUTPUT_DIR   = Path("outputs") / "diagnostics"
BATTING_TRACE = Path("outputs") / "mixed_effects" / "joint_batting_trace.nc"
BOWLING_TRACE = Path("outputs") / "mixed_effects" / "joint_bowling_trace.nc"
BATTING_DATA  = Path("Data") / "batting_joint.csv"
BOWLING_DATA  = Path("Data") / "bowling_joint.csv"


# ── helpers ──────────────────────────────────────────────────────────────────

def rhat_ess_summary(idata: az.InferenceData, var_names: list, label: str):
    """Print and return Rhat / ESS summary table."""
    summ = az.summary(idata, var_names=var_names, round_to=4)
    print(f"\n{'='*60}")
    print(f"  {label} — Convergence summary")
    print(f"{'='*60}")
    cols = [c for c in ["mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess_bulk", "ess_tail"] if c in summ.columns]
    print(summ[cols].to_string())
    return summ


def plot_rhat(summ: pd.DataFrame, label: str, outpath: Path):
    if "r_hat" not in summ.columns:
        return
    rhat = summ["r_hat"].dropna().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(4, len(rhat) * 0.25)))
    colors = ["#d62728" if v > 1.05 else "#ff7f0e" if v > 1.01 else "#2ca02c" for v in rhat]
    ax.barh(range(len(rhat)), rhat.values, color=colors)
    ax.set_yticks(range(len(rhat)))
    ax.set_yticklabels(rhat.index, fontsize=7)
    ax.axvline(1.01, color="orange", linestyle="--", linewidth=1, label="Rhat=1.01")
    ax.axvline(1.05, color="red",    linestyle="--", linewidth=1, label="Rhat=1.05")
    ax.set_xlabel("R-hat")
    ax.set_title(f"{label} — R-hat (green<1.01 = good)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_traces(idata: az.InferenceData, var_names: list, label: str, outpath: Path):
    # Use only non-degenerate vars (skip corr matrices — diagonal is always 1.0)
    trace_vars = ["alpha", "beta", "chol_u_stds", "chol_eps_stds"]
    trace_vars = [v for v in trace_vars if v in var_names]
    axes = az.plot_trace(idata, var_names=trace_vars, compact=True,
                         figsize=(14, 3 * len(trace_vars)))
    fig = axes.ravel()[0].get_figure()
    fig.suptitle(f"{label} — Trace plots", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_posterior_params(idata: az.InferenceData, var_names: list, label: str, outpath: Path):
    post_vars = ["alpha", "beta", "chol_u_stds", "chol_eps_stds"]
    post_vars = [v for v in post_vars if v in var_names]
    axes = az.plot_posterior(idata, var_names=post_vars, figsize=(14, 3 * len(post_vars)))
    fig = np.atleast_1d(axes).ravel()[0].get_figure()
    fig.suptitle(f"{label} — Posterior distributions (94% HDI)", fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def manual_ppc(idata: az.InferenceData, df: pd.DataFrame,
               outcome_cols: tuple, outcome_labels: tuple,
               label: str, outpath: Path):
    """
    Posterior predictive check: compare observed outcomes to model-predicted means.
    Uses posterior means of alpha, beta, u (player effects).
    """
    post = idata.posterior

    alpha_mean = post["alpha"].mean(dim=("chain", "draw")).values   # (2,)
    beta_mean  = post["beta"].mean(dim=("chain", "draw")).values    # (2, n_pred)
    u_mean     = post["u"].mean(dim=("chain", "draw")).values       # (n_players, 2)

    X = df[["format_odi", "format_t20"]].to_numpy(dtype=float)
    player_idx = df["player_idx"].to_numpy(dtype=int)

    mu_pred = alpha_mean + X @ beta_mean.T + u_mean[player_idx]  # (n_obs, 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for k, (col, lbl) in enumerate(zip(outcome_cols, outcome_labels)):
        obs   = df[col].values
        pred  = mu_pred[:, k]
        ax    = axes[k]
        ax.hist(obs,  bins=60, alpha=0.5, density=True, label="Observed",  color="#1f77b4")
        ax.hist(pred, bins=60, alpha=0.5, density=True, label="Predicted (posterior mean)", color="#ff7f0e")
        ax.set_xlabel(lbl)
        ax.set_ylabel("Density")
        ax.set_title(f"PPC — {lbl}")
        ax.legend(fontsize=8)
    fig.suptitle(f"{label} — Posterior Predictive Check", fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_player_corr(idata: az.InferenceData, label: str, corr_var: str, outpath: Path):
    """Plot posterior distribution of the player-level correlation ρ_ψ."""
    post = idata.posterior
    corr_samples = post[corr_var].values[:, :, 0, 1].ravel()  # all chains & draws

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(corr_samples, bins=60, density=True, color="#9467bd", alpha=0.8)
    ax.axvline(corr_samples.mean(), color="black", linewidth=1.5,
               label=f"Mean = {corr_samples.mean():.3f}")
    ax.axvline(np.percentile(corr_samples, 2.5),  color="red", linestyle="--", linewidth=1, label="95% HDI")
    ax.axvline(np.percentile(corr_samples, 97.5), color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Correlation ρ")
    ax.set_ylabel("Density")
    ax.set_title(f"{label} — Player-level outcome correlation (ρ_ψ)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ── main ─────────────────────────────────────────────────────────────────────

def run_diagnostics(trace_path: Path, data_path: Path,
                    outcome_cols: tuple, outcome_labels: tuple,
                    key_vars: list, corr_var: str, label: str):

    print(f"\n{'#'*60}")
    print(f"  Loading {label} trace: {trace_path}")
    idata = az.from_netcdf(str(trace_path))
    df    = pd.read_csv(data_path)

    prefix = label.lower().replace(" ", "_")

    # 1. Convergence summary table
    summ = rhat_ess_summary(idata, key_vars, label)

    # 2. Rhat bar chart
    plot_rhat(summ, label, OUTPUT_DIR / f"{prefix}_rhat.png")

    # 3. Trace plots (compact)
    plot_traces(idata, key_vars, label, OUTPUT_DIR / f"{prefix}_trace.png")

    # 4. Posterior parameter plots
    plot_posterior_params(idata, key_vars, label, OUTPUT_DIR / f"{prefix}_posteriors.png")

    # 5. Manual PPC
    manual_ppc(idata, df, outcome_cols, outcome_labels, label, OUTPUT_DIR / f"{prefix}_ppc.png")

    # 6. Player-level correlation posterior
    plot_player_corr(idata, label, corr_var, OUTPUT_DIR / f"{prefix}_player_corr.png")

    return summ


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Batting ──────────────────────────────────────────────────────────────
    batting_summ = run_diagnostics(
        trace_path    = BATTING_TRACE,
        data_path     = BATTING_DATA,
        outcome_cols  = ("y1", "y2"),
        outcome_labels= ("log(avg)", "log(SR)"),
        key_vars      = ["alpha", "beta", "chol_u_corr", "chol_eps_corr",
                         "chol_u_stds", "chol_eps_stds"],
        corr_var      = "chol_u_corr",
        label         = "Batting",
    )

    # ── Bowling ──────────────────────────────────────────────────────────────
    bowling_summ = run_diagnostics(
        trace_path    = BOWLING_TRACE,
        data_path     = BOWLING_DATA,
        outcome_cols  = ("y1", "y2"),
        outcome_labels= ("log(econ)", "log(avg)"),
        key_vars      = ["alpha", "beta", "chol_u_corr", "chol_eps_corr",
                         "chol_u_stds", "chol_eps_stds"],
        corr_var      = "chol_u_corr",
        label         = "Bowling",
    )

    # ── Combined Rhat summary ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  CONVERGENCE SUMMARY")
    print("="*60)
    for label, summ in [("Batting", batting_summ), ("Bowling", bowling_summ)]:
        if "r_hat" in summ.columns:
            rh = summ["r_hat"].dropna()
            n_bad = (rh > 1.05).sum()
            n_warn = ((rh > 1.01) & (rh <= 1.05)).sum()
            print(f"  {label}: max_Rhat={rh.max():.4f}, "
                  f"Rhat>1.05: {n_bad}, Rhat>1.01: {n_warn}, "
                  f"Rhat<=1.01 (good): {(rh<=1.01).sum()}/{len(rh)}")

    print(f"\nAll diagnostics saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
