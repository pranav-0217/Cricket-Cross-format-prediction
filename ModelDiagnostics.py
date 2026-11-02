"""Model diagnostics for joint batting and bowling PyMC models."""

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
BATTING_DATA  = Path("Data/new/processed/batting_joint.csv")
BOWLING_DATA  = Path("Data/new/processed/bowling_joint.csv")

def rhat_ess_summary(idata: az.InferenceData, var_names: list, label: str):
    """Print and return Rhat / ESS summary table."""
    summ = az.summary(idata, var_names=var_names, round_to=4)
    print(f"\n{'='*60}")
    print(f"  {label} — Convergence summary")
    print(f"{'='*60}")
    cols = [c for c in ["mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess_bulk", "ess_tail"]
            if c in summ.columns]
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

# Drawn with matplotlib directly; ArviZ 1.x dropped the options these plots need.

PLOT_VARS = ["alpha", "beta", "chol_u_stds",
             "sigma_sampling", "sigma_real", "chol_eps_stds"]

def _chain_draw_series(idata: az.InferenceData, var: str):
    """Yield (dim_label, array[chain, draw]) for every scalar element of `var`."""
    da = idata.posterior[var]
    extra = [d for d in da.dims if d not in ("chain", "draw")]
    if not extra:
        yield var, da.values
        return
    stacked = da.stack(element=extra).transpose("chain", "draw", "element")
    coords = stacked.coords["element"].values
    for k in range(stacked.sizes["element"]):
        raw = coords[k]
        parts = raw if isinstance(raw, tuple) else (raw,)
        yield f"{var}[{', '.join(str(p) for p in parts)}]", stacked.values[:, :, k]

def _selected_vars(idata: az.InferenceData, var_names: list) -> list:
    return [v for v in PLOT_VARS if v in var_names and v in idata.posterior]

def plot_traces(idata: az.InferenceData, var_names: list, label: str, outpath: Path):
    """One panel per variable, all of its elements overlaid (compact style)."""
    plot_vars = _selected_vars(idata, var_names)
    fig, axes = plt.subplots(len(plot_vars), 1,
                             figsize=(14, 2.8 * len(plot_vars)), squeeze=False)
    for ax, var in zip(axes[:, 0], plot_vars):
        n_elem = 0
        for _, series in _chain_draw_series(idata, var):
            for chain in range(series.shape[0]):
                ax.plot(series[chain], lw=0.5, alpha=0.55)
            n_elem += 1
        ax.set_title(f"{var}  ({n_elem} element{'s' if n_elem != 1 else ''}, "
                     f"{series.shape[0]} chains)", fontsize=10, fontweight="bold")
        ax.set_xlabel("draw", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{label} — Trace plots", fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")

def plot_posterior_params(idata: az.InferenceData, var_names: list, label: str, outpath: Path):
    """Posterior histogram per variable with the pooled 94% HDI marked."""
    plot_vars = _selected_vars(idata, var_names)
    fig, axes = plt.subplots(len(plot_vars), 1,
                             figsize=(14, 2.8 * len(plot_vars)), squeeze=False)
    for ax, var in zip(axes[:, 0], plot_vars):
        for name, series in _chain_draw_series(idata, var):
            flat = series.reshape(-1)
            ax.hist(flat, bins=60, density=True, alpha=0.45,
                    histtype="stepfilled", label=name)
            lo, hi = az.hdi(flat, prob=0.94)
            ax.axvline(float(flat.mean()), color="darkred", lw=1.0)
            ax.axvspan(float(lo), float(hi), alpha=0.08, color="grey")
        n = len(list(_chain_draw_series(idata, var)))
        if n <= 8:
            ax.legend(fontsize=6, ncol=2)
        ax.set_title(f"{var}  (mean line, 94% HDI shaded)", fontsize=10, fontweight="bold")
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{label} — Posterior distributions (94% HDI)", fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")

def manual_ppc(idata: az.InferenceData, df: pd.DataFrame,
               outcome_cols: tuple, outcome_labels: tuple,
               label: str, outpath: Path):
    """Posterior predictive check: compare observed outcomes to model-predicted means."""
    post = idata.posterior

    alpha_mean = post["alpha"].mean(dim=("chain", "draw")).values
    beta_mean  = post["beta"].mean(dim=("chain", "draw")).values
    u_mean     = post["u"].mean(dim=("chain", "draw")).values

    X = df[["format_odi", "format_t20"]].to_numpy(dtype=float)
    player_idx = df["player_idx"].to_numpy(dtype=int)

    mu_pred = alpha_mean + X @ beta_mean.T + u_mean[player_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for k, (col, lbl) in enumerate(zip(outcome_cols, outcome_labels)):
        obs   = df[col].values
        pred  = mu_pred[:, k]
        ax    = axes[k]
        ax.hist(obs,  bins=60, alpha=0.5, density=True, label="Observed",  color="#1f77b4")
        ax.hist(pred, bins=60, alpha=0.5, density=True,
                label="Predicted (posterior mean)", color="#ff7f0e")
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
    corr_samples = post[corr_var].values[:, :, 0, 1].ravel()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(corr_samples, bins=60, density=True, color="#9467bd", alpha=0.8)
    ax.axvline(corr_samples.mean(), color="black", linewidth=1.5,
               label=f"Mean = {corr_samples.mean():.3f}")
    ax.axvline(np.percentile(corr_samples, 2.5), color="red", linestyle="--", linewidth=1,
               label="95% HDI")
    ax.axvline(np.percentile(corr_samples, 97.5), color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Correlation ρ")
    ax.set_ylabel("Density")
    ax.set_title(f"{label} — Player-level outcome correlation (ρ_ψ)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"  Saved: {outpath}")

def run_diagnostics(trace_path: Path, data_path: Path,
                    outcome_cols: tuple, outcome_labels: tuple,
                    key_vars: list, corr_var: str, label: str):

    print(f"\n{'#'*60}")
    print(f"  Loading {label} trace: {trace_path}")
    idata = az.from_netcdf(str(trace_path))
    df    = pd.read_csv(data_path)

    prefix = label.lower().replace(" ", "_")
    summ = rhat_ess_summary(idata, key_vars, label)
    plot_rhat(summ, label, OUTPUT_DIR / f"{prefix}_rhat.png")
    plot_traces(idata, key_vars, label, OUTPUT_DIR / f"{prefix}_trace.png")
    plot_posterior_params(idata, key_vars, label, OUTPUT_DIR / f"{prefix}_posteriors.png")
    manual_ppc(idata, df, outcome_cols, outcome_labels, label, OUTPUT_DIR / f"{prefix}_ppc.png")
    plot_player_corr(idata, label, corr_var, OUTPUT_DIR / f"{prefix}_player_corr.png")

    return summ

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    batting_summ = run_diagnostics(
        trace_path    = BATTING_TRACE,
        data_path     = BATTING_DATA,
        outcome_cols  = ("y1", "y2"),
        outcome_labels= ("log(avg)", "log(SR)"),
        key_vars      = ["alpha", "beta", "chol_u_corr", "chol_u_stds",
                         "sigma_sampling", "sigma_real", "eps_corr"],
        corr_var      = "chol_u_corr",
        label         = "Batting",
    )

    bowling_summ = run_diagnostics(
        trace_path    = BOWLING_TRACE,
        data_path     = BOWLING_DATA,
        outcome_cols  = ("y1", "y2"),
        outcome_labels= ("log(econ)", "log(avg)"),
        key_vars      = ["alpha", "beta", "chol_u_corr", "chol_u_stds",
                         "sigma_sampling", "sigma_real", "eps_corr"],
        corr_var      = "chol_u_corr",
        label         = "Bowling",
    )

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
