"""Plot the fitted one-knot linear age curve from the enriched models."""

from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("Data/new/processed")
ME_DIR = Path("outputs") / "mixed_effects"
OUT_DIR = Path("outputs") / "plots"

FAMILIES = {
    "batting": ("Batting average", "Batting strike rate"),
    "bowling": ("Bowling economy rate", "Bowling average"),
}

# Outcomes where a lower value is the better performance.
LOWER_IS_BETTER = {"Bowling economy rate", "Bowling average"}

def curve_draws(beta_age, beta_hinge, ages, knot, mean_age, sd_age):
    """Percent change vs the knot, for every posterior draw. Shape (draws, ages)."""
    age_z = (ages - mean_age) / sd_age
    hinge_z = np.maximum(0.0, ages - knot) / sd_age
    knot_z = (knot - mean_age) / sd_age
    # Value at the knot, subtracted so the curve is anchored there.
    eta = (beta_age[:, None] * (age_z - knot_z)[None, :]
           + beta_hinge[:, None] * hinge_z[None, :])
    return (np.exp(eta) - 1.0) * 100.0

def collect() -> list[dict]:
    """Fitted curve summaries for all four outcomes, in panel order."""
    panels: list[dict] = []
    for kind, labels in FAMILIES.items():
        meta = json.loads((DATA_DIR / f"{kind}_joint_enriched_meta.json").read_text())
        preds = meta["predictors"]
        if "age_scaled" not in preds or "age_hinge" not in preds:
            raise SystemExit(f"{kind}: no age terms in the current predictor list -- "
                             "rebuild with PrepareData.py first")
        knot = float(meta["age_curve"]["knot_age_years"])

        data = pd.read_csv(DATA_DIR / f"{kind}_joint_enriched.csv")
        mean_age, sd_age = data["avg_age"].mean(), data["avg_age"].std(ddof=0)

        idata = az.from_netcdf(str(ME_DIR / f"joint_{kind}_enriched_trace.nc"))
        beta = idata.posterior["beta"]
        if beta.shape[-1] != len(preds):
            raise SystemExit(f"{kind}: trace has {beta.shape[-1]} predictors but the data "
                             f"defines {len(preds)} -- refit the enriched model first")
        ia, ih = preds.index("age_scaled"), preds.index("age_hinge")

        lo_age = float(np.percentile(data["avg_age"], 1))
        hi_age = float(np.percentile(data["avg_age"], 99))
        ages = np.linspace(lo_age, hi_age, 160)

        for row, label in enumerate(labels):
            b_age = beta[:, :, row, ia].values.ravel()
            b_hin = beta[:, :, row, ih].values.ravel()
            draws = curve_draws(b_age, b_hin, ages, knot, mean_age, sd_age)
            hdi_lo, hdi_hi = az.hdi(b_hin, prob=0.94)
            panels.append({
                "kind": kind, "label": label, "knot": knot, "ages": ages,
                "mid": draws.mean(axis=0),
                "band": np.percentile(draws, [3, 97], axis=0),
                "before": b_age.mean() / sd_age * 100,
                "after": (b_age.mean() + b_hin.mean()) / sd_age * 100,
                "excludes_zero": (hdi_lo > 0) == (hdi_hi > 0),
                "avg_age": data["avg_age"].to_numpy(),
                "lo_age": lo_age, "hi_age": hi_age,
            })
    return panels

def plot_simple(panels: list[dict]) -> None:
    """Batting only, side by side: curve, band, knot, slopes."""
    batting = [p for p in panels if p["kind"] == "batting"]
    if not batting:
        raise SystemExit("no batting panels to plot")
    # Compact figure with enlarged type: the paper scales this down considerably.
    fig, axes = plt.subplots(1, len(batting), figsize=(11, 4.0))
    for ax, p in zip(np.atleast_1d(axes).ravel(), batting):
        ax.tick_params(labelsize=11)
        ax.fill_between(p["ages"], p["band"][0], p["band"][1],
                        color="#1976D2", alpha=0.18, linewidth=0)
        ax.plot(p["ages"], p["mid"], color="#1976D2", lw=2.5)
        ax.axvline(p["knot"], color="black", ls="--", lw=1.2)
        ax.axhline(0, color="grey", lw=0.8, alpha=0.6)
        note = "  (lower is better)" if p["label"] in LOWER_IS_BETTER else ""
        ax.set_title(f"{p['label']}{note}\n{p['before']:+.1f}% per year before, "
                     f"{p['after']:+.1f}% per year after",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Career-midpoint age (years)", fontsize=12)
        ax.set_ylabel(f"% difference vs age {p['knot']:g}", fontsize=12)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "age_curve_simple.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_simple(collect())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for col, (kind, labels) in enumerate(FAMILIES.items()):
        meta = json.loads((DATA_DIR / f"{kind}_joint_enriched_meta.json").read_text())
        preds = meta["predictors"]
        if "age_scaled" not in preds or "age_hinge" not in preds:
            raise SystemExit(f"{kind}: no age terms in the current predictor list -- "
                             "rebuild with PrepareData.py first")
        knot = float(meta["age_curve"]["knot_age_years"])

        data = pd.read_csv(DATA_DIR / f"{kind}_joint_enriched.csv")
        mean_age, sd_age = data["avg_age"].mean(), data["avg_age"].std(ddof=0)

        idata = az.from_netcdf(str(ME_DIR / f"joint_{kind}_enriched_trace.nc"))
        beta = idata.posterior["beta"]
        if beta.shape[-1] != len(preds):
            raise SystemExit(f"{kind}: trace has {beta.shape[-1]} predictors but the data "
                             f"defines {len(preds)} -- refit the enriched model first")
        ia, ih = preds.index("age_scaled"), preds.index("age_hinge")

        lo_age = float(np.percentile(data["avg_age"], 1))
        hi_age = float(np.percentile(data["avg_age"], 99))
        ages = np.linspace(lo_age, hi_age, 160)

        for row, label in enumerate(labels):
            ax = axes[row, col]
            b_age = beta[:, :, row, ia].values.ravel()
            b_hin = beta[:, :, row, ih].values.ravel()
            draws = curve_draws(b_age, b_hin, ages, knot, mean_age, sd_age)
            mid = draws.mean(axis=0)
            band = np.percentile(draws, [3, 97], axis=0)

            ax.fill_between(ages, band[0], band[1], color="#1976D2", alpha=0.16,
                            linewidth=0)
            ax.plot(ages, mid, color="#1976D2", lw=2)
            ax.axvline(knot, color="black", ls="--", lw=1, alpha=0.65)
            ax.axhline(0, color="grey", lw=0.8, alpha=0.5)

            before = b_age.mean() / sd_age * 100
            after = (b_age.mean() + b_hin.mean()) / sd_age * 100
            hdi_lo, hdi_hi = az.hdi(b_hin, prob=0.94)
            verdict = "excludes 0" if (hdi_lo > 0) == (hdi_hi > 0) else "includes 0"

            ax.set_title(
                f"{label}\nknot {knot:g}  |  {before:+.2f}%/yr before, "
                f"{after:+.2f}%/yr after  |  slope change 94% HDI {verdict}",
                fontsize=10, fontweight="bold")
            ax.set_xlabel("Career-midpoint age in format (years)", fontsize=9)
            ax.set_ylabel(f"% difference vs age {knot:g}", fontsize=9)
            ax.grid(True, alpha=0.3)

            # Where the data actually is: the post-knot segment is thinly supported.
            twin = ax.twinx()
            twin.hist(data["avg_age"], bins=40, range=(lo_age, hi_age),
                      color="grey", alpha=0.16, linewidth=0)
            twin.set_yticks([])
            twin.set_zorder(0)
            ax.set_zorder(1)
            ax.patch.set_visible(False)

            share = (data["avg_age"] > knot).mean() * 100
            ax.annotate(f"{share:.0f}% of rows lie above the knot",
                        xy=(0.98, 0.04), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=8, color="dimgrey")

    fig.suptitle("Fitted population-level age curve (one-knot linear spline)\n"
                 "Posterior mean with 94% credible band; grey shows the age distribution",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "age_curve.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
