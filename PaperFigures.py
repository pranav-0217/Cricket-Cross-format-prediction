"""Every figure in the paper, plus the percent-change table behind Tables 1 and 2.
Run after the four fits and EvaluateKFold.py."""

from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

DATA_DIR = Path("Data/new/processed")
ME_DIR = Path("outputs") / "mixed_effects"
OUT_DIR = Path("outputs") / "paper_figures"

FORMATS = ("Test", "ODI", "T20")
COLOURS = {"Test": "#1976D2", "ODI": "#388E3C", "T20": "#F57C00"}
INK, MUTED = "#0b0b0b", "#52514e"
WIDTH, DPI, TOP_N = 14.4, 200, 10
FAMILIES = {"batting": ("Batting average", "Batting strike rate"),
            "bowling": ("Bowling economy rate", "Bowling average")}
# Drawn 14.4in wide and printed at text width, these sizes come out near 8-10pt.
STYLE = {"font.size": 19, "axes.titlesize": 23, "axes.labelsize": 20,
         "xtick.labelsize": 17, "ytick.labelsize": 17, "axes.titleweight": "normal",
         "font.family": "serif", "font.serif": ["cmr10"],
         "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True}
CLEAN = {**STYLE, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
         "axes.spines.top": False, "axes.spines.right": False}
# Figures 3 and 4 are drawn at their printed size and saved as vector PDFs.
PRINT = {"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
         "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
         "axes.formatter.use_mathtext": True, "pdf.fonttype": 42,
         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
         "xtick.major.size": 2.5, "ytick.major.size": 2.5,
         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
         "axes.spines.top": False, "axes.spines.right": False}


def enriched_beta(kind: str) -> tuple[np.ndarray, list[str]]:
    """Posterior draws of beta, shape (draws, outcome, predictor)."""
    preds = json.loads((DATA_DIR / f"{kind}_joint_enriched_meta.json").read_text())["predictors"]
    beta = az.from_netcdf(ME_DIR / f"joint_{kind}_enriched_trace.nc").posterior["beta"].values
    return beta.reshape(-1, *beta.shape[2:]), preds


def pct(b):
    return 100 * np.expm1(b)


def effects_table() -> None:
    """Every enriched fixed effect as percent change with its 94% HDI."""
    rows = []
    for kind, labels in FAMILIES.items():
        beta, preds = enriched_beta(kind)
        for j, p in enumerate(preds):
            for o, outcome in enumerate(labels):
                lo, hi = pct(az.hdi(beta[:, o, j], prob=0.94))
                rows.append({"model": kind, "predictor": p, "outcome": outcome,
                             "pct_mean": round(pct(beta[:, o, j].mean()), 1),
                             "hdi_3%": round(lo, 1), "hdi_97%": round(hi, 1)})
    pd.DataFrame(rows).to_csv(ME_DIR / "enriched_effects_pct.csv", index=False)


def raw_outcomes() -> None:
    """Figure 1."""
    bat = pd.read_csv(DATA_DIR / "batting_long.csv")
    bowl = pd.read_csv(DATA_DIR / "bowling_long.csv")
    rows = [(bat, "avg", "sr", "Batting average", "Strike rate"),
            (bowl, "avg", "econ", "Bowling average", "Economy rate")]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 6.63), layout="constrained")
        for r, (df, x, y, xl, yl) in enumerate(rows):
            for c, fmt in enumerate(FORMATS):
                ax, s = axes[r, c], df[df["format"] == fmt]
                ax.scatter(s[x], s[y], s=18, alpha=0.5, color=COLOURS[fmt], edgecolors="none")
                ax.set_xlim(df[x].min() * 0.95, df[x].max() * 1.02)
                ax.set_ylim(df[y].min() * 0.95, df[y].max() * 1.02)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel(xl)
                ax.text(0.96, 0.93, f"n = {len(s):,}", transform=ax.transAxes,
                        fontsize=18, va="top", ha="right", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
                if r == 0:
                    ax.set_title(fmt)
                if c == 0:
                    ax.set_ylabel(yl)
                else:
                    ax.tick_params(labelleft=False)
        fig.savefig(OUT_DIR / "poster_raw_outcomes.png", dpi=DPI)
        plt.close(fig)


def format_posteriors() -> None:
    """Figure 2: rows are outcomes, columns are format comparisons."""
    bat, _ = enriched_beta("batting")
    bowl, _ = enriched_beta("bowling")
    rows = [("Batting\naverage", bat, 0), ("Batting\nstrike rate", bat, 1),
            ("Bowling\neconomy rate", bowl, 0), ("Bowling\naverage", bowl, 1)]
    cols = [("ODI vs Test", lambda b, o: b[:, o, 0], COLOURS["ODI"]),
            ("T20 vs Test", lambda b, o: b[:, o, 1], COLOURS["T20"]),
            ("T20 vs ODI", lambda b, o: b[:, o, 1] - b[:, o, 0], "#7a7a7a")]
    with plt.rc_context(CLEAN):
        fig, axes = plt.subplots(4, 3, figsize=(WIDTH, 11.0), layout="constrained")
        for i, (rname, b, o) in enumerate(rows):
            for j, (cname, get, colour) in enumerate(cols):
                ax, beta = axes[i, j], get(b, o)
                lo, hi = pct(az.hdi(beta, prob=0.94))
                ax.axvspan(lo, hi, color=colour, alpha=0.15, lw=0)
                ax.hist(pct(beta), bins=50, density=True, color=colour, alpha=0.85, edgecolor="none")
                ax.axvline(pct(beta).mean(), color=INK, lw=1.5)
                ax.text(0.03, 0.92, f"${pct(beta.mean()):+.1f}\\%$", transform=ax.transAxes,
                        va="top", ha="left", color=INK)
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
                ax.xaxis.set_major_locator(MaxNLocator(4))
                ax.grid(axis="x", alpha=0.25)
                if i == 0:
                    ax.set_title(cname)
                if j == 0:
                    ax.set_ylabel(rname)
                if i == 3:
                    ax.set_xlabel("Change (%)")
        fig.savefig(OUT_DIR / "format_effects_posterior.png", dpi=DPI, facecolor="white")
        plt.close(fig)


def age_curve() -> None:
    """Figure 3: fitted one-knot age spline with 94% bands, 5.0 x 3.3 in."""
    with plt.rc_context({**PRINT, "xtick.labelsize": 8, "ytick.labelsize": 8}):
        fig, axes = plt.subplots(2, 2, figsize=(5.0, 3.3), layout="constrained")
        for col, (kind, labels) in enumerate(FAMILIES.items()):
            meta = json.loads((DATA_DIR / f"{kind}_joint_enriched_meta.json").read_text())
            preds, knot = meta["predictors"], float(meta["age_curve"]["knot_age_years"])
            d = pd.read_csv(DATA_DIR / f"{kind}_joint_enriched.csv")
            sd_age = d["avg_age"].std(ddof=0)
            beta, _ = enriched_beta(kind)
            ia, ih = preds.index("age_scaled"), preds.index("age_hinge")
            ages = np.linspace(np.percentile(d["avg_age"], 1), np.percentile(d["avg_age"], 99), 160)
            for row, label in enumerate(labels):
                ax = axes[row, col]
                b_age, b_hin = beta[:, row, ia], beta[:, row, ih]
                draws = pct(b_age[:, None] * ((ages - knot) / sd_age)[None, :]
                            + b_hin[:, None] * (np.maximum(0.0, ages - knot) / sd_age)[None, :])
                lo, hi = np.percentile(draws, [3, 97], axis=0)
                ax.fill_between(ages, lo, hi, color=COLOURS["Test"], alpha=0.18, lw=0)
                ax.plot(ages, draws.mean(0), color=COLOURS["Test"], lw=1.4)
                ax.axvline(knot, color=INK, ls="--", lw=0.7, alpha=0.7)
                ax.axhline(0, color=MUTED, lw=0.5, alpha=0.6)
                ax.grid(alpha=0.25, lw=0.4)
                before = b_age.mean() / sd_age * 100
                after = (b_age.mean() + b_hin.mean()) / sd_age * 100
                note = f"${before:+.1f}\\%$ per year before {knot:g}\n${after:+.1f}\\%$ per year after"
                if kind == "bowling":
                    note += "\n(lower is better)"
                # Rising curves leave the top-left empty, falling ones the top-right.
                x, ha = (0.03, "left") if before > 0 else (0.97, "right")
                ax.text(x, 0.95, note, transform=ax.transAxes, color=INK, va="top", ha=ha,
                        fontsize=8, linespacing=1.1, multialignment="left", zorder=5,
                        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))
                ax.set_title(label, pad=3)
                if col == 0:
                    ax.set_ylabel(f"Change vs age {knot:g} (%)")
                if row == 1:
                    ax.set_xlabel("Career-midpoint age (years)")
        fig.savefig(OUT_DIR / "age_curve_simple.pdf")
        fig.savefig(OUT_DIR / "age_curve_simple.png", dpi=600, facecolor="white")
        plt.close(fig)


def transitions(bat: pd.DataFrame, bowl: pd.DataFrame, path: Path) -> None:
    """Figure 4: held-out predictions from the combined k-NN + own-record estimate, 6.0 x 6.9 in."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [("Batting\naverage", bat, 1), ("Batting\nstrike rate", bat, 2),
            ("Bowling\neconomy rate", bowl, 1), ("Bowling\naverage", bowl, 2)]
    with plt.rc_context({**PRINT, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5}):
        fig, axes = plt.subplots(4, 3, figsize=(6.0, 6.9), layout="constrained")
        for i, (rname, df, n) in enumerate(rows):
            for j, fmt in enumerate(FORMATS):
                ax, s = axes[i, j], df[df["target_format"] == fmt]
                a, p = s[f"true{n}"].to_numpy(float), s[f"mix{n}"].to_numpy(float)
                lo, hi = min(a.min(), p.min()) * 0.95, max(a.max(), p.max()) * 1.05
                ax.plot([lo, hi], [lo, hi], ls="--", lw=0.7, color=MUTED, zorder=1)
                # Points are rasterised at 600 dpi; axes and text stay vector.
                ax.scatter(a, p, s=3, alpha=0.45, color=COLOURS[fmt], edgecolors="none",
                           zorder=2, rasterized=True)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.grid(alpha=0.25, lw=0.4)
                ax.text(0.05, 0.95, f"$r = {np.corrcoef(a, p)[0, 1]:.2f}$", transform=ax.transAxes,
                        va="top", ha="left", color=INK, fontsize=8.5)
                if i == 0:
                    ax.set_title(f"Target: {fmt}", pad=3)
                if j == 0:
                    ax.set_ylabel(rname)
        fig.supxlabel("Actual value", fontsize=9)
        fig.supylabel("Predicted value", fontsize=9)
        for out in (path.with_suffix(".pdf"), path.with_suffix(".png")):
            fig.savefig(out, dpi=600, facecolor="white")
            print(f"Saved: {out}")
        plt.close(fig)


def label_top(ax, rows, xcol, ycol, gap):
    pts = [((r[xcol], r[ycol]), ax.transLimits.transform((r[xcol], r[ycol]))[1],
            r["player_name"]) for _, r in rows.iterrows()]
    pts.sort(key=lambda q: -q[1])
    span = gap * (len(pts) - 1)
    start = min(0.99, max(span + 0.01, (pts[0][1] + pts[-1][1]) / 2 + span / 2))
    for i, (xy, _, name) in enumerate(pts):
        ax.annotate(name, xy=xy, xytext=(1.03, start - i * gap), textcoords="axes fraction",
                    fontsize=16, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="grey", alpha=0.6,
                                    shrinkA=0, shrinkB=3))


def player_panel(ax, cax, kind):
    df = pd.read_csv(ME_DIR / f"{kind}_player_rankings.csv")
    if kind == "batting":
        x, y = "psi_log_sr", "psi_log_avg"
        xl, yl = r"$\psi$  log(strike rate)", r"$\psi$  log(batting average)"
    else:
        x, y = "psi_log_econ", "psi_log_avg"
        xl, yl = r"$\psi$  log(economy rate)", r"$\psi$  log(bowling average)"
    sc = ax.scatter(df[x], df[y], c=df["psi_combined"], cmap="RdYlGn", s=26, alpha=0.8,
                    linewidths=0.25, edgecolors="grey", zorder=2)
    ax.axhline(df[y].median(), color="steelblue", lw=1.1, ls="--", alpha=0.6)
    ax.axvline(df[x].median(), color="steelblue", lw=1.1, ls="--", alpha=0.6)
    ax.set_title(kind.capitalize())
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.grid(True, alpha=0.3)
    if kind == "bowling":
        ax.invert_xaxis()
        ax.invert_yaxis()
    cb = ax.figure.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label(r"Combined $\psi$ score", fontsize=17)
    cb.ax.tick_params(labelsize=15)
    cb.locator = plt.MaxNLocator(5)
    cb.update_ticks()
    label_top(ax, df.nlargest(TOP_N, "psi_combined"), x, y, gap=0.095)


def player_effects() -> None:
    """Figure 5: format-only player effects, top ten labelled."""
    w, h = WIDTH, 8.69
    left, names, bottom, top = 1.15, 2.05, 1.85, 0.6
    pw, ph = w / 2 - left - names, h - bottom - top
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(w, h))
        axes = [fig.add_axes([(x0 + left) / w, bottom / h, pw / w, ph / h]) for x0 in (0, w / 2)]
        caxes = [fig.add_axes([(x0 + left + 0.1 * pw) / w, 0.62 / h, 0.8 * pw / w, 0.22 / h])
                 for x0 in (0, w / 2)]
        for ax, cax, kind in zip(axes, caxes, ("batting", "bowling")):
            player_panel(ax, cax, kind)
        fig.savefig(OUT_DIR / "poster_player_effects.png", dpi=DPI)
        plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    effects_table()
    raw_outcomes()
    format_posteriors()
    age_curve()
    transitions(pd.read_csv(ME_DIR / "lopo_kfold_predictions_batting.csv"),
                pd.read_csv(ME_DIR / "lopo_kfold_predictions_bowling.csv"),
                OUT_DIR / "cross_format_transition_kfold.png")
    player_effects()
    print(f"Saved the five paper figures to {OUT_DIR}/ and {ME_DIR}/enriched_effects_pct.csv")


if __name__ == "__main__":
    main()
