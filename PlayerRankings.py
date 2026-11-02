"""Plot player random effects (psi) scatter for batting and bowling rankings."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 20

def label_top(ax, rows, xcol, ycol, namecol="player_name", min_gap=0.048):
    """Annotate `rows` with leader lines into a tidy stack right of the axes."""
    pts = [((r[xcol], r[ycol]),
            ax.transLimits.transform((r[xcol], r[ycol]))[1],
            r[namecol])
           for _, r in rows.iterrows()]
    pts.sort(key=lambda p: -p[1])

    # Centre the stack vertically on the points it describes.
    span = min_gap * (len(pts) - 1)
    start = min(0.98, max(span + 0.02, (pts[0][1] + pts[-1][1]) / 2 + span / 2))

    for i, (xy, _, name) in enumerate(pts):
        ax.annotate(
            name,
            xy=xy, xycoords="data",
            xytext=(1.04, start - i * min_gap), textcoords="axes fraction",
            fontsize=7.5, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.45, color="grey",
                            alpha=0.55, shrinkA=0, shrinkB=3),
        )

def panel(ax, df, xcol, ycol, xlabel, ylabel, title, invert=False):
    sc = ax.scatter(df[xcol], df[ycol], c=df["psi_combined"], cmap="RdYlGn",
                    s=28, alpha=0.75, linewidths=0.3, edgecolors="grey", zorder=2)
    ax.axhline(df[ycol].median(), color="steelblue", lw=1, ls="--", alpha=0.6)
    ax.axvline(df[xcol].median(), color="steelblue", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if invert:
        ax.invert_xaxis()
        ax.invert_yaxis()   # lower economy and average are better

    # Colourbar below the panel, so the right-hand side stays free for labels.
    cb = plt.colorbar(sc, ax=ax, orientation="horizontal",
                      pad=0.12, shrink=0.75, aspect=32)
    cb.set_label("Combined ψ score", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # After limits and inversion are settled.
    label_top(ax, df.nlargest(TOP_N, "psi_combined"), xcol, ycol)

bat = pd.read_csv("outputs/mixed_effects/batting_player_rankings.csv")
bowl = pd.read_csv("outputs/mixed_effects/bowling_player_rankings.csv")

fig, axes = plt.subplots(1, 2, figsize=(17, 7))

panel(axes[0], bat, "psi_log_sr", "psi_log_avg",
      "ψ  log(Strike Rate)", "ψ  log(Batting Average)",
      f"Batting Player Random Effects\n(format-only model; top {TOP_N} labelled)")

panel(axes[1], bowl, "psi_log_econ", "psi_log_avg",
      "ψ  log(Economy Rate)", "ψ  log(Bowling Average)",
      f"Bowling Player Random Effects\n(format-only model; top {TOP_N} labelled)",
      invert=True)

# Room for the label gutters on the right of each panel.
fig.subplots_adjust(left=0.06, right=0.80, wspace=0.62)

out = OUT_DIR / "player_random_effects.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)
