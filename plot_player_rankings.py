"""Plot player random effects (psi) scatter for batting and bowling rankings."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path

OUT_DIR = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

bat = pd.read_csv("outputs/mixed_effects/batting_enriched_player_rankings.csv")
bowl = pd.read_csv("outputs/mixed_effects/bowling_enriched_player_rankings.csv")

TOP_N = 20  # label top N players

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#f9f9f9")

# ── Batting ───────────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor("#f9f9f9")

sc = ax.scatter(
    bat["psi_log_sr"], bat["psi_log_avg"],
    c=bat["psi_combined"], cmap="RdYlGn",
    s=30, alpha=0.7, linewidths=0.3, edgecolors="grey", zorder=2
)
plt.colorbar(sc, ax=ax, label="Combined ψ score", shrink=0.85)

# Label top N
top = bat.nlargest(TOP_N, "psi_combined")
for _, row in top.iterrows():
    ax.annotate(
        row["player_name"],
        (row["psi_log_sr"], row["psi_log_avg"]),
        fontsize=6.5, ha="left", va="bottom",
        xytext=(3, 3), textcoords="offset points",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")]
    )

# Median lines
ax.axhline(bat["psi_log_avg"].median(), color="steelblue", lw=1, ls="--", alpha=0.6)
ax.axvline(bat["psi_log_sr"].median(),  color="steelblue", lw=1, ls="--", alpha=0.6)

ax.set_xlabel("ψ  log(Strike Rate)", fontsize=11)
ax.set_ylabel("ψ  log(Batting Average)", fontsize=11)
ax.set_title("Batting Player Random Effects\n(enriched model)", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)

# ── Bowling ───────────────────────────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor("#f9f9f9")

sc2 = ax.scatter(
    bowl["psi_log_econ"], bowl["psi_log_avg"],
    c=bowl["psi_combined"], cmap="RdYlGn",
    s=30, alpha=0.7, linewidths=0.3, edgecolors="grey", zorder=2
)
plt.colorbar(sc2, ax=ax, label="Combined ψ score", shrink=0.85)

# Label top N (lowest = best for bowling)
top_b = bowl.nlargest(TOP_N, "psi_combined")
for _, row in top_b.iterrows():
    ax.annotate(
        row["player_name"],
        (row["psi_log_econ"], row["psi_log_avg"]),
        fontsize=6.5, ha="left", va="bottom",
        xytext=(3, 3), textcoords="offset points",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")]
    )

ax.axhline(bowl["psi_log_avg"].median(),  color="steelblue", lw=1, ls="--", alpha=0.6)
ax.axvline(bowl["psi_log_econ"].median(), color="steelblue", lw=1, ls="--", alpha=0.6)

ax.set_xlabel("ψ  log(Economy Rate)", fontsize=11)
ax.set_ylabel("ψ  log(Bowling Average)", fontsize=11)
ax.set_title("Bowling Player Random Effects\n(enriched model)", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.invert_xaxis()
ax.invert_yaxis()  # lower = better for bowling

note = "Top-right (batting) / bottom-left (bowling) = best performers.\nDashed lines = median."
fig.text(0.5, 0.01, note, ha="center", fontsize=9, color="grey")

plt.tight_layout(rect=[0, 0.04, 1, 1])
out = OUT_DIR / "player_random_effects.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
