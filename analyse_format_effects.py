"""
Analyse and visualise format fixed effects + cross-format transition prediction.

Produces:
  1. outputs/plots/format_effects_posterior.png  -- posterior distributions of format betas
  2. outputs/plots/cross_format_transition.png   -- predicted vs actual for format transitions
  3. outputs/mixed_effects/format_effects_summary.csv -- interpreted effects on original scale
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import arviz as az

OUT_DIR  = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTORS_BAT  = ["format_odi", "format_t20", "debut_year_scaled", "debut_year_missing",
                    "is_lefthanded", "position_order", "age_at_debut_scaled"]
PREDICTORS_BOWL = ["format_odi", "format_t20", "debut_year_scaled", "debut_year_missing",
                    "bowling_hand", "bowling_type", "position_order", "age_at_debut_scaled"]

# ── Load traces ───────────────────────────────────────────────────────────────
bat_trace  = az.from_netcdf("outputs/mixed_effects/joint_batting_enriched_trace.nc")
bowl_trace = az.from_netcdf("outputs/mixed_effects/joint_bowling_enriched_trace.nc")

bat_params  = pd.read_csv("outputs/mixed_effects/joint_batting_enriched_params_mean.csv").iloc[0]
bowl_params = pd.read_csv("outputs/mixed_effects/joint_bowling_enriched_params_mean.csv").iloc[0]

bat_data  = pd.read_csv("Data/batting_joint_enriched.csv")
bowl_data = pd.read_csv("Data/bowling_joint_enriched.csv")
bat_psi   = pd.read_csv("outputs/mixed_effects/batting_enriched_player_rankings.csv")
bowl_psi  = pd.read_csv("outputs/mixed_effects/bowling_enriched_player_rankings.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. FORMAT EFFECTS POSTERIOR PLOT
# ═══════════════════════════════════════════════════════════════════════════════

bat_beta  = bat_trace.posterior["beta"].values
bowl_beta = bowl_trace.posterior["beta"].values
bat_b  = bat_beta.reshape(-1, 2, len(PREDICTORS_BAT))
bowl_b = bowl_beta.reshape(-1, 2, len(PREDICTORS_BOWL))

all_effects = [
    ("Batting avg  | ODI vs Test",  bat_b[:, 0, 0],                       "#2196F3"),
    ("Batting avg  | T20 vs Test",  bat_b[:, 0, 1],                       "#FF5722"),
    ("Batting avg  | T20 vs ODI",   bat_b[:, 0, 1] - bat_b[:, 0, 0],     "#795548"),
    ("Batting SR   | ODI vs Test",  bat_b[:, 1, 0],                       "#4CAF50"),
    ("Batting SR   | T20 vs Test",  bat_b[:, 1, 1],                       "#9C27B0"),
    ("Batting SR   | T20 vs ODI",   bat_b[:, 1, 1] - bat_b[:, 1, 0],     "#607D8B"),
    ("Bowling econ | ODI vs Test",  bowl_b[:, 0, 0],                      "#2196F3"),
    ("Bowling econ | T20 vs Test",  bowl_b[:, 0, 1],                      "#FF5722"),
    ("Bowling econ | T20 vs ODI",   bowl_b[:, 0, 1] - bowl_b[:, 0, 0],   "#795548"),
    ("Bowling avg  | ODI vs Test",  bowl_b[:, 1, 0],                      "#4CAF50"),
    ("Bowling avg  | T20 vs Test",  bowl_b[:, 1, 1],                      "#9C27B0"),
    ("Bowling avg  | T20 vs ODI",   bowl_b[:, 1, 1] - bowl_b[:, 1, 0],   "#607D8B"),
]

fig, axes = plt.subplots(4, 3, figsize=(18, 14))
fig.suptitle("Posterior Distributions of Format Fixed Effects\n(log scale; Test = baseline)",
             fontsize=13, fontweight="bold")
fig.patch.set_facecolor("#f9f9f9")

for idx, (label, samples, color) in enumerate(all_effects):
    row, col = idx % 4, idx // 4  # 4 rows, 3 cols
    ax = axes[row, col]
    ax.set_facecolor("#f9f9f9")
    ax.hist(samples, bins=60, color=color, alpha=0.75, density=True, edgecolor="none")
    ax.axvline(0, color="black", lw=1.2, ls="--", alpha=0.7)
    mean_val = samples.mean()
    hdi = az.hdi(samples, hdi_prob=0.94)
    ax.axvline(mean_val, color="darkred", lw=1.5)
    pct_change = (np.exp(mean_val) - 1) * 100
    direction = "+" if pct_change > 0 else ""
    ax.set_title(f"{label}\nmean={mean_val:.3f}  ({direction}{pct_change:.1f}%)",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("β (log scale)", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.axvspan(float(hdi[0]), float(hdi[1]), alpha=0.15, color=color)

plt.tight_layout()
plt.savefig(OUT_DIR / "format_effects_posterior.png", dpi=180, bbox_inches="tight")
print("Saved: format_effects_posterior.png")
plt.close()

# ── Print interpretation ───────────────────────────────────────────────────────
print("\n=== Format Fixed Effects (% change from Test baseline) ===")
rows = []
for label, samples in {
    "Batting avg  | ODI vs Test": bat_b[:, 0, 0],
    "Batting avg  | T20 vs Test": bat_b[:, 0, 1],
    "Batting avg  | T20 vs ODI":  bat_b[:, 0, 1] - bat_b[:, 0, 0],
    "Batting SR   | ODI vs Test": bat_b[:, 1, 0],
    "Batting SR   | T20 vs Test": bat_b[:, 1, 1],
    "Batting SR   | T20 vs ODI":  bat_b[:, 1, 1] - bat_b[:, 1, 0],
    "Bowling econ | ODI vs Test": bowl_b[:, 0, 0],
    "Bowling econ | T20 vs Test": bowl_b[:, 0, 1],
    "Bowling econ | T20 vs ODI":  bowl_b[:, 0, 1] - bowl_b[:, 0, 0],
    "Bowling avg  | ODI vs Test": bowl_b[:, 1, 0],
    "Bowling avg  | T20 vs Test": bowl_b[:, 1, 1],
    "Bowling avg  | T20 vs ODI":  bowl_b[:, 1, 1] - bowl_b[:, 1, 0],
}.items():
    m = samples.mean()
    hdi = az.hdi(samples, hdi_prob=0.94)
    pct = (np.exp(m) - 1) * 100
    rows.append({"effect": label, "beta_mean": round(m, 4),
                 "pct_change": round(pct, 1),
                 "hdi_3%": round(float(hdi[0]), 4),
                 "hdi_97%": round(float(hdi[1]), 4)})
    print(f"  {label:35s}  beta={m:+.3f}  => {pct:+.1f}%  94%HDI [{hdi[0]:.3f}, {hdi[1]:.3f}]")

summary_df = pd.DataFrame(rows)
summary_df.to_csv("outputs/mixed_effects/format_effects_summary.csv", index=False)
print("\nSaved: format_effects_summary.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CROSS-FORMAT TRANSITION PREDICTION EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
# For each player with data in 2+ formats, use psi + format betas to predict
# each format and compare predicted vs actual.

def predict_batting(player_name, target_format, params, psi_df, data_df):
    row = psi_df[psi_df["player_name"] == player_name]
    if row.empty: return None
    psi = np.array([float(row["psi_log_avg"].iloc[0]), float(row["psi_log_sr"].iloc[0])])
    d = data_df[data_df["player_name"] == player_name].iloc[0]
    fmt_odi = 1.0 if target_format == "ODI" else 0.0
    fmt_t20 = 1.0 if target_format == "T20" else 0.0
    x = np.array([fmt_odi, fmt_t20,
                  float(d.get("debut_year_scaled", 0) or 0),
                  float(d.get("debut_year_missing", 0) or 0),
                  float(d.get("is_lefthanded", 0) or 0),
                  float(d.get("position_order", 0) or 0),
                  float(d.get("age_at_debut_scaled", 0) or 0)])
    alpha = np.array([params["alpha_log_avg"], params["alpha_log_sr"]])
    beta  = np.array([[params[f"beta_{p}_log_avg"] for p in PREDICTORS_BAT],
                      [params[f"beta_{p}_log_sr"]  for p in PREDICTORS_BAT]])
    mu = alpha + beta @ x + psi
    return np.exp(mu[0]), np.exp(mu[1])   # avg, sr


def predict_bowling(player_name, target_format, params, psi_df, data_df):
    row = psi_df[psi_df["player_name"] == player_name]
    if row.empty: return None
    psi = np.array([float(row["psi_log_econ"].iloc[0]), float(row["psi_log_avg"].iloc[0])])
    d = data_df[data_df["player_name"] == player_name].iloc[0]
    fmt_odi = 1.0 if target_format == "ODI" else 0.0
    fmt_t20 = 1.0 if target_format == "T20" else 0.0
    x = np.array([fmt_odi, fmt_t20,
                  float(d.get("debut_year_scaled", 0) or 0),
                  float(d.get("debut_year_missing", 0) or 0),
                  float(d.get("bowling_hand", 0) or 0),
                  float(d.get("bowling_type", 0) or 0),
                  float(d.get("position_order", 0) or 0),
                  float(d.get("age_at_debut_scaled", 0) or 0)])
    alpha = np.array([params["alpha_log_econ"], params["alpha_log_avg"]])
    beta  = np.array([[params[f"beta_{p}_log_econ"] for p in PREDICTORS_BOWL],
                      [params[f"beta_{p}_log_avg"]  for p in PREDICTORS_BOWL]])
    mu = alpha + beta @ x + psi
    return np.exp(mu[0]), np.exp(mu[1])   # econ, avg


# Build transition predictions for batting
bat_results = []
for _, row in bat_data.iterrows():
    pred = predict_batting(row["player_name"], row["format"],
                           bat_params, bat_psi, bat_data)
    if pred:
        bat_results.append({
            "player_name": row["player_name"],
            "format": row["format"],
            "actual_avg": row["avg"],
            "actual_sr":  row["sr"],
            "pred_avg":   pred[0],
            "pred_sr":    pred[1],
        })

bat_res = pd.DataFrame(bat_results).dropna()

# Build transition predictions for bowling
bowl_data["econ"] = np.exp(bowl_data["y1"])
bowl_data["avg2"] = np.exp(bowl_data["y2"])
bowl_results = []
for _, row in bowl_data.iterrows():
    pred = predict_bowling(row["player_name"], row["format"],
                           bowl_params, bowl_psi, bowl_data)
    if pred:
        bowl_results.append({
            "player_name": row["player_name"],
            "format": row["format"],
            "actual_econ": row["econ"],
            "actual_avg":  row["avg2"],
            "pred_econ":   pred[0],
            "pred_avg":    pred[1],
        })

bowl_res = pd.DataFrame(bowl_results).dropna()

# Compute per-format correlations
print("\n=== Cross-format Transition Prediction (Pearson r: predicted vs actual) ===")
for fmt in ["Test", "ODI", "T20"]:
    sub = bat_res[bat_res["format"] == fmt]
    r_avg = sub["actual_avg"].corr(sub["pred_avg"])
    r_sr  = sub["actual_sr"].corr(sub["pred_sr"])
    print(f"  Batting  {fmt:5s}  r(avg)={r_avg:.3f}  r(SR)={r_sr:.3f}  n={len(sub)}")

for fmt in ["Test", "ODI", "T20"]:
    sub = bowl_res[bowl_res["format"] == fmt]
    r_econ = sub["actual_econ"].corr(sub["pred_econ"])
    r_avg  = sub["actual_avg"].corr(sub["pred_avg"])
    print(f"  Bowling  {fmt:5s}  r(econ)={r_econ:.3f}  r(avg)={r_avg:.3f}  n={len(sub)}")

# ── Transition plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 3, figsize=(16, 18))
fig.suptitle("Cross-Format Transition: Predicted vs Actual\n(each dot = one player-format observation)",
             fontsize=13, fontweight="bold")
fig.patch.set_facecolor("#f9f9f9")

fmt_colors = {"Test": "#1976D2", "ODI": "#388E3C", "T20": "#F57C00"}

for col, fmt in enumerate(["Test", "ODI", "T20"]):
    sub  = bat_res[bat_res["format"] == fmt]
    sub2 = bowl_res[bowl_res["format"] == fmt]

    panels = [
        (axes[0, col], sub["actual_avg"],  sub["pred_avg"],  f"Batting Avg — {fmt}",    "Actual avg",    "Predicted avg"),
        (axes[1, col], sub["actual_sr"],   sub["pred_sr"],   f"Batting SR — {fmt}",     "Actual SR",     "Predicted SR"),
        (axes[2, col], sub2["actual_econ"],sub2["pred_econ"],f"Bowling Economy — {fmt}", "Actual economy","Predicted economy"),
        (axes[3, col], sub2["actual_avg"], sub2["pred_avg"], f"Bowling Avg — {fmt}",    "Actual avg",    "Predicted avg"),
    ]

    for ax, actual, pred, title, xlabel, ylabel in panels:
        ax.set_facecolor("#f9f9f9")
        ax.scatter(actual, pred, alpha=0.5, s=18, color=fmt_colors[fmt], edgecolors="none")
        lo = min(actual.min(), pred.min()) * 0.95
        hi = max(actual.max(), pred.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
        r = actual.corr(pred)
        ax.set_title(f"{title}\nr = {r:.3f}", fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "cross_format_transition.png", dpi=180, bbox_inches="tight")
print("\nSaved: cross_format_transition.png")
plt.close()
