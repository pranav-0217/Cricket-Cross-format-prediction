"""Analyse and visualise format fixed effects + cross-format transition prediction."""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import arviz as az

OUT_DIR  = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("Data/new/processed")
PREDICTORS_BAT = json.loads(
    (DATA_DIR / "batting_joint_enriched_meta.json").read_text(encoding="utf-8")
)["predictors"]
PREDICTORS_BOWL = json.loads(
    (DATA_DIR / "bowling_joint_enriched_meta.json").read_text(encoding="utf-8")
)["predictors"]

bat_trace  = az.from_netcdf("outputs/mixed_effects/joint_batting_enriched_trace.nc")
bowl_trace = az.from_netcdf("outputs/mixed_effects/joint_bowling_enriched_trace.nc")

def check_trace_matches_predictors(trace, predictors, kind: str) -> None:
    """Refuse to interpret a trace fitted with a different predictor set."""
    n_trace = trace.posterior["beta"].shape[-1]
    if n_trace != len(predictors):
        raise SystemExit(
            f"{kind}: trace was fitted with {n_trace} predictors but the current "
            f"data defines {len(predictors)}. Refit "
            f"Fit{kind.capitalize()}Enriched.py before analysing format effects."
        )

check_trace_matches_predictors(bat_trace, PREDICTORS_BAT, "batting")
check_trace_matches_predictors(bowl_trace, PREDICTORS_BOWL, "bowling")

bat_params  = pd.read_csv("outputs/mixed_effects/joint_batting_enriched_params_mean.csv").iloc[0]
bowl_params = pd.read_csv("outputs/mixed_effects/joint_bowling_enriched_params_mean.csv").iloc[0]

bat_data  = pd.read_csv(DATA_DIR / "batting_joint_enriched.csv")
bowl_data = pd.read_csv(DATA_DIR / "bowling_joint_enriched.csv")
bat_psi   = pd.read_csv("outputs/mixed_effects/batting_enriched_player_rankings.csv")
bowl_psi  = pd.read_csv("outputs/mixed_effects/bowling_enriched_player_rankings.csv")

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

for idx, (label, samples, color) in enumerate(all_effects):
    row, col = idx % 4, idx // 4
    ax = axes[row, col]
    ax.hist(samples, bins=60, color=color, alpha=0.75, density=True, edgecolor="none")
    ax.axvline(0, color="black", lw=1.2, ls="--", alpha=0.7)
    mean_val = samples.mean()
    hdi = az.hdi(samples, prob=0.94)
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
    hdi = az.hdi(samples, prob=0.94)
    pct = (np.exp(m) - 1) * 100
    rows.append({"effect": label, "beta_mean": round(m, 4),
                 "pct_change": round(pct, 1),
                 "hdi_3%": round(float(hdi[0]), 4),
                 "hdi_97%": round(float(hdi[1]), 4)})
    print(f"  {label:35s}  beta={m:+.3f}  => {pct:+.1f}%  94%HDI [{hdi[0]:.3f}, {hdi[1]:.3f}]")

summary_df = pd.DataFrame(rows)
summary_df.to_csv("outputs/mixed_effects/format_effects_summary.csv", index=False)
print("\nSaved: format_effects_summary.csv")

def predict_batting(player_id, target_format, params, psi_df, data_df):
    row = psi_df[psi_df["player_id"].astype(str) == str(player_id)]
    if row.empty: return None
    psi = np.array([float(row["psi_log_avg"].iloc[0]), float(row["psi_log_sr"].iloc[0])])
    d = data_df[data_df["player_id"].astype(str) == str(player_id)].iloc[0]
    fmt = {"format_odi": float(target_format == "ODI"),
           "format_t20": float(target_format == "T20")}
    x = np.array([fmt[p] if p in fmt else float(d[p]) for p in PREDICTORS_BAT])
    alpha = np.array([params["alpha_log_avg"], params["alpha_log_sr"]])
    beta  = np.array([[params[f"beta_{p}_log_avg"] for p in PREDICTORS_BAT],
                      [params[f"beta_{p}_log_sr"]  for p in PREDICTORS_BAT]])
    mu = alpha + beta @ x + psi
    return np.exp(mu[0]), np.exp(mu[1])   # avg, sr

def predict_bowling(player_id, target_format, params, psi_df, data_df):
    row = psi_df[psi_df["player_id"].astype(str) == str(player_id)]
    if row.empty: return None
    psi = np.array([float(row["psi_log_econ"].iloc[0]), float(row["psi_log_avg"].iloc[0])])
    d = data_df[data_df["player_id"].astype(str) == str(player_id)].iloc[0]
    fmt = {"format_odi": float(target_format == "ODI"),
           "format_t20": float(target_format == "T20")}
    x = np.array([fmt[p] if p in fmt else float(d[p]) for p in PREDICTORS_BOWL])
    alpha = np.array([params["alpha_log_econ"], params["alpha_log_avg"]])
    beta  = np.array([[params[f"beta_{p}_log_econ"] for p in PREDICTORS_BOWL],
                      [params[f"beta_{p}_log_avg"]  for p in PREDICTORS_BOWL]])
    mu = alpha + beta @ x + psi
    return np.exp(mu[0]), np.exp(mu[1])   # econ, avg

bat_results = []
for _, row in bat_data.iterrows():
    pred = predict_batting(row["player_id"], row["format"],
                           bat_params, bat_psi, bat_data)
    if pred:
        bat_results.append({
            "player_id":   row["player_id"],
            "player_name": row["player_name"],
            "format": row["format"],
            "actual_avg": row["avg"],
            "actual_sr":  row["sr"],
            "pred_avg":   pred[0],
            "pred_sr":    pred[1],
        })

bat_res = pd.DataFrame(bat_results).dropna()

bowl_data["econ"] = np.exp(bowl_data["y1"])
bowl_data["avg2"] = np.exp(bowl_data["y2"])
bowl_results = []
for _, row in bowl_data.iterrows():
    pred = predict_bowling(row["player_id"], row["format"],
                           bowl_params, bowl_psi, bowl_data)
    if pred:
        bowl_results.append({
            "player_id":   row["player_id"],
            "player_name": row["player_name"],
            "format": row["format"],
            "actual_econ": row["econ"],
            "actual_avg":  row["avg2"],
            "pred_econ":   pred[0],
            "pred_avg":    pred[1],
        })

bowl_res = pd.DataFrame(bowl_results).dropna()

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

fig, axes = plt.subplots(4, 3, figsize=(16, 18))
fig.suptitle("Cross-Format Transition: Predicted vs Actual\n"
             "(each dot = one player-format observation)",
             fontsize=13, fontweight="bold")
fig.patch.set_facecolor("#f9f9f9")

fmt_colors = {"Test": "#1976D2", "ODI": "#388E3C", "T20": "#F57C00"}

for col, fmt in enumerate(["Test", "ODI", "T20"]):
    sub  = bat_res[bat_res["format"] == fmt]
    sub2 = bowl_res[bowl_res["format"] == fmt]

    panels = [
        (axes[0, col], sub["actual_avg"], sub["pred_avg"],
         f"Batting Avg — {fmt}", "Actual avg", "Predicted avg"),
        (axes[1, col], sub["actual_sr"], sub["pred_sr"],
         f"Batting SR — {fmt}", "Actual SR", "Predicted SR"),
        (axes[2, col], sub2["actual_econ"], sub2["pred_econ"],
         f"Bowling Economy — {fmt}", "Actual economy", "Predicted economy"),
        (axes[3, col], sub2["actual_avg"], sub2["pred_avg"],
         f"Bowling Avg — {fmt}", "Actual avg", "Predicted avg"),
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
