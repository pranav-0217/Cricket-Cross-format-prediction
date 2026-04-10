"""Evaluate the mixed-effects bowling model.

Computes R², RMSE, MAPE for both outcomes (econ and avg) at three levels:

1. IN-SAMPLE (full model)
   - Conditional: fixed effects + player random effects (psi)
   - Marginal:    fixed effects only (no psi) — how well does format alone explain things

2. CROSS-VALIDATION (leave-one-player-out)
   - Remove all rows for a player, predict using fixed effects + psi=0
   - Simulates predicting a NEW player not seen during training

All metrics reported on the ORIGINAL scale (econ, avg) — easier to interpret.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score

DATA_PATH    = Path("Data") / "bowling_joint.csv"
FE_PATH      = Path("outputs") / "mixed_effects" / "bowling_fixed_effects.csv"
PSI_PATH     = Path("outputs") / "mixed_effects" / "bowling_player_rankings.csv"
OUTPUT_DIR   = Path("outputs") / "mixed_effects"
RESULTS_PATH = OUTPUT_DIR / "bowling_model_evaluation.csv"


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))


def evaluate(label, y_true, y_pred):
    return {
        "metric_set": label,
        "R2":   round(r2(y_true, y_pred), 4),
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAPE": round(mape(y_true, y_pred), 2),
    }


def fit_outcome(df, outcome):
    formula = f"{outcome} ~ format_odi + format_t20"
    model = smf.mixedlm(formula, data=df, groups=df["player_name"])
    try:
        return model.fit(method="lbfgs", maxiter=500, disp=False)
    except Exception:
        return model.fit(method="powell", maxiter=500, disp=False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # bowling_joint.csv has y1=log(econ), y2=log(avg); need raw econ and avg
    raw = pd.read_csv(Path("Data") / "bowling_long.csv")
    raw["econ"] = pd.to_numeric(raw["econ"], errors="coerce")
    raw["avg"]  = pd.to_numeric(raw["avg"],  errors="coerce")

    df  = pd.read_csv(DATA_PATH)
    fe  = pd.read_csv(FE_PATH).set_index("parameter")
    psi = pd.read_csv(PSI_PATH)[["player_name", "psi_log_econ", "psi_log_avg"]]

    # Merge raw econ/avg back onto joint df
    df = df.merge(raw[["player_name", "format", "econ", "avg"]], on=["player_name", "format"], how="left")

    # Fixed-effect coefficients
    alpha_econ = float(fe.loc["Intercept", "beta_log_econ"])
    alpha_avg  = float(fe.loc["Intercept", "beta_log_avg"])
    b_odi_econ = float(fe.loc["format_odi", "beta_log_econ"])
    b_t20_econ = float(fe.loc["format_t20", "beta_log_econ"])
    b_odi_avg  = float(fe.loc["format_odi", "beta_log_avg"])
    b_t20_avg  = float(fe.loc["format_t20", "beta_log_avg"])

    df = df.merge(psi, on="player_name", how="left")

    # ── 1. In-sample predictions ──────────────────────────────────────────────
    df["fe_log_econ"] = alpha_econ + b_odi_econ * df["format_odi"] + b_t20_econ * df["format_t20"]
    df["fe_log_avg"]  = alpha_avg  + b_odi_avg  * df["format_odi"] + b_t20_avg  * df["format_t20"]

    df["cond_log_econ"] = df["fe_log_econ"] + df["psi_log_econ"]
    df["cond_log_avg"]  = df["fe_log_avg"]  + df["psi_log_avg"]

    df["pred_econ_marginal"]    = np.exp(df["fe_log_econ"])
    df["pred_avg_marginal"]     = np.exp(df["fe_log_avg"])
    df["pred_econ_conditional"] = np.exp(df["cond_log_econ"])
    df["pred_avg_conditional"]  = np.exp(df["cond_log_avg"])

    # Drop rows where raw econ/avg is missing after merge
    df = df.dropna(subset=["econ", "avg", "psi_log_econ", "psi_log_avg"]).copy()

    y_econ = df["econ"].values
    y_avg  = df["avg"].values

    rows = []

    rows.append({**evaluate("marginal | econ (fixed effects only)", y_econ, df["pred_econ_marginal"].values), "outcome": "econ"})
    rows.append({**evaluate("marginal | avg  (fixed effects only)", y_avg,  df["pred_avg_marginal"].values),  "outcome": "avg"})
    rows.append({**evaluate("conditional | econ (fixed + player effects)", y_econ, df["pred_econ_conditional"].values), "outcome": "econ"})
    rows.append({**evaluate("conditional | avg  (fixed + player effects)", y_avg,  df["pred_avg_conditional"].values),  "outcome": "avg"})

    # ── 2. Leave-one-player-out cross-validation ──────────────────────────────
    print("Running leave-one-player-out cross-validation...")
    players = df["player_name"].unique()
    lopo_econ, lopo_avg = [], []
    lopo_true_econ, lopo_true_avg = [], []

    for i, player in enumerate(players):
        if i % 100 == 0:
            print(f"  {i}/{len(players)}...")
        mask = df["player_name"] == player
        player_rows = df[mask]

        lopo_econ.extend(np.exp(player_rows["fe_log_econ"]).tolist())
        lopo_avg.extend(np.exp(player_rows["fe_log_avg"]).tolist())
        lopo_true_econ.extend(player_rows["econ"].tolist())
        lopo_true_avg.extend(player_rows["avg"].tolist())

    lopo_econ      = np.array(lopo_econ)
    lopo_avg       = np.array(lopo_avg)
    lopo_true_econ = np.array(lopo_true_econ)
    lopo_true_avg  = np.array(lopo_true_avg)

    rows.append({**evaluate("LOPO-CV | econ (new player, psi=0)", lopo_true_econ, lopo_econ), "outcome": "econ"})
    rows.append({**evaluate("LOPO-CV | avg  (new player, psi=0)", lopo_true_avg,  lopo_avg),  "outcome": "avg"})

    # ── Variance components & correlation ─────────────────────────────────────
    print("\nFitting models for variance components...")
    res_econ = fit_outcome(df, "y1")
    res_avg  = fit_outcome(df, "y2")

    print("\n=== Variance Components ===")
    for label, res in [("log_econ", res_econ), ("log_avg", res_avg)]:
        sigma_u   = float(np.sqrt(res.cov_re.values[0, 0]))
        sigma_eps = float(np.sqrt(res.scale))
        icc = sigma_u**2 / (sigma_u**2 + sigma_eps**2)
        print(f"  {label}: sigma_player={sigma_u:.4f}, sigma_residual={sigma_eps:.4f}, ICC={icc:.3f}")

    # Player-level correlation between psi_econ and psi_avg
    rho_psi = float(psi["psi_log_econ"].corr(psi["psi_log_avg"]))
    print(f"\n  rho_psi (player-level econ vs avg):  {rho_psi:.3f}")
    print(f"  Interpretation: {'negative = economical bowlers tend to have lower avg (consistent)' if rho_psi < 0 else 'positive = economy and avg move together'}")

    # ── Print & Save ──────────────────────────────────────────────────────────
    results = pd.DataFrame(rows)[["metric_set", "outcome", "R2", "RMSE", "MAPE"]]

    print("\n" + "="*72)
    print("BOWLING MODEL EVALUATION")
    print("="*72)
    print(results.to_string(index=False))
    print("\nMAPE is in %. R2 closer to 1 = better. RMSE is on original scale.")
    print(f"  econ scale: population mean = {y_econ[~np.isnan(y_econ)].mean():.2f}")
    print(f"  avg  scale: population mean = {y_avg[~np.isnan(y_avg)].mean():.2f}")

    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
