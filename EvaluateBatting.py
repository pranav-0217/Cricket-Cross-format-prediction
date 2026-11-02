"""Evaluate the mixed-effects batting model."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

DATA_PATH    = Path("Data/new/processed/batting_joint.csv")
FE_PATH      = Path("outputs") / "mixed_effects" / "batting_fixed_effects.csv"
PSI_PATH     = Path("outputs") / "mixed_effects" / "batting_player_rankings.csv"
OUTPUT_DIR   = Path("outputs") / "mixed_effects"
RESULTS_PATH = OUTPUT_DIR / "batting_model_evaluation.csv"

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred):
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))

def evaluate(label, y_true, y_pred):
    return {
        "metric_set": label,
        "R2":   round(r2(y_true, y_pred), 4),
        "MAE":  round(mae(y_true, y_pred), 4),
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAPE": round(mape(y_true, y_pred), 2),
    }

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df  = pd.read_csv(DATA_PATH)
    fe  = pd.read_csv(FE_PATH).set_index("parameter")
    psi = pd.read_csv(PSI_PATH)[["player_id", "psi_log_avg", "psi_log_sr"]]

    alpha_avg = float(fe.loc["Intercept", "beta_log_avg"])
    alpha_sr  = float(fe.loc["Intercept", "beta_log_sr"])
    b_odi_avg = float(fe.loc["format_odi", "beta_log_avg"])
    b_t20_avg = float(fe.loc["format_t20", "beta_log_avg"])
    b_odi_sr  = float(fe.loc["format_odi", "beta_log_sr"])
    b_t20_sr  = float(fe.loc["format_t20", "beta_log_sr"])

    df = df.merge(psi, on="player_id", how="left")

    df["fe_log_avg"] = alpha_avg + b_odi_avg * df["format_odi"] + b_t20_avg * df["format_t20"]
    df["fe_log_sr"]  = alpha_sr  + b_odi_sr  * df["format_odi"] + b_t20_sr  * df["format_t20"]

    df["cond_log_avg"] = df["fe_log_avg"] + df["psi_log_avg"]
    df["cond_log_sr"]  = df["fe_log_sr"]  + df["psi_log_sr"]

    df["pred_avg_marginal"]     = np.exp(df["fe_log_avg"])
    df["pred_sr_marginal"]      = np.exp(df["fe_log_sr"])
    df["pred_avg_conditional"]  = np.exp(df["cond_log_avg"])
    df["pred_sr_conditional"]   = np.exp(df["cond_log_sr"])

    y_avg = df["avg"].values
    y_sr  = df["sr"].values

    rows = []

    rows.append({**evaluate("marginal | avg (fixed effects only)",
                             y_avg, df["pred_avg_marginal"].values),
                 "outcome": "avg"})
    rows.append({**evaluate("marginal | sr  (fixed effects only)",
                             y_sr,  df["pred_sr_marginal"].values),
                 "outcome": "sr"})

    rows.append({**evaluate("conditional | avg (fixed + player effects)",
                             y_avg, df["pred_avg_conditional"].values),
                 "outcome": "avg"})
    rows.append({**evaluate("conditional | sr  (fixed + player effects)",
                             y_sr,  df["pred_sr_conditional"].values),
                 "outcome": "sr"})

    # Only in-sample fits are reported here; EvaluateKFold.py holds the held-out results.

    results = pd.DataFrame(rows)[["metric_set", "outcome", "R2", "MAE", "RMSE", "MAPE"]]

    print("\n" + "="*72)
    print("BATTING MODEL EVALUATION")
    print("="*72)
    print(results.to_string(index=False))
    print("\nMAPE is in %. R2 closer to 1 = better. RMSE is on original scale.")
    print(f"  avg scale:  population mean = {y_avg.mean():.1f}")
    print(f"  sr  scale:  population mean = {y_sr.mean():.1f}")

    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved: {RESULTS_PATH}")

if __name__ == "__main__":
    main()
