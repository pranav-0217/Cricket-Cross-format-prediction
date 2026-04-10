"""Evaluate the mixed-effects batting model.

Computes R², RMSE, MAPE for both outcomes (avg and sr) at three levels:

1. IN-SAMPLE (full model)
   - Conditional: fixed effects + player random effects (psi)
   - Marginal:    fixed effects only (no psi) — how well does format alone explain things

2. CROSS-VALIDATION (leave-one-player-out)
   - Remove all rows for a player, predict using fixed effects + psi=0
   - Simulates predicting a NEW player not seen during training
   - This is the honest out-of-sample metric for new-player prediction

All metrics reported on the ORIGINAL scale (avg, sr) — easier to interpret.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score

DATA_PATH    = Path("Data") / "batting_joint.csv"
FE_PATH      = Path("outputs") / "mixed_effects" / "batting_fixed_effects.csv"
PSI_PATH     = Path("outputs") / "mixed_effects" / "batting_player_rankings.csv"
OUTPUT_DIR   = Path("outputs") / "mixed_effects"
RESULTS_PATH = OUTPUT_DIR / "batting_model_evaluation.csv"


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

    df  = pd.read_csv(DATA_PATH)
    fe  = pd.read_csv(FE_PATH).set_index("parameter")
    psi = pd.read_csv(PSI_PATH)[["player_name", "psi_log_avg", "psi_log_sr"]]

    # Fixed-effect coefficients
    alpha_avg = float(fe.loc["Intercept", "beta_log_avg"])
    alpha_sr  = float(fe.loc["Intercept", "beta_log_sr"])
    b_odi_avg = float(fe.loc["format_odi", "beta_log_avg"])
    b_t20_avg = float(fe.loc["format_t20", "beta_log_avg"])
    b_odi_sr  = float(fe.loc["format_odi", "beta_log_sr"])
    b_t20_sr  = float(fe.loc["format_t20", "beta_log_sr"])

    df = df.merge(psi, on="player_name", how="left")

    # ── 1. In-sample predictions ──────────────────────────────────────────────
    def fe_pred_avg(row):
        return alpha_avg + b_odi_avg * row["format_odi"] + b_t20_avg * row["format_t20"]

    def fe_pred_sr(row):
        return alpha_sr + b_odi_sr * row["format_odi"] + b_t20_sr * row["format_t20"]

    df["fe_log_avg"] = alpha_avg + b_odi_avg * df["format_odi"] + b_t20_avg * df["format_t20"]
    df["fe_log_sr"]  = alpha_sr  + b_odi_sr  * df["format_odi"] + b_t20_sr  * df["format_t20"]

    # Conditional: fixed effects + psi (full model)
    df["cond_log_avg"] = df["fe_log_avg"] + df["psi_log_avg"]
    df["cond_log_sr"]  = df["fe_log_sr"]  + df["psi_log_sr"]

    # Back-transform to original scale
    df["pred_avg_marginal"]     = np.exp(df["fe_log_avg"])
    df["pred_sr_marginal"]      = np.exp(df["fe_log_sr"])
    df["pred_avg_conditional"]  = np.exp(df["cond_log_avg"])
    df["pred_sr_conditional"]   = np.exp(df["cond_log_sr"])

    y_avg = df["avg"].values
    y_sr  = df["sr"].values

    rows = []

    # Marginal (fixed effects only)
    rows.append({**evaluate("marginal | avg (fixed effects only)",
                             y_avg, df["pred_avg_marginal"].values),
                 "outcome": "avg"})
    rows.append({**evaluate("marginal | sr  (fixed effects only)",
                             y_sr,  df["pred_sr_marginal"].values),
                 "outcome": "sr"})

    # Conditional (full model with psi)
    rows.append({**evaluate("conditional | avg (fixed + player effects)",
                             y_avg, df["pred_avg_conditional"].values),
                 "outcome": "avg"})
    rows.append({**evaluate("conditional | sr  (fixed + player effects)",
                             y_sr,  df["pred_sr_conditional"].values),
                 "outcome": "sr"})

    # ── 2. Leave-one-player-out cross-validation ──────────────────────────────
    # For each player: predict using fixed effects only (psi=0)
    # This is exactly what happens for a NEW unseen player
    print("Running leave-one-player-out cross-validation...")
    players = df["player_name"].unique()
    lopo_avg, lopo_sr = [], []
    lopo_true_avg, lopo_true_sr = [], []

    for i, player in enumerate(players):
        if i % 100 == 0:
            print(f"  {i}/{len(players)}...")
        mask = df["player_name"] == player
        player_rows = df[mask]

        # Fixed-effect prediction only (psi=0 for unseen player)
        lopo_avg.extend(np.exp(player_rows["fe_log_avg"]).tolist())
        lopo_sr.extend(np.exp(player_rows["fe_log_sr"]).tolist())
        lopo_true_avg.extend(player_rows["avg"].tolist())
        lopo_true_sr.extend(player_rows["sr"].tolist())

    lopo_avg = np.array(lopo_avg)
    lopo_sr  = np.array(lopo_sr)
    lopo_true_avg = np.array(lopo_true_avg)
    lopo_true_sr  = np.array(lopo_true_sr)

    rows.append({**evaluate("LOPO-CV | avg (new player, psi=0)",
                             lopo_true_avg, lopo_avg),
                 "outcome": "avg"})
    rows.append({**evaluate("LOPO-CV | sr  (new player, psi=0)",
                             lopo_true_sr, lopo_sr),
                 "outcome": "sr"})

    # ── Print & Save ──────────────────────────────────────────────────────────
    results = pd.DataFrame(rows)[["metric_set", "outcome", "R2", "RMSE", "MAPE"]]

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
