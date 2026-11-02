"""Cross-format bowling performance predictor & classifier (enriched model)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH   = Path("Data/new/processed/bowling_joint_enriched.csv")
META_PATH   = Path("Data/new/processed/bowling_joint_enriched_meta.json")
PARAMS_PATH = Path("outputs") / "mixed_effects" / "joint_bowling_enriched_params_mean.csv"
PLAYER_PATH = Path("outputs") / "mixed_effects" / "bowling_enriched_player_rankings.csv"
OUTPUT_DIR  = Path("outputs") / "predictions"

PREDICTORS = json.loads(META_PATH.read_text(encoding="utf-8"))["predictors"]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-format bowling predictor (enriched)")
    p.add_argument("--player",        type=str,   default=None)
    p.add_argument("--player_id",     type=int,   default=None,
                   help="Stable ESPN player ID; preferred when a name is shared.")
    p.add_argument("--new_player",    type=str,   default=None)
    p.add_argument("--known_format",  type=str,   default=None, choices=["Test", "ODI", "T20"])
    p.add_argument("--known_econ",    type=float, default=None)
    p.add_argument("--known_avg",     type=float, default=None)
    p.add_argument("--target_format", type=str,   default=None, choices=["Test", "ODI", "T20"])
    p.add_argument("--k",             type=int,   default=10)
    p.add_argument("--classify",      action="store_true")
    return p.parse_args()

def format_dummies(fmt: str) -> dict:
    return {"format_odi": 1.0 if fmt == "ODI" else 0.0,
            "format_t20": 1.0 if fmt == "T20" else 0.0}

def load_params():
    """Load alpha and full beta vector from enriched params CSV."""
    row = pd.read_csv(PARAMS_PATH).iloc[0]
    alpha = np.array([row["alpha_log_econ"], row["alpha_log_avg"]])
    beta  = np.array([
        [row[f"beta_{p}_log_econ"] for p in PREDICTORS],
        [row[f"beta_{p}_log_avg"]  for p in PREDICTORS],
    ])
    return alpha, beta

def player_x_vector(player_id: int, target_format: str, df: pd.DataFrame) -> np.ndarray:
    """Build X vector for a known player in a target format."""
    row = df[df["player_id"] == player_id].iloc[0]
    fmt = format_dummies(target_format)
    x = np.array([
        fmt[predictor] if predictor in fmt else float(row[predictor])
        for predictor in PREDICTORS
    ])
    return x

def predict_known(player_name: str | None, target_format: str,
                  alpha, beta, df, player_effects, player_id: int | None = None) -> dict:
    if player_id is None:
        matches = df.loc[df["player_name"] == player_name, "player_id"].unique()
        if len(matches) > 1:
            return {"error": f"'{player_name}' is ambiguous; "
                             f"use --player_id with one of {matches.tolist()}."}
        if len(matches) == 1:
            player_id = int(matches[0])
    row = player_effects[player_effects["player_id"] == player_id]
    if row.empty:
        label = player_name if player_name is not None else player_id
        return {"error": f"'{label}' not found. Use --new_player for unseen players."}
    player_name = str(row["player_name"].iloc[0])
    psi = np.array([float(row["psi_log_econ"].iloc[0]),
                    float(row["psi_log_avg"].iloc[0])])
    x   = player_x_vector(player_id, target_format, df)
    mu  = alpha + beta @ x + psi
    return {
        "player":          player_name,
        "player_id":       int(player_id),
        "target_format":   target_format,
        "predicted_econ":  round(float(np.exp(mu[0])), 2),
        "predicted_avg":   round(float(np.exp(mu[1])), 2),
        "psi_log_econ":    round(float(psi[0]), 4),
        "psi_log_avg":     round(float(psi[1]), 4),
        "psi_source":      "enriched model BLUP",
    }

def estimate_psi_knn(known_econ, known_avg, known_format, df, player_effects,
                     alpha, beta, k) -> tuple:
    """Estimate psi for an unseen player via k-NN on format-adjusted residuals."""
    fmt = format_dummies(known_format)
    x0  = np.array([fmt["format_odi"], fmt["format_t20"]] + [0.0] * (len(PREDICTORS) - 2))
    mu0 = alpha + beta @ x0

    new_log = np.array([np.log(max(known_econ, 0.01)), np.log(max(known_avg, 0.01))])
    new_residual = new_log - mu0

    sub = df[df["format"] == known_format].merge(
        player_effects[["player_id", "psi_log_econ", "psi_log_avg"]],
        on="player_id", how="inner"
    )
    known_residuals = np.stack([sub["y1"].to_numpy() - mu0[0],
                                sub["y2"].to_numpy() - mu0[1]], axis=1)
    dists = np.linalg.norm(known_residuals - new_residual, axis=1)
    top_k = sub.iloc[np.argsort(dists)[:k]]
    psi_est = np.array([float(top_k["psi_log_econ"].mean()),
                        float(top_k["psi_log_avg"].mean())])
    return psi_est, top_k["player_name"].tolist()

def predict_new(player_label, known_format, known_econ, known_avg, target_format,
                alpha, beta, df, player_effects, k) -> dict:
    psi, neighbours = estimate_psi_knn(known_econ, known_avg, known_format,
                                       df, player_effects, alpha, beta, k)
    fmt = format_dummies(target_format)
    x   = np.array([fmt["format_odi"], fmt["format_t20"]] + [0.0] * (len(PREDICTORS) - 2))
    mu  = alpha + beta @ x + psi
    return {
        "player":          player_label,
        "target_format":   target_format,
        "predicted_econ":  round(float(np.exp(mu[0])), 2),
        "predicted_avg":   round(float(np.exp(mu[1])), 2),
        "psi_log_econ":    round(float(psi[0]), 4),
        "psi_log_avg":     round(float(psi[1]), 4),
        "psi_source":      f"k-NN (k={k}) from {known_format} stats",
        "nearest_neighbours": neighbours,
        f"known_{known_format}_econ": known_econ,
        f"known_{known_format}_avg":  known_avg,
    }

def build_classifier(df, player_effects) -> None:
    """Classifier: given psi + format, predict below-average (good) bowling."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df2 = df.copy()
    df2["econ"] = np.exp(df2["y1"])
    df2["avg"]  = np.exp(df2["y2"])

    merged = df2.merge(
        player_effects[["player_id", "psi_log_econ", "psi_log_avg", "psi_combined", "rank"]],
        on="player_id", how="left"
    ).dropna(subset=["psi_log_econ", "psi_log_avg"])

    fmt_med_econ = merged.groupby("format")["econ"].median()
    fmt_med_avg  = merged.groupby("format")["avg"].median()
    merged["good_bowler"] = (
        (merged["econ"] < merged["format"].map(fmt_med_econ)) &
        (merged["avg"]  < merged["format"].map(fmt_med_avg))
    ).astype(int)

    feature_cols = ["psi_log_econ", "psi_log_avg", "format_odi", "format_t20",
                    "bowling_hand_left", "bowling_type_spin",
                    "role_allrounder", "role_bowler"]
    for c in feature_cols:
        if c not in merged.columns:
            merged[c] = 0.0
        merged[c] = merged[c].fillna(0.0)

    X = merged[feature_cols].to_numpy(dtype=float)
    y = merged["good_bowler"].to_numpy(dtype=int)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n=== Bowling Classifier: predict below-median econ AND avg (good bowler) ===")
    print(f"Dataset: {len(X)} rows | positive rate: {y.mean():.1%}\n")

    best_clf = None
    for name, clf in [
        ("Logistic Regression", LogisticRegression(max_iter=500)),
        ("Gradient Boosting",   GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]:
        auc = cross_val_score(clf, X_sc, y, cv=cv, scoring="roc_auc")
        acc = cross_val_score(clf, X_sc, y, cv=cv, scoring="accuracy")
        print(f"{name}:")
        print(f"  ROC-AUC : {auc.mean():.3f} ± {auc.std():.3f}")
        print(f"  Accuracy: {acc.mean():.3f} ± {acc.std():.3f}")
        if name == "Gradient Boosting":
            clf.fit(X_sc, y)
            fi = dict(zip(feature_cols, clf.feature_importances_))
            print(f"  Feature importances: { {k: round(v,4) for k,v in fi.items()} }")
            best_clf = clf
        print()

    merged["pred_good_bowler"] = best_clf.predict(X_sc)
    merged["pred_prob"]        = best_clf.predict_proba(X_sc)[:, 1]
    out = OUTPUT_DIR / "bowling_cross_format_predictions.csv"
    merged[["player_id", "player_name", "format", "econ", "avg", "good_bowler",
            "pred_good_bowler", "pred_prob", "psi_combined", "rank"]].to_csv(out, index=False)
    print(f"Predictions saved: {out}")

def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in [PARAMS_PATH, PLAYER_PATH, DATA_PATH]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run FitBowlingEnriched.py first.")
            return

    df             = pd.read_csv(DATA_PATH)
    player_effects = pd.read_csv(PLAYER_PATH)
    alpha, beta    = load_params()

    if (args.player or args.player_id is not None) and args.target_format:
        result = predict_known(args.player, args.target_format,
                               alpha, beta, df, player_effects, args.player_id)
        print("\n=== Prediction (known player — enriched model) ===")
        for k, v in result.items():
            print(f"  {k}: {v}")

    if args.new_player:
        if not all([args.known_format, args.known_econ, args.known_avg, args.target_format]):
            print("For --new_player provide: "
                  "--known_format --known_econ --known_avg --target_format")
            return
        result = predict_new(args.new_player, args.known_format,
                             args.known_econ, args.known_avg, args.target_format,
                             alpha, beta, df, player_effects, args.k)
        print("\n=== Prediction (new player via k-NN — enriched model) ===")
        for k, v in result.items():
            print(f"  {k}: {v}")

    if args.classify or (not args.player and args.player_id is None and not args.new_player):
        build_classifier(df, player_effects)

if __name__ == "__main__":
    main()
