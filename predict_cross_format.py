"""Cross-format batting performance predictor & classifier (enriched model).

Uses the enriched multivariate mixed-effects model:
    y_ij = alpha + beta_format * format + beta_X * X_i + psi_i + eps_ij
    y = [log(avg), log(sr)]

Two prediction modes:
  1. Known player  -- player is in the training set; look up their psi directly.
  2. New player    -- player is unseen; estimate psi via k-nearest-neighbours
                      from players with similar stats in the known format.

Classifier: given psi, predict above/below format-average batting performance.

Usage:
    # Known player
    python predict_cross_format.py --player "V Kohli" --target_format T20

    # New player (provide their known format stats)
    python predict_cross_format.py --new_player "New Guy" \\
        --known_format ODI --known_avg 42.5 --known_sr 88.0 --target_format Test

    # Classifier evaluation on full dataset
    python predict_cross_format.py --classify
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH      = Path("Data") / "batting_joint_enriched.csv"
META_PATH      = Path("Data") / "batting_joint_meta.json"
PARAMS_PATH    = Path("outputs") / "mixed_effects" / "joint_batting_enriched_params_mean.csv"
PLAYER_PATH    = Path("outputs") / "mixed_effects" / "batting_enriched_player_rankings.csv"
OUTPUT_DIR     = Path("outputs") / "predictions"

PREDICTORS = ["format_odi", "format_t20", "debut_year_scaled", "debut_year_missing",
              "is_lefthanded", "position_order", "age_at_debut_scaled"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-format batting predictor (enriched)")
    p.add_argument("--player",        type=str,   default=None)
    p.add_argument("--new_player",    type=str,   default=None)
    p.add_argument("--known_format",  type=str,   default=None, choices=["Test", "ODI", "T20"])
    p.add_argument("--known_avg",     type=float, default=None)
    p.add_argument("--known_sr",      type=float, default=None)
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
    alpha = np.array([row["alpha_log_avg"], row["alpha_log_sr"]])
    # beta shape: (2, n_predictors) — rows = [avg, sr], cols = predictors
    beta = np.array([
        [row[f"beta_{p}_log_avg"] for p in PREDICTORS],
        [row[f"beta_{p}_log_sr"]  for p in PREDICTORS],
    ])
    return alpha, beta


def player_x_vector(player_name: str, target_format: str, df: pd.DataFrame) -> np.ndarray:
    """Build X vector for a known player in a target format."""
    # Get player's non-format covariates from any row (same for all formats)
    row = df[df["player_name"] == player_name].iloc[0]
    fmt = format_dummies(target_format)
    x = np.array([
        fmt["format_odi"],
        fmt["format_t20"],
        float(row.get("debut_year_scaled", 0.0) or 0.0),
        float(row.get("debut_year_missing", 0.0) or 0.0),
        float(row.get("is_lefthanded", 0.0) or 0.0),
        float(row.get("position_order", 0.0) or 0.0),
        float(row.get("age_at_debut_scaled", 0.0) or 0.0),
    ])
    return x


def predict_known(player_name: str, target_format: str,
                  alpha, beta, df, player_effects) -> dict:
    row = player_effects[player_effects["player_name"] == player_name]
    if row.empty:
        return {"error": f"'{player_name}' not found. Use --new_player for unseen players."}
    psi = np.array([float(row["psi_log_avg"].iloc[0]), float(row["psi_log_sr"].iloc[0])])
    x   = player_x_vector(player_name, target_format, df)
    mu  = alpha + beta @ x + psi
    return {
        "player":        player_name,
        "target_format": target_format,
        "predicted_avg": round(float(np.exp(mu[0])), 2),
        "predicted_sr":  round(float(np.exp(mu[1])), 2),
        "psi_log_avg":   round(float(psi[0]), 4),
        "psi_log_sr":    round(float(psi[1]), 4),
        "psi_source":    "enriched model BLUP",
    }


def estimate_psi_knn(known_avg, known_sr, known_format, df, player_effects,
                     alpha, beta, k) -> tuple:
    """Estimate psi for an unseen player via k-NN on format-adjusted residuals."""
    # Mean X for known format (use median of non-format covariates as representative)
    fmt = format_dummies(known_format)
    x0 = np.array([fmt["format_odi"], fmt["format_t20"], 0.0, 0.0, 0.0, 0.0, 0.0])
    mu0 = alpha + beta @ x0      # baseline fixed-effect mean for that format

    new_log = np.array([np.log(max(known_avg, 0.01)), np.log(max(known_sr, 0.01))])
    new_residual = new_log - mu0

    sub = df[df["format"] == known_format].merge(
        player_effects[["player_name", "psi_log_avg", "psi_log_sr"]],
        on="player_name", how="inner"
    )
    known_residuals = np.stack([sub["y1"].to_numpy() - mu0[0],
                                sub["y2"].to_numpy() - mu0[1]], axis=1)
    dists = np.linalg.norm(known_residuals - new_residual, axis=1)
    top_k = sub.iloc[np.argsort(dists)[:k]]
    psi_est = np.array([float(top_k["psi_log_avg"].mean()),
                        float(top_k["psi_log_sr"].mean())])
    return psi_est, top_k["player_name"].tolist()


def predict_new(player_label, known_format, known_avg, known_sr, target_format,
                alpha, beta, df, player_effects, k) -> dict:
    psi, neighbours = estimate_psi_knn(known_avg, known_sr, known_format,
                                       df, player_effects, alpha, beta, k)
    fmt = format_dummies(target_format)
    x   = np.array([fmt["format_odi"], fmt["format_t20"], 0.0, 0.0, 0.0, 0.0, 0.0])
    mu  = alpha + beta @ x + psi
    return {
        "player":        player_label,
        "target_format": target_format,
        "predicted_avg": round(float(np.exp(mu[0])), 2),
        "predicted_sr":  round(float(np.exp(mu[1])), 2),
        "psi_log_avg":   round(float(psi[0]), 4),
        "psi_log_sr":    round(float(psi[1]), 4),
        "psi_source":    f"k-NN (k={k}) from {known_format} stats",
        "nearest_neighbours": neighbours,
        f"known_{known_format}_avg": known_avg,
        f"known_{known_format}_sr":  known_sr,
    }


def build_classifier(df, player_effects) -> None:
    """Classifier: given psi + format covariates, predict above-average batting."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    merged = df.merge(
        player_effects[["player_name", "psi_log_avg", "psi_log_sr", "psi_combined", "rank"]],
        on="player_name", how="left"
    ).dropna(subset=["psi_log_avg", "psi_log_sr"])

    # Label: 1 if player's avg > format median (more robust than mean)
    fmt_median = merged.groupby("format")["avg"].median()
    merged["above_avg"] = (merged["avg"] > merged["format"].map(fmt_median)).astype(int)

    feature_cols = ["psi_log_avg", "psi_log_sr", "format_odi", "format_t20",
                    "is_lefthanded", "position_order"]
    for c in feature_cols:
        if c not in merged.columns:
            merged[c] = 0.0
        merged[c] = merged[c].fillna(0.0)

    X = merged[feature_cols].to_numpy(dtype=float)
    y = merged["above_avg"].to_numpy(dtype=int)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n=== Batting Classifier: predict above-median performance per format ===")
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

    merged["pred_above_avg"] = best_clf.predict(X_sc)
    merged["pred_prob"]      = best_clf.predict_proba(X_sc)[:, 1]
    out = OUTPUT_DIR / "batting_cross_format_predictions.csv"
    merged[["player_name", "format", "avg", "sr", "above_avg",
            "pred_above_avg", "pred_prob", "psi_combined", "rank"]].to_csv(out, index=False)
    print(f"Predictions saved: {out}")


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in [PARAMS_PATH, PLAYER_PATH, DATA_PATH]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run fit_joint_batting_pymc_enriched.py first.")
            return

    df             = pd.read_csv(DATA_PATH)
    player_effects = pd.read_csv(PLAYER_PATH)
    alpha, beta    = load_params()

    if args.player and args.target_format:
        result = predict_known(args.player, args.target_format,
                               alpha, beta, df, player_effects)
        print("\n=== Prediction (known player — enriched model) ===")
        for k, v in result.items():
            print(f"  {k}: {v}")

    if args.new_player:
        if not all([args.known_format, args.known_avg, args.known_sr, args.target_format]):
            print("For --new_player provide: --known_format --known_avg --known_sr --target_format")
            return
        result = predict_new(args.new_player, args.known_format,
                             args.known_avg, args.known_sr, args.target_format,
                             alpha, beta, df, player_effects, args.k)
        print("\n=== Prediction (new player via k-NN — enriched model) ===")
        for k, v in result.items():
            print(f"  {k}: {v}")

    if args.classify or (not args.player and not args.new_player):
        build_classifier(df, player_effects)


if __name__ == "__main__":
    main()
