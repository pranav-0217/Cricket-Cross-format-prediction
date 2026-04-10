"""Prepare joint batting data for multivariate mixed-effects model.

Outputs:
  Data/batting_joint.csv          -- model-ready rows
  Data/batting_joint_meta.json    -- player index mapping & encoding info

Model structure (from professor's design):
  y_ij = X_ij * beta + Z_ij * psi_i + eps_ij
  y   = [batting average, strike rate]   (log-transformed)
  X   = [format_odi, format_t20]         (Test is baseline)
  Z   = player indicator (one-hot)       --> psi = player random effects
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path("Data") / "batting_long.csv"
OUTPUT_CSV_PATH = Path("Data") / "batting_joint.csv"
OUTPUT_META_PATH = Path("Data") / "batting_joint_meta.json"

# Minimum quality filters
MIN_INNINGS = 5
MIN_AVG = 1.0   # strict positive so log is defined
MIN_SR = 1.0
EPS = 1e-6


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    for col in ["avg", "sr", "inns"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Row-quality filters
    df = df[
        df["avg"].notna() & df["sr"].notna() & df["inns"].notna()
        & (df["avg"] >= MIN_AVG)
        & (df["sr"] >= MIN_SR)
        & (df["inns"] >= MIN_INNINGS)
    ].copy()

    # Keep repeated-measures players only (seen in >=2 distinct formats)
    format_counts = df.groupby("player_name")["format"].nunique()
    eligible = format_counts[format_counts >= 2].index
    df = df[df["player_name"].isin(eligible)].copy()

    # Encode format with Test as baseline (matches professor's diagram: T20=[1,0], ODI=[0,1])
    format_order = ["Test", "ODI", "T20"]
    df["format"] = pd.Categorical(df["format"], categories=format_order)
    df = df[df["format"].notna()].copy()
    df["format_odi"] = (df["format"] == "ODI").astype(int)
    df["format_t20"] = (df["format"] == "T20").astype(int)
    df["format_idx"] = df["format"].cat.codes.astype(int)  # Test=0, ODI=1, T20=2

    # Player indexing for the Z matrix (one-hot design matrix for random effects)
    player_names = sorted(df["player_name"].unique().tolist())
    player_to_idx = {name: idx for idx, name in enumerate(player_names)}
    idx_to_player = {str(idx): name for name, idx in player_to_idx.items()}
    df["player_idx"] = df["player_name"].map(player_to_idx).astype(int)

    # Joint outcomes on log scale: y1 = log(avg), y2 = log(sr)
    df["y1"] = np.log(df["avg"] + EPS)
    df["y2"] = np.log(df["sr"] + EPS)

    out_cols = [
        "player_idx",
        "player_name",
        "format",
        "format_idx",
        "format_odi",
        "format_t20",
        "y1",   # log(avg)
        "y2",   # log(sr)
        "avg",
        "sr",
        "inns",
    ]
    out_df = df[out_cols].sort_values(["player_idx", "format_idx"]).reset_index(drop=True)
    out_df.to_csv(OUTPUT_CSV_PATH, index=False)

    meta = {
        "eps": EPS,
        "min_innings": MIN_INNINGS,
        "outcomes": {"y1": "log(batting_average)", "y2": "log(strike_rate)"},
        "player_idx_to_name": idx_to_player,
        "n_players": len(player_names),
        "format_coding": {
            "baseline": "Test",
            "format_idx": {"Test": 0, "ODI": 1, "T20": 2},
            "dummies": {
                "format_odi": "1 if format == ODI else 0",
                "format_t20": "1 if format == T20 else 0",
            },
        },
    }
    OUTPUT_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved: {OUTPUT_CSV_PATH}")
    print(f"Saved: {OUTPUT_META_PATH}")
    print(f"Final rows: {len(out_df)}")
    print(f"Unique players: {out_df['player_idx'].nunique()}")
    fmt_counts = out_df["format"].value_counts().reindex(format_order, fill_value=0)
    print(f"Counts by format: {fmt_counts.to_dict()}")


if __name__ == "__main__":
    main()
