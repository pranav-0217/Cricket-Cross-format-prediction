import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("Data") / "bowling_long.csv"
OUTPUT_CSV_PATH = Path("Data") / "bowling_joint.csv"
OUTPUT_META_PATH = Path("Data") / "bowling_joint_meta.json"
EPS = 1e-6


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    for col in ["overs", "balls", "wkts", "econ", "avg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "wkts" not in df.columns:
        raise ValueError("Required column missing: wkts")

    # Apply row-quality filters: overs>=20 and wkts>=10.
    # If overs does not exist, use balls>=120 as the overs proxy.
    if "overs" in df.columns:
        df = df[df["overs"] >= 20].copy()
    elif "balls" in df.columns:
        df = df[df["balls"] >= 120].copy()
    else:
        raise ValueError("Need either overs or balls column for workload filtering.")

    df = df[df["wkts"] >= 10].copy()

    # Keep only rows where both outcomes are available and strictly positive.
    for col in ["econ", "avg"]:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")
    df = df[df["econ"].notna() & df["avg"].notna()].copy()
    df = df[(df["econ"] > 0) & (df["avg"] > 0)].copy()

    # Keep repeated-measures players only (seen in at least 2 distinct formats).
    format_counts = df.groupby("player_name")["format"].nunique()
    eligible_players = format_counts[format_counts >= 2].index
    df = df[df["player_name"].isin(eligible_players)].copy()

    # Encode format with Test as baseline.
    format_order = ["Test", "ODI", "T20"]
    df["format"] = pd.Categorical(df["format"], categories=format_order)
    df = df[df["format"].notna()].copy()
    df["format_odi"] = (df["format"] == "ODI").astype(int)
    df["format_t20"] = (df["format"] == "T20").astype(int)
    df["format_idx"] = df["format"].cat.codes.astype(int)  # Test=0, ODI=1, T20=2

    # Player indexing for mixed / hierarchical models.
    player_names = sorted(df["player_name"].unique().tolist())
    player_to_idx = {name: idx for idx, name in enumerate(player_names)}
    idx_to_player = {str(idx): name for name, idx in player_to_idx.items()}
    df["player_idx"] = df["player_name"].map(player_to_idx).astype(int)

    # Joint outcomes on log scale.
    df["y1"] = np.log(df["econ"] + EPS)
    df["y2"] = np.log(df["avg"] + EPS)

    out_cols = [
        "player_idx",
        "player_name",
        "format",
        "format_idx",
        "format_odi",
        "format_t20",
        "y1",
        "y2",
    ]
    out_df = df[out_cols].sort_values(["player_idx", "format_idx"]).reset_index(drop=True)
    out_df.to_csv(OUTPUT_CSV_PATH, index=False)

    meta = {
        "eps": EPS,
        "player_idx_to_name": idx_to_player,
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
    print(f"Counts by format: {out_df['format'].value_counts().reindex(format_order, fill_value=0).to_dict()}")


if __name__ == "__main__":
    main()
