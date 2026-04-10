"""Prepare enriched bowling joint data — adds debut_year_scaled to X.

Uses the same debut_year source as batting (combined_cricket_stats.csv Span Start columns).
is_lefthanded / position_order remain NaN until scraping succeeds.

Output:
    Data/bowling_joint_enriched.csv
    Data/bowling_joint_enriched_meta.json
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

JOINT_PATH    = Path("Data") / "bowling_joint.csv"
COMBINED_PATH = Path("Data") / "combined_cricket_stats.csv"
META_PATH     = Path("Data") / "player_metadata_cricinfo.csv"
OUTPUT_PATH   = Path("Data") / "bowling_joint_enriched.csv"
META_OUT      = Path("Data") / "bowling_joint_enriched_meta.json"


def clean_name(n: str) -> str:
    return re.sub(r"\s*\([^)]+\)", "", str(n)).strip()


def load_debut_year() -> pd.Series:
    df = pd.read_csv(COMBINED_PATH)
    df["player_name"] = df["Player Name"].apply(clean_name)
    span_cols = [
        "Span Start_ODI_batting", "Span Start_T20_batting", "Span Start_Test_batting",
        "Span Start_ODI_bowling", "Span Start_T20_bowling", "Span Start_Test_bowling",
    ]
    available = [c for c in span_cols if c in df.columns]
    for c in available:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["debut_year"] = df[available].min(axis=1)
    return df.set_index("player_name")["debut_year"].dropna()


def main() -> None:
    df = pd.read_csv(JOINT_PATH)

    debut_map = load_debut_year()
    df["debut_year"] = df["player_name"].map(debut_map)

    n_players = df["player_name"].nunique()
    n_debut = n_players - df.groupby("player_name")["debut_year"].first().isna().sum()
    print(f"debut_year populated for {n_debut}/{n_players} players "
          f"({n_debut/n_players*100:.0f}%)")

    debut_mean = df["debut_year"].mean()
    df["debut_year_scaled"]  = (df["debut_year"].fillna(debut_mean) - debut_mean) / df["debut_year"].std()
    df["debut_year_missing"] = df["debut_year"].isna().astype(int)

    # Load scraped metadata (batting_style, playing_role, born_year)
    if META_PATH.exists():
        meta_cols = ["player_name", "batting_style", "bowling_style", "playing_role", "born_year"]
        meta = pd.read_csv(META_PATH)[[c for c in meta_cols if c in pd.read_csv(META_PATH).columns]]
        df = df.merge(meta, on="player_name", how="left")

        def encode_batting_style(s):
            if pd.isna(s): return np.nan
            return 1.0 if "left" in str(s).lower() else (0.0 if "right" in str(s).lower() else np.nan)

        def encode_playing_role(s):
            if pd.isna(s): return np.nan
            s = str(s).lower()
            if "wk" in s or ("bats" in s and "all" not in s): return 0.0
            if "batting all" in s: return 1.0
            if "bowling all" in s: return 2.0
            if "bowl" in s: return 3.0
            return np.nan

        def encode_bowling_hand(s):
            """Left-arm -> 1, Right-arm -> 0."""
            if pd.isna(s): return np.nan
            return 1.0 if "left" in str(s).lower() else (0.0 if "right" in str(s).lower() else np.nan)

        def encode_bowling_type(s):
            """Spin -> 1, Pace -> 0."""
            if pd.isna(s): return np.nan
            s = str(s).lower()
            if any(w in s for w in ["spin","break","turn","googly","leg-spin","off-spin","chinaman"]): return 1.0
            if any(w in s for w in ["fast","medium","pace","seam","swing"]): return 0.0
            return np.nan

        df["is_lefthanded"]    = df["batting_style"].apply(encode_batting_style)
        df["position_order"]   = df["playing_role"].apply(encode_playing_role)
        df["bowling_hand"]     = df["bowling_style"].apply(encode_bowling_hand) if "bowling_style" in df.columns else np.nan
        df["bowling_type"]     = df["bowling_style"].apply(encode_bowling_type) if "bowling_style" in df.columns else np.nan
        age_at_debut = df["debut_year"] - df["born_year"]
        age_mean = age_at_debut.mean()
        df["age_at_debut_scaled"] = (age_at_debut.fillna(age_mean) - age_mean) / age_at_debut.std()
        print(f"batting_style populated: {df['is_lefthanded'].notna().sum()}/{len(df)} rows")
        print(f"playing_role  populated: {df['position_order'].notna().sum()}/{len(df)} rows")
        print(f"bowling_hand  populated: {df['bowling_hand'].notna().sum()}/{len(df)} rows")
        print(f"bowling_type  populated: {df['bowling_type'].notna().sum()}/{len(df)} rows")
        print(f"age_at_debut  populated: {age_at_debut.notna().sum()}/{len(df)} rows")
    else:
        df["is_lefthanded"]      = np.nan
        df["position_order"]     = np.nan
        df["age_at_debut_scaled"] = 0.0

    print("\n=== X covariates available ===")
    covariates = {
        "format_odi":         df["format_odi"].notna().sum(),
        "format_t20":         df["format_t20"].notna().sum(),
        "debut_year_scaled":  df["debut_year_scaled"].notna().sum(),
        "debut_year_missing": int((df["debut_year_missing"] == 0).sum()),
        "is_lefthanded":      int(df["is_lefthanded"].notna().sum()),
        "position_order":     int(df["position_order"].notna().sum()),
    }
    total = len(df)
    for k, v in covariates.items():
        print(f"  {k:<25} {v:>5}/{total}  ({v/total*100:.0f}%)")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}  ({len(df)} rows)")

    meta_out = {
        "covariates": list(covariates.keys()),
        "n_rows": len(df),
        "n_players": n_players,
        "debut_year_coverage_pct": round(n_debut / n_players * 100, 1),
        "batting_style_available": False,
        "playing_role_available":  False,
    }
    META_OUT.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print(f"Saved: {META_OUT}")


if __name__ == "__main__":
    main()
