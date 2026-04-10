"""Prepare enriched batting joint data for mixed-effects model.

Adds available metadata to X (fixed effects):
    - format_odi, format_t20   (already present)
    - debut_year_scaled        (career debut year, centred & scaled)
    - is_lefthanded            (0/1) -- added when scraping available
    - position_order           (0=top, 1=middle, 2=lower) -- added when scraping available

Currently populated:
    debut_year  -> from combined_cricket_stats.csv (38% coverage, mean-imputed for rest)
    is_lefthanded, position_order -> NaN until metadata scraping succeeds

Output:
    Data/batting_joint_enriched.csv
    Data/batting_joint_enriched_meta.json
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

JOINT_PATH    = Path("Data") / "batting_joint.csv"
COMBINED_PATH = Path("Data") / "combined_cricket_stats.csv"
META_PATH     = Path("Data") / "player_metadata_cricinfo.csv"   # from scrape_cricbuzz_metadata.py
OUTPUT_PATH   = Path("Data") / "batting_joint_enriched.csv"
META_OUT      = Path("Data") / "batting_joint_enriched_meta.json"


def clean_name(n: str) -> str:
    return re.sub(r"\s*\([^)]+\)", "", str(n)).strip()


def load_debut_year() -> pd.Series:
    """Return Series: player_name -> debut_year from combined_cricket_stats."""
    df = pd.read_csv(COMBINED_PATH)
    df["player_name"] = df["Player Name"].apply(clean_name)
    span_cols = [
        "Span Start_ODI_batting",
        "Span Start_T20_batting",
        "Span Start_Test_batting",
    ]
    for c in span_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["debut_year"] = df[span_cols].min(axis=1)
    return df.set_index("player_name")["debut_year"].dropna()


def load_scraped_metadata() -> pd.DataFrame:
    """Load batting_style and playing_role if scraping has been done."""
    if not META_PATH.exists():
        return pd.DataFrame(columns=["player_name", "batting_style", "playing_role", "born_year"])
    meta = pd.read_csv(META_PATH)
    cols = ["player_name", "batting_style", "playing_role"]
    if "born_year" in meta.columns:
        cols.append("born_year")
    return meta[cols]


def encode_batting_style(s) -> float:
    """Right-hand bat -> 0, Left-hand bat -> 1, NaN if unknown."""
    if pd.isna(s):
        return np.nan
    s = str(s).lower()
    if "left" in s:
        return 1.0
    if "right" in s:
        return 0.0
    return np.nan


def encode_playing_role(s) -> float:
    """Cricbuzz roles: Batsman/WK-Batsman->0, Batting Allrounder->1, Bowling Allrounder->2, Bowler->3."""
    if pd.isna(s):
        return np.nan
    s = str(s).lower()
    if "wk" in s or ("bats" in s and "all" not in s):
        return 0.0
    if "batting all" in s:
        return 1.0
    if "bowling all" in s:
        return 2.0
    if "bowl" in s:
        return 3.0
    return np.nan


def main() -> None:
    df = pd.read_csv(JOINT_PATH)

    # ── debut_year ────────────────────────────────────────────────────────────
    debut_map = load_debut_year()
    df["debut_year"] = df["player_name"].map(debut_map)

    n_debut = df["player_name"].nunique() - df.groupby("player_name")["debut_year"].first().isna().sum()
    print(f"debut_year populated for {n_debut}/{df['player_name'].nunique()} players "
          f"({n_debut/df['player_name'].nunique()*100:.0f}%)")

    # Mean-impute missing debut_year so model can still use full dataset
    debut_mean = df["debut_year"].mean()
    df["debut_year_scaled"] = (df["debut_year"].fillna(debut_mean) - debut_mean) / df["debut_year"].std()
    df["debut_year_missing"] = df["debut_year"].isna().astype(int)  # missingness indicator

    # ── batting_style / playing_role (from scraping when available) ───────────
    scraped = load_scraped_metadata()
    if scraped["batting_style"].notna().sum() > 0:
        df = df.merge(scraped, on="player_name", how="left")
        df["is_lefthanded"]  = df["batting_style"].apply(encode_batting_style)
        df["position_order"] = df["playing_role"].apply(encode_playing_role)
        n_hand = df["player_name"].map(
            df.drop_duplicates("player_name").set_index("player_name")["is_lefthanded"]
        ).notna().sum()
        print(f"batting_style populated for {df['is_lefthanded'].notna().sum()} rows")
        print(f"playing_role  populated for {df['position_order'].notna().sum()} rows")
    else:
        print("batting_style / playing_role: not yet available (scraping pending)")
        df["is_lefthanded"]  = np.nan
        df["position_order"] = np.nan

    # ── born_year → age at debut ──────────────────────────────────────────────
    if "born_year" in df.columns and df["born_year"].notna().sum() > 0:
        df["age_at_debut"] = df["debut_year"] - df["born_year"]
        age_mean = df["age_at_debut"].mean()
        df["age_at_debut_scaled"] = (df["age_at_debut"].fillna(age_mean) - age_mean) / df["age_at_debut"].std()
        print(f"age_at_debut populated for {df['age_at_debut'].notna().sum()} rows")
    else:
        df["age_at_debut"]        = np.nan
        df["age_at_debut_scaled"] = 0.0

    # ── Summary of X covariates ───────────────────────────────────────────────
    print("\n=== X covariates available ===")
    covariates = {
        "format_odi":          df["format_odi"].notna().sum(),
        "format_t20":          df["format_t20"].notna().sum(),
        "debut_year_scaled":   df["debut_year_scaled"].notna().sum(),
        "debut_year_missing":  int((df["debut_year_missing"] == 0).sum()),
        "is_lefthanded":       int(df["is_lefthanded"].notna().sum()),
        "position_order":      int(df["position_order"].notna().sum()),
        "age_at_debut_scaled": int((df["age_at_debut_scaled"] != 0).sum()),
    }
    total = len(df)
    for k, v in covariates.items():
        print(f"  {k:<25} {v:>5}/{total}  ({v/total*100:.0f}%)")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}  ({len(df)} rows)")

    # Save meta
    meta_out = {
        "covariates": list(covariates.keys()),
        "n_rows": len(df),
        "n_players": int(df["player_name"].nunique()),
        "debut_year_coverage_pct": round(n_debut / df["player_name"].nunique() * 100, 1),
        "batting_style_available": bool(df["is_lefthanded"].notna().sum() > 0),
        "playing_role_available":  bool(df["position_order"].notna().sum() > 0),
    }
    META_OUT.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print(f"Saved: {META_OUT}")


if __name__ == "__main__":
    main()
