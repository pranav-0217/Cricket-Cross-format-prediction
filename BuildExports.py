"""Build every ranking table and figure used in the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path("Data/new/processed")
RAW_DIR = Path("Data/new/raw")
OUT_DIR = Path("outputs")
ME_DIR = OUT_DIR / "mixed_effects"
LOG_DIR = OUT_DIR / "log_checks"

MIN_RUNS = 1000
MIN_WICKETS = 50
FORMATS = ["Test", "ODI", "T20"]
FMT_COLOURS = {"Test": "#1976D2", "ODI": "#388E3C", "T20": "#F57C00"}

def raw_career_totals(kind: str, columns: list[str]) -> pd.DataFrame:
    """Sum the requested raw career columns across the three formats, by player_id."""
    frames = []
    for fmt in FORMATS:
        r = pd.read_csv(RAW_DIR / f"{fmt}_{kind}.csv")
        pid = r["Player URL"].astype(str).str.extract(r"(\d+)")[0]
        out = pd.DataFrame({"player_id": pd.to_numeric(pid, errors="coerce")})
        for c in columns:
            out[c] = pd.to_numeric(r[c], errors="coerce")
        frames.append(out.dropna(subset=["player_id"]))
    tot = pd.concat(frames)
    tot["player_id"] = tot["player_id"].astype("int64")
    return tot.groupby("player_id", as_index=False)[columns].sum()

def write_rankings() -> None:
    bat = pd.read_csv(ME_DIR / "batting_player_rankings.csv")
    bowl = pd.read_csv(ME_DIR / "bowling_player_rankings.csv")

    runs = raw_career_totals("Batting", ["Runs"]).rename(columns={"Runs": "career_runs"})
    wkts = raw_career_totals("Bowling", ["Wkts"]).rename(columns={"Wkts": "career_wickets"})

    bat = bat.merge(runs, on="player_id", how="left")
    bowl = bowl.merge(wkts, on="player_id", how="left")

    combined = (bat[["player_id", "player_name", "psi_combined", "career_runs"]]
                .rename(columns={"psi_combined": "composite_score"})
                .sort_values("composite_score", ascending=False)
                .reset_index(drop=True))
    combined.insert(0, "rank", np.arange(1, len(combined) + 1))
    combined["composite_score"] = combined["composite_score"].round(4)
    combined.to_csv(OUT_DIR / "all_players_ranking.csv", index=False)
    print(f"  all_players_ranking.csv          {len(combined):>5} players")

    def top(df: pd.DataFrame, n: int | None, path: Path, note: str,
            threshold_col: str | None = None, threshold: int | None = None) -> None:
        d = df.copy()
        if threshold_col is not None:
            d = d[d[threshold_col].fillna(0) >= threshold]
        d = d.sort_values("psi_combined", ascending=False).reset_index(drop=True)
        if n is not None:
            d = d.head(n)
        d = d.drop(columns=[c for c in ("rank",) if c in d.columns])
        d.insert(0, "rank", np.arange(1, len(d) + 1))
        d.to_csv(path, index=False)
        print(f"  {path.name:32} {len(d):>5} {note}")

    top(bat, 100, OUT_DIR / "top100_batting_rankings.csv", "batters")
    top(bowl, 100, OUT_DIR / "top100_bowling_rankings.csv", "bowlers")
    top(bat, None, OUT_DIR / "top_batting_rankings_1000runs.csv",
        f"batters with >={MIN_RUNS} runs", "career_runs", MIN_RUNS)
    top(bowl, None, OUT_DIR / "top_bowling_rankings_50wkts.csv",
        f"bowlers with >={MIN_WICKETS} wkts", "career_wickets", MIN_WICKETS)

def scatter_by_format(df: pd.DataFrame, xcol: str, ycol: str, xlabel: str,
                      ylabel: str, title: str, path: Path,
                      invert: bool = False) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for ax, fmt in zip(axes, FORMATS):
        s = df[df["format"] == fmt]
        ax.scatter(s[xcol], s[ycol], s=16, alpha=0.55,
                   color=FMT_COLOURS[fmt], edgecolors="none")
        ax.set_title(f"{fmt}  (n={len(s):,})", fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(ylabel, fontsize=9)
    if invert:
        axes[0].invert_xaxis()
        axes[0].invert_yaxis()
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")

def write_scatters() -> None:
    bat = pd.read_csv(DATA_DIR / "batting_long.csv")
    bowl = pd.read_csv(DATA_DIR / "bowling_long.csv")

    bowl_sr = raw_career_totals_sr()
    bowl = bowl.merge(bowl_sr, on=["player_id", "format"], how="left")

    scatter_by_format(bat, "avg", "sr", "Batting average", "Strike rate",
                      "Batting strike rate vs average, by format",
                      OUT_DIR / "sr_vs_avg_by_format.png")
    scatter_by_format(bowl, "avg", "econ", "Bowling average", "Economy rate",
                      "Bowling economy vs average, by format",
                      OUT_DIR / "bowling_econ_vs_avg_by_format.png")
    sub = bowl.dropna(subset=["bowling_sr"])
    scatter_by_format(sub, "bowling_sr", "econ", "Bowling strike rate (balls/wicket)",
                      "Economy rate", "Bowling economy vs strike rate, by format",
                      OUT_DIR / "bowling_sr_vs_econ_by_format.png")

def raw_career_totals_sr() -> pd.DataFrame:
    """Bowling strike rate is only in the raw files, and is per player-format."""
    frames = []
    for fmt in FORMATS:
        r = pd.read_csv(RAW_DIR / f"{fmt}_Bowling.csv")
        pid = r["Player URL"].astype(str).str.extract(r"(\d+)")[0]
        frames.append(pd.DataFrame({
            "player_id": pd.to_numeric(pid, errors="coerce"),
            "format": fmt,
            "bowling_sr": pd.to_numeric(r["SR"], errors="coerce"),
        }).dropna(subset=["player_id"]))
    out = pd.concat(frames)
    out["player_id"] = out["player_id"].astype("int64")
    return out

# The four modelled outcomes, checked raw and on the log scale.
LOG_METRICS = [
    ("batting_avg",  "batting_long.csv",  "avg",  "Batting average"),
    ("batting_sr",   "batting_long.csv",  "sr",   "Batting strike rate"),
    ("bowling_econ", "bowling_long.csv",  "econ", "Bowling economy rate"),
    ("bowling_avg",  "bowling_long.csv",  "avg",  "Bowling average"),
]

def write_log_checks() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for metric, source, col, label in LOG_METRICS:
        s = pd.to_numeric(pd.read_csv(DATA_DIR / source)[col], errors="coerce").dropna()
        s = s[s > 0]
        for scale, values, nice in (("raw", s, label), ("log", np.log(s), f"log({label})")):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(values, bins=50, color="#1976D2", alpha=0.8, edgecolor="none")
            ax.set_title(f"{nice} — histogram (n={len(values):,})", fontsize=10, fontweight="bold")
            ax.set_xlabel(nice, fontsize=9); ax.set_ylabel("count", fontsize=9)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(LOG_DIR / f"{metric}_{scale}_hist.png", dpi=130, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 4))
            stats.probplot(values, dist="norm", plot=ax)
            ax.set_title(f"{nice} — normal Q-Q", fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(LOG_DIR / f"{metric}_{scale}_qq.png", dpi=130, bbox_inches="tight")
            plt.close(fig)

            # Shapiro-Wilk is capped at 5000 observations; sample for stability.
            sample = values.sample(5000, random_state=0) if len(values) > 5000 else values
            rows.append({
                "metric": metric, "scale": scale, "n": len(values),
                "skew": round(float(stats.skew(values)), 4),
                "kurtosis": round(float(stats.kurtosis(values)), 4),
                "shapiro_W": round(float(stats.shapiro(sample).statistic), 4),
                "shapiro_p": float(f"{stats.shapiro(sample).pvalue:.3g}"),
            })
        print(f"  log_checks/{metric}_*  (raw + log, hist + qq)")

    summary = pd.DataFrame(rows)
    summary.to_csv(LOG_DIR / "log_normality_summary.csv", index=False)
    print("\n  Normality before vs after the log transform:")
    piv = summary.pivot(index="metric", columns="scale", values=["skew", "shapiro_W"])
    print(piv.to_string())

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Rankings (from the format-only player effects):")
    write_rankings()
    print("\nFigures:")
    write_scatters()
    print("\nDistribution checks:")
    write_log_checks()

if __name__ == "__main__":
    main()
