"""Build the format-only fixed effects and player random effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA_DIR = Path("Data/new/processed")
OUTPUT_DIR = Path("outputs") / "mixed_effects"

# kind -> (joint csv, meta json, trace, outcome column labels for y1 and y2)
FAMILIES = {
    "batting": ("batting_joint.csv", "batting_joint_enriched_meta.json",
                "joint_batting_trace.nc", ("log_avg", "log_sr")),
    "bowling": ("bowling_joint.csv", "bowling_joint_enriched_meta.json",
                "joint_bowling_trace.nc", ("log_econ", "log_avg")),
}

FE_ROWS = ["Intercept", "format_odi", "format_t20"]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", choices=["statsmodels", "pymc", "both"], default="both")
    p.add_argument("--canonical", choices=["statsmodels", "pymc"], default="pymc",
                   help="engine that writes the unsuffixed filenames the evaluators read. "
                        "Defaults to pymc so the evaluation layer uses the same "
                        "multivariate fit the paper describes; falls back to "
                        "statsmodels automatically if no current trace exists.")
    return p.parse_args()

def combined_score(kind: str, psi1: np.ndarray, psi2: np.ndarray) -> np.ndarray:
    """Single ranking score, higher = better, matching the fit scripts."""
    if kind == "batting":
        # psi1 = log(avg) effect, psi2 = log(sr) effect; both higher = better
        return (psi1 + psi2) / 2.0
    # bowling: psi1 = log(econ), psi2 = log(avg); both lower = better, so negate
    return -(2.0 * psi2 + 1.0 * psi1) / 3.0

def psi_frame(kind: str, player_ids, player_names, psi1, psi2) -> pd.DataFrame:
    lab1, lab2 = FAMILIES[kind][3]
    df = pd.DataFrame({
        "player_id": np.asarray(player_ids, dtype=np.int64),
        "player_name": player_names,
        f"psi_{lab1}": psi1,
        f"psi_{lab2}": psi2,
        "psi_combined": combined_score(kind, np.asarray(psi1), np.asarray(psi2)),
        f"{lab1.replace('log_', '')}_multiplier": np.exp(psi1),
        f"{lab2.replace('log_', '')}_multiplier": np.exp(psi2),
    })
    df["rank"] = df["psi_combined"].rank(ascending=False).astype(int)
    return df.sort_values("rank").reset_index(drop=True)

def fe_frame(kind: str, alpha: np.ndarray, beta: np.ndarray) -> pd.DataFrame:
    """alpha shape (2,), beta shape (2 outcomes, 2 format predictors)."""
    lab1, lab2 = FAMILIES[kind][3]
    df = pd.DataFrame({
        "parameter": FE_ROWS,
        f"beta_{lab1}": [alpha[0], beta[0, 0], beta[0, 1]],
        f"beta_{lab2}": [alpha[1], beta[1, 0], beta[1, 1]],
    })
    df[f"exp_beta_{lab1}"] = np.exp(df[f"beta_{lab1}"])
    df[f"exp_beta_{lab2}"] = np.exp(df[f"beta_{lab2}"])
    return df

def run_statsmodels(kind: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    joint_csv, _, _, (lab1, lab2) = FAMILIES[kind]
    df = pd.read_csv(DATA_DIR / joint_csv)

    alpha, beta_rows, psi_cols, summaries = [], [], [], []
    for outcome, label in (("y1", lab1), ("y2", lab2)):
        res = smf.mixedlm(f"{outcome} ~ format_odi + format_t20",
                          data=df, groups=df["player_id"]).fit(reml=True)
        alpha.append(float(res.fe_params["Intercept"]))
        beta_rows.append([float(res.fe_params["format_odi"]),
                          float(res.fe_params["format_t20"])])
        re = {gid: float(np.asarray(v)[0]) for gid, v in res.random_effects.items()}
        psi_cols.append(re)
        summaries.append(f"=== {kind} {label}: {outcome} ~ format + (1|player_id) ===\n"
                         f"{res.summary()}\n")

    ids = sorted(psi_cols[0])
    names = (df.drop_duplicates("player_id").set_index("player_id")["player_name"]
               .reindex(ids).to_numpy())
    psi = psi_frame(kind, ids, names,
                    np.array([psi_cols[0][i] for i in ids]),
                    np.array([psi_cols[1][i] for i in ids]))
    fe = fe_frame(kind, np.array(alpha), np.array(beta_rows))
    return fe, psi, "\n".join(summaries)

def run_pymc(kind: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    import arviz as az

    _, meta_json, trace_name, _ = FAMILIES[kind]
    trace_path = OUTPUT_DIR / trace_name
    if not trace_path.exists():
        raise FileNotFoundError(
            f"{trace_path} missing -- run Fit{kind.capitalize()}.py first")

    meta_path = DATA_DIR / meta_json
    data_path = DATA_DIR / FAMILIES[kind][0]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # A matching player count is not proof the trace matches; compare timestamps.
    newest_input = max(data_path.stat().st_mtime, meta_path.stat().st_mtime)
    if trace_path.stat().st_mtime < newest_input:
        raise ValueError(
            f"{kind}: {trace_path.name} predates the current data -- refit "
            f"Fit{kind.capitalize()}.py before using the pymc engine."
        )

    idata = az.from_netcdf(str(trace_path))
    post = idata.posterior

    n_trace = post["u"].shape[2]
    n_meta = int(meta["n_players"])
    if n_trace != n_meta:
        raise ValueError(
            f"{kind}: trace has {n_trace} players but current data has {n_meta}. "
            f"The trace is stale -- refit Fit{kind.capitalize()}.py on the current data."
        )

    alpha = post["alpha"].mean(dim=("chain", "draw")).values
    beta = post["beta"].mean(dim=("chain", "draw")).values
    u = post["u"].mean(dim=("chain", "draw")).values

    idx_to_id = meta["player_idx_to_id"]
    idx_to_name = meta["player_idx_to_name"]
    ids = [int(idx_to_id[str(i)]) for i in range(n_meta)]
    names = [idx_to_name.get(str(i), f"player_{i}") for i in range(n_meta)]

    psi = psi_frame(kind, ids, names, u[:, 0], u[:, 1])
    fe = fe_frame(kind, alpha, beta)
    return fe, psi, ""

ENGINES = {"statsmodels": run_statsmodels, "pymc": run_pymc}

def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    engines = ["statsmodels", "pymc"] if args.engine == "both" else [args.engine]

    produced: dict[tuple[str, str], pd.DataFrame] = {}
    for kind in FAMILIES:
        for engine in engines:
            try:
                fe, psi, summary = ENGINES[engine](kind)
            except (FileNotFoundError, ValueError) as exc:
                print(f"  SKIP {kind}/{engine}: {exc}")
                continue

            fe.to_csv(OUTPUT_DIR / f"{kind}_fixed_effects_{engine}.csv", index=False)
            psi.to_csv(OUTPUT_DIR / f"{kind}_player_rankings_{engine}.csv", index=False)
            if summary:
                (OUTPUT_DIR / f"{kind}_format_only_mixedlm_summary.txt").write_text(
                    summary, encoding="utf-8")
            produced[(kind, engine)] = fe
            print(f"  {kind}/{engine}: {len(psi):,} players, "
                  f"fixed effects {list(fe.columns[1:3])}")

    # Canonical copies -- these are the filenames the evaluation scripts read.
    for kind in FAMILIES:
        src = args.canonical if (kind, args.canonical) in produced else None
        if src is None:
            src = next((e for e in engines if (kind, e) in produced), None)
            if src is None:
                continue
            print(f"  NOTE {kind}: --canonical {args.canonical} unavailable, using {src}")
        for stem in ("fixed_effects", "player_rankings"):
            data = pd.read_csv(OUTPUT_DIR / f"{kind}_{stem}_{src}.csv")
            data.to_csv(OUTPUT_DIR / f"{kind}_{stem}.csv", index=False)
        print(f"  canonical {kind}_*.csv <- {src}")

    # Engine comparison, when both ran for a family.
    rows = []
    for kind in FAMILIES:
        if (kind, "statsmodels") not in produced or (kind, "pymc") not in produced:
            continue
        sm, pm = produced[(kind, "statsmodels")], produced[(kind, "pymc")]
        for col in [c for c in sm.columns if c.startswith("beta_")]:
            for i, param in enumerate(FE_ROWS):
                a, b = float(sm[col].iloc[i]), float(pm[col].iloc[i])
                rows.append({"model": kind, "parameter": param, "coefficient": col,
                             "statsmodels": round(a, 6), "pymc": round(b, 6),
                             "abs_diff": round(abs(a - b), 6)})
    if rows:
        comp = pd.DataFrame(rows)
        path = OUTPUT_DIR / "format_only_engine_comparison.csv"
        comp.to_csv(path, index=False)
        print(f"\nEngine comparison -> {path}")
        print(f"  max |statsmodels - pymc| = {comp['abs_diff'].max():.6f}")

if __name__ == "__main__":
    main()
