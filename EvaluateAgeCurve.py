"""Held-out comparison of age-curve specifications for the enriched model."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

DATA_DIR = Path("Data/new/processed")
OUT_DIR = Path("outputs") / "mixed_effects"

FAMILIES = {
    "batting": {"y1": "batting average", "y2": "batting strike rate"},
    "bowling": {"y1": "bowling economy", "y2": "bowling average"},
}
AGE_TERMS = ("age_scaled", "age_hinge")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def natural_cubic_columns(ages: np.ndarray, df: int = 4) -> pd.DataFrame:
    """Natural cubic spline basis, linear beyond the boundary knots."""
    basis = dmatrix(f"cr(x, df={df}) - 1", {"x": ages}, return_type="dataframe")
    basis.columns = [f"ncs{i+1}" for i in range(basis.shape[1])]
    return basis.reset_index(drop=True)

def build_specs(data: pd.DataFrame, base: list[str]) -> dict[str, list[str]]:
    """Predictor lists for each competing specification, all sharing `base`."""
    ncs = natural_cubic_columns(data["avg_age"].to_numpy())
    for col in ncs.columns:
        data[col] = ncs[col].to_numpy()
    return {
        "no age": base,
        "linear age": [*base, "age_scaled"],
        "linear + hinge": [*base, "age_scaled", "age_hinge"],
        "natural cubic (df=4)": [*base, *ncs.columns],
    }

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot

def cross_validate(data: pd.DataFrame, predictors: list[str], outcome: str,
                   folds: int) -> tuple[float, float]:
    """Grouped CV. Returns (R2 on log scale, R2 on original scale)."""
    groups = data["player_id"].to_numpy()
    splitter = GroupKFold(n_splits=folds)
    log_true, log_pred = [], []

    for train_idx, test_idx in splitter.split(data, groups=groups):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        formula = f"{outcome} ~ " + " + ".join(predictors)
        fit = smf.mixedlm(formula, data=train, groups=train["player_id"]).fit(reml=False)

        # A held-out player has no known ability, so psi = 0 and only fixed effects predict.
        pred = fit.predict(exog=test)
        log_true.append(test[outcome].to_numpy())
        log_pred.append(np.asarray(pred, dtype=float))

    lt = np.concatenate(log_true)
    lp = np.concatenate(log_pred)
    return r2(lt, lp), r2(np.exp(lt), np.exp(lp))

def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for kind, outcomes in FAMILIES.items():
        meta = json.loads((DATA_DIR / f"{kind}_joint_enriched_meta.json").read_text())
        preds = meta["predictors"]
        missing = [t for t in AGE_TERMS if t not in preds]
        if missing:
            raise SystemExit(f"{kind}: {missing} absent from the current predictor list -- "
                             "rebuild with PrepareData.py first")

        data = pd.read_csv(DATA_DIR / f"{kind}_joint_enriched.csv")
        base = [p for p in preds if p not in AGE_TERMS]
        # Drop any base predictor that is constant, which breaks a fold's design.
        base = [c for c in base if data[c].nunique() > 1]
        specs = build_specs(data, base)

        print(f"\n═══ {kind.upper()}  ({len(data):,} rows, "
              f"{data['player_id'].nunique():,} players, {args.folds}-fold grouped CV) ═══")
        for outcome, label in outcomes.items():
            print(f"\n  {label}  (held-out R2, psi = 0)")
            print(f"    {'specification':24} {'log scale':>10} {'original':>10} {'vs no age':>11}")
            baseline = None
            for name, predictors in specs.items():
                r2_log, r2_orig = cross_validate(data, predictors, outcome, args.folds)
                if baseline is None:
                    baseline = r2_orig
                delta = r2_orig - baseline
                flag = "" if name == "no age" else f"{delta:+.4f}"
                print(f"    {name:24} {r2_log:>10.4f} {r2_orig:>10.4f} {flag:>11}")
                rows.append({"model": kind, "outcome": label, "specification": name,
                             "r2_log_scale": round(r2_log, 4),
                             "r2_original_scale": round(r2_orig, 4),
                             "delta_vs_no_age": round(delta, 4),
                             "folds": args.folds})

    out = OUT_DIR / "age_curve_cv_comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
