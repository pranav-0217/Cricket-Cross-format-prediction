"""Held-out test of precision weighting by exposure, for all four outcomes."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

DATA_DIR = Path("Data/new/processed")
OUT_DIR = Path("outputs") / "mixed_effects"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folds", type=int, default=5)
    return p.parse_args()

LABELS = {
    "batting": {"y1": "batting average", "y2": "batting strike rate"},
    "bowling": {"y1": "bowling economy", "y2": "bowling average"},
}

def load_family(kind: str) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Data, predictors and the exposure column per outcome."""
    meta = json.loads((DATA_DIR / f"{kind}_joint_enriched_meta.json").read_text())
    preds = list(meta["predictors"])
    data = pd.read_csv(DATA_DIR / f"{kind}_joint_enriched.csv")

    spec = meta.get("exposure")
    if spec is None:
        raise SystemExit(f"{kind}: metadata has no 'exposure' block -- "
                         "rebuild with PrepareData.py")
    weights = dict(spec["columns"])

    for col in weights.values():
        if col not in data:
            raise SystemExit(f"{kind}: exposure column {col!r} absent from the data")
        if data[col].isna().any() or (data[col] <= 0).any():
            raise SystemExit(f"{kind}: exposure {col} must be present and positive")

    preds = [c for c in preds if data[c].nunique() > 1]
    return data, preds, {"weights": weights, "labels": LABELS[kind]}

def r2(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray | None = None) -> float:
    if w is None:
        w = np.ones_like(y_true)
    mean = np.average(y_true, weights=w)
    ss_res = float(np.sum(w * (y_true - y_pred) ** 2))
    ss_tot = float(np.sum(w * (y_true - mean) ** 2))
    return 1.0 - ss_res / ss_tot

def cv_fit(data: pd.DataFrame, preds: list[str], outcome: str, wcol: str,
           folds: int, weighted: bool) -> tuple[float, float]:
    groups = data["player_id"].to_numpy()
    y_all, p_all, w_all = [], [], []

    for tr, te in GroupKFold(n_splits=folds).split(data, groups=groups):
        train, test = data.iloc[tr], data.iloc[te]
        Xtr = sm.add_constant(train[preds].to_numpy(dtype=float), has_constant="add")
        ytr = train[outcome].to_numpy(dtype=float)

        if weighted:
            s = np.sqrt(train[wcol].to_numpy(dtype=float))
            fit = sm.MixedLM(ytr * s, Xtr * s[:, None], groups=train["player_id"].to_numpy(),
                             exog_re=s[:, None]).fit(reml=False)
        else:
            fit = sm.MixedLM(ytr, Xtr, groups=train["player_id"].to_numpy(),
                             exog_re=np.ones((len(train), 1))).fit(reml=False)

        beta = np.asarray(fit.fe_params, dtype=float)
        Xte = sm.add_constant(test[preds].to_numpy(dtype=float), has_constant="add")
        y_all.append(test[outcome].to_numpy(dtype=float))
        p_all.append(Xte @ beta)
        w_all.append(test[wcol].to_numpy(dtype=float))

    y, p, w = np.concatenate(y_all), np.concatenate(p_all), np.concatenate(w_all)
    # Back-transform: the outcomes are on the log scale.
    return r2(np.exp(y), np.exp(p)), r2(np.exp(y), np.exp(p), w)

def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for kind in ("batting", "bowling"):
        data, preds, spec = load_family(kind)
        print(f"\n═══ {kind.upper()}  ({len(data):,} rows, "
              f"{data['player_id'].nunique():,} players, {args.folds}-fold grouped CV) ═══")
        for outcome, label in spec["labels"].items():
            wcol = spec["weights"][outcome]
            lo, hi = data[wcol].min(), data[wcol].max()
            print(f"\n  {label}   exposure = {wcol} (range {lo:.0f}-{hi:.0f})")
            print(f"    {'model':14} {'R2 unweighted':>15} {'R2 weighted':>13}")
            res = {}
            for weighted in (False, True):
                ru, rw = cv_fit(data, preds, outcome, wcol, args.folds, weighted)
                name = "weighted" if weighted else "unweighted"
                res[name] = (ru, rw)
                print(f"    {name:14} {ru:>15.4f} {rw:>13.4f}")
                rows.append({"model": kind, "outcome": label, "fit": name,
                             "exposure": wcol,
                             "r2_unweighted": round(ru, 4), "r2_weighted": round(rw, 4)})
            du = res["weighted"][0] - res["unweighted"][0]
            dw = res["weighted"][1] - res["unweighted"][1]
            print(f"    {'change':14} {du:>+15.4f} {dw:>+13.4f}")

    out = OUT_DIR / "exposure_weighting_cv_comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
