"""Leakage-free K-fold evaluation of the k-NN latent transfer.

A held-out prediction uses nothing from the player's target-format record: debut
year, age at debut and career-midpoint age all come from the known format.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from TransferMethods import run_lopo_methods, summarize_methods

DATA_DIR = Path("Data/new/processed")
OUT_DIR = Path("outputs") / "mixed_effects"
FOLD_DIR = Path("outputs") / "kfold"
FORMAT_DUMMIES = {"Test": (0, 0), "ODI": (1, 0), "T20": (0, 1)}
K_CHECK = (5, 20, 30)

FAMILIES = {
    "batting": {"script": "FitBattingEnriched.py",
                "labels": ("log_avg", "log_sr"), "raw": ("avg", "sr"),
                "psi": ("psi_log_avg", "psi_log_sr"),
                "fit_args": ["--chains", "4", "--cores", "1",
                             "--tune", "1500", "--draws", "1000",
                             "--target_accept", "0.95"]},
    "bowling": {"script": "FitBowlingEnriched.py",
                "labels": ("log_econ", "log_avg"), "raw": ("econ", "avg"),
                "psi": ("psi_log_econ", "psi_log_avg"),
                "fit_args": ["--chains", "4", "--cores", "1",
                             "--tune", "2000", "--draws", "1000",
                             "--target_accept", "0.95"]},
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-k", type=int, default=10, help="neighbours in the transfer")
    p.add_argument("--skip-fits", action="store_true",
                   help="reuse fold fits already on disk")
    p.add_argument("--plot", action="store_true",
                   help="also write the predicted-vs-actual transition figure "
                        "from the pooled fold predictions, so the figure and the "
                        "table describe the same evaluation")
    p.add_argument("--k-check", action="store_true",
                   help=f"also score the k-NN transfer at k = {', '.join(map(str, K_CHECK))}")
    return p.parse_args()

def make_folds(player_ids: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.sort(player_ids))
    return [np.sort(f) for f in np.array_split(shuffled, k)]

def fold_fixed_effects(kind: str, joint: pd.DataFrame, suffix: str,
                       labels: tuple[str, str]) -> pd.DataFrame:
    """alpha + X.beta per row, from this fold's fitted parameters.

    fe{n}_row uses each row's own covariates and scores the training pool.
    fe{n}_as_<format> is what a prediction from this row may use: debut year and
    age at debut from this format's debut, and the format indicators of <format>.
    """
    meta = json.loads((DATA_DIR / f"{kind}_joint_enriched_meta.json").read_text())
    preds = meta["predictors"]
    params = pd.read_csv(OUT_DIR / f"joint_{kind}_enriched_params_mean{suffix}.csv").iloc[0]
    X = joint[preds].to_numpy(dtype=float)
    known = X.copy()
    # PrepareData's standardisation is linear in the raw value, so refit it to rescale.
    for col, raw in (("debut_year_scaled", joint["debut_year_raw"]),
                     ("age_at_debut_scaled", joint["debut_year_raw"] - joint["born_year"])):
        slope, intercept = np.polyfit(joint[col.removesuffix("_scaled")], joint[col], 1)
        known[:, preds.index(col)] = intercept + slope * raw.to_numpy(dtype=float)
    i_odi, i_t20 = preds.index("format_odi"), preds.index("format_t20")
    joint = joint.copy()
    for n, lab in ((1, labels[0]), (2, labels[1])):
        alpha = float(params[f"alpha_{lab}"])
        beta = np.array([float(params[f"beta_{p}_{lab}"]) for p in preds])
        joint[f"fe{n}_row"] = alpha + X @ beta
        for fmt, (odi, t20) in FORMAT_DUMMIES.items():
            as_fmt = known.copy()
            as_fmt[:, i_odi], as_fmt[:, i_t20] = odi, t20
            joint[f"fe{n}_as_{fmt}"] = alpha + as_fmt @ beta
    return joint

def fold_own_record(kind: str, joint: pd.DataFrame, suffix: str,
                    labels: tuple[str, str]) -> pd.DataFrame:
    """Each row's residual, shrunk by how much cricket stands behind it."""
    meta = json.loads((DATA_DIR / f"{kind}_joint_enriched_meta.json").read_text())
    params = pd.read_csv(OUT_DIR / f"joint_{kind}_enriched_params_mean{suffix}.csv").iloc[0]
    joint = joint.copy()
    for n, lab in ((1, labels[0]), (2, labels[1])):
        w = joint[meta["exposure"]["columns"][f"y{n}"]]
        ability = params[f"chol_u_stds_{lab}"] ** 2
        luck = params[f"sigma_sampling_{lab}"] ** 2 / w + params[f"sigma_real_{lab}"] ** 2
        fe_own = np.select([joint["format"] == f for f in FORMAT_DUMMIES],
                           [joint[f"fe{n}_as_{f}"] for f in FORMAT_DUMMIES])
        joint[f"own{n}_row"] = ability / (ability + luck) * (joint[f"y{n}"] - fe_own)
    return joint

def main() -> None:
    args = parse_args()
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    fold_inputs: dict[str, list] = {}
    by_fold: list[dict] = []

    for kind, cfg in FAMILIES.items():
        data = pd.read_csv(DATA_DIR / f"{kind}_joint_enriched.csv")
        for col in cfg["raw"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=list(cfg["raw"])).reset_index(drop=True)

        folds = make_folds(data["player_id"].unique(), args.folds, args.seed)
        print(f"\n{'='*70}\n{kind.upper()}: {args.folds} folds over "
              f"{data['player_id'].nunique():,} players\n{'='*70}")

        per_fold, fold_inputs[kind] = [], []
        for i, held in enumerate(folds):
            suffix = f"_fold{i}"
            exclude = FOLD_DIR / f"{kind}_fold{i}_holdout.txt"
            exclude.write_text("\n".join(map(str, held)))

            if not args.skip_fits:
                print(f"\n-- fold {i}: refitting without {len(held)} players")
                cmd = [sys.executable, cfg["script"],
                       "--exclude-players", str(exclude),
                       "--out-suffix", suffix, *cfg["fit_args"]]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    print(r.stdout[-2000:]); print(r.stderr[-2000:])
                    raise SystemExit(f"{kind} fold {i} fit failed")

            psi = pd.read_csv(OUT_DIR / f"{kind}_enriched_player_rankings{suffix}.csv")
            train_ids = set(psi["player_id"])
            if train_ids & set(held):
                raise SystemExit(f"{kind} fold {i}: held-out players present in the fit")

            # Query rows: the held-out players. Neighbour pool: training players.
            scored = fold_fixed_effects(kind, data, suffix, cfg["labels"])
            scored = fold_own_record(kind, scored, suffix, cfg["labels"])
            pool = scored[scored["player_id"].isin(train_ids)]
            queries = scored[scored["player_id"].isin(set(held))]

            fold_inputs[kind].append((pd.concat([queries, pool], ignore_index=True),
                                      psi, set(held), train_ids))
            res = score_fold(kind, cfg, *fold_inputs[kind][-1], k=args.k, label=f"fold {i}")
            per_fold.append(res)
            by_fold += [{"model": kind, "fold": i, **row}
                        for row in summarize_methods(res, *cfg["raw"])]
            print(f"   fold {i}: {len(res)} held-out transition pairs")

        results[kind] = pd.concat(per_fold, ignore_index=True)

    bat, bowl = results["batting"], results["bowling"]
    rows = summarize_methods(bat, "avg", "sr") + summarize_methods(bowl, "econ", "bowl_avg")
    out = pd.DataFrame(rows)
    out.insert(0, "evaluation", f"{args.folds}-fold leakage-free")
    print("\n" + "=" * 72)
    print(f"LEAKAGE-FREE {args.folds}-FOLD LOPO (k={args.k})")
    print("=" * 72)
    print(out.to_string(index=False))
    path = OUT_DIR / "lopo_kfold_evaluation.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved: {path}")

    for kind, res in results.items():
        pred_path = OUT_DIR / f"lopo_kfold_predictions_{kind}.csv"
        res.to_csv(pred_path, index=False)
        print(f"Saved: {pred_path}")
    fold_path = OUT_DIR / "lopo_kfold_by_fold.csv"
    pd.DataFrame(by_fold).to_csv(fold_path, index=False)
    print(f"Saved: {fold_path}")

    if args.k_check:
        check_k(fold_inputs, args.k)

    if args.plot:
        from PaperFigures import transitions
        transitions(bat, bowl, Path("outputs") / "plots" / "cross_format_transition_kfold.png")

def score_fold(kind: str, cfg: dict, frame: pd.DataFrame, psi: pd.DataFrame,
               held: set, train_ids: set, k: int, label: str) -> pd.DataFrame:
    return run_lopo_methods(f"{kind} {label}", frame, psi,
                            psi1=cfg["psi"][0], psi2=cfg["psi"][1],
                            raw1=cfg["raw"][0], raw2=cfg["raw"][1], k=k,
                            restrict_to=held, pool_ids=train_ids)

def check_k(fold_inputs: dict[str, list], k_main: int) -> None:
    """Held-out R^2 of the k-NN transfer for other neighbour counts."""
    rows = []
    for k in sorted({k_main, *K_CHECK}):
        row = {"k": k}
        for kind, cfg in FAMILIES.items():
            res = pd.concat([score_fold(kind, cfg, *inputs, k=k, label=f"k={k}")
                             for inputs in fold_inputs[kind]], ignore_index=True)
            for n, name in ((1, cfg["raw"][0]), (2, cfg["raw"][1])):
                row[f"{kind}_{name}"] = round(float(r2_score(res[f"true{n}"], res[f"knn{n}"])), 4)
        rows.append(row)
    out = pd.DataFrame(rows)
    print("\n=== k-NN transfer R^2 by k ===")
    print(out.to_string(index=False))
    path = OUT_DIR / "knn_k_sensitivity.csv"
    out.to_csv(path, index=False)
    print(f"Saved: {path}")

if __name__ == "__main__":
    main()
