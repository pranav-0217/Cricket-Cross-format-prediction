"""Leakage-free K-fold evaluation of the k-NN latent transfer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from TransferMethods import run_lopo_methods, summarize_methods

DATA_DIR = Path("Data/new/processed")
OUT_DIR = Path("outputs") / "mixed_effects"
FOLD_DIR = Path("outputs") / "kfold"

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
    return p.parse_args()

def make_folds(player_ids: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.sort(player_ids))
    return [np.sort(f) for f in np.array_split(shuffled, k)]

def fold_fixed_effects(kind: str, joint: pd.DataFrame, suffix: str,
                       labels: tuple[str, str]) -> pd.DataFrame:
    """alpha + X.beta per row, from this fold's fitted parameters."""
    meta = json.loads((DATA_DIR / f"{kind}_joint_enriched_meta.json").read_text())
    preds = meta["predictors"]
    params = pd.read_csv(OUT_DIR / f"joint_{kind}_enriched_params_mean{suffix}.csv").iloc[0]
    X = joint[preds].to_numpy(dtype=float)
    joint = joint.copy()
    for tag, lab in (("fe1_row", labels[0]), ("fe2_row", labels[1])):
        beta = np.array([float(params[f"beta_{p}_{lab}"]) for p in preds])
        joint[tag] = float(params[f"alpha_{lab}"]) + X @ beta
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
        joint[f"own{n}_row"] = ability / (ability + luck) * (joint[f"y{n}"] - joint[f"fe{n}_row"])
    return joint

def main() -> None:
    args = parse_args()
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}

    for kind, cfg in FAMILIES.items():
        data = pd.read_csv(DATA_DIR / f"{kind}_joint_enriched.csv")
        for col in cfg["raw"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=list(cfg["raw"])).reset_index(drop=True)

        folds = make_folds(data["player_id"].unique(), args.folds, args.seed)
        print(f"\n{'='*70}\n{kind.upper()}: {args.folds} folds over "
              f"{data['player_id'].nunique():,} players\n{'='*70}")

        per_fold = []
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

            res = run_lopo_methods(
                f"{kind} fold {i}",
                pd.concat([queries, pool], ignore_index=True),
                psi, psi1=cfg["psi"][0], psi2=cfg["psi"][1],
                raw1=cfg["raw"][0], raw2=cfg["raw"][1], k=args.k,
                restrict_to=set(held), pool_ids=train_ids,
            )
            per_fold.append(res)
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

    if args.plot:
        plot_transitions(bat, bowl)

def plot_transitions(bat: pd.DataFrame, bowl: pd.DataFrame) -> None:
    """Predicted vs actual by target format, from the pooled fold predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = Path("outputs") / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colours = {"Test": "#1976D2", "ODI": "#388E3C", "T20": "#F57C00"}
    fig, axes = plt.subplots(4, 3, figsize=(16, 19))

    print("\n=== Pearson r, leakage-free folds ===")
    for col, fmt in enumerate(["Test", "ODI", "T20"]):
        b = bat[bat["target_format"] == fmt]
        w = bowl[bowl["target_format"] == fmt]
        panels = [
            (axes[0, col], b["true1"], b["knn1"],
             f"Batting Avg -- {fmt}", "Actual avg", "Predicted avg"),
            (axes[1, col], b["true2"], b["knn2"],
             f"Batting SR -- {fmt}", "Actual SR", "Predicted SR"),
            (axes[2, col], w["true1"], w["knn1"],
             f"Bowling Economy -- {fmt}", "Actual economy", "Predicted economy"),
            (axes[3, col], w["true2"], w["knn2"],
             f"Bowling Avg -- {fmt}", "Actual avg", "Predicted avg"),
        ]
        print(f"  {fmt:5s} bat avg r={b['true1'].corr(b['knn1']):.3f}  "
              f"bat sr r={b['true2'].corr(b['knn2']):.3f}  "
              f"bowl econ r={w['true1'].corr(w['knn1']):.3f}  "
              f"bowl avg r={w['true2'].corr(w['knn2']):.3f}  n={len(b)}/{len(w)}")
        for ax, actual, pred, title, xlabel, ylabel in panels:
            ax.scatter(actual, pred, alpha=0.5, s=18, color=colours[fmt], edgecolors="none")
            lo = min(actual.min(), pred.min()) * 0.95
            hi = max(actual.max(), pred.max()) * 1.05
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
            ax.set_title(f"{title}\nr = {actual.corr(pred):.3f}", fontsize=10, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = plot_dir / "cross_format_transition_kfold.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
