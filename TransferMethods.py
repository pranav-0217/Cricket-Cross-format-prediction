"""Leave-one-player-out evaluation of the format-aware k-NN psi transfer."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

OUTPUT_DIR = Path("outputs") / "mixed_effects"

K_DEFAULT = 10

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred):
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

DATA_DIR = Path("Data/new/processed")

def fitted_fixed_effects(kind, joint, spec, labels):
    """Per-row fixed-effect fitted values for both outcomes."""
    joint = joint.copy()
    if spec == "format-only":
        fe = pd.read_csv(OUTPUT_DIR / f"{kind}_fixed_effects.csv").set_index("parameter")
        col1, col2 = f"beta_{labels[0]}", f"beta_{labels[1]}"
        for tag, col in (("fe1_row", col1), ("fe2_row", col2)):
            joint[tag] = joint["format"].map(
                {f: fe_value(fe, f, col) for f in ["Test", "ODI", "T20"]}
            )
        return joint

    meta = json.loads((DATA_DIR / f"{kind}_joint_enriched_meta.json").read_text())
    preds = meta["predictors"]
    params = pd.read_csv(OUTPUT_DIR / f"joint_{kind}_enriched_params_mean.csv").iloc[0]
    missing = [c for c in preds if c not in joint.columns]
    if missing:
        raise SystemExit(
            f"{kind}: enriched spec needs predictor columns absent from the data: {missing}. "
            "Pass the enriched joint file."
        )
    X = joint[preds].to_numpy(dtype=float)
    for tag, lab in (("fe1_row", labels[0]), ("fe2_row", labels[1])):
        beta = np.array([float(params[f"beta_{p}_{lab}"]) for p in preds])
        joint[tag] = float(params[f"alpha_{lab}"]) + X @ beta
    return joint

def fe_value(fe, fmt, col):
    alpha = float(fe.loc["Intercept", col])
    if fmt == "Test":
        return alpha
    if fmt == "ODI":
        return alpha + float(fe.loc["format_odi", col])
    if fmt == "T20":
        return alpha + float(fe.loc["format_t20", col])
    raise ValueError(fmt)

def run_lopo_methods(name, joint, psi, psi1, psi2, raw1, raw2, k=K_DEFAULT,
                     restrict_to=None, pool_ids=None):
    """Compute per-pair LOPO predictions for every method."""
    # Left join: held-out players have no psi of their own, only neighbours' psi is read.
    joint = joint.merge(psi[["player_id", psi1, psi2]], on="player_id", how="left")
    formats = ["Test", "ODI", "T20"]
    if "fe1_row" not in joint or "fe2_row" not in joint:
        raise SystemExit("joint must carry fe1_row/fe2_row from fitted_fixed_effects()")

    by_format = {f: joint[joint["format"] == f].reset_index(drop=True) for f in formats}
    # player_id -> (y1, y2) lookup per format, for the direct k-NN baseline
    by_format_log = {
        f: by_format[f].drop_duplicates("player_id").set_index("player_id")[["y1", "y2"]]
        for f in formats
    }

    # Neighbours come only from the training pool.
    if pool_ids is not None:
        by_format = {f: df[df["player_id"].isin(pool_ids)].reset_index(drop=True)
                     for f, df in by_format.items()}
        by_format_log = {
            f: by_format[f].drop_duplicates("player_id").set_index("player_id")[["y1", "y2"]]
            for f in formats
        }

    players = joint.groupby("player_id")
    records = []
    n_dknn_empty = 0

    n_players = joint["player_id"].nunique()
    done = 0
    for player_id, rows in players:
        done += 1
        if done % 200 == 0:
            print(f"  {name}: {done}/{n_players} players...")

        if restrict_to is not None and player_id not in restrict_to:
            continue
        player_formats = rows["format"].unique().tolist()
        if len(player_formats) < 2:
            continue

        for known_fmt in player_formats:
            known_row = rows[rows["format"] == known_fmt].iloc[0]
            q_res1 = known_row["y1"] - known_row["fe1_row"]
            q_res2 = known_row["y2"] - known_row["fe2_row"]

            cand = by_format[known_fmt]
            cand = cand[cand["player_id"] != player_id]
            if len(cand) == 0:
                continue

            cand_res1 = cand["y1"].to_numpy() - cand["fe1_row"].to_numpy()
            cand_res2 = cand["y2"].to_numpy() - cand["fe2_row"].to_numpy()
            dists = np.sqrt((cand_res1 - q_res1) ** 2 + (cand_res2 - q_res2) ** 2)

            # The direct k-NN baseline is model-free, so it matches on raw log statistics.
            dists_raw = np.sqrt(
                (cand["y1"].to_numpy() - known_row["y1"]) ** 2
                + (cand["y2"].to_numpy() - known_row["y2"]) ** 2
            )

            kk = min(k, len(cand))
            top_idx = np.argsort(dists)[:kk]
            psi_hat1 = float(cand.iloc[top_idx][psi1].mean())
            psi_hat2 = float(cand.iloc[top_idx][psi2].mean())

            for target_fmt in player_formats:
                if target_fmt == known_fmt:
                    continue
                target_row = rows[rows["format"] == target_fmt].iloc[0]
                if pd.isna(target_row[raw1]) or pd.isna(target_row[raw2]):
                    continue

                # 1. fixed effects only (psi = 0)
                fixed_log1 = target_row["fe1_row"]
                fixed_log2 = target_row["fe2_row"]

                # 2. direct format conversion of the player's own value
                conv_log1 = known_row["y1"] + (target_row["fe1_row"] - known_row["fe1_row"])
                conv_log2 = known_row["y2"] + (target_row["fe2_row"] - known_row["fe2_row"])

                # 3. direct k-NN on log stats (neighbours need BOTH formats)
                target_lookup = by_format_log[target_fmt]
                cand_both_mask = cand["player_id"].isin(target_lookup.index).to_numpy()
                if cand_both_mask.any():
                    dists_both = dists_raw[cand_both_mask]
                    cand_both = cand[cand_both_mask]
                    kk_b = min(k, len(cand_both))
                    top_b = np.argsort(dists_both)[:kk_b]
                    nb_ids = cand_both.iloc[top_b]["player_id"]
                    nb_targets = target_lookup.loc[nb_ids]
                    dknn_log1 = float(nb_targets["y1"].mean())
                    dknn_log2 = float(nb_targets["y2"].mean())
                else:
                    n_dknn_empty += 1
                    dknn_log1 = np.nan
                    dknn_log2 = np.nan

                # 4. k-NN latent (psi) transfer -- the paper's method
                knn_log1 = target_row["fe1_row"] + psi_hat1
                knn_log2 = target_row["fe2_row"] + psi_hat2

                # 5. the player's own record, shrunk by how much they have played
                own_log1 = target_row["fe1_row"] + known_row.get("own1_row", np.nan)
                own_log2 = target_row["fe2_row"] + known_row.get("own2_row", np.nan)

                # 6. equal mix of the k-NN and own-record estimates
                mix_log1 = (knn_log1 + own_log1) / 2
                mix_log2 = (knn_log2 + own_log2) / 2

                records.append({
                    "player_id": player_id,
                    "player_name": known_row["player_name"],
                    "known_format": known_fmt,
                    "target_format": target_fmt,
                    "true1": target_row[raw1],
                    "true2": target_row[raw2],
                    "fixed1": np.exp(fixed_log1), "fixed2": np.exp(fixed_log2),
                    "conv1": np.exp(conv_log1),   "conv2": np.exp(conv_log2),
                    "dknn1": np.exp(dknn_log1),   "dknn2": np.exp(dknn_log2),
                    "knn1": np.exp(knn_log1),     "knn2": np.exp(knn_log2),
                    "own1": np.exp(own_log1),     "own2": np.exp(own_log2),
                    "mix1": np.exp(mix_log1),     "mix2": np.exp(mix_log2),
                })

    if n_dknn_empty:
        print(f"  WARNING: {n_dknn_empty} pairs had no both-format neighbours for direct k-NN")
    return pd.DataFrame(records)

def summarize(label, y_true, y_pred):
    return {
        "metric_set": label,
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "MAE": round(mae(y_true, y_pred), 4),
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAPE": round(mape(y_true, y_pred), 2),
        "n_pairs": len(y_true),
    }

METHOD_LABELS = {
    "fixed": "LOPO fixed-effects only",
    "conv":  "LOPO direct conversion",
    "dknn":  "LOPO direct k-NN",
    "knn":   "LOPO + k-NN transfer",
    "own":   "LOPO own record, shrunk",
    "mix":   "LOPO k-NN + own record",
}

def summarize_methods(res, outcome1_label, outcome2_label):
    rows = []
    for m in METHOD_LABELS:
        sub = res.dropna(subset=[f"{m}1", f"{m}2"])
        if sub.empty:
            continue
        rows.append(summarize(f"{METHOD_LABELS[m]} | {outcome1_label}",
                              sub["true1"].to_numpy(), sub[f"{m}1"].to_numpy()))
        rows.append(summarize(f"{METHOD_LABELS[m]} | {outcome2_label}",
                              sub["true2"].to_numpy(), sub[f"{m}2"].to_numpy()))
    return rows

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", choices=["format-only", "enriched"], default="format-only",
                   help="Which fitted model supplies the fixed effects and psi. "
                        "Enriched also uses the player covariates, which are known "
                        "for an unseen player, so it is the fairer prediction setting; "
                        "format-only is what the rankings use.")
    p.add_argument("-k", type=int, default=K_DEFAULT)
    return p.parse_args()

SPEC_FILES = {
    "format-only": {"data": "{kind}_joint.csv", "psi": "{kind}_player_rankings.csv"},
    "enriched":    {"data": "{kind}_joint_enriched.csv",
                    "psi": "{kind}_enriched_player_rankings.csv"},
}

def load_family(kind, spec, labels, raw_cols):
    files = SPEC_FILES[spec]
    joint = pd.read_csv(DATA_DIR / files["data"].format(kind=kind))
    for col in raw_cols:
        joint[col] = pd.to_numeric(joint[col], errors="coerce")
    joint = joint.dropna(subset=list(raw_cols)).reset_index(drop=True)
    joint = fitted_fixed_effects(kind, joint, spec, labels)
    psi = pd.read_csv(OUTPUT_DIR / files["psi"].format(kind=kind))
    return joint, psi

def main():
    args = parse_args()
    spec = args.spec
    suffix = "" if spec == "format-only" else "_enriched"
    print(f"Specification: {spec}\n")

    print("=== Batting: LOPO methods ===")
    bat_joint, bat_psi = load_family("batting", spec, ("log_avg", "log_sr"), ("avg", "sr"))
    bat_res = run_lopo_methods("batting", bat_joint, bat_psi,
                               psi1="psi_log_avg", psi2="psi_log_sr",
                               raw1="avg", raw2="sr", k=args.k)

    print("\n=== Bowling: LOPO methods ===")
    bowl_joint, bowl_psi = load_family("bowling", spec, ("log_econ", "log_avg"), ("econ", "avg"))
    bowl_res = run_lopo_methods("bowling", bowl_joint, bowl_psi,
                                psi1="psi_log_econ", psi2="psi_log_avg",
                                raw1="econ", raw2="avg", k=args.k)

    knn_rows = [
        summarize("LOPO+kNN | avg", bat_res["true1"].to_numpy(), bat_res["knn1"].to_numpy()),
        summarize("LOPO+kNN | sr", bat_res["true2"].to_numpy(), bat_res["knn2"].to_numpy()),
        summarize("LOPO+kNN | econ", bowl_res["true1"].to_numpy(), bowl_res["knn1"].to_numpy()),
        summarize("LOPO+kNN | bowl_avg", bowl_res["true2"].to_numpy(), bowl_res["knn2"].to_numpy()),
    ]
    results = pd.DataFrame(knn_rows)
    results.insert(0, "spec", spec)
    print("\n" + "=" * 72)
    print(f"LOPO + k-NN EVALUATION (k={args.k}, {spec})")
    print("=" * 72)
    print(results.to_string(index=False))
    out_path = OUTPUT_DIR / f"lopo_knn_evaluation{suffix}.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    base_rows = (summarize_methods(bat_res, "avg", "sr")
                 + summarize_methods(bowl_res, "econ", "bowl_avg"))
    baselines = pd.DataFrame(base_rows)
    baselines.insert(0, "spec", spec)
    print("\n" + "=" * 72)
    print(f"LOPO BASELINE COMPARISON (identical transition pairs, k={args.k}, {spec})")
    print("=" * 72)
    print(baselines.to_string(index=False))
    base_path = OUTPUT_DIR / f"lopo_baselines_evaluation{suffix}.csv"
    baselines.to_csv(base_path, index=False)
    print(f"\nSaved: {base_path}")

if __name__ == "__main__":
    main()
