"""Fail fast if the active model data is incomplete or identity-unsafe."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("Data/new/processed")

def validate_model(kind: str, metadata_ids: set[str]) -> None:
    data_path = PROCESSED / f"{kind}_joint_enriched.csv"
    meta_path = PROCESSED / f"{kind}_joint_enriched_meta.json"
    data = pd.read_csv(data_path, dtype={"player_id": str})
    model_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    predictors = model_meta["predictors"]

    missing_columns = sorted(set(predictors) - set(data.columns))
    if missing_columns:
        raise ValueError(f"{kind}: missing predictor columns: {missing_columns}")
    if data.duplicated(["player_id", "format"]).any():
        raise ValueError(f"{kind}: duplicate player_id/format rows")
    if not set(data["player_id"]).issubset(metadata_ids):
        raise ValueError(f"{kind}: model players absent from complete metadata")
    if data[predictors].isna().any().any():
        raise ValueError(f"{kind}: missing predictor values")
    if not np.isfinite(data[predictors].to_numpy(dtype=float)).all():
        raise ValueError(f"{kind}: non-finite predictor values")

    constant = [column for column in predictors if data[column].nunique() < 2]
    if constant:
        raise ValueError(f"{kind}: constant predictors: {constant}")

    # The age terms vary within a player by design and are exempt from the constancy check.
    format_varying = set(model_meta.get("format_varying_predictors", []))
    unexpected = format_varying - set(predictors)
    if unexpected:
        raise ValueError(f"{kind}: format_varying_predictors not in predictors: "
                         f"{sorted(unexpected)}")

    player_covariates = [
        p for p in predictors
        if not p.startswith("format_") and p not in format_varying
    ]
    changing = [
        column
        for column in player_covariates
        if data.groupby("player_id")[column].nunique(dropna=False).max() > 1
    ]
    if changing:
        raise ValueError(f"{kind}: player covariates change by format: {changing}")

    # A format-varying covariate that never varies signals a broken build.
    for column in sorted(format_varying):
        if data.groupby("player_id")[column].nunique(dropna=False).max() <= 1:
            raise ValueError(
                f"{kind}: {column} is declared format-varying but is constant within every player"
            )

    if model_meta.get("missing_covariate_values") != 0:
        raise ValueError(f"{kind}: metadata JSON does not certify zero missing values")

    print(
        f"{kind.capitalize()}: {len(data):,} rows, "
        f"{data['player_id'].nunique():,} players, {len(predictors)} predictors, "
        "0 missing values"
    )

def main() -> None:
    metadata = pd.read_csv(
        PROCESSED / "player_metadata_complete.csv", dtype={"player_id": str}
    )
    if metadata["player_id"].duplicated().any():
        raise ValueError("Metadata contains duplicate player IDs")
    if metadata.isna().any().any() or (metadata.astype(str) == "").any().any():
        raise ValueError("Metadata contains missing or empty values")
    if metadata["playing_role_source"].eq("derived_from_model_eligibility").any():
        raise ValueError("Metadata contains model-eligibility-derived player roles")

    metadata_ids = set(metadata["player_id"])
    validate_model("batting", metadata_ids)
    validate_model("bowling", metadata_ids)
    print(f"Metadata: {len(metadata):,} stable-ID players, 0 missing values")

if __name__ == "__main__":
    main()
