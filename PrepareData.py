"""Build the current, complete cricket model datasets from stable player IDs."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

DATA_ROOT = Path("Data") / "new"
RAW_DIR = DATA_ROOT / "raw"
EXTERNAL_DIR = DATA_ROOT / "external"
PROCESSED_DIR = DATA_ROOT / "processed"
BASE_METADATA_PATH = EXTERNAL_DIR / "player_meta_cricketdata.csv"
SUPPLEMENT_PATH = EXTERNAL_DIR / "player_metadata_espn_supplement.json"
VERIFIED_FALLBACK_PATH = EXTERNAL_DIR / "player_metadata_cricbuzz_legacy.csv"
ROLE_OVERRIDES_PATH = EXTERNAL_DIR / "player_role_overrides_wikipedia.csv"
VERIFIED_ROLE_OVERRIDES_PATH = EXTERNAL_DIR / "player_role_overrides_verified.csv"

FORMATS = ("Test", "ODI", "T20")
FORMAT_ORDER = ["Test", "ODI", "T20"]
NON_COUNTRY_TEAMS = {"AFR", "ASIA", "ICC", "WORLD"}
COUNTRY_MIN_PLAYERS = 10
EPS = 1e-6

# Knot for the one-knot linear age spline, in years.
AGE_KNOT_DEFAULT = 29.0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-missing",
        action="store_true",
        help="Fetch only missing model metadata from ESPN's ID-based endpoint.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--age-knot",
        type=float,
        default=AGE_KNOT_DEFAULT,
        help="Knot age for the one-knot linear age spline (years).",
    )
    return parser.parse_args()

def extract_player_id(url: object) -> str:
    match = re.search(r"/player/(\d+)\.html", str(url))
    if not match:
        raise ValueError(f"Cannot extract ESPN player ID from URL: {url!r}")
    return match.group(1)

def extract_country(player_name: object) -> str:
    match = re.search(r"\(([^)]+)\)\s*$", str(player_name))
    return match.group(1).strip().upper() if match else ""

def clean_player_name(player_name: object) -> str:
    return re.sub(r"\s*\([^)]+\)\s*$", "", str(player_name)).strip()

def parse_span_start(span: object) -> int:
    match = re.match(r"^(\d{4})", str(span).strip())
    if not match:
        raise ValueError(f"Cannot parse career-span start: {span!r}")
    return int(match.group(1))

def parse_span_end(span: object) -> int:
    """Last year of the career span. A single-season span reads as start == end."""
    years = re.findall(r"\d{4}", str(span).strip())
    if not years:
        raise ValueError(f"Cannot parse career-span end: {span!r}")
    return int(years[-1])

def read_raw(kind: str, format_name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{format_name}_{kind}.csv"
    df = pd.read_csv(path)
    df["player_id"] = df["Player URL"].map(extract_player_id)
    df["player_name"] = df["Player Name"].map(clean_player_name)
    df["country_raw"] = df["Player Name"].map(extract_country)
    df["debut_year_raw"] = df["Span"].map(parse_span_start)
    df["last_year_raw"] = df["Span"].map(parse_span_end)
    df["format"] = format_name
    return df

def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")

def build_filtered_long_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    batting_parts: list[pd.DataFrame] = []
    bowling_parts: list[pd.DataFrame] = []
    identity_parts: list[pd.DataFrame] = []

    for format_name in FORMATS:
        bat = read_raw("Batting", format_name)
        bowl = read_raw("Bowling", format_name)
        identity_parts.extend(
            [
                bat[["player_id", "player_name", "country_raw", "debut_year_raw", "format"]],
                bowl[["player_id", "player_name", "country_raw", "debut_year_raw", "format"]],
            ]
        )

        for col in ("Ave", "SR", "Inns", "NO", "Runs", "BF"):
            if col in bat:
                bat[col] = numeric(bat[col])
        bat = bat[
            bat["Ave"].notna()
            & bat["SR"].notna()
            & bat["Inns"].notna()
            & (bat["Ave"] >= 1.0)
            & (bat["SR"] >= 1.0)
            & (bat["Inns"] >= 5)
        ].copy()
        bat = bat.rename(columns={"Ave": "avg", "SR": "sr", "Inns": "inns"})
        # Exposure denominators, used to weight each row by its precision.
        bat["w_avg"] = (bat["inns"] - bat.get("NO", 0)).clip(lower=1.0)
        balls = bat["BF"] if "BF" in bat else pd.Series(np.nan, index=bat.index)
        bat["w_sr"] = balls.fillna(100.0 * bat["Runs"] / bat["sr"]).clip(lower=1.0)
        batting_parts.append(
            bat[
                [
                    "player_id",
                    "player_name",
                    "format",
                    "avg",
                    "sr",
                    "inns",
                    "country_raw",
                    "debut_year_raw",
                    "last_year_raw",
                    "w_avg",
                    "w_sr",
                ]
            ]
        )

        for col in ("Overs", "Balls", "Wkts", "Econ", "Ave"):
            if col in bowl:
                bowl[col] = numeric(bowl[col])
        if "Overs" in bowl:
            workload_ok = bowl["Overs"] >= 20
            overs = bowl["Overs"]
        else:
            workload_ok = bowl["Balls"] >= 120
            overs = bowl["Balls"] / 6.0
        bowl = bowl[
            workload_ok
            & bowl["Wkts"].notna()
            & (bowl["Wkts"] >= 10)
            & bowl["Econ"].notna()
            & bowl["Ave"].notna()
            & (bowl["Econ"] > 0)
            & (bowl["Ave"] > 0)
        ].copy()
        bowl["overs"] = overs.loc[bowl.index]
        bowl = bowl.rename(columns={"Wkts": "wkts", "Econ": "econ", "Ave": "avg"})
        # Economy is runs per over, so balls bowled; bowling average is per wicket.
        bowl["w_econ"] = (bowl["overs"] * 6.0).clip(lower=1.0)
        bowl["w_bavg"] = bowl["wkts"].clip(lower=1.0)
        bowling_parts.append(
            bowl[
                [
                    "player_id",
                    "player_name",
                    "format",
                    "overs",
                    "wkts",
                    "econ",
                    "avg",
                    "country_raw",
                    "debut_year_raw",
                    "last_year_raw",
                    "w_econ",
                    "w_bavg",
                ]
            ]
        )

    batting = pd.concat(batting_parts, ignore_index=True)
    bowling = pd.concat(bowling_parts, ignore_index=True)
    identity = pd.concat(identity_parts, ignore_index=True)

    bat_counts = batting.groupby("player_id")["format"].nunique()
    bowl_counts = bowling.groupby("player_id")["format"].nunique()
    batting = batting[batting["player_id"].isin(bat_counts[bat_counts >= 2].index)].copy()
    bowling = bowling[bowling["player_id"].isin(bowl_counts[bowl_counts >= 2].index)].copy()
    return batting, bowling, identity

def is_missing(value: object) -> bool:
    return pd.isna(value) or str(value).strip().lower() in {
        "", "na", "nan", "none", "unknown",
    }

def style_from_api(payload: dict, style_type: str) -> str:
    for style in payload.get("styles") or payload.get("style") or []:
        if style.get("type") == style_type:
            return str(style.get("description") or "").strip()
    return ""

def role_from_api(payload: dict) -> str:
    position = payload.get("position") or {}
    return str(position.get("name") or "").strip()

def fetch_espn_player(player_id: str, attempts: int = 3) -> dict:
    url = f"http://core.espnuk.org/v2/sports/cricket/athletes/{player_id}"
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=20) as response:
                payload = json.load(response)
            return {
                "player_id": player_id,
                "full_name": str(
                    payload.get("fullName") or payload.get("displayName") or ""
                ).strip(),
                "dob": str(payload.get("dateOfBirth") or payload.get("dateOfBirthStr") or "")[:10],
                "batting_style": style_from_api(payload, "batting"),
                "bowling_style": style_from_api(payload, "bowling"),
                "playing_role": role_from_api(payload),
                "source": url,
            }
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch ESPN metadata for {player_id}: {last_error}")

def load_supplement() -> dict[str, dict]:
    if not SUPPLEMENT_PATH.exists():
        return {}
    records = json.loads(SUPPLEMENT_PATH.read_text(encoding="utf-8"))
    return {str(record["player_id"]): record for record in records}

def refresh_incomplete_metadata(
    final_ids: set[str], bowling_ids: set[str], workers: int
) -> dict[str, dict]:
    base = pd.read_csv(BASE_METADATA_PATH, dtype={"cricinfo_id": str}).set_index("cricinfo_id")
    supplement = load_supplement()
    needed: list[str] = []
    for player_id in sorted(final_ids, key=int):
        row = base.loc[player_id] if player_id in base.index else None
        required = ["full_name", "dob", "batting_style", "playing_role"]
        if player_id in bowling_ids:
            required.append("bowling_style")
        if row is None or any(is_missing(row.get(field)) for field in required):
            cached = supplement.get(player_id, {})
            if any(is_missing(cached.get(field)) for field in required):
                needed.append(player_id)

    if needed:
        print(f"Refreshing {len(needed)} incomplete ESPN metadata records...")
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            future_to_id = {executor.submit(fetch_espn_player, player_id): player_id
                            for player_id in needed}
            for count, future in enumerate(as_completed(future_to_id), 1):
                player_id = future_to_id[future]
                try:
                    supplement[player_id] = future.result()
                except Exception as exc:
                    failures.append(f"{player_id}: {exc}")
                if count % 25 == 0 or count == len(needed):
                    print(f"  {count}/{len(needed)}")
        if failures:
            raise RuntimeError("Metadata refresh failures:\n" + "\n".join(failures))

        EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        ordered = [supplement[player_id] for player_id in sorted(supplement, key=int)]
        SUPPLEMENT_PATH.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
        print(f"Saved: {SUPPLEMENT_PATH}")
    return supplement

def canonical_country(values: pd.Series) -> str:
    country_tokens: set[str] = set()
    for value in values.dropna().astype(str):
        for token in value.upper().split("/"):
            token = token.strip()
            if token and token not in NON_COUNTRY_TEAMS:
                country_tokens.add(token)
    if not country_tokens:
        raise ValueError("Player has no country in raw source rows")
    return "/".join(sorted(country_tokens))

def most_common_name(values: pd.Series) -> str:
    return Counter(values.dropna().astype(str)).most_common(1)[0][0]

def coalesce_metadata_value(base_row: pd.Series | None, supplement: dict, field: str) -> str:
    refreshed = supplement.get(field)
    if not is_missing(refreshed):
        return str(refreshed).strip()
    if base_row is not None:
        original = base_row.get(field)
        if not is_missing(original):
            return str(original).strip()
    return ""

def normalize_role(value: str) -> str:
    role = value.lower().replace("-", " ")
    if "allround" in role or "all round" in role:
        return "allrounder"
    if "bowl" in role:
        return "bowler"
    if any(word in role for word in ("batter", "batsman", "wicketkeeper", "keeper")):
        return "batter_or_wicketkeeper"
    raise ValueError(f"Unrecognized playing role: {value!r}")

def batting_hand(value: str) -> str:
    lower = value.lower()
    if "left" in lower:
        return "left"
    if "right" in lower:
        return "right"
    raise ValueError(f"Unrecognized batting style: {value!r}")

def bowling_hand(value: str) -> str:
    lower = value.lower()
    if "left" in lower:
        return "left"
    if "right" in lower:
        return "right"
    # An unqualified legbreak or googly is bowled right-arm.
    if "legbreak" in lower or "googly" in lower:
        return "right"
    raise ValueError(f"Unrecognized bowling style: {value!r}")

def bowling_type(value: str) -> str:
    lower = value.lower()
    spin_terms = ("spin", "break", "orthodox", "chinaman", "googly", "slow left")
    pace_terms = ("fast", "medium", "pace", "seam", "swing")
    if any(term in lower for term in spin_terms):
        return "spin"
    if any(term in lower for term in pace_terms):
        return "pace"
    raise ValueError(f"Unrecognized bowling style: {value!r}")

def build_complete_metadata(
    final_ids: set[str],
    batting_ids: set[str],
    bowling_ids: set[str],
    identity: pd.DataFrame,
    supplement: dict[str, dict],
) -> pd.DataFrame:
    base = pd.read_csv(BASE_METADATA_PATH, dtype={"cricinfo_id": str}).set_index("cricinfo_id")
    identity = identity[identity["player_id"].isin(final_ids)].copy()
    fallback = pd.read_csv(VERIFIED_FALLBACK_PATH, dtype=str).fillna("")
    fallback = fallback.set_index("player_name")
    if ROLE_OVERRIDES_PATH.exists():
        role_overrides = pd.read_csv(ROLE_OVERRIDES_PATH, dtype={"player_id": str}).fillna("")
        role_overrides = role_overrides.set_index("player_id")
    else:
        role_overrides = pd.DataFrame()
    if VERIFIED_ROLE_OVERRIDES_PATH.exists():
        verified_role_overrides = pd.read_csv(
            VERIFIED_ROLE_OVERRIDES_PATH, dtype={"player_id": str}
        ).fillna("")
        verified_role_overrides = verified_role_overrides.set_index("player_id")
    else:
        verified_role_overrides = pd.DataFrame()
    name_to_ids = identity.groupby("player_name")["player_id"].agg(lambda values: set(values))
    records: list[dict] = []

    for player_id in sorted(final_ids, key=int):
        raw = identity[identity["player_id"] == player_id]
        base_row = base.loc[player_id] if player_id in base.index else None
        refreshed = supplement.get(player_id, {})
        full_name = coalesce_metadata_value(base_row, refreshed, "full_name")
        dob = coalesce_metadata_value(base_row, refreshed, "dob")
        bat_style = coalesce_metadata_value(base_row, refreshed, "batting_style")
        bowl_style = coalesce_metadata_value(base_row, refreshed, "bowling_style")
        role = coalesce_metadata_value(base_row, refreshed, "playing_role")

        player_name = most_common_name(raw["player_name"])
        role_source = "espn"
        if (is_missing(role) and len(name_to_ids.get(player_name, set())) == 1
                and player_name in fallback.index):
            fallback_row = fallback.loc[player_name]
            if isinstance(fallback_row, pd.DataFrame):
                fallback_row = fallback_row.iloc[0]
            fallback_role = str(fallback_row.get("playing_role") or "").strip()
            fallback_born = str(fallback_row.get("born_year") or "").strip()
            espn_born = str(dob)[:4]
            if not is_missing(fallback_role) and fallback_born.startswith(espn_born):
                role = fallback_role
                role_source = "cricbuzz_dob_verified"

        if is_missing(role) and player_id in verified_role_overrides.index:
            override_role = str(
                verified_role_overrides.loc[player_id].get("playing_role") or ""
            ).strip()
            if not is_missing(override_role):
                normalize_role(override_role)
                role = override_role
                role_source = "source_verified_override"

        if is_missing(role) and player_id in role_overrides.index:
            override_role = str(role_overrides.loc[player_id].get("playing_role") or "").strip()
            if not is_missing(override_role):
                try:
                    normalize_role(override_role)
                except ValueError:
                    pass
                else:
                    role = override_role
                    role_source = "wikipedia_infobox"

        if is_missing(role):
            if player_id in batting_ids and player_id in bowling_ids:
                role = "Allrounder"
            elif player_id in bowling_ids:
                role = "Bowler"
            else:
                role = "Batter"
            role_source = "derived_from_model_eligibility"

        required = {
            "full_name": full_name,
            "dob": dob,
            "batting_style": bat_style,
            "playing_role": role,
        }
        if player_id in bowling_ids:
            required["bowling_style"] = bowl_style
        missing = [field for field, value in required.items() if is_missing(value)]
        if missing:
            raise ValueError(f"Player {player_id} still lacks required metadata: {missing}")

        debut_year = int(raw["debut_year_raw"].min())
        born_year = int(str(dob)[:4])
        record = {
            "player_id": player_id,
            "player_name": player_name,
            "full_name": full_name,
            "country": canonical_country(raw["country_raw"]),
            "debut_year": debut_year,
            "date_of_birth": str(dob)[:10],
            "born_year": born_year,
            "age_at_debut": debut_year - born_year,
            "batting_style": bat_style,
            "batting_hand": batting_hand(bat_style),
            "bowling_style": bowl_style if not is_missing(bowl_style) else "not_applicable",
            "bowling_hand": (bowling_hand(bowl_style) if player_id in bowling_ids
                             else "not_applicable"),
            "bowling_type": (bowling_type(bowl_style) if player_id in bowling_ids
                             else "not_applicable"),
            "playing_role_raw": role,
            "playing_role": normalize_role(role),
            "playing_role_source": role_source,
            "in_batting_model": int(player_id in batting_ids),
            "in_bowling_model": int(player_id in bowling_ids),
        }
        records.append(record)

    metadata = pd.DataFrame(records)
    if metadata.isna().any().any() or (metadata.astype(str) == "").any().any():
        raise ValueError("Complete metadata table unexpectedly contains empty values")
    if not metadata["age_at_debut"].between(10, 60).all():
        bad = metadata.loc[~metadata["age_at_debut"].between(10, 60)]
        raise ValueError(f"Implausible age-at-debut values:\n{bad.to_string(index=False)}")
    return metadata

def add_country_group(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()
    counts = metadata["country"].value_counts()
    metadata["country_group"] = metadata["country"].where(
        metadata["country"].map(counts).ge(COUNTRY_MIN_PLAYERS)
        & ~metadata["country"].str.contains("/"),
        "OTHER_OR_MULTI",
    )
    return metadata

def standardize_player_column(metadata: pd.DataFrame, column: str) -> pd.Series:
    values = metadata[column].astype(float)
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        raise ValueError(f"Cannot standardize {column}: standard deviation is {std}")
    return (values - values.mean()) / std

def encode_common_covariates(rows: pd.DataFrame,
                             metadata: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    metadata = metadata.copy()
    metadata["debut_year_scaled"] = standardize_player_column(metadata, "debut_year")
    metadata["age_at_debut_scaled"] = standardize_player_column(metadata, "age_at_debut")
    metadata["role_allrounder"] = (metadata["playing_role"] == "allrounder").astype(float)
    metadata["role_bowler"] = (metadata["playing_role"] == "bowler").astype(float)

    active_ids = set(rows["player_id"])
    active_country_counts = metadata.loc[
        metadata["player_id"].isin(active_ids), "country"
    ].value_counts()
    metadata["country_model_group"] = metadata["country"].where(
        metadata["country"].map(active_country_counts).ge(COUNTRY_MIN_PLAYERS)
        & ~metadata["country"].str.contains("/"),
        "OTHER_OR_MULTI",
    )
    country_dummies = pd.get_dummies(
        metadata["country_model_group"], prefix="country", dtype=float
    )
    reference = "country_AUS"
    if reference not in country_dummies:
        raise ValueError("Expected Australia reference category is unavailable")
    country_predictors = sorted(column for column in country_dummies if column != reference)
    metadata = pd.concat([metadata, country_dummies[country_predictors]], axis=1)

    metadata_for_merge = metadata.drop(columns=["player_name"])
    encoded = rows.merge(metadata_for_merge, on="player_id", how="left", validate="many_to_one")
    common = [
        "format_odi",
        "format_t20",
        "debut_year_scaled",
        "role_allrounder",
        "role_bowler",
        "age_at_debut_scaled",
        *country_predictors,
    ]
    return encoded, common

def add_age_covariates(rows: pd.DataFrame, knot: float) -> tuple[pd.DataFrame, list[str]]:
    """Population-level age curve terms: a linear slope plus one hinge."""
    rows = rows.copy()
    missing = rows[["debut_year_raw", "last_year_raw", "born_year"]].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"Cannot derive career-midpoint age for {int(missing.sum())} rows")

    span_start = rows["debut_year_raw"].astype(float)
    span_end = rows["last_year_raw"].astype(float)
    if (span_end < span_start).any():
        raise ValueError("Career span ends before it starts")

    rows["avg_age"] = (span_start + span_end) / 2.0 - rows["born_year"].astype(float)
    if not rows["avg_age"].between(10, 60).all():
        bad = rows.loc[~rows["avg_age"].between(10, 60), "avg_age"]
        raise ValueError(f"Implausible career-midpoint ages: {bad.head().tolist()}")

    scale = rows["avg_age"].std(ddof=0)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError(f"Cannot scale avg_age: standard deviation is {scale}")

    rows["age_scaled"] = (rows["avg_age"] - rows["avg_age"].mean()) / scale
    rows["age_hinge"] = np.maximum(0.0, rows["avg_age"] - knot) / scale
    return rows, ["age_scaled", "age_hinge"]

def add_indices(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    player_ids = sorted(rows["player_id"].unique(), key=int)
    player_index = {player_id: index for index, player_id in enumerate(player_ids)}
    rows["player_idx"] = rows["player_id"].map(player_index).astype(int)
    rows["format"] = pd.Categorical(rows["format"], categories=FORMAT_ORDER, ordered=True)
    rows["format_idx"] = rows["format"].cat.codes.astype(int)
    rows["format_odi"] = (rows["format"] == "ODI").astype(int)
    rows["format_t20"] = (rows["format"] == "T20").astype(int)
    return rows.sort_values(["player_idx", "format_idx"]).reset_index(drop=True)

def assert_complete_covariates(
    df: pd.DataFrame,
    predictors: list[str],
    label: str,
    format_varying: tuple[str, ...] = (),
) -> None:
    missing = df[predictors].isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        raise ValueError(f"{label} has missing covariates:\n{missing.to_string()}")
    if not np.isfinite(df[predictors].to_numpy(dtype=float)).all():
        raise ValueError(f"{label} has non-finite covariate values")
    # predictors[:2] are the format dummies; format_varying names the age terms.
    player_level = [p for p in predictors[2:] if p not in format_varying]
    varying = df.groupby("player_id")[player_level].nunique(dropna=False).max()
    varying = varying[varying > 1]
    if not varying.empty:
        raise ValueError(f"{label} has player covariates varying across formats: "
                         f"{varying.index.tolist()}")

def save_meta(
    path: Path,
    df: pd.DataFrame,
    predictors: list[str],
    outcomes: dict[str, str],
    age_knot: float | None = None,
    format_varying: tuple[str, ...] = (),
    exposure: dict[str, str] | None = None,
) -> None:
    player_map = (
        df[["player_idx", "player_id", "player_name"]]
        .drop_duplicates()
        .sort_values("player_idx")
    )
    payload = {
        "eps": EPS,
        "n_rows": int(len(df)),
        "n_players": int(df["player_id"].nunique()),
        "outcomes": outcomes,
        "predictors": predictors,
        "country_reference": "AUS",
        "role_reference": "batter_or_wicketkeeper",
        "format_reference": "Test",
        "player_idx_to_id": dict(zip(player_map["player_idx"].astype(str),
                                     player_map["player_id"])),
        "player_idx_to_name": dict(zip(player_map["player_idx"].astype(str),
                                       player_map["player_name"])),
        "missing_covariate_values": int(df[predictors].isna().sum().sum()),
        "format_varying_predictors": list(format_varying),
    }
    if exposure is not None:
        payload["exposure"] = {
            "columns": exposure,
            "meaning": "denominator of each outcome; residual variance is "
                       "sigma_sampling^2 / w + sigma_real^2",
        }
    if age_knot is not None:
        payload["age_curve"] = {
            "knot_age_years": age_knot,
            "basis": "age_scaled + max(0, age - knot)/sd(age)",
            "age_definition": "career midpoint age within the format: "
                              "(span_start + span_end)/2 - birth_year",
            "slope_before_knot": "beta_age_scaled",
            "slope_after_knot": "beta_age_scaled + beta_age_hinge",
        }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def main() -> None:
    args = parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    batting, bowling, identity = build_filtered_long_data()
    batting_ids = set(batting["player_id"])
    bowling_ids = set(bowling["player_id"])
    final_ids = batting_ids | bowling_ids
    supplement = load_supplement()
    if args.refresh_missing:
        supplement = refresh_incomplete_metadata(final_ids, bowling_ids, args.workers)

    metadata = build_complete_metadata(final_ids, batting_ids, bowling_ids, identity, supplement)
    metadata = add_country_group(metadata)
    metadata.to_csv(PROCESSED_DIR / "player_metadata_complete.csv", index=False)

    batting = add_indices(batting)
    bowling = add_indices(bowling)
    batting["y1"] = np.log(batting["avg"] + EPS)
    batting["y2"] = np.log(batting["sr"] + EPS)
    bowling["y1"] = np.log(bowling["econ"] + EPS)
    bowling["y2"] = np.log(bowling["avg"] + EPS)

    batting.to_csv(PROCESSED_DIR / "batting_long.csv", index=False)
    bowling.to_csv(PROCESSED_DIR / "bowling_long.csv", index=False)

    batting_joint_cols = [
        "player_idx", "player_id", "player_name", "format", "format_idx",
        "format_odi", "format_t20", "y1", "y2", "avg", "sr", "inns",
        "w_avg", "w_sr",
    ]
    bowling_joint_cols = [
        "player_idx", "player_id", "player_name", "format", "format_idx",
        "format_odi", "format_t20", "y1", "y2", "econ", "avg", "overs", "wkts",
        "w_econ", "w_bavg",
    ]
    batting[batting_joint_cols].to_csv(PROCESSED_DIR / "batting_joint.csv", index=False)
    bowling[bowling_joint_cols].to_csv(PROCESSED_DIR / "bowling_joint.csv", index=False)

    bat_encoded, bat_common = encode_common_covariates(batting, metadata)
    bat_encoded["is_lefthanded"] = (bat_encoded["batting_hand"] == "left").astype(float)
    bat_encoded, age_predictors = add_age_covariates(bat_encoded, args.age_knot)
    bat_predictors = [*bat_common[:3], "is_lefthanded", *bat_common[3:], *age_predictors]

    bowl_encoded, bowl_common = encode_common_covariates(bowling, metadata)
    bowl_encoded["bowling_hand_left"] = (bowl_encoded["bowling_hand"] == "left").astype(float)
    bowl_encoded["bowling_type_spin"] = (bowl_encoded["bowling_type"] == "spin").astype(float)
    bowl_encoded, _ = add_age_covariates(bowl_encoded, args.age_knot)
    bowl_predictors = [
        *bowl_common[:3],
        "bowling_hand_left",
        "bowling_type_spin",
        *bowl_common[3:],
        *age_predictors,
    ]

    fmt_varying = tuple(age_predictors)
    assert_complete_covariates(bat_encoded, bat_predictors, "batting enriched data", fmt_varying)
    assert_complete_covariates(bowl_encoded, bowl_predictors, "bowling enriched data", fmt_varying)
    bat_encoded.to_csv(PROCESSED_DIR / "batting_joint_enriched.csv", index=False)
    bowl_encoded.to_csv(PROCESSED_DIR / "bowling_joint_enriched.csv", index=False)

    save_meta(
        PROCESSED_DIR / "batting_joint_enriched_meta.json",
        bat_encoded,
        bat_predictors,
        {"y1": "log(batting_average)", "y2": "log(strike_rate)"},
        age_knot=args.age_knot,
        format_varying=fmt_varying,
        exposure={"y1": "w_avg", "y2": "w_sr"},
    )
    save_meta(
        PROCESSED_DIR / "bowling_joint_enriched_meta.json",
        bowl_encoded,
        bowl_predictors,
        {"y1": "log(economy_rate)", "y2": "log(bowling_average)"},
        age_knot=args.age_knot,
        format_varying=fmt_varying,
        exposure={"y1": "w_econ", "y2": "w_bavg"},
    )

    print(f"Batting: {len(bat_encoded)} rows, {len(batting_ids)} stable-ID players")
    print(f"Bowling: {len(bowl_encoded)} rows, {len(bowling_ids)} stable-ID players")
    print(f"Metadata: {len(metadata)} players, zero empty cells")
    print(f"Batting predictors: {len(bat_predictors)}")
    print(f"Bowling predictors: {len(bowl_predictors)}")
    print(f"Age curve: knot at {args.age_knot:g} yrs, terms {age_predictors}")
    print(f"Saved current data under: {PROCESSED_DIR}")

if __name__ == "__main__":
    main()
