"""
Loads race data into a clean pandas DataFrame regardless of source
(API cache or Kaggle CSVs). The rest of the project only talks to this layer.
"""

import json
import pandas as pd
from pathlib import Path

from src.config import DATA_CACHE, DATA_RAW, HISTORY_START_YEAR
from src.data.fetcher import fetch_seasons


def load_race_results(
    start_year: int = HISTORY_START_YEAR,
    end_year: int = None,
    use_kaggle: bool = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per driver per race, sorted by
    (year, round). Automatically uses Kaggle CSVs if present, otherwise
    falls back to the API.

    Columns:
        year, round, race_name, circuit,
        driver_id, driver_name,
        constructor_id, constructor_name,
        grid, position, status, points,
        finished (bool), fastest_lap_rank
    """
    from datetime import datetime
    if end_year is None:
        end_year = datetime.now().year

    kaggle_available = _kaggle_files_present()

    if use_kaggle is None:
        use_kaggle = kaggle_available

    if use_kaggle and kaggle_available:
        print("Loading from Kaggle CSVs...")
        df = _load_from_kaggle(start_year, end_year)
    else:
        print("Loading from API cache / fetching missing seasons...")
        records = fetch_seasons(start_year, end_year)
        df = pd.DataFrame(records)

    df = _clean(df)
    print(f"Loaded {len(df):,} driver-race records ({df['year'].min()}–{df['year'].max()})")
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _kaggle_files_present() -> bool:
    required = ["results.csv", "races.csv", "drivers.csv", "constructors.csv"]
    return all((DATA_RAW / f).exists() for f in required)


def _load_from_kaggle(start_year: int, end_year: int) -> pd.DataFrame:
    races = pd.read_csv(DATA_RAW / "races.csv")
    results = pd.read_csv(DATA_RAW / "results.csv")
    drivers = pd.read_csv(DATA_RAW / "drivers.csv")
    constructors = pd.read_csv(DATA_RAW / "constructors.csv")

    # Filter to year range
    races = races[(races["year"] >= start_year) & (races["year"] <= end_year)]

    # Merge everything together
    df = results.merge(races[["raceId", "year", "round", "name", "circuitId"]], on="raceId")
    df = df.merge(drivers[["driverId", "driverRef", "forename", "surname"]], on="driverId")
    df = df.merge(constructors[["constructorId", "constructorRef", "name"]], on="constructorId", suffixes=("", "_constructor"))

    df = df.rename(columns={
        "name":             "race_name",
        "circuitId":        "circuit",
        "driverRef":        "driver_id",
        "constructorRef":   "constructor_id",
        "name_constructor": "constructor_name",
        "positionOrder":    "position",
        "grid":             "grid",
        "points":           "points",
        "statusId":         "status",
    })

    df["driver_name"] = df["forename"] + " " + df["surname"]
    df["fastest_lap_rank"] = 0  # not in base Kaggle file, fill later if needed

    # position is stored as int in Kaggle (DNFs get high numbers via positionOrder)
    # We want None for classified DNFs — use positionText if available
    if "positionText" in df.columns:
        df["position"] = pd.to_numeric(df["positionText"], errors="coerce")

    return df[[
        "year", "round", "race_name", "circuit",
        "driver_id", "driver_name",
        "constructor_id", "constructor_name",
        "grid", "position", "status", "points", "fastest_lap_rank"
    ]]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["year"].astype(int)
    df["round"] = df["round"].astype(int)
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce").fillna(0).astype(int)
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0.0)
    df["position"] = pd.to_numeric(df["position"], errors="coerce")  # NaN = DNF

    # A driver "finished" if they have a classified position
    df["finished"] = df["position"].notna()

    df = df.sort_values(["year", "round"]).reset_index(drop=True)
    return df
