"""
Stage 1a — Build driver ratings from historical results.

Usage
-----
    python scripts/model/build_ratings.py
    python scripts/model/build_ratings.py --start 2014
    python scripts/model/build_ratings.py --no-quali       # skip the quali update
    python scripts/model/build_ratings.py --as-of 2021     # train through 2021 only
                                                             (holdout-building mode)

Outputs
-------
    data/processed/ratings.csv            current per-driver (μ, σ, ordinal) snapshot
    data/processed/ratings_history.csv    per-weekend trajectory
    data/processed/rater.pkl              pickled F1Rater (for Stage 2 reuse)

Sanity section printed at the end shows the top-20 active drivers by μ and
by ordinal (μ − 3σ, the conservative skill estimate). A good v1 should put
Verstappen, Hamilton, Alonso, Leclerc near the top, and established
backmarkers (Latifi, Mazepin, etc.) near the bottom of active drivers.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

# Make `src` importable when invoked via `python scripts/model/build_ratings.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import DATA_PROCESSED, DATA_RAW, RATING_START_YEAR
from src.model.rating import build_ratings_from_results


def main(start_year: int, as_of_year: int | None, use_quali: bool) -> None:
    print(f"\n=== Stage 1a — Driver rating ({start_year} → "
          f"{'present' if as_of_year is None else as_of_year}) ===\n")

    # --- Load Ergast CSVs --------------------------------------------------
    print("Loading raw data...")
    results    = pd.read_csv(DATA_RAW / "results.csv")
    races      = pd.read_csv(DATA_RAW / "races.csv")
    drivers    = pd.read_csv(DATA_RAW / "drivers.csv")
    status     = pd.read_csv(DATA_RAW / "status.csv")
    qualifying = pd.read_csv(DATA_RAW / "qualifying.csv") if use_quali else None

    # Optional training cutoff (holdout builder)
    if as_of_year is not None:
        races = races[races["year"] <= as_of_year]

    n_races_in_scope = races[races["year"] >= start_year].shape[0]
    print(f"  results rows      : {len(results):,}")
    print(f"  races rows        : {len(races):,}")
    print(f"  in-scope races    : {n_races_in_scope:,}  ({start_year}+)")
    if qualifying is not None:
        print(f"  qualifying rows   : {len(qualifying):,}")
    else:
        print("  qualifying        : skipped (--no-quali)")

    # --- Build -------------------------------------------------------------
    print("\nProcessing races chronologically...")
    rater = build_ratings_from_results(
        results=results,
        races=races,
        drivers=drivers,
        status=status,
        qualifying=qualifying,
        start_year=start_year,
    )
    print(f"  drivers rated     : {len(rater.drivers):,}")
    print(f"  history rows      : {len(rater.history):,}")

    # --- Persist -----------------------------------------------------------
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    snap = rater.snapshot()
    hist = rater.history_df()

    snap_path  = DATA_PROCESSED / "ratings.csv"
    hist_path  = DATA_PROCESSED / "ratings_history.csv"
    model_path = DATA_PROCESSED / "rater.pkl"

    snap.to_csv(snap_path, index=False)
    hist.to_csv(hist_path, index=False)
    with open(model_path, "wb") as fh:
        pickle.dump(rater, fh)

    print(f"\nSaved:")
    print(f"  {snap_path}")
    print(f"  {hist_path}")
    print(f"  {model_path}")

    # --- Sanity-check print ------------------------------------------------
    # Define "active" as last race in the top two most recent years we covered.
    max_year = int(snap["last_year"].dropna().max())
    active_cutoff = max_year - 1
    active = snap[snap["last_year"] >= active_cutoff].copy()
    print(f"\n--- Top 20 active drivers ({active_cutoff}-{max_year}) by μ ---")
    print(active.sort_values("mu", ascending=False).head(20)
              [["driver", "mu", "sigma", "ordinal", "n_races", "n_qualis",
                "last_year"]]
              .to_string(index=False))
    print(f"\n--- Top 20 active drivers by ordinal (μ−3σ, conservative) ---")
    print(active.sort_values("ordinal", ascending=False).head(20)
              [["driver", "mu", "sigma", "ordinal", "n_races", "n_qualis",
                "last_year"]]
              .to_string(index=False))
    print(f"\n--- Bottom 10 active drivers by μ (sanity: known backmarkers) ---")
    # At least 10 races, to avoid noise from short-stint reserve drivers
    established = active[active["n_races"] >= 10]
    print(established.sort_values("mu", ascending=True).head(10)
              [["driver", "mu", "sigma", "ordinal", "n_races", "last_year"]]
              .to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=RATING_START_YEAR,
                        help=f"Earliest year to include (default {RATING_START_YEAR}).")
    parser.add_argument("--as-of", type=int, default=None,
                        help="Train through this year only (holdout builder).")
    parser.add_argument("--no-quali", action="store_true",
                        help="Skip the qualifying update (race-only baseline).")
    args = parser.parse_args()
    main(start_year=args.start,
         as_of_year=args.as_of,
         use_quali=not args.no_quali)
