"""
Stage 1b — Build constructor reliability estimates from historical results.

Usage
-----
    python scripts/model/build_reliability.py
    python scripts/model/build_reliability.py --start 2014
    python scripts/model/build_reliability.py --as-of 2021

Outputs
-------
    data/processed/reliability.csv          per (constructor, year) posterior
    data/processed/reliability_history.csv  per-race rolling trajectory
    data/processed/reliability.pkl          pickled ConstructorReliability

Sanity section prints:
  - Most/least reliable (constructor, year) pairs in the most recent year
  - Known failure seasons as a gut-check (2015 McLaren-Honda, 2017 Red Bull-Renault)
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import DATA_PROCESSED, DATA_RAW, RATING_START_YEAR
from src.model.reliability import build_reliability_from_results


def main(start_year: int, as_of_year: int | None) -> None:
    print(f"\n=== Stage 1b — Constructor reliability ({start_year} → "
          f"{'present' if as_of_year is None else as_of_year}) ===\n")

    # --- Load --------------------------------------------------------------
    print("Loading raw data...")
    results      = pd.read_csv(DATA_RAW / "results.csv")
    races        = pd.read_csv(DATA_RAW / "races.csv")
    constructors = pd.read_csv(DATA_RAW / "constructors.csv")
    status       = pd.read_csv(DATA_RAW / "status.csv")

    if as_of_year is not None:
        races = races[races["year"] <= as_of_year]

    n_in_scope = races[races["year"] >= start_year].shape[0]
    print(f"  results rows   : {len(results):,}")
    print(f"  races in scope : {n_in_scope:,}  ({start_year}+)")

    # --- Build -------------------------------------------------------------
    print("\nProcessing races chronologically...")
    engine = build_reliability_from_results(
        results=results,
        races=races,
        constructors=constructors,
        status=status,
        start_year=start_year,
    )
    print(f"  (constructor, year) pairs : {len(engine.state):,}")
    print(f"  history rows              : {len(engine.history):,}")

    # --- Persist -----------------------------------------------------------
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    snap = engine.snapshot()
    hist = engine.history_df()

    snap_path = DATA_PROCESSED / "reliability.csv"
    hist_path = DATA_PROCESSED / "reliability_history.csv"
    pkl_path  = DATA_PROCESSED / "reliability.pkl"

    snap.to_csv(snap_path, index=False)
    hist.to_csv(hist_path, index=False)
    with open(pkl_path, "wb") as fh:
        pickle.dump(engine, fh)

    print(f"\nSaved:")
    print(f"  {snap_path}")
    print(f"  {hist_path}")
    print(f"  {pkl_path}")

    # --- Sanity-check print ------------------------------------------------
    max_year = int(snap["year"].max())
    cols = ["constructor", "year", "n_entries", "n_mech_dnfs",
            "empirical_dnf_rate", "reliability_mean", "reliability_sigma"]

    print(f"\n--- Most reliable teams, {max_year} (posterior) ---")
    print(snap[snap["year"] == max_year].sort_values("reliability_mean", ascending=False)
              .head(10)[cols].to_string(index=False))

    print(f"\n--- Least reliable teams, {max_year} (posterior) ---")
    print(snap[snap["year"] == max_year].sort_values("reliability_mean", ascending=True)
              .head(10)[cols].to_string(index=False))

    # Historical gut-check: specific seasons where reliability was famously awful.
    known_bad = [
        ("McLaren",  2015),   # Honda comeback disaster
        ("McLaren",  2016),   # Still Honda
        ("McLaren",  2017),   # Still Honda
        ("Red Bull", 2017),   # Renault-era Red Bull chronic failures
        ("Ferrari",  2020),   # Engine regression after FIA dispute
        ("Williams", 2019),   # Late-FW42 era
    ]
    print("\n--- Named sanity checks (low reliability expected) ---")
    for name, yr in known_bad:
        row = snap[(snap["constructor"] == name) & (snap["year"] == yr)]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {name:10s} {yr}: entries={r['n_entries']:>2}  "
                  f"mech_dnfs={r['n_mech_dnfs']:>2}  "
                  f"reliability={r['reliability_mean']:.3f} "
                  f"± {r['reliability_sigma']:.3f}")

    # Historical gut-check: seasons where reliability was famously excellent
    known_good = [
        ("Mercedes", 2020),
        ("Mercedes", 2021),
        ("Red Bull", 2023),
        ("McLaren",  2024),
    ]
    print("\n--- Named sanity checks (high reliability expected) ---")
    for name, yr in known_good:
        row = snap[(snap["constructor"] == name) & (snap["year"] == yr)]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {name:10s} {yr}: entries={r['n_entries']:>2}  "
                  f"mech_dnfs={r['n_mech_dnfs']:>2}  "
                  f"reliability={r['reliability_mean']:.3f} "
                  f"± {r['reliability_sigma']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=RATING_START_YEAR)
    parser.add_argument("--as-of", type=int, default=None)
    args = parser.parse_args()
    main(start_year=args.start, as_of_year=args.as_of)
