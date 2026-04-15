"""
Stage 2 — Predict a single race and print the ranked grid.

Usage
-----
    # Predict a historical race from the dataset (using rater/reliability
    # state AS OF THAT MOMENT, i.e. no look-ahead). This is the "would we
    # have called this race right?" mode.
    python scripts/model/predict_race.py --year 2024 --round 22

    # Predict the MOST RECENT race in the dataset.
    python scripts/model/predict_race.py --latest

    # Predict a hypothetical race from a CSV you hand it. The CSV must have
    # columns: driverId, driverName, constructorId, constructorName, grid.
    # Use this for "what if the grid for next Sunday looks like this..."
    python scripts/model/predict_race.py --year 2025 --field my_field.csv

How this stays honest
---------------------
When predicting a historical race we REPLAY the rater + reliability state
chronologically only up to (but NOT including) the target race. So the
prediction uses strictly pre-race information — same as the back-test.

We don't use the saved rater.pkl directly because that file was updated
with ALL races including the one you're predicting. That would leak
outcome information.

Outputs
-------
A single ranked table to stdout, plus a CSV at
reports/predictions/<year>-R<round>.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    DATA_PROCESSED, DATA_RAW, DRIVER_SCORE_MODE, RATING_START_YEAR,
    SOFTMAX_TAU_INIT,
)
from src.model.race_predictor import RacePredictor, compute_pace_map
from src.model.rating import F1Rater
from src.model.reliability import ConstructorReliability

OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_reference_data() -> dict:
    """Load the raw-data tables we need to replay / look up race fields."""
    results      = pd.read_csv(DATA_RAW / "results.csv")
    races        = pd.read_csv(DATA_RAW / "races.csv")
    drivers      = pd.read_csv(DATA_RAW / "drivers.csv")
    constructors = pd.read_csv(DATA_RAW / "constructors.csv")
    qualifying   = pd.read_csv(DATA_RAW / "qualifying.csv")
    status       = pd.read_csv(DATA_RAW / "status.csv")
    cs           = pd.read_csv(DATA_RAW / "constructor_standings.csv")

    race_meta = races[races["year"] >= RATING_START_YEAR][
        ["raceId", "year", "round", "name"]
    ].sort_values(["year", "round"]).reset_index(drop=True)

    driver_name_map = (
        drivers.assign(driver=lambda d: d["forename"] + " " + d["surname"])
               .set_index("driverId")["driver"]
    )
    constructor_name_map = constructors.set_index("constructorId")["name"]
    status_map = status.set_index("statusId")["status"]

    res = results.merge(race_meta, on="raceId", how="inner").copy()
    res["driverName"]      = res["driverId"].map(driver_name_map)
    res["constructorName"] = res["constructorId"].map(constructor_name_map)
    res["status"]          = res["statusId"].map(status_map)

    qua = qualifying.merge(race_meta, on="raceId", how="inner").copy()
    qua["driverName"] = qua["driverId"].map(driver_name_map)

    pace_map = compute_pace_map(cs, races)

    return {
        "race_meta": race_meta,
        "results":   res,
        "qualifying": qua,
        "pace_map":  pace_map,
        "driver_name_map":      driver_name_map,
        "constructor_name_map": constructor_name_map,
    }


def replay_up_to(
    race_meta: pd.DataFrame, res: pd.DataFrame, qua: pd.DataFrame,
    target_year: int, target_round: int,
) -> tuple[F1Rater, ConstructorReliability]:
    """
    Rebuild rater + reliability from scratch, processing every race with
    (year, round) STRICTLY BEFORE (target_year, target_round).

    For the target race itself we only process its QUALIFYING (quali happens
    before the race so it's fair game) — this pre-qualifying skill update is
    what Stage 2 uses as the driver-skill feature when we call predict().
    """
    rater       = F1Rater()
    reliability = ConstructorReliability()

    for race_id, year, round_, _name in race_meta.itertuples(index=False):
        y, r = int(year), int(round_)

        if (y, r) == (target_year, target_round):
            # Only apply qualifying; stop before the race itself.
            q_rows = qua[qua["raceId"] == race_id]
            if not q_rows.empty:
                rater.update_qualifying(y, r, q_rows)
            break

        if (y, r) > (target_year, target_round):
            # We've walked past the target without hitting it.
            break

        q_rows = qua[qua["raceId"] == race_id]
        if not q_rows.empty:
            rater.update_qualifying(y, r, q_rows)
        r_rows = res[res["raceId"] == race_id]
        if not r_rows.empty:
            rater.update_race(y, r, r_rows)
            reliability.update_race(y, r, r_rows[[
                "constructorId", "constructorName", "status"
            ]])

    return rater, reliability


# ---------------------------------------------------------------------------
# Field construction
# ---------------------------------------------------------------------------
def field_from_history(
    res: pd.DataFrame, target_year: int, target_round: int
) -> tuple[pd.DataFrame, str, int]:
    """
    Look up the actual starting grid for a historical race.
    Returns: (field_df, race_name, raceId).
    """
    rows = res[(res["year"] == target_year) & (res["round"] == target_round)]
    if rows.empty:
        raise ValueError(
            f"No race found in data for year={target_year}, round={target_round}"
        )
    race_name = rows["name"].iloc[0]
    race_id   = int(rows["raceId"].iloc[0])
    field = rows[[
        "driverId", "driverName", "constructorId", "constructorName", "grid",
    ]].copy()
    return field, race_name, race_id


def field_from_csv(
    csv_path: Path,
    driver_name_map: pd.Series,
    constructor_name_map: pd.Series,
) -> pd.DataFrame:
    """
    Read a user-supplied field CSV. Flexible about which columns are present:
      required: driverId, constructorId, grid
      optional: driverName, constructorName (filled from lookups if missing)
    """
    df = pd.read_csv(csv_path)
    required = {"driverId", "constructorId", "grid"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Field CSV missing columns: {missing}")

    if "driverName" not in df.columns:
        df["driverName"] = df["driverId"].map(driver_name_map)
    if "constructorName" not in df.columns:
        df["constructorName"] = df["constructorId"].map(constructor_name_map)

    df["driverId"]      = df["driverId"].astype(int)
    df["constructorId"] = df["constructorId"].astype(int)
    df["grid"]          = df["grid"].astype(int)

    return df[["driverId", "driverName", "constructorId", "constructorName", "grid"]]


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------
def _prob_to_decimal_odds(p: float, overround: float = 1.0) -> float:
    """
    Decimal (European) odds:  odds = 1 / (p * overround)
      overround=1.00 → fair odds ("true" odds, 100% book)
      overround=1.20 → bookmaker's posted odds with a 20% built-in margin
    """
    if p <= 0:
        return float("inf")
    return 1.0 / (p * overround)


def format_table(preds: pd.DataFrame, race_label: str, mode: str,
                 tau: float, actuals: pd.DataFrame | None,
                 overround: float = 1.0) -> str:
    """
    Build the printable prediction table. If `actuals` is provided (historical
    race) we append finish/DNF info. If `overround` > 1, we append posted
    decimal odds next to the fair odds.
    """
    has_book = overround > 1.0
    bookhdr = f" {'book':>7s}" if has_book else ""

    lines = [
        "",
        f"{race_label}   (mode={mode!r}, τ={tau}"
        + (f", overround={overround:.2f}" if has_book else "") + ")",
        "-" * (95 + (8 if has_book else 0)),
        f"{'#':>2}  {'driver':<22s} {'team':<22s} "
        f"{'grid':>4}  {'ord':>5s} {'rel':>5s} {'pace':>5s} "
        f"{'p_win':>6s} {'p_pod':>6s} {'p_pts':>6s} {'fair':>7s}" + bookhdr,
        "-" * (95 + (8 if has_book else 0)),
    ]

    if actuals is not None:
        preds = preds.merge(
            actuals[["driverId", "positionOrder", "status"]],
            on="driverId", how="left",
        )

    for i, row in preds.iterrows():
        p = float(row["p_win_mc"])
        fair = _prob_to_decimal_odds(p, 1.0)
        base = (
            f"{i+1:>2}  {row['driver']:<22s} {row['constructor']:<22s} "
            f"{int(row['grid']):>4}  "
            f"{row['ordinal']:>5.0f} {row['reliability']:>5.2f} "
            f"{row['pace_score']:>5.2f} "
            f"{p:>6.3f} "
            f"{row['p_podium_mc']:>6.3f} {row['p_points_mc']:>6.3f} "
            f"{fair:>7.2f}"
        )
        if has_book:
            book = _prob_to_decimal_odds(p, overround)
            base += f" {book:>7.2f}"
        if actuals is not None and pd.notna(row.get("positionOrder", np.nan)):
            pos = int(row["positionOrder"])
            st  = str(row.get("status", ""))
            marker = f"   → P{pos}" if st == "Finished" else f"   → P{pos} ({st})"
            base += marker
        lines.append(base)

    lines.append("-" * (95 + (8 if has_book else 0)))
    lines.append(
        "  ord = driver ordinal μ−3σ  (higher = better)"
    )
    lines.append(
        "  rel = P(car finishes)  |  pace = prev-year constructor pace score [0,1]"
    )
    lines.append(
        "  p_win = Monte-Carlo win probability  |  p_pod = P(top 3)  |  p_pts = P(top 10)"
    )
    lines.append(
        f"  fair = 1 / p_win  (zero-margin decimal odds)"
    )
    if has_book:
        lines.append(
            f"  book = 1 / (p_win × {overround:.2f})  "
            f"(posted odds with {(overround-1)*100:.0f}% margin baked in)"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(year: int | None, round_: int | None, field_csv: str | None,
         use_latest: bool, tau: float, mode: str, n_mc: int, seed: int,
         overround: float):
    data = load_reference_data()
    race_meta = data["race_meta"]
    res       = data["results"]
    qua       = data["qualifying"]

    # Resolve (year, round)
    if use_latest:
        tail = race_meta.iloc[-1]
        year, round_ = int(tail["year"]), int(tail["round"])
        print(f"--latest resolved to {year} R{round_}  ({tail['name']!r})")

    if year is None or round_ is None:
        raise SystemExit(
            "Need either --latest OR (--year AND --round). "
            "If supplying a custom --field CSV, still pass --year/--round "
            "so pace_map and reliability can pick the right year."
        )

    # Field: from CSV or from history
    if field_csv:
        field = field_from_csv(
            Path(field_csv),
            driver_name_map=data["driver_name_map"],
            constructor_name_map=data["constructor_name_map"],
        )
        race_label = f"{year} — custom field from {field_csv}"
        actuals = None
    else:
        field, race_name, _ = field_from_history(res, year, round_)
        race_label = f"{year} R{round_}  {race_name}"
        actuals = res[(res["year"] == year) & (res["round"] == round_)][
            ["driverId", "positionOrder", "status"]
        ]

    # Rebuild state from scratch up to the target race
    print(f"Replaying history up to {year} R{round_} (no look-ahead)...")
    rater, reliability = replay_up_to(race_meta, res, qua, year, round_)
    print(f"  rater drivers tracked  : {len(rater.drivers):,}")
    print(f"  (constructor, year) rel: {len(reliability.state):,}")

    # Predict
    predictor = RacePredictor(
        rater=rater, reliability=reliability, pace_map=data["pace_map"],
        tau=tau, mode=mode, n_mc=n_mc, seed=seed,
    )
    preds = predictor.predict(field, year=year, round_=round_)

    # Print
    text = format_table(preds, race_label, mode, tau, actuals, overround)
    print(text)

    # Persist CSV
    out_path = OUT_DIR / f"{year}-R{round_:02d}_{mode}.csv"
    preds_out = preds.copy()
    preds_out["fair_decimal_odds"] = 1.0 / preds_out["p_win_mc"].clip(lower=1e-6)
    if overround > 1.0:
        preds_out["posted_decimal_odds"] = 1.0 / (
            preds_out["p_win_mc"].clip(lower=1e-6) * overround
        )
    if actuals is not None:
        preds_out = preds_out.merge(
            actuals[["driverId", "positionOrder", "status"]],
            on="driverId", how="left",
        )
    preds_out.to_csv(out_path, index=False)
    print(f"\nSaved prediction table: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",   type=int, default=None)
    parser.add_argument("--round",  type=int, default=None, dest="round_")
    parser.add_argument("--latest", action="store_true",
                        help="Predict the most recent race in the dataset.")
    parser.add_argument("--field",  type=str, default=None,
                        help="Optional CSV with driverId/constructorId/grid "
                             "for a hypothetical race.")
    parser.add_argument("--tau",    type=float, default=SOFTMAX_TAU_INIT)
    parser.add_argument("--mode",   type=str, default=DRIVER_SCORE_MODE,
                        choices=["ordinal", "mu", "z_mu", "z_ordinal"])
    parser.add_argument("--n-mc",   type=int, default=5000)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--overround", type=float, default=1.0,
                        help="Bookmaker margin as a multiplier on probabilities "
                             "(1.00 = fair odds, 1.20 = 20% overround). "
                             "When >1, posted odds columns are shown.")
    args = parser.parse_args()
    main(
        year=args.year, round_=args.round_, field_csv=args.field,
        use_latest=args.latest, tau=args.tau, mode=args.mode,
        n_mc=args.n_mc, seed=args.seed, overround=args.overround,
    )
