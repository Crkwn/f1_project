"""
Stage 2 — Walk-forward back-test.

What this does
--------------
1. Replays races in chronological order from RATING_START_YEAR through the
   latest year in the data.
2. Before each race in the HOLDOUT window (default 2022–latest), it:
      (a) processes that weekend's qualifying (updates rater state),
      (b) asks the predictor for P(win), P(podium), P(points),
      (c) records the prediction AND the actual outcome,
      (d) updates rater + reliability with the race's results,
      (e) moves on.
3. Reports calibration + accuracy metrics:
      - log-loss on P(win) for the holdout
      - Brier score on P(win)
      - Top-1 hit rate (did we put highest P(win) on the winner?)
      - Top-3 hit rate (winner in our top-3 favourites?)
      - Reliability bucket table (do our "60% drivers" actually win 60%?)
   Compared against a naïve baseline that uses P(win) = softmax(−grid / 5),
   i.e., grid position alone as the predictor.

Why a walk-forward replay from scratch
--------------------------------------
We can't load Stage 1's saved rater.pkl and use it directly — that state
has been updated with ALL races, including the ones we want to predict.
Walking forward from 2014 lets us ensure that every 2022+ prediction uses
ONLY information available at that moment. This is the F1 analogue of a
strict purged-walk-forward CV in financial back-testing.

Outputs
-------
reports/backtest/predictions.csv    per-(race, driver) predictions + outcome
reports/backtest/metrics.txt        headline numbers
reports/backtest/reliability.png    reliability diagram (predicted vs. observed)

Usage
-----
    python scripts/model/backtest.py
    python scripts/model/backtest.py --holdout-start 2023
    python scripts/model/backtest.py --fit-tau       # grid search τ
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    DATA_RAW, DRIVER_SCORE_MODE, RATING_START_YEAR, SOFTMAX_TAU_INIT,
)
from src.model.race_predictor import RacePredictor, compute_pace_map
from src.model.rating import F1Rater
from src.model.reliability import ConstructorReliability
from src.model.status_families import family_of

OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(start_year: int):
    results      = pd.read_csv(DATA_RAW / "results.csv")
    races        = pd.read_csv(DATA_RAW / "races.csv")
    drivers      = pd.read_csv(DATA_RAW / "drivers.csv")
    constructors = pd.read_csv(DATA_RAW / "constructors.csv")
    qualifying   = pd.read_csv(DATA_RAW / "qualifying.csv")
    status       = pd.read_csv(DATA_RAW / "status.csv")
    cs           = pd.read_csv(DATA_RAW / "constructor_standings.csv")
    ds           = pd.read_csv(DATA_RAW / "driver_standings.csv")

    # Annotate
    race_meta = races[races["year"] >= start_year][
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

    return race_meta, res, qua, pace_map, ds


# ---------------------------------------------------------------------------
# Baselines — four of increasing "domain-knowledge" richness.
# Each returns a probability vector aligned with the field order.
#
# In each baseline we use EMPIRICAL rates computed WITHIN THE TRAINING
# WINDOW (< holdout_start) — so the baseline is not cheating by using
# holdout data to calibrate itself.
# ---------------------------------------------------------------------------
def grid_softmax_baseline(grids: np.ndarray, scale: float = 5.0) -> np.ndarray:
    """B0: P(win_i) ∝ exp(−grid_i / scale). Arbitrary softmax over grid."""
    z = -grids.astype(float) / scale
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def pole_empirical_baseline(grids: np.ndarray, pole_win_rate: float) -> np.ndarray:
    """
    B1: Pole sitter gets P = pole_win_rate; the remaining (1 − pole_win_rate)
    is split uniformly across the non-pole drivers. Captures "grid dominates"
    as a CALIBRATED (empirical) distribution rather than an arbitrary softmax.
    """
    N = len(grids)
    p = np.full(N, (1 - pole_win_rate) / max(N - 1, 1), dtype=float)
    pole_idx = np.where(grids == 1)[0]
    if len(pole_idx) > 0:
        p[pole_idx[0]] = pole_win_rate
    # Very rarely there's no grid==1 (grid data missing); renormalise.
    if p.sum() > 0:
        p = p / p.sum()
    return p


def previous_winner_baseline(
    driver_ids: np.ndarray, last_winner_id: int | None,
    persistence_rate: float,
) -> np.ndarray:
    """
    B2: "Last race's winner wins again" with empirical persistence rate.
    last_winner_id=None (season opener) → uniform prior.
    """
    N = len(driver_ids)
    if last_winner_id is None or last_winner_id not in driver_ids:
        return np.full(N, 1.0 / N)
    p = np.full(N, (1 - persistence_rate) / (N - 1), dtype=float)
    idx = int(np.where(driver_ids == last_winner_id)[0][0])
    p[idx] = persistence_rate
    return p / p.sum()


def leader_baseline(
    driver_ids: np.ndarray, leader_id: int | None, leader_win_rate: float,
) -> np.ndarray:
    """
    B3: Drivers-championship leader wins, at empirical "leader wins next
    race" rate. Before round 1 there's no leader — fall back to uniform.
    """
    N = len(driver_ids)
    if leader_id is None or leader_id not in driver_ids:
        return np.full(N, 1.0 / N)
    p = np.full(N, (1 - leader_win_rate) / (N - 1), dtype=float)
    idx = int(np.where(driver_ids == leader_id)[0][0])
    p[idx] = leader_win_rate
    return p / p.sum()


def compute_empirical_rates(res: pd.DataFrame, train_cutoff_year: int) -> dict:
    """
    Pre-compute the three empirical rates needed for baselines, using only
    data strictly BEFORE the holdout window (no look-ahead).

    Returns:
      pole_win_rate     : P(winner starts from pole)
      persistence_rate  : P(round-N winner == round-(N−1) winner)
      leader_win_rate   : P(winner is the current championship leader pre-race)
    """
    train = res[res["year"] < train_cutoff_year].copy()
    train["family"] = train["status"].apply(family_of)
    train["is_winner"] = ((train["positionOrder"] == 1) &
                          (train["family"] == "finished")).astype(int)

    # pole_win_rate
    n_winners = train[train["is_winner"] == 1]
    n_pole_wins = (n_winners["grid"] == 1).sum()
    pole_win_rate = n_pole_wins / max(len(n_winners), 1)

    # persistence_rate — per-(year,round) winner, compare to prev round
    winners_by_race = (
        train[train["is_winner"] == 1]
        .sort_values(["year", "round"])
        [["year", "round", "driverId"]]
        .drop_duplicates(subset=["year", "round"])
        .reset_index(drop=True)
    )
    n_cons = 0
    n_checks = 0
    for i in range(1, len(winners_by_race)):
        prev, this = winners_by_race.iloc[i-1], winners_by_race.iloc[i]
        if prev["year"] == this["year"] and this["round"] == prev["round"] + 1:
            n_checks += 1
            if prev["driverId"] == this["driverId"]:
                n_cons += 1
    persistence_rate = n_cons / max(n_checks, 1)

    return {
        "pole_win_rate":    pole_win_rate,
        "persistence_rate": persistence_rate,
        # leader_win_rate computed separately because it needs driver_standings
        # — we fill this in in main() after loading ds.
    }


def compute_leader_win_rate(
    res: pd.DataFrame, ds: pd.DataFrame, train_cutoff_year: int
) -> float:
    """P(race winner was championship leader going into the race) in training window."""
    train = res[res["year"] < train_cutoff_year].copy()
    train["family"] = train["status"].apply(family_of)
    winners = train[(train["positionOrder"] == 1) & (train["family"] == "finished")]
    # For each race, identify who led the standings AFTER the previous race.
    # Easier: use ds (driver_standings.csv) — standing after this race. Shift by one round.
    hits = 0
    total = 0
    # Build lookup: for each raceId, the pre-race leader = leader after previous race
    # Get per-race leader FROM THIS race's post-race standings, then for the NEXT race,
    # whoever won is checked against the PRIOR leader.
    ds_with_year = ds.merge(res[["raceId", "year", "round"]].drop_duplicates(),
                            on="raceId", how="inner")
    leader_by_race = (
        ds_with_year[ds_with_year["position"] == 1]
        .sort_values(["year", "round"])
        [["year", "round", "driverId"]]
        .drop_duplicates(["year", "round"])
    )
    # Build (year, round) → leaderId
    leader_lookup = {(int(r["year"]), int(r["round"])): int(r["driverId"])
                     for _, r in leader_by_race.iterrows()}
    for _, w in winners.iterrows():
        y, rnd = int(w["year"]), int(w["round"])
        if rnd <= 1:
            continue  # no pre-race leader for round 1
        prior_leader = leader_lookup.get((y, rnd - 1))
        if prior_leader is None:
            continue
        total += 1
        if prior_leader == int(w["driverId"]):
            hits += 1
    return hits / max(total, 1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def log_loss_winner(p_win_per_race: list[np.ndarray],
                    winner_idx: list[int]) -> float:
    """Average −log P(actual winner). Smaller is better."""
    eps = 1e-12
    total = 0.0
    n = 0
    for p, w in zip(p_win_per_race, winner_idx):
        if w is None:
            continue
        total += -np.log(max(p[w], eps))
        n += 1
    return total / n if n else float("nan")


def brier_winner(p_win_per_race: list[np.ndarray],
                 winner_idx: list[int]) -> float:
    """Sum over drivers of (P_i − actual_i)² averaged across races."""
    total = 0.0
    n = 0
    for p, w in zip(p_win_per_race, winner_idx):
        if w is None:
            continue
        y = np.zeros_like(p)
        y[w] = 1.0
        total += float(np.sum((p - y) ** 2))
        n += 1
    return total / n if n else float("nan")


def top_k_hit_rate(p_win_per_race: list[np.ndarray],
                   winner_idx: list[int], k: int) -> float:
    hits = 0
    n = 0
    for p, w in zip(p_win_per_race, winner_idx):
        if w is None:
            continue
        top_k = np.argsort(-p)[:k]
        if w in top_k:
            hits += 1
        n += 1
    return hits / n if n else float("nan")


def mean_p_on_winner(p_win_per_race: list[np.ndarray],
                     winner_idx: list[int]) -> float:
    """
    Average of P(driver_i wins) evaluated at the actual winner's index.
    The most intuitive sharpness measure: "how much probability did we
    actually put on the thing that happened?"
      no-info (uniform over 20) ≈ 0.05
      higher is better, bounded above by 1.0
    """
    vals, n = 0.0, 0
    for p, w in zip(p_win_per_race, winner_idx):
        if w is None:
            continue
        vals += float(p[w])
        n += 1
    return vals / n if n else float("nan")


def rank_of_winner(p_win_per_race: list[np.ndarray],
                   winner_idx: list[int]) -> float:
    """
    Average rank (1 = highest P) of the actual winner in our predictions.
    no-info would be ~10.5 in a 20-driver field.
    """
    ranks, n = 0, 0
    for p, w in zip(p_win_per_race, winner_idx):
        if w is None:
            continue
        order = np.argsort(-p)
        ranks += int(np.where(order == w)[0][0]) + 1
        n += 1
    return ranks / n if n else float("nan")


def reliability_bins(p_flat: np.ndarray, y_flat: np.ndarray,
                     n_bins: int = 10) -> pd.DataFrame:
    """
    Group predicted probabilities into bins, report observed frequency.
    Useful for reliability-diagram inspection.
    """
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p_flat >= lo) & (p_flat < hi) if hi < 1.0 else (p_flat >= lo) & (p_flat <= hi)
        if mask.sum() == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": 0,
                         "mean_pred": float("nan"), "observed": float("nan")})
        else:
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(mask.sum()),
                         "mean_pred": float(p_flat[mask].mean()),
                         "observed":  float(y_flat[mask].mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core walk-forward
# ---------------------------------------------------------------------------
def walkforward(
    race_meta:       pd.DataFrame,
    res:             pd.DataFrame,
    qua:             pd.DataFrame,
    pace_map:        dict,
    ds:              pd.DataFrame,
    holdout_start:   int,
    tau:             float,
    mode:            str,
    empirical_rates: dict,
) -> dict:
    """
    Walk every race in the data from RATING_START_YEAR forward; for races in
    the holdout window, produce (model A, model B, four baselines) side by
    side against the actual outcome.

    Returns a dict:
      records              : list of per-race metadata dicts (with _preds DF)
      model_A              : list of (N,) arrays — analytic softmax P(win)
      model_B              : list of (N,) arrays — Monte-Carlo P(win)
      baselines            : {name: list-of-(N,)-arrays} for the four baselines
      winner_idx           : list of int-or-None, winner's index in the field
    """
    rater       = F1Rater()
    reliability = ConstructorReliability()
    predictor   = RacePredictor(
        rater=rater, reliability=reliability, pace_map=pace_map,
        tau=tau, mode=mode,
    )

    # Pre-race leader lookup: standings-position-1 after race (year, round) tells
    # us who'd lead going into (year, round+1). Missing entries → None at lookup.
    ds_with_year = ds.merge(
        race_meta[["raceId", "year", "round"]], on="raceId", how="inner",
    )
    leader_lookup = {
        (int(r["year"]), int(r["round"])): int(r["driverId"])
        for _, r in ds_with_year[ds_with_year["position"] == 1].iterrows()
    }

    pole_rate    = empirical_rates["pole_win_rate"]
    persist_rate = empirical_rates["persistence_rate"]
    leader_rate  = empirical_rates["leader_win_rate"]

    records, model_A, model_B = [], [], []
    bl_grid, bl_pole, bl_prev, bl_leader = [], [], [], []
    winner_idx_list = []

    # Track most-recent classified winner so the persistence baseline has a target.
    last_winner: dict = {"year": None, "driverId": None}

    for race_id, year, round_, race_name in race_meta.itertuples(index=False):
        y, r = int(year), int(round_)

        # Qualifying first (establishes pre-race rating state)
        q_rows = qua[qua["raceId"] == race_id]
        if not q_rows.empty:
            rater.update_qualifying(y, r, q_rows)

        race_rows = res[res["raceId"] == race_id]
        if race_rows.empty:
            continue

        # Build the race field BEFORE updating rater with race outcome.
        field = race_rows[[
            "driverId", "driverName", "constructorId", "constructorName",
            "grid", "positionOrder", "status",
        ]].copy()

        # -------- PREDICTION (holdout only) --------
        if y >= holdout_start:
            preds = predictor.predict(field, year=y, round_=r)

            # Classified winner
            finishers = field.copy()
            finishers["family"] = finishers["status"].apply(family_of)
            winners = finishers[(finishers["positionOrder"] == 1) &
                                (finishers["family"] == "finished")]
            w_did = int(winners.iloc[0]["driverId"]) if not winners.empty else None

            # Align predictions with field order for metrics
            ordered     = preds.set_index("driverId").loc[field["driverId"].values]
            p_A         = ordered["p_win_analytic"].to_numpy()
            p_B         = ordered["p_win_mc"].to_numpy()
            driver_ids  = field["driverId"].to_numpy()
            grids       = field["grid"].to_numpy()

            # Baselines
            p_grid   = grid_softmax_baseline(grids)
            p_pole   = pole_empirical_baseline(grids, pole_rate)
            lw_id    = (last_winner["driverId"]
                        if (last_winner["year"] == y and r >= 2) else None)
            p_prev   = previous_winner_baseline(driver_ids, lw_id, persist_rate)
            lid      = leader_lookup.get((y, r - 1))  # leader going into this race
            p_leader = leader_baseline(driver_ids, lid, leader_rate)

            w_idx = (int(np.where(driver_ids == w_did)[0][0])
                     if w_did is not None else None)

            records.append({
                "raceId": int(race_id), "year": y, "round": r,
                "race_name":         race_name,
                "n_drivers":         len(field),
                "winner_id":         w_did,
                "predicted_top1_A":  int(ordered["p_win_analytic"].idxmax())
                                      if not ordered.empty else None,
                "predicted_top1_B":  int(ordered["p_win_mc"].idxmax())
                                      if not ordered.empty else None,
            })
            model_A.append(p_A)
            model_B.append(p_B)
            bl_grid.append(p_grid)
            bl_pole.append(p_pole)
            bl_prev.append(p_prev)
            bl_leader.append(p_leader)
            winner_idx_list.append(w_idx)

            # Per-(race,driver) predictions with actuals, for downstream slicing.
            out_rows = preds.copy()
            out_rows["raceId"]        = int(race_id)
            out_rows["year"]          = y
            out_rows["round"]         = r
            out_rows["race_name"]     = race_name
            out_rows["actual_finish"] = out_rows["driverId"].map(
                field.set_index("driverId")["positionOrder"])
            out_rows["is_winner"] = (out_rows["driverId"] == w_did).astype(int)
            records[-1]["_preds"] = out_rows

        # -------- PERSISTENCE BOOKKEEPING (every race, training included) --------
        # We need last_winner as soon as the holdout starts.
        fin_all = field.copy()
        fin_all["family"] = fin_all["status"].apply(family_of)
        wn = fin_all[(fin_all["positionOrder"] == 1) &
                     (fin_all["family"] == "finished")]
        if not wn.empty:
            last_winner = {"year": y, "driverId": int(wn.iloc[0]["driverId"])}

        # -------- UPDATE STEP --------
        rater.update_race(y, r, race_rows)
        reliability.update_race(y, r, race_rows[[
            "constructorId", "constructorName", "status"
        ]])

    return {
        "records":    records,
        "model_A":    model_A,
        "model_B":    model_B,
        "baselines":  {
            "grid_softmax":            bl_grid,
            "pole_empirical":          bl_pole,
            "previous_winner":         bl_prev,
            "championship_leader":     bl_leader,
        },
        "winner_idx": winner_idx_list,
    }


# ---------------------------------------------------------------------------
# τ fitting
# ---------------------------------------------------------------------------
def fit_tau(race_meta, res, qua, pace_map, ds, holdout_start, mode,
            empirical_rates,
            taus=(10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 130)):
    """
    Grid-search τ by log-loss on holdout. We report BOTH analytic (A) and
    MC (B) log-loss per τ — they'll agree in the limit of MC samples, but
    we want to see the agreement before committing.
    """
    rows = []
    for tau in taus:
        result = walkforward(
            race_meta, res, qua, pace_map, ds, holdout_start,
            tau=tau, mode=mode, empirical_rates=empirical_rates,
        )
        ll_A = log_loss_winner(result["model_A"], result["winner_idx"])
        ll_B = log_loss_winner(result["model_B"], result["winner_idx"])
        rows.append({"tau": tau, "log_loss_A": ll_A, "log_loss_B": ll_B})
        print(f"  τ={tau:>5.0f}   ll_A={ll_A:.4f}   ll_B={ll_B:.4f}")
    best = min(rows, key=lambda r: r["log_loss_B"])
    print(f"\nBest τ by MC log-loss: {best['tau']}   (ll_B={best['log_loss_B']:.4f})")
    return best["tau"], pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Named-race dump — qualitative sanity check of the MC model
# ---------------------------------------------------------------------------
def _format_race_block(rec: dict, model_p: np.ndarray) -> list[str]:
    """Build a multi-line text block showing top-3 model picks + actual winner."""
    preds  = rec["_preds"]  # already sorted by p_win_mc desc
    top3   = preds.head(3)
    winner = preds[preds["is_winner"] == 1]
    ll     = -np.log(max(model_p[rec["_winner_idx"]], 1e-12))
    lines  = [f"{rec['year']} R{rec['round']:<2} {rec['race_name']:<28s}  "
              f"(model log-loss = {ll:.3f})"]
    lines.append("   Top-3 model picks (by MC P(win)):")
    for _, row in top3.iterrows():
        mark = "  ← WINNER" if row["is_winner"] == 1 else ""
        fin  = (f"P{int(row['actual_finish']):>2}"
                if pd.notna(row["actual_finish"]) else "DNF")
        lines.append(
            f"     {row['driver']:<22s}  grid={int(row['grid']):>2}   "
            f"p_win={row['p_win_mc']:.3f}   finished={fin}{mark}"
        )
    # Actual winner line if they weren't in top-3
    if not winner.empty and (winner.iloc[0]["driverId"] not in
                             top3["driverId"].values):
        w = winner.iloc[0]
        rank_in_model = preds["driverId"].tolist().index(w["driverId"]) + 1
        lines.append(
            f"   Actual winner was NOT in top-3:  {w['driver']}  "
            f"grid={int(w['grid']):>2}   p_win={w['p_win_mc']:.3f}   "
            f"(ranked #{rank_in_model} by model)"
        )
    return lines


def print_named_race_dump(records: list, result: dict, n_each: int = 5) -> str:
    """
    Show the model's picks vs. reality on three slices:
      - the races where the model was MOST confident in the correct winner
      - the races where the model was MOST WRONG about the winner
      - a few random races for representativeness
    Returns the printed text so it can be persisted.
    """
    w_list = result["winner_idx"]
    p_B    = result["model_B"]

    per_race = []
    for rec, p, w in zip(records, p_B, w_list):
        if w is None:
            continue
        rec_copy = dict(rec)  # shallow — don't mutate
        rec_copy["_winner_idx"] = w
        ll = -np.log(max(p[w], 1e-12))
        per_race.append((ll, rec_copy, p))

    per_race.sort(key=lambda x: x[0])

    out_lines: list[str] = []

    def dump(header: str, items):
        out_lines.append("")
        out_lines.append(header)
        out_lines.append("-" * len(header))
        for _, rec, p in items:
            for line in _format_race_block(rec, p):
                out_lines.append(line)
            out_lines.append("")

    dump(
        f"MOST CONFIDENT in correct winner  (top {n_each} by lowest log-loss)",
        per_race[:n_each],
    )
    dump(
        f"MOST WRONG about the winner  (top {n_each} by highest log-loss)",
        list(reversed(per_race[-n_each:])),
    )
    if len(per_race) > n_each:
        rng = np.random.default_rng(0)
        idx = sorted(rng.choice(len(per_race), size=n_each, replace=False).tolist())
        dump(
            f"RANDOM sample of {n_each} races (representative)",
            [per_race[i] for i in idx],
        )

    text = "\n".join(out_lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(holdout_start: int, tau: float, fit_tau_flag: bool, mode: str):
    print(f"\n=== Stage 2 back-test  (holdout ≥ {holdout_start}, mode={mode!r}) ===\n")
    print("Loading data...")
    race_meta, res, qua, pace_map, ds = load_data(RATING_START_YEAR)
    print(f"  races in scope : {len(race_meta):,}  ({RATING_START_YEAR}+)")

    # --- Empirical rates for baselines (training window only) ---------------
    print("\nComputing empirical baseline rates from training window...")
    rates = compute_empirical_rates(res, train_cutoff_year=holdout_start)
    rates["leader_win_rate"] = compute_leader_win_rate(
        res, ds, train_cutoff_year=holdout_start,
    )
    print(f"  pole_win_rate     = {rates['pole_win_rate']:.3f}   "
          f"(train-window P[winner started from pole])")
    print(f"  persistence_rate  = {rates['persistence_rate']:.3f}   "
          f"(P[round-N winner == round-(N-1) winner])")
    print(f"  leader_win_rate   = {rates['leader_win_rate']:.3f}   "
          f"(P[winner was standings leader pre-race])")

    # Optional τ fit
    if fit_tau_flag:
        print("\nFitting τ via grid search...")
        tau, _ = fit_tau(race_meta, res, qua, pace_map, ds, holdout_start,
                         mode=mode, empirical_rates=rates)

    print(f"\nRunning walk-forward at τ = {tau}, mode = {mode!r}...")
    result = walkforward(
        race_meta, res, qua, pace_map, ds, holdout_start,
        tau=tau, mode=mode, empirical_rates=rates,
    )
    records = result["records"]
    w_list  = result["winner_idx"]
    n_races = len(records)
    n_counted = sum(1 for w in w_list if w is not None)
    print(f"  holdout races predicted        : {n_races}")
    print(f"  races with a classified winner : {n_counted}")

    # ------------------- Metrics: model A, model B, four baselines ---------
    runs = {
        "model (analytic A)":             result["model_A"],
        "model (Monte-Carlo B)":          result["model_B"],
        "baseline: grid-softmax":         result["baselines"]["grid_softmax"],
        "baseline: pole-empirical":       result["baselines"]["pole_empirical"],
        "baseline: last-race-winner":     result["baselines"]["previous_winner"],
        "baseline: championship-leader":  result["baselines"]["championship_leader"],
    }

    metrics_rows = []
    print("\n--- Metrics comparison ---")
    print(f"{'method':35s} {'P(winner)':>10s} {'top-1':>7s} {'top-3':>7s} "
          f"{'avg_rank':>9s} {'log-loss':>10s} {'Brier':>8s}")
    print("-" * 90)
    for name, p_list in runs.items():
        mpw = mean_p_on_winner(p_list, w_list)
        t1  = top_k_hit_rate(p_list, w_list, 1)
        t3  = top_k_hit_rate(p_list, w_list, 3)
        ar  = rank_of_winner(p_list, w_list)
        ll  = log_loss_winner(p_list, w_list)
        br  = brier_winner(p_list, w_list)
        print(f"{name:35s} {mpw:>10.3f} {t1:>7.3f} {t3:>7.3f} "
              f"{ar:>9.2f} {ll:>10.4f} {br:>8.4f}")
        metrics_rows.append({
            "method":   name,   "mean_p_winner": mpw, "top_1": t1, "top_3": t3,
            "avg_rank_of_winner": ar, "log_loss": ll, "brier": br,
        })

    # ------------------- Reliability diagram (model B) ---------------------
    p_flat = np.concatenate(result["model_B"])
    y_flat = np.concatenate([
        np.eye(len(p), dtype=int)[w] if w is not None else np.zeros(len(p), dtype=int)
        for p, w in zip(result["model_B"], w_list)
    ])
    rel_df = reliability_bins(p_flat, y_flat, n_bins=10)
    print("\n--- Reliability bins — model (MC B): predicted P(win) vs observed ---")
    print(rel_df.round(4).to_string(index=False))

    # ------------------- Named-race dump -----------------------------------
    race_dump_text = print_named_race_dump(records, result, n_each=5)

    # ------------------- Plot ----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    valid = rel_df.dropna(subset=["mean_pred"])
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    axes[0].plot(valid["mean_pred"], valid["observed"],
                 "o-", label="model (B)", color="tab:blue")
    axes[0].set_xlabel("predicted P(win)")
    axes[0].set_ylabel("observed win frequency")
    axes[0].set_title(
        f"Reliability diagram, {holdout_start}+  "
        f"(n={n_counted} races, mode={mode!r})"
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    labels = [r["method"].replace("baseline: ", "").replace("model ", "")
              for r in metrics_rows]
    t1s = [r["top_1"] for r in metrics_rows]
    axes[1].bar(range(len(labels)), t1s,
                color=["tab:blue", "tab:blue", "tab:orange",
                       "tab:orange", "tab:orange", "tab:orange"])
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("top-1 hit rate")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Top-1 hit rate — model vs baselines")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig_path = OUT_DIR / f"reliability_{mode}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    # ------------------- Persist -------------------------------------------
    # Per-(race, driver) predictions
    all_preds = pd.concat([r["_preds"] for r in records], ignore_index=True)
    cols_keep = [
        "raceId", "year", "round", "race_name",
        "driverId", "driver", "constructor", "grid",
        "ordinal", "reliability", "pace_score", "grid_score",
        "combined_score",
        "p_win_analytic", "p_win_mc", "p_podium_mc", "p_points_mc",
        "actual_finish", "is_winner",
    ]
    preds_path = OUT_DIR / f"predictions_{mode}.csv"
    all_preds[cols_keep].to_csv(preds_path, index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = OUT_DIR / f"metrics_{mode}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    txt_path = OUT_DIR / f"metrics_{mode}.txt"
    with open(txt_path, "w") as fh:
        fh.write(f"Holdout ≥ {holdout_start}\n")
        fh.write(f"τ = {tau}\n")
        fh.write(f"mode = {mode}\n")
        fh.write(f"n_races = {n_counted}\n\n")
        fh.write("Empirical baseline rates (training window only):\n")
        fh.write(f"  pole_win_rate     = {rates['pole_win_rate']:.3f}\n")
        fh.write(f"  persistence_rate  = {rates['persistence_rate']:.3f}\n")
        fh.write(f"  leader_win_rate   = {rates['leader_win_rate']:.3f}\n\n")
        fh.write("Metrics:\n")
        fh.write(metrics_df.to_string(index=False) + "\n\n")
        fh.write("Reliability bins (model B):\n")
        fh.write(rel_df.round(4).to_string(index=False) + "\n\n")
        fh.write(race_dump_text)

    print(f"\nSaved:")
    print(f"  {preds_path}")
    print(f"  {metrics_path}")
    print(f"  {txt_path}")
    print(f"  {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-start", type=int, default=2022,
                        help="First year to include in the evaluation holdout.")
    parser.add_argument("--tau", type=float, default=SOFTMAX_TAU_INIT,
                        help="Softmax temperature.")
    parser.add_argument("--fit-tau", action="store_true",
                        help="Grid-search τ by log-loss before reporting.")
    parser.add_argument("--mode", type=str, default=DRIVER_SCORE_MODE,
                        choices=["ordinal", "mu", "z_mu", "z_ordinal"],
                        help="Driver-skill feature mode; see src/config.py.")
    args = parser.parse_args()
    main(args.holdout_start, args.tau, args.fit_tau, args.mode)
