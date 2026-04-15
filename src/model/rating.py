"""
Stage 1a — Driver rating engine.

Goal
----
Produce a per-driver skill estimate (μ, σ) that can be used standalone
(ranking table) or fed into Stage 2 (race-level win probabilities). The
estimate has to be defensible on three fronts:

  1. Handles race-level noise properly.  OpenSkill's Plackett-Luce (PL)
     update takes the full finishing order per race in one pass. That's
     strictly more information than pairwise Elo updates, and the
     uncertainty bookkeeping (σ) falls out of the model rather than
     being bolted on.

  2. Avoids survivorship bias.  Mechanical DNFs (engine, gearbox,
     hydraulics, …) are car-attributable and get CENSORED from the
     rating — we do not let the driver absorb a "bad race" signal for
     their engine exploding. Accident DNFs are kept at reduced weight
     (part driver, part situational). Driver-side DNFs and DQs are
     also censored. This is the F1 analogue of the standard PD/LGD
     decomposition in credit: separate the "would I survive" question
     from the "how do I perform conditional on surviving" question,
     and recombine downstream.

  3. Widens uncertainty when our priors deserve to widen.  Two
     structural situations make us genuinely less sure of a driver's
     current skill:
       * team switch  — new car, new garage, new set-up team
       * long inactivity — skill may have drifted (EDA Q16)
     We bump σ upward (in variance space) at those transitions, so
     the next race has a larger-than-normal update.

What this file does NOT do
--------------------------
- It does not estimate CAR reliability. That's Stage 1b
  (`reliability.py`), and it's exactly why we censor mechanical DNFs
  here — so Stage 1b can own that signal end-to-end.
- It does not produce win probabilities. Stage 2 combines driver μ,σ
  with car reliability and race context to output P(win | field).
- The qualifying update in v1 uses QUALI_UPDATE_WEIGHT = 1.0. We start
  at parity with race and will re-tune from back-testing.

Inputs expected (Ergast schema)
-------------------------------
results.csv    : per (raceId, driverId), with constructorId, positionOrder,
                 statusId — statusId resolves to a text status that
                 status_families.family_of() classifies into one of six
                 families.
qualifying.csv : per (raceId, driverId), with constructorId, position.
races.csv      : maps raceId → year, round (chronological order key).
drivers.csv    : maps driverId → forename, surname.
status.csv     : maps statusId → status string.

Outputs
-------
F1Rater.snapshot()    : current (μ, σ, ordinal, metadata) per driver
F1Rater.history_df()  : per-weekend (μ, σ) snapshot per driver — for
                         trajectory plots and back-test diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from openskill.models import PlackettLuce

from src.config import (
    QUALI_UPDATE_WEIGHT,
    RATING_INITIAL_MU,
    RATING_INITIAL_SIGMA,
    RATING_START_YEAR,
    SIGMA_BUMP_PER_YEAR_INACTIVE,
    SIGMA_BUMP_TEAM_SWITCH,
    SIGMA_FLOOR,
)
from src.model.status_families import family_of, update_weight


# ---------------------------------------------------------------------------
# Per-driver state
# ---------------------------------------------------------------------------
@dataclass
class DriverState:
    """Rolling state for a single driver across their career window.

    μ, σ are the current OpenSkill-PL rating. The `last_*` fields are
    how we detect team switches and inactivity gaps on the next update.
    """
    driver_id: int
    driver_name: str
    mu: float = RATING_INITIAL_MU
    sigma: float = RATING_INITIAL_SIGMA
    last_year: Optional[int] = None
    last_round: Optional[int] = None
    last_constructor: Optional[int] = None
    n_races: int = 0
    n_qualis: int = 0


# ---------------------------------------------------------------------------
# Rater
# ---------------------------------------------------------------------------
class F1Rater:
    """
    OpenSkill-PL wrapper with F1-specific corrections.

    Usage
    -----
        rater = F1Rater()
        for each weekend in chronological order:
            rater.update_qualifying(year, round, quali_df)
            rater.update_race(year, round, race_df)
        ratings = rater.snapshot()
        history = rater.history_df()
    """

    def __init__(self) -> None:
        self.model = PlackettLuce()
        self.drivers: dict[int, DriverState] = {}
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get_or_create(self, driver_id: int, driver_name: str) -> DriverState:
        if driver_id not in self.drivers:
            self.drivers[driver_id] = DriverState(
                driver_id=driver_id, driver_name=driver_name
            )
        return self.drivers[driver_id]

    def _to_rating(self, state: DriverState):
        """Hand OpenSkill a fresh Rating object carrying this driver's μ, σ."""
        return self.model.rating(
            mu=state.mu, sigma=state.sigma, name=str(state.driver_id)
        )

    def _apply_structural_inflation(
        self, state: DriverState, year: int, constructor_id: int
    ) -> None:
        """
        Pre-update σ bump, variance-additive (σ² ← σ² + bump²).

        Fires at most once per weekend per driver because we write back
        `last_year`/`last_constructor` after every update: a quali update
        that triggers the bump will leave last_constructor = this constructor,
        so the race update later that weekend sees no change and skips.

          * team switch    → SIGMA_BUMP_TEAM_SWITCH   (first race at new team)
          * year gap > 1   → SIGMA_BUMP_PER_YEAR_INACTIVE per missed season
                              (a driver who raced last year and this year
                              has gap=1 → no bump. Gap=2 means they skipped
                              a full season → 1 missed year → small bump.)
        """
        var_bump = 0.0

        # Team switch — only counts after the driver has raced at least once
        if (
            state.last_constructor is not None
            and constructor_id != state.last_constructor
        ):
            var_bump += SIGMA_BUMP_TEAM_SWITCH ** 2

        # Inactivity — year gap strictly greater than 1 means seasons missed
        if state.last_year is not None:
            gap = year - state.last_year
            if gap > 1:
                missed = gap - 1
                var_bump += (SIGMA_BUMP_PER_YEAR_INACTIVE ** 2) * missed

        if var_bump > 0:
            state.sigma = math.sqrt(state.sigma ** 2 + var_bump)

    def _write_back(self, state: DriverState, new_rating) -> None:
        """Store new μ, σ — apply SIGMA_FLOOR to prevent σ → 0."""
        state.mu = float(new_rating.mu)
        state.sigma = max(float(new_rating.sigma), SIGMA_FLOOR)

    def _record(
        self, state: DriverState, year: int, round_: int, event: str
    ) -> None:
        self.history.append({
            "year":     year,
            "round":    round_,
            "driverId": state.driver_id,
            "driver":   state.driver_name,
            "mu":       state.mu,
            "sigma":    state.sigma,
            "event":    event,
        })

    # ------------------------------------------------------------------
    # Race update
    # ------------------------------------------------------------------
    def update_race(self, year: int, round_: int, race_df: pd.DataFrame) -> None:
        """
        Apply one race's update to all participating drivers.

        race_df required columns:
          driverId, driverName, constructorId, positionOrder, status

        Filtering:
          - status → family → update_weight.
          - weight == 0 rows are DROPPED from rate() entirely (censored).
          - weight in (0, 1) rows are kept with per-team weight scaling.
          - weight == 1 rows are normal-weight finishers.
        """
        if race_df.empty:
            return

        df = race_df.copy()
        df["family"] = df["status"].apply(family_of)
        df["w"] = df["family"].apply(update_weight)

        active = df[df["w"] > 0].copy()
        if len(active) < 2:
            return  # can't run a PL update with <2 participants

        # Ergast's positionOrder is 1..N with DNFs after finishers, ordered by
        # laps completed — exactly what we need as an OpenSkill rank vector.
        active = active.sort_values("positionOrder").reset_index(drop=True)

        # Build participant tuples, applying pre-update inflation as we go.
        participants: list[tuple[DriverState, float, int]] = []
        for _, row in active.iterrows():
            state = self._get_or_create(int(row["driverId"]), row["driverName"])
            cid = int(row["constructorId"])
            self._apply_structural_inflation(state, year, cid)
            participants.append((state, float(row["w"]), cid))

        teams   = [[self._to_rating(st)] for st, _, _ in participants]
        ranks   = list(range(1, len(participants) + 1))  # rank parallels sorted df
        weights = [[w] for _, w, _ in participants]

        new_teams = self.model.rate(teams, ranks=ranks, weights=weights)

        for (state, _, cid), new_team in zip(participants, new_teams):
            self._write_back(state, new_team[0])
            state.last_year        = year
            state.last_round       = round_
            state.last_constructor = cid
            state.n_races         += 1
            self._record(state, year, round_, event="race")

    # ------------------------------------------------------------------
    # Qualifying update
    # ------------------------------------------------------------------
    def update_qualifying(
        self,
        year: int,
        round_: int,
        quali_df: pd.DataFrame,
        weight: float = QUALI_UPDATE_WEIGHT,
    ) -> None:
        """
        Apply a qualifying weekend's update.

        quali_df required columns:
          driverId, driverName, constructorId, position

        Quali is a cleaner skill signal than race outcomes (no strategy,
        no safety cars, no tyre management), so we include it as its own
        update. Drivers who failed to set a time (position NaN) are
        dropped — no information to update on.

        Structural σ inflation is applied here as well — on weekends
        where quali runs first, this is where the team-switch / inactivity
        bump fires, and the race update later that weekend will be a
        no-op for inflation (last_constructor / last_year are now up to
        date).
        """
        if quali_df.empty:
            return

        active = quali_df.dropna(subset=["position"]).copy()
        if len(active) < 2:
            return
        active = active.sort_values("position").reset_index(drop=True)

        participants: list[tuple[DriverState, int]] = []
        for _, row in active.iterrows():
            state = self._get_or_create(int(row["driverId"]), row["driverName"])
            cid = int(row["constructorId"])
            self._apply_structural_inflation(state, year, cid)
            participants.append((state, cid))

        teams   = [[self._to_rating(st)] for st, _ in participants]
        ranks   = list(range(1, len(participants) + 1))
        weights = [[weight] for _ in participants]

        new_teams = self.model.rate(teams, ranks=ranks, weights=weights)

        for (state, cid), new_team in zip(participants, new_teams):
            self._write_back(state, new_team[0])
            state.last_year        = year
            state.last_round       = round_
            state.last_constructor = cid
            state.n_qualis        += 1
            self._record(state, year, round_, event="quali")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def snapshot(self) -> pd.DataFrame:
        """
        Current (μ, σ, ordinal) per driver, sorted by μ descending.

        `ordinal = μ − 3σ` is OpenSkill's conservative skill estimate (the
        lower end of a 3-σ interval). Using it for ranking is more
        conservative than μ: a driver with few races has wide σ and
        therefore a lower ordinal than their μ alone would suggest.
        """
        rows = []
        for st in self.drivers.values():
            rows.append({
                "driverId":         st.driver_id,
                "driver":           st.driver_name,
                "mu":               round(st.mu, 3),
                "sigma":            round(st.sigma, 3),
                "ordinal":          round(st.mu - 3.0 * st.sigma, 3),
                "last_year":        st.last_year,
                "last_round":       st.last_round,
                "last_constructor": st.last_constructor,
                "n_races":          st.n_races,
                "n_qualis":         st.n_qualis,
            })
        return (
            pd.DataFrame(rows)
            .sort_values("mu", ascending=False)
            .reset_index(drop=True)
        )

    def history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def build_ratings_from_results(
    results: pd.DataFrame,
    races: pd.DataFrame,
    drivers: pd.DataFrame,
    status: pd.DataFrame,
    qualifying: Optional[pd.DataFrame] = None,
    start_year: int = RATING_START_YEAR,
) -> F1Rater:
    """
    Process every race in [start_year, max_year] in (year, round) order
    and return a fully-updated F1Rater.

    For each weekend we run qualifying first (if available), then the race.
    Weekend ordering matters only for the σ-inflation bookkeeping — the
    bump fires on the first event of the weekend.
    """
    # --- filter & sort races to our scope ---
    race_meta = (
        races[races["year"] >= start_year][["raceId", "year", "round"]]
        .sort_values(["year", "round"])
        .reset_index(drop=True)
    )

    # --- helper maps ---
    driver_name_map = (
        drivers.assign(driver=lambda d: d["forename"] + " " + d["surname"])
               .set_index("driverId")["driver"]
    )
    status_map = status.set_index("statusId")["status"]

    # --- annotate results ---
    res = results.merge(race_meta, on="raceId", how="inner").copy()
    res["driverName"] = res["driverId"].map(driver_name_map)
    res["status"]     = res["statusId"].map(status_map)

    # --- annotate qualifying ---
    if qualifying is not None:
        qua = qualifying.merge(race_meta, on="raceId", how="inner").copy()
        qua["driverName"] = qua["driverId"].map(driver_name_map)
    else:
        qua = None

    rater = F1Rater()

    for race_id, year, round_ in race_meta.itertuples(index=False):
        if qua is not None:
            q_rows = qua[qua["raceId"] == race_id]
            if not q_rows.empty:
                rater.update_qualifying(int(year), int(round_), q_rows)

        r_rows = res[res["raceId"] == race_id]
        if not r_rows.empty:
            rater.update_race(int(year), int(round_), r_rows)

    return rater
