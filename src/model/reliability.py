"""
Stage 1b — Constructor reliability model.

Goal
----
For every (constructor, year), estimate the probability that one of its
cars will suffer a mechanical DNF in a given race, with uncertainty. This
is the car-side counterpart to Stage 1a's driver-side rating:

  Stage 2 predictor:
      P(driver wins race)
        = P(car finishes)                        ← Stage 1b
        × P(driver wins | car finishes)           ← Stage 1a + grid / pace features

Treating these two as separate posteriors is the standard survivorship-
bias fix. It is exactly analogous to how credit-risk models factor a
default event into a probability-of-default (PD) term and a
loss-given-default (LGD) term and estimate each on its own data — here,
"does the car survive" and "given the car survives, how fast is the
driver".

What counts as "car failed"?
----------------------------
Only the MECHANICAL family from status_families.py. Everything else —
finished, accident, driver-side illness, disqualified, generic retired
— is treated as "car did NOT mechanically fail". In particular, accident
DNFs don't penalise reliability: the car was taken out of the race by a
driving event, not by engineering.

Caveat we're accepting in v1:
  Accident DNFs truncate our observation of reliability — the gearbox
  might have failed on lap 40 if the car hadn't crashed on lap 5. We're
  treating those as fully "non-failure" rather than "censored". For a
  grid-wide estimate this slightly overstates reliability; it's a v1
  simplification. Survival-analysis style censoring is a v2 option.

Bayesian model
--------------
Per (constructor, year), we model the mechanical-DNF rate θ as:

  Prior    :  θ  ~  Beta(α₀, β₀)              α₀=2, β₀=10
  Likelihood:  k  ~  Binomial(n, θ)           n = car-entries this season
  Posterior:  θ | data  ~  Beta(α₀+k, β₀+n-k)

  posterior_mean_θ  = (α₀ + k)              / (α₀ + β₀ + n)
  reliability_mean  = 1 − posterior_mean_θ
  reliability_var   = θ_mean(1−θ_mean) / (α₀+β₀+n+1)

The prior is weakly informative:
  Beta(2, 10) → prior mean θ = 0.167 (≈17% mech DNF rate historically)
                prior effective sample size = 12
So a team with 2 races and 0 DNFs (n=2, k=0) gets
  θ_mean = 2/14 = 0.143 → reliability ≈ 85.7%
rather than "0 DNFs in 2 races ⇒ 100% reliable" which is obviously
over-confident.

Year reset
----------
Posteriors are RESET to the prior at the start of each new year. A
2020 Mercedes tells us little about a 2024 Mercedes — new car, new
rules, new reliability baseline. Cross-year smoothing is a v2 option
(e.g., EWMA or year-end posterior → next-year prior with discount).

Inputs expected (Ergast)
------------------------
results.csv    : per (raceId, driverId, constructorId) with statusId.
races.csv      : maps raceId → year, round.
status.csv     : maps statusId → status (family_of() classifies).

Outputs
-------
ConstructorReliability.snapshot()  : current (year, constructor)-level state
ConstructorReliability.history_df(): per-(year, round, constructor) trajectory
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.config import RATING_START_YEAR
from src.model.status_families import family_of


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------
# Beta(α₀, β₀) on mechanical DNF rate θ.
# Mean = α₀/(α₀+β₀). Effective sample size = α₀+β₀.
# α₀=2, β₀=10 gives prior mean 16.7% DNF, effective ~12 observations.
# Chosen to be weak-but-informative — won't dominate an 20-entry season
# (new info outweighs prior) but will prevent "0 DNFs in 2 races ⇒ 100%".
ALPHA_PRIOR = 2.0
BETA_PRIOR  = 10.0


# ---------------------------------------------------------------------------
# Per-(constructor, year) state
# ---------------------------------------------------------------------------
@dataclass
class ConstructorYearState:
    """Rolling state for one constructor in one season."""
    constructor_id:   int
    constructor_name: str
    year:             int
    n_entries:        int = 0   # car-races (each driver-race = 1 entry)
    n_mechanical:     int = 0   # mechanical DNFs
    last_round:       Optional[int] = None

    # Bayesian posterior
    @property
    def alpha_post(self) -> float:
        return ALPHA_PRIOR + self.n_mechanical

    @property
    def beta_post(self) -> float:
        return BETA_PRIOR + (self.n_entries - self.n_mechanical)

    @property
    def dnf_rate_mean(self) -> float:
        """Posterior mean of mechanical-DNF rate per entry."""
        return self.alpha_post / (self.alpha_post + self.beta_post)

    @property
    def dnf_rate_var(self) -> float:
        """Posterior variance of mechanical-DNF rate."""
        m  = self.dnf_rate_mean
        n_eff = self.alpha_post + self.beta_post
        return m * (1 - m) / (n_eff + 1)

    @property
    def reliability_mean(self) -> float:
        """Posterior mean of P(car survives the race)."""
        return 1.0 - self.dnf_rate_mean

    @property
    def reliability_sigma(self) -> float:
        """Posterior std of reliability (same as DNF-rate std by symmetry)."""
        return math.sqrt(self.dnf_rate_var)


# ---------------------------------------------------------------------------
# Online processor
# ---------------------------------------------------------------------------
class ConstructorReliability:
    """
    Process races chronologically, maintain per-(constructor, year) Bayesian
    posterior over mechanical-DNF rate. Mirrors F1Rater's online design so
    Stage 2 can call `.current_estimate(constructor_id, year)` at any race
    and get the correct as-of-that-race posterior.
    """

    def __init__(self) -> None:
        self.state: dict[tuple[int, int], ConstructorYearState] = {}
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------
    def _get_or_create(
        self, constructor_id: int, constructor_name: str, year: int
    ) -> ConstructorYearState:
        key = (constructor_id, year)
        if key not in self.state:
            self.state[key] = ConstructorYearState(
                constructor_id=constructor_id,
                constructor_name=constructor_name,
                year=year,
            )
        return self.state[key]

    def update_race(self, year: int, round_: int, race_df: pd.DataFrame) -> None:
        """
        race_df required columns:
          constructorId, constructorName, status

        Every driver-row counts as one entry for that constructor. Mechanical
        DNFs increment the failure count; everything else increments only
        the total-entries count.
        """
        if race_df.empty:
            return

        df = race_df.copy()
        df["family"] = df["status"].apply(family_of)

        # Aggregate by constructor within this race
        for cid, grp in df.groupby("constructorId"):
            cname = grp["constructorName"].iloc[0]
            state = self._get_or_create(int(cid), cname, year)
            n_this_race       = len(grp)
            n_mech_this_race  = int((grp["family"] == "mechanical").sum())
            state.n_entries    += n_this_race
            state.n_mechanical += n_mech_this_race
            state.last_round    = round_
            self.history.append({
                "year":             year,
                "round":            round_,
                "constructorId":    int(cid),
                "constructor":      cname,
                "n_entries_season": state.n_entries,
                "n_mech_season":    state.n_mechanical,
                "reliability_mean": state.reliability_mean,
                "reliability_sigma": state.reliability_sigma,
            })

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def current_estimate(
        self, constructor_id: int, year: int
    ) -> Optional[ConstructorYearState]:
        """
        Return the current posterior state for (constructor, year), or None
        if this constructor has no recorded entries in that year yet.
        Stage 2 calls this when building features for a race.
        """
        return self.state.get((constructor_id, year))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def snapshot(self) -> pd.DataFrame:
        """All (constructor, year) states — current end-of-processing view."""
        rows = []
        for st in self.state.values():
            rows.append({
                "constructorId":      st.constructor_id,
                "constructor":        st.constructor_name,
                "year":               st.year,
                "n_entries":          st.n_entries,
                "n_mech_dnfs":        st.n_mechanical,
                "empirical_dnf_rate": (st.n_mechanical / st.n_entries
                                        if st.n_entries > 0 else float("nan")),
                "dnf_rate_mean":      round(st.dnf_rate_mean, 4),
                "reliability_mean":   round(st.reliability_mean, 4),
                "reliability_sigma":  round(st.reliability_sigma, 4),
                "last_round":         st.last_round,
            })
        return (
            pd.DataFrame(rows)
              .sort_values(["year", "reliability_mean"],
                           ascending=[True, False])
              .reset_index(drop=True)
        )

    def history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def build_reliability_from_results(
    results:      pd.DataFrame,
    races:        pd.DataFrame,
    constructors: pd.DataFrame,
    status:       pd.DataFrame,
    start_year:   int = RATING_START_YEAR,
) -> ConstructorReliability:
    """
    Process every race in [start_year, max_year] chronologically and
    return a fully-populated ConstructorReliability.

    Parameters mirror build_ratings_from_results for consistency.
    """
    race_meta = (
        races[races["year"] >= start_year][["raceId", "year", "round"]]
        .sort_values(["year", "round"])
        .reset_index(drop=True)
    )
    constructor_name = constructors.set_index("constructorId")["name"]
    status_map       = status.set_index("statusId")["status"]

    res = results.merge(race_meta, on="raceId", how="inner").copy()
    res["constructorName"] = res["constructorId"].map(constructor_name)
    res["status"]          = res["statusId"].map(status_map)

    engine = ConstructorReliability()

    for race_id, year, round_ in race_meta.itertuples(index=False):
        r_rows = res[res["raceId"] == race_id]
        if not r_rows.empty:
            engine.update_race(int(year), int(round_), r_rows)

    return engine
