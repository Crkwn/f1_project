"""
Driver Ability Score
====================
Each driver carries a single time-varying score that represents their
current estimated ability relative to the rest of the field.

The mechanism is a multi-competitor Elo system:
  - After each race, every finishing position implies a set of pairwise
    outcomes (driver A finished ahead of driver B → A "beat" B).
  - We run a standard Elo update for every such pair.
  - Over time, the scores converge to reflect true relative ability.

Why Elo and not just average finish position?
  - It adjusts for field quality automatically: beating a field of
    high-scored drivers moves your score more than beating low-scored ones.
  - It naturally decays stale information: a driver who stops racing has
    their score frozen, while active drivers continue to update.
  - It's interpretable: score differences have a direct probabilistic
    meaning (see `win_probability`).

DNF handling:
  - Classified DNFs (driver started but didn't finish) are placed at the
    back of the finishing order. They still carry information — a driver
    who DNFs frequently relative to teammates is genuinely less reliable.
  - Drivers not classified at all (DNS, excluded) are excluded from updates.
"""

import pandas as pd
from dataclasses import dataclass, field
from src.config import ELO_INITIAL, ELO_K, ELO_SCALE


@dataclass
class DriverRecord:
    driver_id: str
    driver_name: str
    score: float = ELO_INITIAL
    races: int = 0
    last_year: int = 0
    last_round: int = 0


class DriverScoreModel:
    """
    Processes races in chronological order and maintains a current Elo score
    for every driver seen.
    """

    def __init__(self, k: float = ELO_K, scale: float = ELO_SCALE):
        self.k = k
        self.scale = scale
        self.drivers: dict[str, DriverRecord] = {}
        self.history: list[dict] = []  # score snapshot after every race

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_season(self, df: pd.DataFrame) -> None:
        """Process all races in a DataFrame (must be sorted by year, round)."""
        for (year, round_), group in df.groupby(["year", "round"]):
            self._process_race(year, round_, group)

    def current_scores(self) -> pd.DataFrame:
        """Returns current scores for all drivers, sorted descending."""
        rows = [
            {
                "driver_id":   d.driver_id,
                "driver_name": d.driver_name,
                "score":       round(d.score, 1),
                "races":       d.races,
                "last_year":   d.last_year,
                "last_round":  d.last_round,
            }
            for d in self.drivers.values()
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

    def score_history(self) -> pd.DataFrame:
        """Full history of scores after each race — useful for plotting trajectories."""
        return pd.DataFrame(self.history)

    def win_probability(self, driver_ids: list[str]) -> dict[str, float]:
        """
        Given a list of driver IDs (the race field), return each driver's
        probability of winning using a softmax over their scores.

        The softmax here uses the same scale as Elo so probabilities are
        consistent with head-to-head Elo expectations.
        """
        scores = {
            did: self.drivers[did].score
            for did in driver_ids
            if did in self.drivers
        }
        # Softmax with Elo scale
        import math
        exp_scores = {did: math.exp(s / self.scale) for did, s in scores.items()}
        total = sum(exp_scores.values())
        return {did: v / total for did, v in exp_scores.items()}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create(self, driver_id: str, driver_name: str) -> DriverRecord:
        if driver_id not in self.drivers:
            self.drivers[driver_id] = DriverRecord(
                driver_id=driver_id,
                driver_name=driver_name,
                score=ELO_INITIAL,
            )
        return self.drivers[driver_id]

    def _process_race(self, year: int, round_: int, group: pd.DataFrame) -> None:
        """
        Core update step.
        1. Build finishing order (classified finishers first, then DNFs).
        2. For every pair (i, j) where i finished ahead of j, run Elo update.
        3. Average each driver's net update across all their pairwise comparisons.
        4. Record a snapshot for history.
        """
        # Separate finishers from DNFs; both are ranked
        finishers = group[group["finished"]].sort_values("position")
        dnfs = group[~group["finished"]]

        # Combined ordered list: finishers first, DNFs after (order among DNFs is arbitrary)
        ordered = pd.concat([finishers, dnfs])

        participants = []
        for _, row in ordered.iterrows():
            rec = self._get_or_create(row["driver_id"], row["driver_name"])
            participants.append(rec)

        n = len(participants)
        if n < 2:
            return

        # Accumulate net score changes per driver
        delta: dict[str, float] = {p.driver_id: 0.0 for p in participants}
        comparisons: dict[str, int] = {p.driver_id: 0 for p in participants}

        for i in range(n):
            for j in range(i + 1, n):
                winner = participants[i]   # i finished ahead of j
                loser  = participants[j]

                expected_winner = self._expected(winner.score, loser.score)
                expected_loser  = 1.0 - expected_winner

                delta[winner.driver_id] += self.k * (1.0 - expected_winner)
                delta[loser.driver_id]  += self.k * (0.0 - expected_loser)

                comparisons[winner.driver_id] += 1
                comparisons[loser.driver_id]  += 1

        # Apply averaged updates and record history
        for p in participants:
            did = p.driver_id
            n_comp = comparisons[did]
            if n_comp > 0:
                p.score += delta[did] / n_comp
            p.races += 1
            p.last_year  = year
            p.last_round = round_

            self.history.append({
                "year":        year,
                "round":       round_,
                "driver_id":   p.driver_id,
                "driver_name": p.driver_name,
                "score":       p.score,
                "races":       p.races,
            })

    def _expected(self, rating_a: float, rating_b: float) -> float:
        """Standard Elo expected score for player A against player B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / self.scale))
