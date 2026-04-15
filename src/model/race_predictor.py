"""
Stage 2 — Race predictor.

Combines:
  Stage 1a  : per-driver (μ, σ) from F1Rater
  Stage 1b  : per-(constructor, year) reliability from ConstructorReliability
  Pace proxy: previous-year constructor championship rank (EDA 09 validated)
  Grid pos  : from the actual starting grid for this race (after penalties)

and produces, per driver in the race field:

  p_win_analytic  — softmax(score / τ) scaled by P(car finishes)
  p_win_mc        — Monte-Carlo Plackett-Luce with reliability as Bernoulli
  p_podium_mc     — P(finish in top 3), from the same MC draw
  p_points_mc     — P(finish in top 10)

Analytic (A) and MC (B) both come from the same combined_score; A is the
direct softmax, B is what that softmax IS as a sampler for the winner —
they converge for P(win) as N_samples → ∞. B earns its keep by producing
P(podium) and P(points) for free.

Design choices locked in via src/config.py:
  DRIVER_SCORE_MODE = "ordinal"        (see config for the full rationale)
  SOFTMAX_TAU_INIT  = 80.0             (fitted from back-test)
  W_DRIVER_ORDINAL  = 1.0
  W_PACE_PREV_RANK  = 50.0
  W_GRID            = 30.0
  MC_SAMPLES        = 5000

combined_score_i =
      W_DRIVER_ORDINAL * driver_feature_i      (ordinal = μ−3σ)
    + W_PACE_PREV_RANK * pace_score_i          (from prev-year rank, [0,1])
    + W_GRID           * grid_score_i          ((21−grid)/20, [0,1])

P(win_analytic)   ∝ reliability_i * exp(combined_score_i / τ)
P(win_mc) etc.    from Gumbel-max sampling + Bernoulli(reliability) censoring

Monte-Carlo sampling (Gumbel-max trick)
---------------------------------------
If s_i are scores and u_i ~ Gumbel(0,1), then sorting by (s_i/τ + u_i)
descending yields a draw from Plackett-Luce(s/τ). For N samples of a
20-driver field, that's a single vectorised sort — milliseconds.

With reliability, we independently draw did_finish_i ~ Bernoulli(rel_i)
per sample per driver, and push non-finishers to -∞ before sorting.
Their finish "position" is thus ordered after all finishers; downstream
we require (rank==0) AND finished to count a win.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    DRIVER_SCORE_MODE,
    MC_SAMPLES,
    SOFTMAX_TAU_INIT,
    W_DRIVER_ORDINAL,
    W_GRID,
    W_PACE_PREV_RANK,
)
from src.model.rating import DriverState, F1Rater
from src.model.reliability import ConstructorReliability


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------
def compute_pace_map(
    constructor_standings: pd.DataFrame,
    races: pd.DataFrame,
) -> dict[tuple[int, int], float]:
    """
    Build a (constructor_id, year) → pace_score lookup, where pace_score
    is based on the PREVIOUS year's final championship rank.

      rank 1  → 1.00  (last year's champion)
      rank 5  → 0.60
      rank 10 → 0.10
      rank 11+→ 0.00
      no prev year (new team) → not in map; caller falls back to 0.50

    This is the proxy validated in EDA 09 (Spearman ρ = −0.60 for
    rank_prev vs this-year pole rate; ρ = −0.73 in the budget-cap era).
    """
    last_race_per_year = (
        races.sort_values(["year", "round"])
             .groupby("year").tail(1)[["year", "raceId"]]
    )
    final_stand = constructor_standings.merge(last_race_per_year, on="raceId")

    pace_map: dict[tuple[int, int], float] = {}
    for _, row in final_stand.iterrows():
        cid = int(row["constructorId"])
        # year we're computing pace FOR = standings-year + 1
        year_target = int(row["year"]) + 1
        rank = int(row["position"])
        pace_map[(cid, year_target)] = max(0.0, (11 - rank) / 10.0)
    return pace_map


def driver_skill_feature(state: DriverState, mode: str) -> float:
    """Pluggable driver-skill feature — see config for mode definitions."""
    if mode == "ordinal":
        return state.mu - 3.0 * state.sigma
    if mode == "mu":
        return state.mu
    if mode in ("z_mu", "z_ordinal"):
        # Per-race z applied downstream; return raw value here.
        return state.mu if mode == "z_mu" else state.mu - 3.0 * state.sigma
    raise ValueError(f"Unknown DRIVER_SCORE_MODE: {mode!r}")


def grid_score(grid_position: int) -> float:
    """
    Grid position → [0, 1] where 1 = pole, 0.05 = back of grid.
    `grid_position <= 0` (pit lane start, penalty-to-back) maps to 0.
    """
    if grid_position <= 0:
        return 0.0
    return max(0.0, (21 - grid_position) / 20.0)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class RacePredictor:
    """
    Stateless race predictor. All history lives in the rater/reliability
    objects you pass in — this class just reads them and produces probs.
    """

    def __init__(
        self,
        rater:       F1Rater,
        reliability: ConstructorReliability,
        pace_map:    dict[tuple[int, int], float],
        tau:         float = SOFTMAX_TAU_INIT,
        w_driver:    float = W_DRIVER_ORDINAL,
        w_pace:      float = W_PACE_PREV_RANK,
        w_grid:      float = W_GRID,
        n_mc:        int   = MC_SAMPLES,
        mode:        str   = DRIVER_SCORE_MODE,
        prior_mu:    float = 25.0,
        prior_sigma: float = 25.0 / 3.0,
        seed:        Optional[int] = 42,
    ):
        self.rater       = rater
        self.reliability = reliability
        self.pace_map    = pace_map
        self.tau         = tau
        self.w_driver    = w_driver
        self.w_pace      = w_pace
        self.w_grid      = w_grid
        self.n_mc        = n_mc
        self.mode        = mode
        self.prior_mu    = prior_mu
        self.prior_sigma = prior_sigma
        self.rng         = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------
    def build_features(
        self, race_field: pd.DataFrame, year: int
    ) -> pd.DataFrame:
        """
        race_field required columns:
          driverId, driverName, constructorId, constructorName, grid

        Returns a DataFrame with the features and the combined_score per driver.

        In z-modes, the per-race z-scoring happens here across the field
        AFTER raw features are collected — so it sees exactly the grid
        that's about to race, as intended.
        """
        feats = []
        for _, row in race_field.iterrows():
            did  = int(row["driverId"])
            cid  = int(row["constructorId"])
            grid = int(row["grid"])

            # --- driver skill (raw; z-scoring handled below if needed) ---
            if did in self.rater.drivers:
                state    = self.rater.drivers[did]
                mu       = state.mu
                sigma    = state.sigma
                raw_skill = driver_skill_feature(state, self.mode)
            else:
                # Driver never seen — use the prior, as the rater itself would.
                mu    = self.prior_mu
                sigma = self.prior_sigma
                raw_skill = (
                    self.prior_mu - 3.0 * self.prior_sigma
                    if self.mode in ("ordinal", "z_ordinal") else self.prior_mu
                )

            # --- car reliability ---
            rel_state = self.reliability.current_estimate(cid, year)
            if rel_state is not None:
                reliability = rel_state.reliability_mean
                rel_sigma   = rel_state.reliability_sigma
            else:
                from src.model.reliability import ALPHA_PRIOR, BETA_PRIOR
                reliability = 1.0 - ALPHA_PRIOR / (ALPHA_PRIOR + BETA_PRIOR)
                rel_sigma   = float("nan")

            # --- pace (prev year rank) ---
            pace = self.pace_map.get((cid, year), 0.5)

            # --- grid ---
            g_score = grid_score(grid)

            feats.append({
                "driverId":          did,
                "driver":            row["driverName"],
                "constructorId":     cid,
                "constructor":       row["constructorName"],
                "grid":              grid,
                "mu":                mu,
                "sigma":             sigma,
                "ordinal":           mu - 3.0 * sigma,
                "raw_skill":         raw_skill,
                "reliability":       reliability,
                "reliability_sigma": rel_sigma,
                "pace_score":        pace,
                "grid_score":        g_score,
            })

        df = pd.DataFrame(feats)

        # --- z-score the skill feature across this race's field, if requested ---
        if self.mode in ("z_mu", "z_ordinal"):
            mean = df["raw_skill"].mean()
            std  = df["raw_skill"].std(ddof=0)
            if std > 0:
                df["skill_feature"] = (df["raw_skill"] - mean) / std
            else:
                df["skill_feature"] = 0.0
        else:
            df["skill_feature"] = df["raw_skill"]

        df["combined_score"] = (
            self.w_driver * df["skill_feature"]
            + self.w_pace * df["pace_score"]
            + self.w_grid * df["grid_score"]
        )
        return df

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self, race_field: pd.DataFrame, year: int, round_: int | None = None
    ) -> pd.DataFrame:
        """
        Produce per-driver probability outputs for one race.

        Returns a DataFrame with feature columns plus:
          p_win_analytic, p_win_mc, p_podium_mc, p_points_mc
        sorted by p_win_mc descending.
        """
        feats = self.build_features(race_field, year)
        if feats.empty:
            return feats

        scores       = feats["combined_score"].to_numpy(dtype=float)
        reliabilities = feats["reliability"].to_numpy(dtype=float)
        N = len(feats)

        # -------- A: analytic softmax, reliability-weighted ---------------
        # Subtract the max for numerical stability before exp.
        z = scores / self.tau
        z = z - z.max()
        raw = reliabilities * np.exp(z)
        if raw.sum() <= 0:
            p_win_analytic = np.full(N, 1.0 / N)
        else:
            p_win_analytic = raw / raw.sum()

        # -------- B: Monte-Carlo PL with reliability censoring ------------
        p_win_mc, p_podium_mc, p_points_mc = self._mc_sample(
            scores_over_tau=scores / self.tau,
            reliabilities=reliabilities,
        )

        feats["p_win_analytic"] = p_win_analytic
        feats["p_win_mc"]       = p_win_mc
        feats["p_podium_mc"]    = p_podium_mc
        feats["p_points_mc"]    = p_points_mc

        return (
            feats.sort_values("p_win_mc", ascending=False)
                 .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Monte-Carlo kernel (Gumbel-max PL + Bernoulli reliability)
    # ------------------------------------------------------------------
    def _mc_sample(
        self,
        scores_over_tau: np.ndarray,
        reliabilities:   np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (p_win, p_podium, p_points), each of shape (N,).

        Gumbel-max: for each sample, perturb scores_over_tau by Gumbel(0,1)
        noise and sort descending — the result is a Plackett-Luce draw.

        Reliability: independently Bernoulli-draw 'finished' per driver per
        sample. Non-finishers get perturbed score -inf so they sort to the
        back. A win requires BOTH rank==0 AND finished==True, otherwise
        the race had an all-DNF sample and P(win) contributes 0 there.
        """
        N = scores_over_tau.shape[0]
        n = self.n_mc

        finished = self.rng.random((n, N)) < reliabilities[None, :]
        gumbels  = self.rng.gumbel(0.0, 1.0, size=(n, N))
        perturbed = scores_over_tau[None, :] + gumbels
        perturbed = np.where(finished, perturbed, -np.inf)

        # order[s, k] = driver index at finishing position k (0 = winner)
        order = np.argsort(-perturbed, axis=1)
        # ranks[s, i] = finishing position of driver i (0 = winner, N-1 = last)
        ranks = np.argsort(order, axis=1)

        is_winner = (ranks == 0) & finished
        is_podium = (ranks <  3) & finished
        is_points = (ranks < 10) & finished

        p_win    = is_winner.mean(axis=0)
        p_podium = is_podium.mean(axis=0)
        p_points = is_points.mean(axis=0)

        return p_win, p_podium, p_points
