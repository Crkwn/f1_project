"""
Race outcome predictor.
Given a set of driver scores, produces:
  - Win probability per driver
  - Podium probability per driver (top 3)
  - Expected finishing position per driver
"""

import math
import pandas as pd
from src.model.score import DriverScoreModel
from src.config import MIN_RACES_FOR_RELIABLE_SCORE


def predict_race(
    model: DriverScoreModel,
    driver_ids: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Returns a DataFrame of predictions for the given field of drivers.
    Drivers not yet in the model (rookies) get the initial Elo score.

    Columns:
        driver_id, driver_name, score, races,
        win_prob, podium_prob, expected_position, reliable
    """
    from src.config import ELO_INITIAL

    records = []
    for did in driver_ids:
        if did in model.drivers:
            rec = model.drivers[did]
            score = rec.score
            races = rec.races
            name  = rec.driver_name
        else:
            score = ELO_INITIAL
            races = 0
            name  = did   # fallback to ID if name unknown

        records.append({
            "driver_id":   did,
            "driver_name": name,
            "score":       score,
            "races":       races,
        })

    df = pd.DataFrame(records)

    # --- Win probability: softmax over scores ---
    exp_scores = df["score"].apply(lambda s: math.exp(s / 400))
    total_exp  = exp_scores.sum()
    df["win_prob"] = exp_scores / total_exp

    # --- Podium probability: simulate via pairwise ---
    # P(driver i finishes top 3) ≈ sum over all subsets of 2 others that
    # i beats all of — this is expensive for large fields, so we use a
    # fast approximation: run 3 sequential "winner removal" rounds.
    df["podium_prob"] = _podium_probability(df["score"].tolist(), df["driver_id"].tolist())

    # --- Expected position: rank by score (higher score = better expected position) ---
    df["expected_position"] = df["score"].rank(ascending=False).astype(int)

    # Flag drivers with few races — scores are less reliable
    df["reliable"] = df["races"] >= MIN_RACES_FOR_RELIABLE_SCORE

    return (
        df.sort_values("win_prob", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def _podium_probability(scores: list[float], ids: list[str]) -> pd.Series:
    """
    Approximate P(top 3) for each driver using the Plackett-Luce model.
    We compute P(win) + P(2nd) + P(3rd) by iterating over who wins each slot.

    This is O(n^2) — fast enough for a 20-driver F1 grid.
    """
    n = len(scores)
    exp_s = [math.exp(s / 400) for s in scores]

    podium_prob = [0.0] * n
    total = sum(exp_s)

    for i in range(n):
        p_win = exp_s[i] / total
        # Given i won, probability of each j finishing 2nd
        remaining_after_i = total - exp_s[i]
        for j in range(n):
            if j == i:
                continue
            p_2nd = (exp_s[i] / total) * (exp_s[j] / remaining_after_i)
            # Given i won and j is 2nd, probability of each k finishing 3rd
            remaining_after_ij = remaining_after_i - exp_s[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                p_3rd = p_2nd * (exp_s[k] / remaining_after_ij)
                podium_prob[i] += p_3rd
                podium_prob[j] += p_3rd
                podium_prob[k] += p_3rd

    # Normalise so it sums to 3 (3 podium spots)
    total_prob = sum(podium_prob)
    if total_prob > 0:
        podium_prob = [p * 3 / total_prob for p in podium_prob]

    return pd.Series(podium_prob, name="podium_prob")
