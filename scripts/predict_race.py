"""
Predict win and podium probabilities for an upcoming race.

Usage:
    # Predict next race using current 2025 grid
    python scripts/predict_race.py

    # Predict for a specific set of drivers
    python scripts/predict_race.py --drivers max_verstappen lando_norris charles_leclerc

    # Fetch latest race results first, then predict
    python scripts/predict_race.py --update
"""

import argparse
import pickle
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_PROCESSED
from src.data.fetcher import fetch_season_results
from src.model.predictor import predict_race
from src.model.score import DriverScoreModel


# Default 2025 grid (driver IDs as used in the Ergast/Jolpica API)
GRID_2025 = [
    "max_verstappen", "liam_lawson",        # Red Bull
    "lando_norris", "oscar_piastri",        # McLaren
    "charles_leclerc", "lewis_hamilton",    # Ferrari
    "george_russell", "andrea_kimi_antonelli",  # Mercedes
    "fernando_alonso", "lance_stroll",      # Aston Martin
    "pierre_gasly", "jack_doohan",          # Alpine
    "carlos_sainz", "alexander_albon",      # Williams
    "yuki_tsunoda", "isack_hadjar",         # Racing Bulls
    "nico_hulkenberg", "oliver_bearman",    # Haas
    "zhou_guanyu", "valtteri_bottas",       # Sauber (placeholder — check 2025 lineup)
]


def load_model() -> DriverScoreModel:
    model_path = DATA_PROCESSED / "driver_score_model.pkl"
    if not model_path.exists():
        print("No model found. Run `python scripts/build_scores.py` first.")
        sys.exit(1)
    with open(model_path, "rb") as f:
        return pickle.load(f)


def update_with_current_season(model: DriverScoreModel) -> None:
    """Fetch 2025 results so far and apply them to the model."""
    year = datetime.now().year
    print(f"Fetching {year} results...")
    records = fetch_season_results(year, force=True)
    if not records:
        print("No results yet for current season.")
        return

    import pandas as pd
    from src.data.loader import _clean
    df = _clean(pd.DataFrame(records))
    model.process_season(df)
    print(f"Applied {len(df)} records from {year}.")


def main(driver_ids: list[str], update: bool):
    model = load_model()

    if update:
        update_with_current_season(model)

    print(f"\n=== Race Prediction ===\n")
    predictions = predict_race(model, driver_ids, top_n=len(driver_ids))

    print(f"{'Driver':<30} {'Score':>7}  {'Win %':>6}  {'Podium %':>8}  {'Exp. Pos':>8}  {'Reliable':>8}")
    print("-" * 75)
    for _, row in predictions.iterrows():
        reliable_flag = "" if row["reliable"] else " *"
        print(
            f"{row['driver_name']:<30} "
            f"{row['score']:>7.1f}  "
            f"{row['win_prob']*100:>6.2f}%  "
            f"{row['podium_prob']*100:>7.2f}%  "
            f"{row['expected_position']:>8}  "
            f"{'yes' if row['reliable'] else 'no *':>8}"
        )

    print("\n* = fewer than 5 races, score may not be reliable yet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drivers", nargs="+", default=GRID_2025,
        help="Driver IDs to include in prediction"
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Fetch latest current-season results before predicting"
    )
    args = parser.parse_args()
    main(args.drivers, args.update)
