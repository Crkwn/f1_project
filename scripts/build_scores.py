"""
Build and save driver ability scores from all historical race data.

Usage:
    python scripts/build_scores.py
    python scripts/build_scores.py --start 2010  # from a specific year
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import HISTORY_START_YEAR, DATA_PROCESSED
from src.data.loader import load_race_results
from src.model.score import DriverScoreModel


def main(start_year: int):
    print(f"\n=== Building driver scores ({start_year} → present) ===\n")

    # Load all race results
    df = load_race_results(start_year=start_year)

    # Build scores
    model = DriverScoreModel()
    print("Processing races...")
    model.process_season(df)

    # Save model
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    model_path = DATA_PROCESSED / "driver_score_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")

    # Print top 20 current scores
    scores = model.current_scores()
    active = scores[scores["last_year"] >= 2023].head(20)

    print("\n--- Top 20 active drivers by current score ---")
    print(active[["driver_name", "score", "races", "last_year"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=HISTORY_START_YEAR)
    args = parser.parse_args()
    main(args.start)
