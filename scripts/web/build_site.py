"""
Build the static demo site under /docs from the back-test artefacts.

What it produces
----------------
    docs/data.json          everything the front-end reads, in one file
    docs/reliability.png    the calibration diagram (copied)

How it stays simple
-------------------
No framework. The front-end is plain HTML/CSS/JS reading this JSON. To
refresh the demo after a new back-test run, just re-run this script and
push the /docs directory — GitHub Pages picks it up automatically.

Usage
-----
    python scripts/web/build_site.py
    python scripts/web/build_site.py --mode ordinal      # default
    python scripts/web/build_site.py --mode z_ordinal    # alternative run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import SOFTMAX_TAU_INIT

ROOT    = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "backtest"
DOCS    = ROOT / "docs"


def _round_decimal(p: float, overround: float = 1.0) -> float:
    if p <= 0:
        return 0.0
    odds = 1.0 / (p * overround)
    # Cap very-long odds for readability
    return round(min(odds, 999.0), 2)


def build(mode: str, overround: float) -> None:
    preds_path = REPORTS / f"predictions_{mode}.csv"
    mets_path  = REPORTS / f"metrics_{mode}.csv"
    if not preds_path.exists() or not mets_path.exists():
        raise SystemExit(
            f"Missing back-test output for mode={mode!r}. Run:\n"
            f"  python scripts/model/backtest.py --mode {mode}\n"
            f"first."
        )

    print(f"Reading {preds_path.name}...")
    preds = pd.read_csv(preds_path)

    # ------------------------------------------------------------------
    # Build per-race table
    # ------------------------------------------------------------------
    preds = preds.sort_values(["year", "round", "p_win_mc"],
                              ascending=[True, True, False])
    races_json = []
    for (y, rnd, race_name), g in preds.groupby(
        ["year", "round", "race_name"], sort=False
    ):
        winner_row = g[g["is_winner"] == 1]
        winner = winner_row.iloc[0]["driver"] if not winner_row.empty else None
        top_pick = g.iloc[0]
        top_pick_won = int(top_pick["is_winner"])
        race_log_loss = (
            -np.log(max(float(winner_row.iloc[0]["p_win_mc"]), 1e-12))
            if not winner_row.empty else None
        )
        drivers = []
        for _, r in g.iterrows():
            pw = float(r["p_win_mc"])
            drivers.append({
                "driver":       r["driver"],
                "constructor":  r["constructor"],
                "grid":         int(r["grid"]),
                "ordinal":      round(float(r["ordinal"]), 1),
                "reliability":  round(float(r["reliability"]), 3),
                "pace":         round(float(r["pace_score"]), 2),
                "p_win":        round(pw, 4),
                "p_podium":     round(float(r["p_podium_mc"]), 4),
                "p_points":     round(float(r["p_points_mc"]), 4),
                "fair_odds":    _round_decimal(pw, 1.0),
                "book_odds":    _round_decimal(pw, overround),
                "actual_finish": (int(r["actual_finish"])
                                   if pd.notna(r["actual_finish"]) else None),
                "is_winner":    int(r["is_winner"]),
            })
        races_json.append({
            "raceId":        int(g.iloc[0]["raceId"]),
            "year":          int(y),
            "round":         int(rnd),
            "name":          race_name,
            "winner":        winner,
            "top_pick":      top_pick["driver"],
            "top_pick_conf": round(float(top_pick["p_win_mc"]), 3),
            "top_pick_hit":  bool(top_pick_won),
            "model_log_loss": (round(race_log_loss, 3)
                                if race_log_loss is not None else None),
            "drivers":       drivers,
        })

    # ------------------------------------------------------------------
    # Headline metrics
    # ------------------------------------------------------------------
    mets = pd.read_csv(mets_path)
    model_row = mets[mets["method"] == "model (Monte-Carlo B)"].iloc[0]

    # Per-race top-pick bucketed hit rates
    top_picks = (
        preds.sort_values(["raceId", "p_win_mc"], ascending=[True, False])
             .groupby("raceId")
             .head(1)
    )
    bins = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    labels = ["<30%", "30–40%", "40–50%", "50–60%", "60–70%", "70–80%", "80%+"]
    top_picks = top_picks.assign(
        bucket=pd.cut(top_picks["p_win_mc"], bins=bins, labels=labels,
                      include_lowest=True)
    )
    bucket_stats = (
        top_picks.groupby("bucket", observed=False)
                 .agg(n=("is_winner", "size"),
                      hits=("is_winner", "sum"),
                      mean_conf=("p_win_mc", "mean"))
                 .reset_index()
    )
    bucket_stats["hit_rate"] = (
        bucket_stats["hits"] / bucket_stats["n"].clip(lower=1)
    )
    bucket_stats_json = [
        {"bucket": str(r["bucket"]),
         "n":      int(r["n"]),
         "hits":   int(r["hits"]),
         "mean_conf": (round(float(r["mean_conf"]), 3)
                        if pd.notna(r["mean_conf"]) else None),
         "hit_rate": (round(float(r["hit_rate"]), 3)
                      if r["n"] else None)}
        for _, r in bucket_stats.iterrows()
    ]

    # Reliability (calibration) table — 10 bins over all (race,driver)
    edges = np.linspace(0, 1, 11)
    calib = []
    p_all = preds["p_win_mc"].to_numpy()
    y_all = preds["is_winner"].to_numpy()
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p_all >= lo) & (p_all <= hi) if hi == 1.0 else (p_all >= lo) & (p_all < hi)
        if mask.sum() == 0:
            calib.append({"bin_lo": round(lo, 2), "bin_hi": round(hi, 2),
                          "n": 0, "mean_pred": None, "observed": None})
        else:
            calib.append({
                "bin_lo": round(lo, 2),
                "bin_hi": round(hi, 2),
                "n":      int(mask.sum()),
                "mean_pred": round(float(p_all[mask].mean()), 4),
                "observed":  round(float(y_all[mask].mean()), 4),
            })

    # Method comparison
    methods_json = []
    for _, r in mets.iterrows():
        methods_json.append({
            "method":          r["method"],
            "mean_p_winner":   round(float(r["mean_p_winner"]), 4),
            "top_1":           round(float(r["top_1"]), 4),
            "top_3":           round(float(r["top_3"]), 4),
            "avg_rank":        round(float(r["avg_rank_of_winner"]), 3),
        })

    # Best / worst / representative highlights
    ranked = [r for r in races_json if r["model_log_loss"] is not None]
    ranked_by_ll = sorted(ranked, key=lambda r: r["model_log_loss"])
    best_raceids  = [r["raceId"] for r in ranked_by_ll[:5]]
    worst_raceids = [r["raceId"] for r in ranked_by_ll[-5:][::-1]]

    out = {
        "meta": {
            "mode":           mode,
            "tau":            SOFTMAX_TAU_INIT,
            "overround":      overround,
            "n_races":        int(len(races_json)),
            "holdout_start":  int(min(r["year"] for r in races_json)),
            "holdout_end":    int(max(r["year"] for r in races_json)),
            "mean_p_winner":  round(float(model_row["mean_p_winner"]), 4),
            "top_1_rate":     round(float(model_row["top_1"]), 4),
            "top_3_rate":     round(float(model_row["top_3"]), 4),
            "avg_rank_winner": round(float(model_row["avg_rank_of_winner"]), 3),
        },
        "methods":       methods_json,
        "races":         races_json,
        "bucket_stats":  bucket_stats_json,
        "calibration":   calib,
        "best_raceids":  best_raceids,
        "worst_raceids": worst_raceids,
    }

    DOCS.mkdir(parents=True, exist_ok=True)
    data_path = DOCS / "data.json"
    with open(data_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  wrote {data_path}   ({len(races_json)} races, "
          f"{data_path.stat().st_size/1024:.0f} KB)")

    # Copy reliability PNG for reference
    png_src = REPORTS / f"reliability_{mode}.png"
    if png_src.exists():
        shutil.copy(png_src, DOCS / "reliability.png")
        print(f"  copied {png_src.name} -> docs/reliability.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="ordinal",
                        choices=["ordinal", "mu", "z_mu", "z_ordinal"])
    parser.add_argument("--overround", type=float, default=1.20,
                        help="Overround baked into book_odds column.")
    args = parser.parse_args()
    build(args.mode, args.overround)
