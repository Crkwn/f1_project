"""
Fetches F1 race result data from the Jolpica API (free Ergast replacement).
Results are cached locally as JSON so we never re-fetch the same season twice.
Current season is always re-fetched to pick up new race weekends.
"""

import json
import time
import requests
from pathlib import Path
from datetime import datetime

from src.config import JOLPICA_BASE, API_DELAY_SECONDS, DATA_CACHE

_MAX_RETRIES = 5


def _get_with_backoff(url: str) -> requests.Response:
    """GET with exponential backoff on 429 / transient errors."""
    delay = 2.0
    for attempt in range(_MAX_RETRIES):
        resp = requests.get(url, timeout=15)
        if resp.status_code == 429:
            wait = delay * (2 ** attempt)
            print(f"    Rate limited — waiting {wait:.0f}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError(f"Failed after {_MAX_RETRIES} retries: {url}")


def _cache_path(year: int) -> Path:
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    return DATA_CACHE / f"results_{year}.json"


def _is_current_season(year: int) -> bool:
    return year == datetime.now().year


def fetch_season_results(year: int, force: bool = False) -> list[dict]:
    """
    Returns a flat list of race result records for every race in `year`.
    Each record represents one driver in one race.

    Fields:
        year, round, race_name, circuit, driver_id, driver_name,
        constructor, grid, position, status, points, fastest_lap_rank
    """
    cache = _cache_path(year)

    if cache.exists() and not force and not _is_current_season(year):
        with open(cache) as f:
            return json.load(f)

    print(f"  Fetching {year} from API...")
    records = []
    offset = 0
    limit = 100

    while True:
        url = f"{JOLPICA_BASE}/{year}/results.json?limit={limit}&offset={offset}"
        resp = _get_with_backoff(url)
        data = resp.json()

        races = data["MRData"]["RaceTable"]["Races"]
        if not races:
            break

        for race in races:
            for result in race["Results"]:
                driver = result["Driver"]
                constructor = result["Constructor"]

                # position is None for DNFs that didn't finish classified
                position_str = result.get("position", "")
                try:
                    position = int(position_str)
                except (ValueError, TypeError):
                    position = None

                records.append({
                    "year":             int(race["season"]),
                    "round":            int(race["round"]),
                    "race_name":        race["raceName"],
                    "circuit":          race["Circuit"]["circuitId"],
                    "driver_id":        driver["driverId"],
                    "driver_name":      f"{driver['givenName']} {driver['familyName']}",
                    "constructor_id":   constructor["constructorId"],
                    "constructor_name": constructor["name"],
                    "grid":             int(result.get("grid", 0)),
                    "position":         position,
                    "status":           result["status"],
                    "points":           float(result.get("points", 0)),
                    "fastest_lap_rank": int(result.get("FastestLap", {}).get("rank", 0) or 0),
                })

        total = int(data["MRData"]["total"])
        offset += limit
        if offset >= total:
            break
        time.sleep(API_DELAY_SECONDS)

    # Cache completed seasons permanently
    if not _is_current_season(year):
        with open(cache, "w") as f:
            json.dump(records, f)

    return records


def fetch_seasons(start_year: int, end_year: int) -> list[dict]:
    """Fetch and combine multiple seasons. Returns flat list of all records."""
    all_records = []
    for year in range(start_year, end_year + 1):
        records = fetch_season_results(year)
        all_records.extend(records)
        time.sleep(API_DELAY_SECONDS)
    print(f"  Total records: {len(all_records):,} across {end_year - start_year + 1} seasons")
    return all_records
