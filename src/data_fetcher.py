"""Fetch F1 race data from the OpenF1 API (https://openf1.org)."""

import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.openf1.org/v1"
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
SEASONS = range(2023, 2027)  # OpenF1 has data from 2023 onwards

# Rate limit: 3 req/s and 30 req/min on free tier
# We use 2.5s between requests to stay well under 30 req/min
REQUEST_DELAY = 2.5


def _get(endpoint: str, **params) -> list[dict]:
    """Make a GET request to the OpenF1 API with retry + 429 backoff."""
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = min(30, 5 * (attempt + 1))
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return []  # endpoint/data not available for this session
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < 4:
                time.sleep(3 * (attempt + 1))
            else:
                print(f"    Failed: {endpoint} {params} -> {e}")
                return []
    return []


def fetch_race_sessions(year: int) -> list[dict]:
    """Get all Race sessions for a given year."""
    sessions = _get("sessions", year=year, session_type="Race")
    time.sleep(REQUEST_DELAY)
    return [s for s in sessions if not s.get("is_cancelled", False)]


def fetch_drivers(session_key: int) -> list[dict]:
    """Get driver info for a session."""
    data = _get("drivers", session_key=session_key)
    time.sleep(REQUEST_DELAY)
    return data


def fetch_starting_grid(session_key: int) -> list[dict]:
    """Get the starting grid for a race session."""
    data = _get("starting_grid", session_key=session_key)
    time.sleep(REQUEST_DELAY)
    return data


def fetch_initial_positions(session_key: int) -> list[dict]:
    """Fallback: get earliest position data to infer grid order."""
    data = _get("position", session_key=session_key)
    time.sleep(REQUEST_DELAY)
    if not data:
        return []
    # Get the earliest timestamp's positions as the starting order
    earliest_date = min(d["date"] for d in data)
    return [d for d in data if d["date"] == earliest_date]


def fetch_session_results(session_key: int) -> list[dict]:
    """Get final results for a session."""
    data = _get("session_result", session_key=session_key)
    time.sleep(REQUEST_DELAY)
    return data


def fetch_all():
    """Download all race data from OpenF1 and save raw JSON per season."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for year in SEASONS:
        print(f"\n=== {year} Season ===")
        races = fetch_race_sessions(year)
        if not races:
            print(f"  No races found for {year}")
            continue

        season_data = []
        for race in races:
            sk = race["session_key"]
            circuit = race.get("circuit_short_name", "Unknown")
            country = race.get("country_name", "?")
            print(f"  {country} ({circuit}) - session {sk}")

            drivers = fetch_drivers(sk)

            grid = fetch_starting_grid(sk)
            if not grid:
                print(f"    No starting_grid data, falling back to position endpoint")
                grid_fallback = fetch_initial_positions(sk)
                # Convert position format to starting_grid format
                grid = [
                    {"driver_number": p["driver_number"], "position": p["position"]}
                    for p in grid_fallback
                ]

            results = fetch_session_results(sk)

            if not results:
                print(f"    No results for session {sk}, skipping")
                continue

            season_data.append({
                "session": race,
                "drivers": drivers,
                "starting_grid": grid,
                "results": results,
            })

        with open(DATA_DIR / f"season_{year}.json", "w") as f:
            json.dump(season_data, f, indent=2)
        print(f"  Saved {len(season_data)} races -> data/raw/season_{year}.json")

    print("\nDone fetching.")


def parse_results_to_dataframe() -> pd.DataFrame:
    """Parse raw OpenF1 JSON into a flat DataFrame matching the expected schema.

    Output columns: season, round, circuit_id, circuit_name, driver_id,
                    driver_name, constructor_id, constructor_name,
                    grid_position, finish_position, status, points
    """
    rows = []

    for year in SEASONS:
        filepath = DATA_DIR / f"season_{year}.json"
        if not filepath.exists():
            continue
        with open(filepath) as f:
            season_data = json.load(f)

        for round_num, race_data in enumerate(season_data, start=1):
            session = race_data["session"]
            circuit_id = session.get("circuit_short_name", "unknown").lower().replace(" ", "_")
            circuit_name = session.get("circuit_short_name", "Unknown")
            country_name = session.get("country_name", "")
            # e.g. "Silverstone, Great Britain" or "Monza, Italy"
            circuit_display = f"{circuit_name}, {country_name}" if country_name else circuit_name

            # Build driver lookup: driver_number -> info
            driver_lookup = {}
            for d in race_data["drivers"]:
                dn = d["driver_number"]
                driver_lookup[dn] = {
                    "driver_id": d.get("name_acronym", str(dn)),
                    "driver_name": d.get("full_name", d.get("broadcast_name", str(dn))),
                    "constructor_id": d.get("team_name", "Unknown").lower().replace(" ", "_"),
                    "constructor_name": d.get("team_name", "Unknown"),
                }

            # Build grid lookup: driver_number -> grid position
            grid_lookup = {g["driver_number"]: g["position"] for g in race_data["starting_grid"]}

            # Process results
            for result in race_data["results"]:
                dn = result["driver_number"]
                driver_info = driver_lookup.get(dn, {
                    "driver_id": str(dn), "driver_name": str(dn),
                    "constructor_id": "unknown", "constructor_name": "Unknown",
                })

                dnf = result.get("dnf", False)
                dns = result.get("dns", False)
                dsq = result.get("dsq", False)

                if dns:
                    status = "DNS"
                elif dnf:
                    status = "DNF"
                elif dsq:
                    status = "Disqualified"
                else:
                    status = "Finished"

                finish_pos = result.get("position")

                rows.append({
                    "season": year,
                    "round": round_num,
                    "circuit_id": circuit_id,
                    "circuit_name": circuit_display,
                    "driver_id": driver_info["driver_id"],
                    "driver_name": driver_info["driver_name"],
                    "constructor_id": driver_info["constructor_id"],
                    "constructor_name": driver_info["constructor_name"],
                    "grid_position": grid_lookup.get(dn, 20),
                    "finish_position": finish_pos,
                    "status": status,
                    "points": 0.0,
                })

    df = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "race_results.csv", index=False)
    print(f"\nParsed {len(df)} result rows -> data/processed/race_results.csv")
    return df


if __name__ == "__main__":
    fetch_all()
    parse_results_to_dataframe()
