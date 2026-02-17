"""
Historical weather data fetcher using Open-Meteo APIs.

Fetches real past weather and marine data for route midpoints, then builds
a training dataset by detecting conditions that would have triggered sailing
bans. This replaces the synthetic data generation with real-world patterns.

APIs used:
  - Marine API: wave_height, wave_period, swell (supports start_date/end_date)
  - Archive Weather API: wind, gusts, visibility (historical data back to 1940)
"""

import csv
import json
import urllib.request
import urllib.parse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

from src.config.constants import (
    PORTS,
    ROUTES,
    SAILING_BAN_THRESHOLDS,
    VESSEL_TYPES,
    knots_to_beaufort,
)
from src.utils.logger import get_logger
from src.utils.rate_limiter import api_rate_limiter

logger = get_logger(__name__)

# Open-Meteo endpoints
MARINE_API_URL = "https://marine-api.open-meteo.com/v1/marine"
ARCHIVE_WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "historical"

# Maximum date range per API request (Open-Meteo recommends <=92 days)
MAX_CHUNK_DAYS = 90


def _fetch_json(url: str, timeout: int = 30, retries: int = 3) -> dict:
    """Fetch URL with retries and exponential backoff."""
    if not api_rate_limiter.acquire(timeout=10):
        raise RuntimeError("API rate limit exceeded — try again later")

    last_error = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning("Fetch failed (attempt %d/%d): %s — retrying in %ds",
                               attempt + 1, retries, e, wait)
                time.sleep(wait)
    raise last_error


def fetch_historical_marine(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> dict:
    """
    Fetch historical marine data (waves, swell) for a date range.

    Args:
        latitude/longitude: Location coordinates
        start_date/end_date: ISO format dates (YYYY-MM-DD)

    Returns:
        Open-Meteo response dict with hourly marine data
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join([
            "wave_height",
            "wave_direction",
            "wave_period",
            "wind_wave_height",
            "swell_wave_height",
            "swell_wave_direction",
            "swell_wave_period",
        ]),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Europe/Athens",
    }
    url = f"{MARINE_API_URL}?{urllib.parse.urlencode(params)}"
    logger.info("Fetching historical marine data (%.4f, %.4f) %s to %s",
                latitude, longitude, start_date, end_date)
    return _fetch_json(url)


def fetch_historical_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> dict:
    """
    Fetch historical weather data (wind, gusts, visibility) for a date range.

    Uses the Open-Meteo Archive API which provides data back to 1940.

    Args:
        latitude/longitude: Location coordinates
        start_date/end_date: ISO format dates (YYYY-MM-DD)

    Returns:
        Open-Meteo response dict with hourly weather data
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join([
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "visibility",
            "precipitation",
        ]),
        "wind_speed_unit": "kn",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Europe/Athens",
    }
    url = f"{ARCHIVE_WEATHER_API_URL}?{urllib.parse.urlencode(params)}"
    logger.info("Fetching historical weather data (%.4f, %.4f) %s to %s",
                latitude, longitude, start_date, end_date)
    return _fetch_json(url)


def _date_chunks(start_date: str, end_date: str, chunk_days: int = MAX_CHUNK_DAYS):
    """Split a date range into chunks of max chunk_days."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while start <= end:
        chunk_end = min(start + timedelta(days=chunk_days - 1), end)
        yield start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        start = chunk_end + timedelta(days=1)


def fetch_historical_for_route(
    route_id: str,
    start_date: str,
    end_date: str,
) -> dict:
    """
    Fetch historical weather + marine data for a route's midpoint.

    Uses the midpoint between origin and destination as the most
    representative point for open-sea conditions.

    Returns dict with merged hourly data keyed by ISO timestamp.
    """
    origin_code, dest_code = route_id.split("-")
    origin = PORTS[origin_code]
    dest = PORTS[dest_code]

    mid_lat = (origin["lat"] + dest["lat"]) / 2
    mid_lon = (origin["lon"] + dest["lon"]) / 2

    # Fetch in chunks to respect API limits
    all_hours = {}

    for chunk_start, chunk_end in _date_chunks(start_date, end_date):
        marine_data = fetch_historical_marine(mid_lat, mid_lon, chunk_start, chunk_end)
        weather_data = fetch_historical_weather(mid_lat, mid_lon, chunk_start, chunk_end)

        marine_hourly = marine_data.get("hourly", {})
        weather_hourly = weather_data.get("hourly", {})

        marine_times = marine_hourly.get("time", [])
        weather_times = weather_hourly.get("time", [])

        # Index marine data by time
        for i, t in enumerate(marine_times):
            if t not in all_hours:
                all_hours[t] = {}
            all_hours[t]["wave_height"] = marine_hourly.get("wave_height", [None] * len(marine_times))[i]
            all_hours[t]["wave_direction"] = marine_hourly.get("wave_direction", [None] * len(marine_times))[i]
            all_hours[t]["wave_period"] = marine_hourly.get("wave_period", [None] * len(marine_times))[i]
            all_hours[t]["swell_wave_height"] = marine_hourly.get("swell_wave_height", [None] * len(marine_times))[i]

        # Index weather data by time
        for i, t in enumerate(weather_times):
            if t not in all_hours:
                all_hours[t] = {}
            all_hours[t]["wind_speed_kn"] = weather_hourly.get("wind_speed_10m", [None] * len(weather_times))[i]
            all_hours[t]["wind_direction"] = weather_hourly.get("wind_direction_10m", [None] * len(weather_times))[i]
            all_hours[t]["wind_gusts_kn"] = weather_hourly.get("wind_gusts_10m", [None] * len(weather_times))[i]
            all_hours[t]["visibility"] = weather_hourly.get("visibility", [None] * len(weather_times))[i]

        # Be polite to the API
        time.sleep(0.5)

    return {
        "route_id": route_id,
        "midpoint": {"lat": mid_lat, "lon": mid_lon},
        "start_date": start_date,
        "end_date": end_date,
        "hourly": all_hours,
    }


def _determine_status(wind_kn: float, wave_m: float, vessel_type: str) -> tuple[str, str]:
    """
    Determine likely sailing status based on real weather conditions.

    Returns (status, reason) tuple.
    """
    category = VESSEL_TYPES.get(vessel_type, "conventional")
    bf_threshold = SAILING_BAN_THRESHOLDS[category]
    bf = knots_to_beaufort(wind_kn)

    # Wave thresholds (matching sailing_ban_checker.py)
    wave_thresholds = {
        "high_speed": 2.5,
        "conventional": 5.0,
        "small_craft": 2.0,
    }
    wave_threshold = wave_thresholds[category]

    if bf >= bf_threshold:
        return "CANCELLED", "sailing_ban_wind"
    if wave_m >= wave_threshold:
        return "CANCELLED", "sailing_ban_waves"
    if bf >= bf_threshold - 1:
        return "AT_RISK", "near_threshold"
    return "SAILED", ""


def build_historical_dataset(
    days_back: int = 365,
    routes: list[str] | None = None,
    vessel_types: list[str] | None = None,
    output_dir: Path | str | None = None,
) -> dict:
    """
    Build a historical training dataset from real Open-Meteo weather data.

    For each route and time point, records the actual weather conditions
    and the sailing status that those conditions would have produced
    under Coast Guard rules.

    Args:
        days_back: How many days of history to fetch (default: 365)
        routes: List of route IDs to include (default: all routes)
        vessel_types: Vessel types to simulate (default: CONVENTIONAL + HIGH_SPEED)
        output_dir: Where to save the CSV (default: data/historical/)

    Returns:
        Dict with stats about the generated dataset.
    """
    out = Path(output_dir) if output_dir else DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    if routes is None:
        routes = list(ROUTES.keys())
    if vessel_types is None:
        vessel_types = ["CONVENTIONAL", "HIGH_SPEED"]

    end_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back + 7)).strftime("%Y-%m-%d")

    csv_path = out / "historical_weather_dataset.csv"
    fieldnames = [
        "date", "hour", "route_id", "vessel_type", "status", "reason",
        "wind_speed_kn", "wind_gusts_kn", "wind_beaufort", "wind_direction",
        "wave_height_m", "wave_period_s", "swell_height_m", "visibility_m",
    ]

    total_records = 0
    cancelled_count = 0
    routes_done = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for route_id in routes:
            origin_code, dest_code = route_id.split("-")
            if origin_code not in PORTS or dest_code not in PORTS:
                logger.warning("Skipping %s — missing port data", route_id)
                continue

            logger.info("Fetching historical data for %s (%s to %s)...",
                        route_id, start_date, end_date)
            print(f"  Fetching {route_id}...", end=" ", flush=True)

            try:
                data = fetch_historical_for_route(route_id, start_date, end_date)
            except Exception as e:
                logger.error("Failed to fetch %s: %s", route_id, e)
                print(f"FAILED ({e})")
                continue

            # Process each hour
            route_count = 0
            for timestamp, conditions in sorted(data["hourly"].items()):
                wind_kn = conditions.get("wind_speed_kn")
                wave_m = conditions.get("wave_height")

                # Skip hours with missing critical data
                if wind_kn is None or wave_m is None:
                    continue

                wind_gusts = conditions.get("wind_gusts_kn") or wind_kn * 1.3
                wave_period = conditions.get("wave_period") or 0.0
                swell = conditions.get("swell_wave_height") or 0.0
                visibility = conditions.get("visibility") or 50000.0
                wind_dir = conditions.get("wind_direction") or 0.0

                try:
                    dt = datetime.fromisoformat(timestamp)
                except (ValueError, TypeError):
                    continue

                # Ferries typically operate 06:00-23:00
                if dt.hour < 6 or dt.hour > 22:
                    continue

                for vessel_type in vessel_types:
                    status, reason = _determine_status(wind_kn, wave_m, vessel_type)

                    row = {
                        "date": dt.strftime("%Y-%m-%d"),
                        "hour": dt.strftime("%H:%M"),
                        "route_id": route_id,
                        "vessel_type": vessel_type,
                        "status": status,
                        "reason": reason,
                        "wind_speed_kn": round(wind_kn, 1),
                        "wind_gusts_kn": round(wind_gusts, 1),
                        "wind_beaufort": knots_to_beaufort(wind_kn),
                        "wind_direction": round(wind_dir, 0),
                        "wave_height_m": round(wave_m, 2),
                        "wave_period_s": round(wave_period, 1),
                        "swell_height_m": round(swell, 2),
                        "visibility_m": round(visibility, 0),
                    }
                    writer.writerow(row)
                    total_records += 1
                    route_count += 1
                    if status == "CANCELLED":
                        cancelled_count += 1

            routes_done += 1
            print(f"OK ({route_count} records)")

    cancel_rate = cancelled_count / total_records if total_records else 0

    stats = {
        "csv_path": str(csv_path),
        "total_records": total_records,
        "cancelled": cancelled_count,
        "sailed_or_risk": total_records - cancelled_count,
        "cancel_rate": round(cancel_rate, 3),
        "routes_fetched": routes_done,
        "date_range": {"start": start_date, "end": end_date},
    }

    # Save stats alongside CSV
    stats_path = out / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Historical dataset built: %d records (%d cancelled, %.1f%%) -> %s",
                total_records, cancelled_count, cancel_rate * 100, csv_path)

    return stats


def load_historical_dataset(csv_path: str | Path | None = None) -> tuple[list, list]:
    """
    Load the historical dataset CSV and convert to (X, y) for ML training.

    Returns:
        (X, y) where X is a list of feature vectors and y is a list of labels
        (1 = cancelled, 0 = sailed/at_risk).
    """
    from src.models.ml_predictor import extract_features

    if csv_path is None:
        csv_path = DATA_DIR / "historical_weather_dataset.csv"
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Historical dataset not found at {csv_path}. "
            "Run `python main.py --fetch-historical` first."
        )

    X = []
    y = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wind_kn = float(row["wind_speed_kn"])
            wind_gusts = float(row["wind_gusts_kn"])
            wave_m = float(row["wave_height_m"])
            wave_period = float(row["wave_period_s"])
            swell = float(row["swell_height_m"])
            visibility = float(row["visibility_m"])
            vessel_type = row["vessel_type"]
            hour = int(row["hour"].split(":")[0])

            date_str = row["date"]
            try:
                month = datetime.strptime(date_str, "%Y-%m-%d").month
            except (ValueError, TypeError):
                month = 6

            route_id = row["route_id"]
            exposed = ROUTES.get(route_id, {}).get("exposed", True)

            features = extract_features(
                wind_speed_knots=wind_kn,
                wave_height_m=wave_m,
                wave_period_s=wave_period,
                wind_gust_knots=wind_gusts,
                visibility_m=visibility,
                swell_height_m=swell,
                vessel_type=vessel_type,
                exposed_route=exposed,
                hour_of_day=hour,
                month=month,
            )
            X.append(features)
            y.append(1 if row["status"] == "CANCELLED" else 0)

    logger.info("Loaded %d records from %s (%d cancelled)",
                len(X), csv_path, sum(y))
    return X, y
