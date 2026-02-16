"""
Client for fetching marine weather data from Open-Meteo APIs.

Open-Meteo provides free weather and marine forecast data without API keys.
We use two endpoints:
  - Marine API: wave height, wave period, swell
  - Weather API: wind speed, gusts, visibility
"""

import json
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.cache import api_cache
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Base URLs for Open-Meteo APIs
MARINE_API_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_marine_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 3,
) -> dict:
    """Fetch marine forecast (waves) for a given location."""
    cache_key = f"marine:{latitude:.4f}:{longitude:.4f}:{forecast_days}"
    cached = api_cache.get(cache_key)
    if cached is not None:
        logger.debug("Cache hit for marine forecast at (%.4f, %.4f)", latitude, longitude)
        return cached

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
        "forecast_days": forecast_days,
        "timezone": "Europe/Athens",
    }
    url = f"{MARINE_API_URL}?{urllib.parse.urlencode(params)}"
    logger.info("Fetching marine forecast for (%.4f, %.4f)", latitude, longitude)
    result = _fetch_json(url)
    api_cache.set(cache_key, result)
    return result


def fetch_weather_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 3,
) -> dict:
    """Fetch weather forecast (wind, visibility) for a given location."""
    cache_key = f"weather:{latitude:.4f}:{longitude:.4f}:{forecast_days}"
    cached = api_cache.get(cache_key)
    if cached is not None:
        logger.debug("Cache hit for weather forecast at (%.4f, %.4f)", latitude, longitude)
        return cached

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
        "forecast_days": forecast_days,
        "timezone": "Europe/Athens",
    }
    url = f"{WEATHER_API_URL}?{urllib.parse.urlencode(params)}"
    logger.info("Fetching weather forecast for (%.4f, %.4f)", latitude, longitude)
    result = _fetch_json(url)
    api_cache.set(cache_key, result)
    return result


def fetch_route_conditions(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    forecast_days: int = 3,
) -> dict:
    """
    Fetch conditions for both endpoints of a route + midpoint.
    Uses parallel requests via ThreadPoolExecutor.
    """
    mid_lat = (origin_lat + dest_lat) / 2
    mid_lon = (origin_lon + dest_lon) / 2

    points = {
        "origin": (origin_lat, origin_lon),
        "midpoint": (mid_lat, mid_lon),
        "destination": (dest_lat, dest_lon),
    }

    results = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {}
        for name, (lat, lon) in points.items():
            futures[pool.submit(fetch_marine_forecast, lat, lon, forecast_days)] = (name, "marine")
            futures[pool.submit(fetch_weather_forecast, lat, lon, forecast_days)] = (name, "weather")

        for future in as_completed(futures):
            name, data_type = futures[future]
            if name not in results:
                results[name] = {}
            results[name][data_type] = future.result()

    return results


def _fetch_json(url: str, timeout: int = 30) -> dict:
    """Fetch a URL and return parsed JSON. Raises on HTTP/network errors."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))
