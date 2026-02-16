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
from datetime import datetime, timedelta
from typing import Optional

# Base URLs for Open-Meteo APIs
MARINE_API_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_marine_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 3,
) -> dict:
    """
    Fetch marine forecast (waves) for a given location.

    Args:
        latitude: Location latitude (e.g. 37.9475 for Piraeus)
        longitude: Location longitude (e.g. 23.6372 for Piraeus)
        forecast_days: Number of days to forecast (1-7)

    Returns:
        dict with 'hourly' key containing time series of marine variables.
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
        "forecast_days": forecast_days,
        "timezone": "Europe/Athens",
    }
    url = f"{MARINE_API_URL}?{urllib.parse.urlencode(params)}"
    return _fetch_json(url)


def fetch_weather_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 3,
) -> dict:
    """
    Fetch weather forecast (wind, visibility) for a given location.

    Args:
        latitude: Location latitude
        longitude: Location longitude
        forecast_days: Number of days to forecast (1-7)

    Returns:
        dict with 'hourly' key containing time series of weather variables.
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
        "wind_speed_unit": "kn",  # knots — matches our Beaufort conversion
        "forecast_days": forecast_days,
        "timezone": "Europe/Athens",
    }
    url = f"{WEATHER_API_URL}?{urllib.parse.urlencode(params)}"
    return _fetch_json(url)


def fetch_route_conditions(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    forecast_days: int = 3,
) -> dict:
    """
    Fetch conditions for both endpoints of a route.

    Returns dict with 'origin' and 'destination' forecasts.
    For a more accurate mid-route forecast we use the midpoint coordinates.
    """
    mid_lat = (origin_lat + dest_lat) / 2
    mid_lon = (origin_lon + dest_lon) / 2

    return {
        "origin": {
            "marine": fetch_marine_forecast(origin_lat, origin_lon, forecast_days),
            "weather": fetch_weather_forecast(origin_lat, origin_lon, forecast_days),
        },
        "midpoint": {
            "marine": fetch_marine_forecast(mid_lat, mid_lon, forecast_days),
            "weather": fetch_weather_forecast(mid_lat, mid_lon, forecast_days),
        },
        "destination": {
            "marine": fetch_marine_forecast(dest_lat, dest_lon, forecast_days),
            "weather": fetch_weather_forecast(dest_lat, dest_lon, forecast_days),
        },
    }


def _fetch_json(url: str, timeout: int = 30) -> dict:
    """Fetch a URL and return parsed JSON. Raises on HTTP/network errors."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))
