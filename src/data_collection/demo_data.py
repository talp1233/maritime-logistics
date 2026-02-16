"""
Realistic demo/offline data for the maritime intelligence platform.

Generates synthetic but realistic weather patterns for the Aegean Sea,
including seasonal Meltemi wind patterns and typical wave responses.
"""

import math
import random
from datetime import datetime, timedelta


# Seasonal wind profiles for Aegean Sea (average knots by month)
# Meltemi peaks in July-August
MONTHLY_WIND_AVG = {
    1: 18, 2: 17, 3: 15, 4: 12, 5: 14, 6: 18,
    7: 25, 8: 26, 9: 20, 10: 16, 11: 17, 12: 19,
}

# Wave height roughly correlates: wave_m ≈ 0.06 * wind_knots^1.3
# but with lag and swell components


def _wind_to_wave(wind_knots: float) -> float:
    """Approximate wave height from wind speed (simplified Sverdrup-Munk)."""
    if wind_knots <= 0:
        return 0.1
    return round(0.06 * (wind_knots ** 1.3) + random.gauss(0, 0.2), 2)


def generate_hourly_forecast(
    base_wind_knots: float,
    hours: int = 48,
    start_time: datetime | None = None,
    storm_probability: float = 0.15,
) -> dict:
    """
    Generate realistic hourly weather data.

    Args:
        base_wind_knots: Average wind for the period
        hours: Number of hours to generate
        start_time: Start datetime (defaults to now)
        storm_probability: Chance of a storm window appearing

    Returns:
        dict matching Open-Meteo response structure
    """
    if start_time is None:
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0)

    times = []
    wind_speeds = []
    wind_gusts = []
    wind_directions = []
    wave_heights = []
    wave_periods = []
    visibilities = []

    # Decide if there's a storm window
    has_storm = random.random() < storm_probability
    storm_start = random.randint(8, hours - 12) if has_storm else -1
    storm_duration = random.randint(6, 18) if has_storm else 0
    storm_peak = base_wind_knots * random.uniform(1.8, 2.8)

    for h in range(hours):
        t = start_time + timedelta(hours=h)
        times.append(t.strftime("%Y-%m-%dT%H:%M"))

        # Diurnal cycle: wind peaks in afternoon (14:00), calms at night (04:00)
        hour_of_day = t.hour
        diurnal = 1.0 + 0.3 * math.sin(math.pi * (hour_of_day - 6) / 12)

        # Storm overlay
        if storm_start <= h < storm_start + storm_duration:
            progress = (h - storm_start) / storm_duration
            # Bell curve for storm
            storm_factor = math.exp(-8 * (progress - 0.5) ** 2)
            wind = base_wind_knots * diurnal + storm_peak * storm_factor
        else:
            wind = base_wind_knots * diurnal + random.gauss(0, 2)

        wind = max(0, round(wind, 1))
        gust = round(wind * random.uniform(1.2, 1.6), 1)

        # Meltemi blows from N/NE (0-45 degrees) in summer
        direction = random.gauss(20, 30) % 360

        wave = max(0.1, _wind_to_wave(wind))
        period = round(3.5 + wind * 0.1 + random.gauss(0, 0.5), 1)
        visibility = max(1000, round(50000 - wind * 500 + random.gauss(0, 3000)))

        wind_speeds.append(wind)
        wind_gusts.append(gust)
        wind_directions.append(round(direction, 0))
        wave_heights.append(wave)
        wave_periods.append(max(2.0, period))
        visibilities.append(visibility)

    return {
        "weather": {
            "hourly": {
                "time": times,
                "wind_speed_10m": wind_speeds,
                "wind_gusts_10m": wind_gusts,
                "wind_direction_10m": wind_directions,
                "visibility": visibilities,
                "precipitation": [0.0] * hours,
            }
        },
        "marine": {
            "hourly": {
                "time": times,
                "wave_height": wave_heights,
                "wave_direction": wind_directions,
                "wave_period": wave_periods,
                "wind_wave_height": [round(w * 0.7, 2) for w in wave_heights],
                "swell_wave_height": [round(w * 0.4, 2) for w in wave_heights],
                "swell_wave_direction": [round((d + 15) % 360, 0) for d in wind_directions],
                "swell_wave_period": [round(p + 2, 1) for p in wave_periods],
            }
        },
    }


def generate_demo_route_conditions(
    forecast_days: int = 2,
    scenario: str = "auto",
) -> dict:
    """
    Generate demo weather data for a route (origin, midpoint, destination).

    Args:
        forecast_days: Number of days
        scenario: One of "calm", "storm", "meltemi", "auto"
            "auto" picks based on current month

    Returns:
        dict matching fetch_route_conditions() output structure
    """
    hours = forecast_days * 24

    if scenario == "auto":
        month = datetime.now().month
        base_wind = MONTHLY_WIND_AVG.get(month, 15)
        storm_prob = 0.3 if month in (7, 8) else 0.15
    elif scenario == "calm":
        base_wind = 8
        storm_prob = 0.0
    elif scenario == "storm":
        base_wind = 22
        storm_prob = 0.95
    elif scenario == "meltemi":
        base_wind = 28
        storm_prob = 0.5
    else:
        base_wind = 15
        storm_prob = 0.15

    # Generate slightly different conditions for each point
    origin = generate_hourly_forecast(base_wind * 0.8, hours, storm_probability=storm_prob * 0.7)
    midpoint = generate_hourly_forecast(base_wind, hours, storm_probability=storm_prob)
    dest = generate_hourly_forecast(base_wind * 0.9, hours, storm_probability=storm_prob * 0.8)

    return {
        "origin": origin,
        "midpoint": midpoint,
        "destination": dest,
    }


# Pre-built scenarios for quick demos
DEMO_SCENARIOS = {
    "calm_summer": {"scenario": "calm", "description": "Calm summer day — light winds, low waves"},
    "storm": {"scenario": "storm", "description": "Winter storm — gale force winds, high seas"},
    "meltemi": {"scenario": "meltemi", "description": "Summer Meltemi — strong N winds, typical Aegean"},
    "auto": {"scenario": "auto", "description": "Automatic — based on current month's typical conditions"},
}
