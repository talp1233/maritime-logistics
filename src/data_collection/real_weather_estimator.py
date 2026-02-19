"""
Weather estimator for real cancellation events.

When Open-Meteo Archive is not accessible, this module estimates the
5-day weather buildup based on:
  - Reported Beaufort levels from news sources (real)
  - Known wind-wave relationships (Pierson-Moskowitz spectrum)
  - Winter Aegean climatology (seasonal baseline)
  - Storm evolution patterns (based on synoptic meteorology)

This bridges the gap between fully synthetic data and real API data.
The TIMING of events is real, the SEVERITY is real (from news reports),
and the BUILDUP pattern follows established meteorological principles.

When internet is available, replace this with Open-Meteo Archive data
using: python main.py --analyze-real --fetch-real-weather
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

from src.config.constants import (
    BEAUFORT_SCALE,
    PORTS,
    ROUTES,
    VESSEL_TYPES,
    SAILING_BAN_THRESHOLDS,
    knots_to_beaufort,
)
from src.data_collection.temporal_dataset import extract_temporal_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Winter Aegean climatology (December-February)
# Based on published statistics for the Central Aegean
WINTER_AEGEAN_BASELINE = {
    "mean_wind_kn": 14.0,    # Average winter wind
    "std_wind_kn": 6.0,      # Standard deviation
    "mean_wave_m": 1.2,      # Average winter wave height
    "std_wave_m": 0.5,       # Standard deviation
    "strong_wind_prob": 0.15, # Probability of Bf >= 6 on any day
}

# Storm evolution patterns for the Aegean
# Based on synoptic meteorology of the Eastern Mediterranean
STORM_PATTERNS = {
    "rapid_onset": {
        # Rapid storms: pressure drops quickly, wind intensifies in 24-48h
        # Typical of cold front passages in winter
        "buildup_days": 2,
        "d5_fraction": 0.35,
        "d4_fraction": 0.40,
        "d3_fraction": 0.50,
        "d2_fraction": 0.65,
        "d1_fraction": 0.85,
        "d0_fraction": 1.00,
        "probability": 0.4,
    },
    "gradual_buildup": {
        # Gradual storms: pressure gradient increases over 3-4 days
        # Typical of deep lows approaching from the west
        "buildup_days": 4,
        "d5_fraction": 0.45,
        "d4_fraction": 0.55,
        "d3_fraction": 0.65,
        "d2_fraction": 0.78,
        "d1_fraction": 0.90,
        "d0_fraction": 1.00,
        "probability": 0.35,
    },
    "persistent": {
        # Persistent strong winds: already elevated, then spike
        # Typical of established northerly flow patterns
        "buildup_days": 5,
        "d5_fraction": 0.60,
        "d4_fraction": 0.65,
        "d3_fraction": 0.70,
        "d2_fraction": 0.80,
        "d1_fraction": 0.90,
        "d0_fraction": 1.00,
        "probability": 0.25,
    },
}


def beaufort_to_knots_mid(bf: int) -> float:
    """Convert Beaufort to midpoint knots value."""
    if bf in BEAUFORT_SCALE:
        lo, hi, _ = BEAUFORT_SCALE[bf]
        return (lo + hi) / 2.0
    return 0.0


def wind_to_wave_pm(wind_kn: float) -> float:
    """
    Estimate significant wave height from wind speed.

    Uses a simplified Pierson-Moskowitz relationship:
    Hs = 0.22 * (U^2 / g)   for fully developed sea

    In practice, fetch and duration limit wave growth.
    For the Aegean (limited fetch ~200km), we use a reduction factor.
    """
    wind_ms = wind_kn * 0.514444
    g = 9.81

    if wind_ms <= 0:
        return 0.1

    # Fully developed PM wave height
    hs_full = 0.22 * (wind_ms ** 2) / g

    # Fetch-limited reduction (Aegean ~200km fetch)
    fetch_km = 200.0
    fetch_factor = min(1.0, math.sqrt(fetch_km / 500.0))

    # Duration factor (storm needs time to build waves)
    hs = hs_full * fetch_factor

    return max(0.1, hs)


def estimate_5day_weather(
    event_date: str,
    reported_beaufort: int,
    route_id: str,
    rng: random.Random | None = None,
) -> dict:
    """
    Estimate the 5-day weather window leading up to a real event.

    Args:
        event_date: Date of the event (YYYY-MM-DD)
        reported_beaufort: Beaufort level reported in news
        route_id: Route for geographic context
        rng: Random generator for reproducibility

    Returns:
        Dict with daily and hourly weather data.
    """
    if rng is None:
        rng = random.Random(hash(event_date + route_id))

    # Convert reported Beaufort to knots
    peak_wind_kn = beaufort_to_knots_mid(reported_beaufort)
    # Add some variation (news reports are approximate)
    peak_wind_kn *= rng.uniform(0.9, 1.1)

    # Select storm pattern
    pattern_name = rng.choices(
        list(STORM_PATTERNS.keys()),
        weights=[p["probability"] for p in STORM_PATTERNS.values()],
    )[0]
    pattern = STORM_PATTERNS[pattern_name]

    # Build daily wind progression
    fractions = [
        pattern["d5_fraction"],
        pattern["d4_fraction"],
        pattern["d3_fraction"],
        pattern["d2_fraction"],
        pattern["d1_fraction"],
        pattern["d0_fraction"],
    ]

    daily_wind_means = []
    daily_wind_maxes = []
    daily_wave_means = []
    daily_wave_maxes = []
    all_hourly_winds = []
    all_hourly_waves = []

    for day_idx, frac in enumerate(fractions):
        day_base_wind = peak_wind_kn * frac

        # Add natural variability
        day_base_wind *= rng.uniform(0.85, 1.15)

        # Wave from wind (with lag for early days)
        wave_lag_factor = 0.8 if day_idx < 3 else 1.0
        day_base_wave = wind_to_wave_pm(day_base_wind) * wave_lag_factor

        # Generate 24 hourly values
        hourly_winds = []
        hourly_waves = []
        for h in range(24):
            # Diurnal cycle: peak at ~14:00, trough at ~06:00
            diurnal = 1.0 + 0.12 * math.sin(math.pi * (h - 6) / 12.0)
            # Synoptic variation
            synoptic = rng.gauss(0, day_base_wind * 0.08)
            wind_h = max(1, day_base_wind * diurnal + synoptic)

            # Wave inertia (smoother than wind)
            wave_noise = rng.gauss(0, day_base_wave * 0.06)
            wave_h = max(0.1, day_base_wave * (0.95 + 0.05 * diurnal) + wave_noise)

            hourly_winds.append(wind_h)
            hourly_waves.append(wave_h)

        daily_wind_means.append(sum(hourly_winds) / 24.0)
        daily_wind_maxes.append(max(hourly_winds))
        daily_wave_means.append(sum(hourly_waves) / 24.0)
        daily_wave_maxes.append(max(hourly_waves))
        all_hourly_winds.extend(hourly_winds)
        all_hourly_waves.extend(hourly_waves)

    return {
        "daily_wind_means": daily_wind_means,
        "daily_wind_maxes": daily_wind_maxes,
        "daily_wave_means": daily_wave_means,
        "daily_wave_maxes": daily_wave_maxes,
        "hourly_winds": all_hourly_winds,
        "hourly_waves": all_hourly_waves,
        "pattern_type": pattern_name,
        "peak_wind_kn": peak_wind_kn,
    }


def estimate_normal_day_weather(
    event_date: str,
    route_id: str,
    rng: random.Random | None = None,
) -> dict:
    """
    Estimate weather for a normal sailing day (no ban).

    Uses winter Aegean climatology with variability.
    Normal days have moderate winds (Bf 3-5) without buildup.
    """
    if rng is None:
        rng = random.Random(hash(event_date + route_id + "normal"))

    baseline = WINTER_AEGEAN_BASELINE

    # Normal day wind: centered around baseline, occasional gusty
    day_mean_wind = max(3, rng.gauss(baseline["mean_wind_kn"], baseline["std_wind_kn"]))

    # Ensure it stays below ban threshold
    if knots_to_beaufort(day_mean_wind) >= 7:
        day_mean_wind = rng.uniform(12, 20)  # Cap at moderate winds

    daily_wind_means = []
    daily_wind_maxes = []
    daily_wave_means = []
    daily_wave_maxes = []
    all_hourly_winds = []
    all_hourly_waves = []

    for day_idx in range(6):
        # Mild day-to-day variation (persistence)
        day_wind = day_mean_wind * rng.uniform(0.7, 1.3)
        day_wave = wind_to_wave_pm(day_wind)

        hourly_winds = []
        hourly_waves = []
        for h in range(24):
            diurnal = 1.0 + 0.10 * math.sin(math.pi * (h - 6) / 12.0)
            wind_h = max(1, day_wind * diurnal + rng.gauss(0, 2))
            wave_h = max(0.1, day_wave * (0.95 + 0.05 * diurnal) + rng.gauss(0, 0.1))
            hourly_winds.append(wind_h)
            hourly_waves.append(wave_h)

        daily_wind_means.append(sum(hourly_winds) / 24.0)
        daily_wind_maxes.append(max(hourly_winds))
        daily_wave_means.append(sum(hourly_waves) / 24.0)
        daily_wave_maxes.append(max(hourly_waves))
        all_hourly_winds.extend(hourly_winds)
        all_hourly_waves.extend(hourly_waves)

    return {
        "daily_wind_means": daily_wind_means,
        "daily_wind_maxes": daily_wind_maxes,
        "daily_wave_means": daily_wave_means,
        "daily_wave_maxes": daily_wave_maxes,
        "hourly_winds": all_hourly_winds,
        "hourly_waves": all_hourly_waves,
        "pattern_type": "normal",
        "peak_wind_kn": max(daily_wind_maxes),
    }


def build_real_temporal_features(records: list[dict], seed: int = 42) -> list[dict]:
    """
    Build temporal features for real ground truth records.

    For cancelled events: uses reported Beaufort + storm pattern estimation
    For sailed events: uses winter Aegean climatology

    Returns records augmented with temporal features.
    """
    from src.data_collection.real_cancellation_analyzer import CONFIRMED_BAN_DATES

    rng = random.Random(seed)
    augmented = []

    for r in records:
        event_date = r["date"]
        route_id = r["route_id"]
        ban_info = CONFIRMED_BAN_DATES.get(event_date)

        if r["status"] == "CANCELLED" and ban_info:
            weather = estimate_5day_weather(
                event_date, ban_info["beaufort"], route_id, rng
            )
        else:
            weather = estimate_normal_day_weather(event_date, route_id, rng)

        features = extract_temporal_features(
            daily_wind_means=weather["daily_wind_means"],
            daily_wind_maxes=weather["daily_wind_maxes"],
            daily_wave_means=weather["daily_wave_means"],
            daily_wave_maxes=weather["daily_wave_maxes"],
            hourly_winds=weather["hourly_winds"],
            hourly_waves=weather["hourly_waves"],
            route_id=route_id,
            vessel_type=r.get("vessel_type", "CONVENTIONAL"),
            event_date=event_date,
        )

        augmented_record = dict(r)
        augmented_record["temporal_features"] = features
        augmented_record["pattern_type"] = weather["pattern_type"]
        augmented.append(augmented_record)

    cancelled = sum(1 for r in augmented if r["status"] == "CANCELLED")
    logger.info("Built temporal features for %d records (%d cancelled)",
                len(augmented), cancelled)
    return augmented
