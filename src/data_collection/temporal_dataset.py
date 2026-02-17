"""
Temporal dataset builder for early cancellation prediction.

Instead of looking at conditions at departure time only, this module
builds feature vectors from the 5-day weather window leading up to
each event.  This captures storm buildup patterns that allow
prediction 3 days before departure.

Approach:
  1. For each ground truth record (date, route, status):
     - Collect hourly weather for D-5 through D-0 (144 hours)
     - Aggregate into daily summaries + trend features
  2. Feature vector captures:
     - Per-day wind/wave aggregates (mean, max) for D-5..D-0
     - Wind and wave trend slopes (linear regression over 5 days)
     - Storm buildup intensity (recent vs earlier conditions)
     - Hours above critical thresholds in the window
     - Route and vessel context

Data source:
  - PRIMARY: Open-Meteo Archive API (real historical weather)
  - FALLBACK: Synthetic generation from event-day conditions
"""

from __future__ import annotations

import csv
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

from src.config.constants import (
    PORTS,
    ROUTES,
    VESSEL_TYPES,
    SAILING_BAN_THRESHOLDS,
    knots_to_beaufort,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "temporal"

# ── Feature names for the temporal model ─────────────────────────────

TEMPORAL_FEATURE_NAMES = [
    # Per-day wind (D-5 to D-0 = 6 values each)
    "wind_mean_d5", "wind_mean_d4", "wind_mean_d3",
    "wind_mean_d2", "wind_mean_d1", "wind_mean_d0",
    "wind_max_d5", "wind_max_d4", "wind_max_d3",
    "wind_max_d2", "wind_max_d1", "wind_max_d0",
    # Per-day waves (D-5 to D-0)
    "wave_mean_d5", "wave_mean_d4", "wave_mean_d3",
    "wave_mean_d2", "wave_mean_d1", "wave_mean_d0",
    "wave_max_d5", "wave_max_d4", "wave_max_d3",
    "wave_max_d2", "wave_max_d1", "wave_max_d0",
    # Trend features
    "wind_trend_slope",        # kn/day increase over 5 days
    "wave_trend_slope",        # m/day increase over 5 days
    "wind_accel",              # (D-1+D0 avg) / (D-5+D-4 avg) ratio
    "wave_accel",              # same for waves
    # Storm buildup features
    "hours_above_bf6",         # total hours with Bf >= 6 in window
    "hours_above_bf8",         # total hours with Bf >= 8 in window
    "hours_above_wave2m",      # total hours with Hs >= 2m
    "hours_above_wave4m",      # total hours with Hs >= 4m
    "max_consecutive_bf6",     # longest streak above Bf 6
    "peak_wind_kn",            # absolute max wind in 5-day window
    "peak_wave_m",             # absolute max wave in 5-day window
    "peak_hours_before",       # hours between peak and departure
    # Context
    "route_exposed",           # 1.0 if route is exposed
    "vessel_is_highspeed",     # 1.0 if high-speed vessel
    "bf_threshold",            # vessel category threshold
    "month_norm",              # month / 12
    "day_of_week_norm",        # day_of_week / 7
]


# ── Linear regression helper ─────────────────────────────────────────

def _linear_slope(values: list[float]) -> float:
    """
    Simple linear regression slope for evenly-spaced values.

    Returns the daily rate of change.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _max_consecutive(hourly_flags: list[bool]) -> int:
    """Find longest consecutive True streak."""
    max_run = 0
    current = 0
    for flag in hourly_flags:
        if flag:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


# ── Feature extraction from a 5-day hourly window ────────────────────

def extract_temporal_features(
    daily_wind_means: list[float],
    daily_wind_maxes: list[float],
    daily_wave_means: list[float],
    daily_wave_maxes: list[float],
    hourly_winds: list[float],
    hourly_waves: list[float],
    route_id: str | None = None,
    vessel_type: str = "CONVENTIONAL",
    event_date: str | None = None,
) -> list[float]:
    """
    Extract temporal feature vector from a 5-day weather window.

    Args:
        daily_wind_means: [D-5, D-4, D-3, D-2, D-1, D-0] mean wind (kn)
        daily_wind_maxes: [D-5, ..., D-0] max wind (kn)
        daily_wave_means: [D-5, ..., D-0] mean wave height (m)
        daily_wave_maxes: [D-5, ..., D-0] max wave height (m)
        hourly_winds:     Full hourly wind values (up to 144 hours)
        hourly_waves:     Full hourly wave values (up to 144 hours)
        route_id:         Route key for geographic context
        vessel_type:      Vessel type key
        event_date:       Date string for temporal context

    Returns:
        Feature vector (list of floats) matching TEMPORAL_FEATURE_NAMES
    """
    # Pad to 6 days if shorter
    while len(daily_wind_means) < 6:
        daily_wind_means.insert(0, daily_wind_means[0] if daily_wind_means else 0)
    while len(daily_wind_maxes) < 6:
        daily_wind_maxes.insert(0, daily_wind_maxes[0] if daily_wind_maxes else 0)
    while len(daily_wave_means) < 6:
        daily_wave_means.insert(0, daily_wave_means[0] if daily_wave_means else 0)
    while len(daily_wave_maxes) < 6:
        daily_wave_maxes.insert(0, daily_wave_maxes[0] if daily_wave_maxes else 0)

    features: list[float] = []

    # Per-day wind means (D-5 to D-0)
    features.extend(daily_wind_means[:6])
    # Per-day wind maxes
    features.extend(daily_wind_maxes[:6])
    # Per-day wave means
    features.extend(daily_wave_means[:6])
    # Per-day wave maxes
    features.extend(daily_wave_maxes[:6])

    # Trend slopes (kn/day and m/day)
    wind_slope = _linear_slope(daily_wind_means[:6])
    wave_slope = _linear_slope(daily_wave_means[:6])
    features.append(wind_slope)
    features.append(wave_slope)

    # Acceleration: recent (D-1+D-0) vs earlier (D-5+D-4)
    recent_wind = (daily_wind_means[4] + daily_wind_means[5]) / 2.0
    early_wind = (daily_wind_means[0] + daily_wind_means[1]) / 2.0
    wind_accel = recent_wind / early_wind if early_wind > 1 else 1.0
    features.append(wind_accel)

    recent_wave = (daily_wave_means[4] + daily_wave_means[5]) / 2.0
    early_wave = (daily_wave_means[0] + daily_wave_means[1]) / 2.0
    wave_accel = recent_wave / early_wave if early_wave > 0.1 else 1.0
    features.append(wave_accel)

    # Storm threshold features (from hourly data)
    bf_flags_6 = [knots_to_beaufort(w) >= 6 for w in hourly_winds]
    bf_flags_8 = [knots_to_beaufort(w) >= 8 for w in hourly_winds]
    wave_flags_2 = [w >= 2.0 for w in hourly_waves]
    wave_flags_4 = [w >= 4.0 for w in hourly_waves]

    features.append(float(sum(bf_flags_6)))
    features.append(float(sum(bf_flags_8)))
    features.append(float(sum(wave_flags_2)))
    features.append(float(sum(wave_flags_4)))

    features.append(float(_max_consecutive(bf_flags_6)))

    # Peak conditions
    peak_wind = max(hourly_winds) if hourly_winds else 0
    peak_wave = max(hourly_waves) if hourly_waves else 0
    features.append(peak_wind)
    features.append(peak_wave)

    # Hours between peak wind and end (departure)
    if hourly_winds and peak_wind > 0:
        peak_idx = hourly_winds.index(peak_wind)
        features.append(float(len(hourly_winds) - 1 - peak_idx))
    else:
        features.append(0.0)

    # Context features
    route = ROUTES.get(route_id, {})
    features.append(1.0 if route.get("exposed", True) else 0.0)

    category = VESSEL_TYPES.get(vessel_type, "conventional")
    features.append(1.0 if category == "high_speed" else 0.0)
    features.append(float(SAILING_BAN_THRESHOLDS.get(category, 8)))

    # Temporal context
    month = 6
    dow = 3
    if event_date:
        try:
            dt = datetime.strptime(event_date, "%Y-%m-%d")
            month = dt.month
            dow = dt.weekday()
        except (ValueError, TypeError):
            pass
    features.append(month / 12.0)
    features.append(dow / 7.0)

    return features


# ── Build temporal dataset from ground truth + Open-Meteo Archive ─────

def build_temporal_dataset_from_archive(
    ground_truth_path: Path | None = None,
    output_dir: Path | None = None,
    lookback_days: int = 5,
) -> dict:
    """
    Build temporal dataset by fetching 5-day weather windows from
    Open-Meteo Archive API for each ground truth event.

    REQUIRES INTERNET. Use build_temporal_dataset_synthetic() as fallback.

    Returns:
        Stats dict with path to saved dataset.
    """
    from src.data_collection.ground_truth import GroundTruthCollector
    from src.data_collection.historical_weather import (
        fetch_historical_marine,
        fetch_historical_weather,
    )

    gt_path = ground_truth_path or Path("data/ground_truth/cancellation_records.csv")
    out = Path(output_dir) if output_dir else DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    collector = GroundTruthCollector()
    if ground_truth_path:
        collector.records_file = gt_path
    records = collector.load_records()

    if not records:
        return {"error": "No ground truth records found"}

    # Group records by (date, route) to avoid duplicate fetches
    event_keys: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        key = (r["date"], r["route_id"])
        event_keys.setdefault(key, []).append(r)

    csv_path = out / "temporal_dataset.csv"
    fieldnames = TEMPORAL_FEATURE_NAMES + ["label"]  # label = 1 cancelled, 0 sailed

    total = 0
    cancelled = 0
    errors = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (event_date, route_id), route_records in event_keys.items():
            # Calculate window
            try:
                d = datetime.strptime(event_date, "%Y-%m-%d")
            except ValueError:
                errors += 1
                continue

            start = (d - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            end = event_date

            origin_code, dest_code = route_id.split("-")
            if origin_code not in PORTS or dest_code not in PORTS:
                errors += 1
                continue

            origin = PORTS[origin_code]
            dest = PORTS[dest_code]
            mid_lat = (origin["lat"] + dest["lat"]) / 2
            mid_lon = (origin["lon"] + dest["lon"]) / 2

            # Fetch weather window
            try:
                marine = fetch_historical_marine(mid_lat, mid_lon, start, end)
                weather = fetch_historical_weather(mid_lat, mid_lon, start, end)
            except Exception as e:
                logger.warning("Failed to fetch %s/%s: %s", event_date, route_id, e)
                errors += 1
                continue

            # Parse hourly data
            w_hourly = weather.get("hourly", {})
            m_hourly = marine.get("hourly", {})
            winds = [w or 0 for w in w_hourly.get("wind_speed_10m", [])]
            waves = [w or 0 for w in m_hourly.get("wave_height", [])]
            times = w_hourly.get("time", [])

            # Aggregate into daily summaries
            daily_winds: dict[str, list[float]] = {}
            daily_waves: dict[str, list[float]] = {}

            for i, t in enumerate(times):
                day = t[:10]
                wind = winds[i] if i < len(winds) else 0
                wave = waves[i] if i < len(waves) else 0
                daily_winds.setdefault(day, []).append(wind)
                daily_waves.setdefault(day, []).append(wave)

            # Build sorted daily aggregates (D-5 to D-0)
            sorted_days = sorted(daily_winds.keys())[-6:]
            wind_means = [sum(daily_winds[d]) / len(daily_winds[d]) for d in sorted_days]
            wind_maxes = [max(daily_winds[d]) for d in sorted_days]
            wave_means = [sum(daily_waves.get(d, [0])) / max(1, len(daily_waves.get(d, [0])))
                          for d in sorted_days]
            wave_maxes = [max(daily_waves.get(d, [0])) for d in sorted_days]

            # Process each record for this date/route
            for r in route_records:
                features = extract_temporal_features(
                    daily_wind_means=list(wind_means),
                    daily_wind_maxes=list(wind_maxes),
                    daily_wave_means=list(wave_means),
                    daily_wave_maxes=list(wave_maxes),
                    hourly_winds=winds,
                    hourly_waves=waves,
                    route_id=route_id,
                    vessel_type=r.get("vessel_type", "CONVENTIONAL"),
                    event_date=event_date,
                )

                label = 1 if r["status"] == "CANCELLED" else 0

                row = dict(zip(TEMPORAL_FEATURE_NAMES, features))
                row["label"] = label
                writer.writerow(row)

                total += 1
                if label:
                    cancelled += 1

    stats = {
        "csv_path": str(csv_path),
        "total_records": total,
        "cancelled": cancelled,
        "cancel_rate": round(cancelled / total, 3) if total else 0,
        "unique_events": len(event_keys),
        "fetch_errors": errors,
    }

    stats_path = out / "temporal_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Built temporal dataset: %d records (%d cancelled) -> %s",
                total, cancelled, csv_path)
    return stats


# ── Synthetic temporal dataset (no internet needed) ───────────────────

def build_temporal_dataset_synthetic(
    ground_truth_path: Path | None = None,
    output_dir: Path | None = None,
    lookback_days: int = 5,
    seed: int = 42,
) -> dict:
    """
    Build temporal dataset by synthesizing the 5-day lead-up from
    the event-day conditions in ground truth.

    Uses meteorological heuristics:
      - Storm buildup: wind increases ~3-5 kn/day before peak
      - Wave lag: waves lag wind by 6-12 hours, grow more slowly
      - Persistence: weather systems last 2-4 days
      - Seasonal: Meltemi (Jul-Aug) has different patterns

    This is NOT real data, but captures realistic temporal patterns
    for model architecture validation.
    """
    from src.data_collection.ground_truth import GroundTruthCollector

    rng = random.Random(seed)

    gt_path = ground_truth_path or Path("data/ground_truth/cancellation_records.csv")
    out = Path(output_dir) if output_dir else DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    collector = GroundTruthCollector()
    if ground_truth_path:
        collector.records_file = gt_path
    records = collector.load_records()

    if not records:
        return {"error": "No ground truth records found"}

    csv_path = out / "temporal_dataset.csv"
    fieldnames = TEMPORAL_FEATURE_NAMES + ["label"]

    total = 0
    cancelled = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            event_wind = r["wind_speed_kn"]
            event_wave = r["wave_height_m"]
            event_date = r["date"]

            # Synthesize the 5-day buildup
            daily_wind_means: list[float] = []
            daily_wind_maxes: list[float] = []
            daily_wave_means: list[float] = []
            daily_wave_maxes: list[float] = []
            all_hourly_winds: list[float] = []
            all_hourly_waves: list[float] = []

            is_cancellation = r["status"] == "CANCELLED"

            for day_offset in range(lookback_days, -1, -1):
                # day_offset: 5=D-5, 4=D-4, ..., 0=D-0
                if is_cancellation:
                    # Storm buildup: progressive increase
                    # D-5: ~40% of event wind, D-0: 100%
                    progress = 1.0 - (day_offset / (lookback_days + 1))
                    # S-curve for more realistic buildup
                    s_curve = 1.0 / (1.0 + math.exp(-8.0 * (progress - 0.5)))
                    day_base_wind = event_wind * (0.3 + 0.7 * s_curve)
                    # Waves lag wind by ~0.5 day
                    wave_progress = max(0, progress - 0.08)
                    s_wave = 1.0 / (1.0 + math.exp(-8.0 * (wave_progress - 0.5)))
                    day_base_wave = event_wave * (0.3 + 0.7 * s_wave)
                else:
                    # Normal conditions: relatively stable, mild
                    day_base_wind = event_wind * rng.uniform(0.7, 1.3)
                    day_base_wave = event_wave * rng.uniform(0.7, 1.3)

                # Generate 24 hourly values with diurnal pattern
                hourly_winds: list[float] = []
                hourly_waves: list[float] = []
                for h in range(24):
                    # Diurnal cycle: winds peak at 14:00, trough at 06:00
                    diurnal = 1.0 + 0.15 * math.sin(math.pi * (h - 6) / 12.0)
                    wind_h = max(0, day_base_wind * diurnal + rng.gauss(0, 2))
                    # Waves are smoother (sea state has inertia)
                    wave_h = max(0.1, day_base_wave * (0.9 + 0.1 * diurnal)
                                 + rng.gauss(0, 0.15))
                    hourly_winds.append(wind_h)
                    hourly_waves.append(wave_h)

                daily_wind_means.append(sum(hourly_winds) / 24.0)
                daily_wind_maxes.append(max(hourly_winds))
                daily_wave_means.append(sum(hourly_waves) / 24.0)
                daily_wave_maxes.append(max(hourly_waves))
                all_hourly_winds.extend(hourly_winds)
                all_hourly_waves.extend(hourly_waves)

            features = extract_temporal_features(
                daily_wind_means=daily_wind_means,
                daily_wind_maxes=daily_wind_maxes,
                daily_wave_means=daily_wave_means,
                daily_wave_maxes=daily_wave_maxes,
                hourly_winds=all_hourly_winds,
                hourly_waves=all_hourly_waves,
                route_id=r["route_id"],
                vessel_type=r.get("vessel_type", "CONVENTIONAL"),
                event_date=event_date,
            )

            label = 1 if is_cancellation else 0
            row = dict(zip(TEMPORAL_FEATURE_NAMES, features))
            row["label"] = label
            writer.writerow(row)

            total += 1
            if label:
                cancelled += 1

    stats = {
        "csv_path": str(csv_path),
        "total_records": total,
        "cancelled": cancelled,
        "cancel_rate": round(cancelled / total, 3) if total else 0,
        "source": "synthetic",
    }

    stats_path = out / "temporal_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Built synthetic temporal dataset: %d records (%d cancelled) -> %s",
                total, cancelled, csv_path)
    return stats


# ── Load temporal dataset for training ────────────────────────────────

def load_temporal_dataset(
    csv_path: Path | str | None = None,
) -> tuple[list[list[float]], list[int]]:
    """
    Load temporal dataset CSV into (X, y) for ML training.

    Returns:
        (X, y) where X is list of feature vectors, y is list of labels.
    """
    if csv_path is None:
        csv_path = DATA_DIR / "temporal_dataset.csv"
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Temporal dataset not found at {csv_path}. "
            "Run `python main.py --build-temporal` first."
        )

    X: list[list[float]] = []
    y: list[int] = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            features = [float(row[name]) for name in TEMPORAL_FEATURE_NAMES]
            X.append(features)
            y.append(int(float(row["label"])))

    logger.info("Loaded temporal dataset: %d records (%d cancelled) from %s",
                len(X), sum(y), csv_path)
    return X, y
