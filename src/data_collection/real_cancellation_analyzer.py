"""
Real Cancellation Analyzer — fetches weather for actual cancellation events
and discovers recurring patterns that precede sailing bans.

Workflow:
  1. Load real cancellation records (scraped from news sources)
  2. For each event, fetch 5-day weather window from Open-Meteo Archive
  3. Also fetch weather for NON-cancellation days (sailed days)
  4. Extract temporal features for all events
  5. Analyze patterns: what weather signatures precede cancellations?
  6. Build and validate prediction model on real data

Data Sources:
  - Cancellation dates: news articles (greekcitytimes, keeptalkinggreece,
    euronews, greekreporter, in.gr, tovima, athens24)
  - Weather: Open-Meteo Archive API (free, no key needed)

Usage:
    python -m src.data_collection.real_cancellation_analyzer
    python -m src.data_collection.real_cancellation_analyzer --analyze-only
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

from src.config.constants import PORTS, ROUTES, VESSEL_TYPES, SAILING_BAN_THRESHOLDS, knots_to_beaufort
from src.data_collection.temporal_dataset import (
    extract_temporal_features,
    TEMPORAL_FEATURE_NAMES,
    _linear_slope,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
REAL_EVENTS_CSV = DATA_DIR / "ground_truth" / "real_cancellation_events.csv"
REAL_TEMPORAL_CSV = DATA_DIR / "temporal" / "real_temporal_dataset.csv"
REAL_STATS_JSON = DATA_DIR / "temporal" / "real_temporal_stats.json"
PATTERN_REPORT = DATA_DIR / "temporal" / "pattern_analysis.json"

# Cancellation dates confirmed from news sources (our time window)
# Format: date -> {beaufort, ports_affected, source}
CONFIRMED_BAN_DATES = {
    "2026-01-03": {"beaufort": 7, "ports": ["PIR", "RAF"], "source": "greekcitytimes.com"},
    "2026-01-04": {"beaufort": 7, "ports": ["PIR", "RAF"], "source": "greekcitytimes.com"},
    "2026-01-08": {"beaufort": 9, "ports": ["PIR", "RAF", "LAV"], "source": "keeptalkinggreece.com"},
    "2026-01-10": {"beaufort": 9, "ports": ["PIR"], "source": "greekreporter.com"},
    "2026-01-21": {"beaufort": 9, "ports": ["PIR", "RAF", "LAV"], "source": "tovima.com"},
    "2026-01-26": {"beaufort": 8, "ports": ["PIR", "RAF", "LAV"], "source": "athens24.com"},
    "2026-02-15": {"beaufort": 9, "ports": ["PIR", "RAF", "LAV"], "source": "in.gr"},
}

# Routes we track
TRACKED_ROUTES = ["PIR-MYK", "PIR-NAX", "PIR-SAN", "PIR-HER", "RAF-MYK"]


def load_real_events() -> list[dict]:
    """Load real cancellation events from CSV."""
    if not REAL_EVENTS_CSV.exists():
        logger.error("Real events file not found: %s", REAL_EVENTS_CSV)
        return []

    records = []
    with open(REAL_EVENTS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    logger.info("Loaded %d real cancellation event records", len(records))
    return records


def build_real_ground_truth(
    start_date: str = "2025-12-01",
    end_date: str = "2026-02-15",
) -> list[dict]:
    """
    Build a complete ground truth dataset combining:
    - Real cancellation dates (from news sources)
    - Sailed dates (days with no reported ban)

    For each date in the range, for each route and vessel type:
    - If date is a confirmed ban date -> CANCELLED
    - Otherwise -> SAILED

    This gives us labeled events for training.
    """
    records = []

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    current = start

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        ban_info = CONFIRMED_BAN_DATES.get(date_str)

        for route_id in TRACKED_ROUTES:
            origin_code = route_id.split("-")[0]

            for vessel_type in ["CONVENTIONAL", "HIGH_SPEED"]:
                if ban_info and origin_code in ban_info["ports"]:
                    # For high-speed: always cancelled during ban
                    # For conventional: cancelled only if reported Bf >= 8
                    category = VESSEL_TYPES.get(vessel_type, "conventional")
                    threshold = SAILING_BAN_THRESHOLDS[category]

                    if ban_info["beaufort"] >= threshold:
                        status = "CANCELLED"
                        reason = "sailing_ban"
                    elif ban_info["beaufort"] >= threshold - 1:
                        # Near threshold — some HSC cancelled even below
                        status = "CANCELLED" if vessel_type == "HIGH_SPEED" else "SAILED"
                        reason = "precautionary" if status == "CANCELLED" else ""
                    else:
                        status = "SAILED"
                        reason = ""
                else:
                    status = "SAILED"
                    reason = ""

                records.append({
                    "date": date_str,
                    "route_id": route_id,
                    "vessel_type": vessel_type,
                    "status": status,
                    "reason": reason,
                    "source": ban_info["source"] if ban_info else "no_ban_reported",
                    "reported_beaufort": ban_info["beaufort"] if ban_info else None,
                })

        current += timedelta(days=1)

    cancelled = sum(1 for r in records if r["status"] == "CANCELLED")
    logger.info("Built real ground truth: %d records (%d cancelled, %.1f%%)",
                len(records), cancelled, 100 * cancelled / len(records) if records else 0)
    return records


def fetch_weather_for_events(
    records: list[dict],
    lookback_days: int = 5,
) -> list[dict]:
    """
    For each unique (date, route) pair, fetch the 5-day weather window
    from Open-Meteo Archive.

    Returns records augmented with temporal features.

    REQUIRES INTERNET ACCESS.
    """
    from src.data_collection.historical_weather import (
        fetch_historical_marine,
        fetch_historical_weather,
    )

    # Group by (date, route) to avoid duplicate fetches
    event_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        key = (r["date"], r["route_id"])
        event_groups[key].append(r)

    augmented_records = []
    fetch_count = 0
    error_count = 0

    for (event_date, route_id), group_records in sorted(event_groups.items()):
        # Calculate 5-day window
        try:
            d = datetime.strptime(event_date, "%Y-%m-%d")
        except ValueError:
            error_count += 1
            continue

        window_start = (d - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        # Get midpoint coordinates
        origin_code, dest_code = route_id.split("-")
        if origin_code not in PORTS or dest_code not in PORTS:
            error_count += 1
            continue

        origin = PORTS[origin_code]
        dest = PORTS[dest_code]
        mid_lat = (origin["lat"] + dest["lat"]) / 2
        mid_lon = (origin["lon"] + dest["lon"]) / 2

        # Fetch 5-day weather window
        try:
            marine = fetch_historical_marine(mid_lat, mid_lon, window_start, event_date)
            weather = fetch_historical_weather(mid_lat, mid_lon, window_start, event_date)
            fetch_count += 1
        except Exception as e:
            logger.warning("Failed to fetch weather for %s/%s: %s", event_date, route_id, e)
            error_count += 1
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
        if len(sorted_days) < 2:
            error_count += 1
            continue

        wind_means = [sum(daily_winds[d]) / len(daily_winds[d]) for d in sorted_days]
        wind_maxes = [max(daily_winds[d]) for d in sorted_days]
        wave_means = [
            sum(daily_waves.get(d, [0])) / max(1, len(daily_waves.get(d, [0])))
            for d in sorted_days
        ]
        wave_maxes = [max(daily_waves.get(d, [0])) for d in sorted_days]

        # Extract features for each record in this group
        for r in group_records:
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

            augmented = dict(r)
            augmented["temporal_features"] = features
            augmented["d0_wind_mean"] = wind_means[-1] if wind_means else 0
            augmented["d0_wind_max"] = wind_maxes[-1] if wind_maxes else 0
            augmented["d0_wave_mean"] = wave_means[-1] if wave_means else 0
            augmented["d0_wave_max"] = wave_maxes[-1] if wave_maxes else 0
            augmented_records.append(augmented)

        # Rate limiting
        time.sleep(0.3)

        if fetch_count % 10 == 0:
            print(f"  Fetched {fetch_count} weather windows ({error_count} errors)...")

    logger.info("Fetched weather for %d event groups (%d errors)", fetch_count, error_count)
    return augmented_records


def save_real_temporal_dataset(augmented_records: list[dict]) -> dict:
    """Save the real temporal dataset to CSV."""
    REAL_TEMPORAL_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = TEMPORAL_FEATURE_NAMES + ["label"]
    total = 0
    cancelled = 0

    with open(REAL_TEMPORAL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in augmented_records:
            if "temporal_features" not in r:
                continue

            label = 1 if r["status"] == "CANCELLED" else 0
            row = dict(zip(TEMPORAL_FEATURE_NAMES, r["temporal_features"]))
            row["label"] = label
            writer.writerow(row)

            total += 1
            if label:
                cancelled += 1

    stats = {
        "csv_path": str(REAL_TEMPORAL_CSV),
        "total_records": total,
        "cancelled": cancelled,
        "cancel_rate": round(cancelled / total, 3) if total else 0,
        "source": "real_news_events + open_meteo_archive",
        "ban_dates": list(CONFIRMED_BAN_DATES.keys()),
        "ban_dates_count": len(CONFIRMED_BAN_DATES),
    }

    with open(REAL_STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Saved real temporal dataset: %d records (%d cancelled) -> %s",
                total, cancelled, REAL_TEMPORAL_CSV)
    return stats


def analyze_patterns(augmented_records: list[dict]) -> dict:
    """
    Analyze weather patterns that precede cancellations vs normal sailing.

    Key questions:
    - How does wind evolve in the 5 days before a cancellation?
    - What's the typical wave buildup pattern?
    - How far in advance can we detect an approaching ban?
    - Are there route-specific differences?
    """
    cancelled = [r for r in augmented_records if r["status"] == "CANCELLED" and "temporal_features" in r]
    sailed = [r for r in augmented_records if r["status"] == "SAILED" and "temporal_features" in r]

    if not cancelled or not sailed:
        return {"error": "Need both cancelled and sailed records for pattern analysis"}

    # Feature indices for per-day wind means (D-5 through D-0)
    # 0-5: wind_mean D-5..D-0
    # 6-11: wind_max D-5..D-0
    # 12-17: wave_mean D-5..D-0
    # 18-23: wave_max D-5..D-0

    def avg_feature(records, idx):
        values = [r["temporal_features"][idx] for r in records if len(r["temporal_features"]) > idx]
        return sum(values) / len(values) if values else 0

    # Wind buildup comparison
    wind_buildup_cancel = {}
    wind_buildup_sail = {}
    wave_buildup_cancel = {}
    wave_buildup_sail = {}

    for d in range(6):
        day_label = f"D-{5-d}"
        wind_buildup_cancel[day_label] = round(avg_feature(cancelled, d), 1)
        wind_buildup_sail[day_label] = round(avg_feature(sailed, d), 1)
        wave_buildup_cancel[day_label] = round(avg_feature(cancelled, 12 + d), 2)
        wave_buildup_sail[day_label] = round(avg_feature(sailed, 12 + d), 2)

    # Trend analysis
    cancel_wind_slope = avg_feature(cancelled, 24)
    sail_wind_slope = avg_feature(sailed, 24)
    cancel_wave_slope = avg_feature(cancelled, 25)
    sail_wave_slope = avg_feature(sailed, 25)

    # Storm hours
    cancel_bf6_hours = avg_feature(cancelled, 28)
    sail_bf6_hours = avg_feature(sailed, 28)
    cancel_bf8_hours = avg_feature(cancelled, 29)
    sail_bf8_hours = avg_feature(sailed, 29)

    # Peak conditions
    cancel_peak_wind = avg_feature(cancelled, 33)
    sail_peak_wind = avg_feature(sailed, 33)
    cancel_peak_wave = avg_feature(cancelled, 34)
    sail_peak_wave = avg_feature(sailed, 34)

    # Route-specific analysis
    route_stats = {}
    for route_id in TRACKED_ROUTES:
        route_cancelled = [r for r in cancelled if r["route_id"] == route_id]
        route_sailed = [r for r in sailed if r["route_id"] == route_id]
        if route_cancelled:
            route_stats[route_id] = {
                "cancelled_count": len(route_cancelled),
                "sailed_count": len(route_sailed),
                "avg_peak_wind_cancel": round(
                    sum(r["temporal_features"][33] for r in route_cancelled) / len(route_cancelled), 1
                ),
                "avg_peak_wind_sail": round(
                    sum(r["temporal_features"][33] for r in route_sailed) / len(route_sailed), 1
                ) if route_sailed else 0,
            }

    # Early warning signals — when does divergence start?
    divergence = {}
    for d in range(6):
        day_label = f"D-{5-d}"
        cancel_wind = avg_feature(cancelled, d)
        sail_wind = avg_feature(sailed, d)
        ratio = cancel_wind / sail_wind if sail_wind > 0 else 1.0
        divergence[day_label] = {
            "cancel_wind": round(cancel_wind, 1),
            "sail_wind": round(sail_wind, 1),
            "ratio": round(ratio, 2),
            "detectable": ratio > 1.3,
        }

    report = {
        "summary": {
            "total_records": len(augmented_records),
            "cancelled_with_features": len(cancelled),
            "sailed_with_features": len(sailed),
            "ban_dates": list(CONFIRMED_BAN_DATES.keys()),
        },
        "wind_buildup": {
            "cancellation_days": wind_buildup_cancel,
            "normal_sailing_days": wind_buildup_sail,
            "interpretation": "Wind mean (kn) per day leading up to event",
        },
        "wave_buildup": {
            "cancellation_days": wave_buildup_cancel,
            "normal_sailing_days": wave_buildup_sail,
            "interpretation": "Wave height mean (m) per day leading up to event",
        },
        "trends": {
            "wind_slope_cancel": round(cancel_wind_slope, 2),
            "wind_slope_sail": round(sail_wind_slope, 2),
            "wave_slope_cancel": round(cancel_wave_slope, 3),
            "wave_slope_sail": round(sail_wave_slope, 3),
            "interpretation": "Positive slope = increasing conditions. "
                            "Cancellations should show steeper increase.",
        },
        "storm_intensity": {
            "hours_above_bf6_cancel": round(cancel_bf6_hours, 1),
            "hours_above_bf6_sail": round(sail_bf6_hours, 1),
            "hours_above_bf8_cancel": round(cancel_bf8_hours, 1),
            "hours_above_bf8_sail": round(sail_bf8_hours, 1),
            "interpretation": "Hours with wind above Beaufort 6/8 in the 5-day window",
        },
        "peak_conditions": {
            "peak_wind_cancel_kn": round(cancel_peak_wind, 1),
            "peak_wind_sail_kn": round(sail_peak_wind, 1),
            "peak_wave_cancel_m": round(cancel_peak_wave, 2),
            "peak_wave_sail_m": round(sail_peak_wave, 2),
        },
        "early_warning_divergence": divergence,
        "route_specific": route_stats,
    }

    # Save report
    with open(PATTERN_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    return report


def print_pattern_report(report: dict) -> None:
    """Pretty-print the pattern analysis report."""
    if "error" in report:
        print(f"  Error: {report['error']}")
        return

    s = report["summary"]
    print()
    print("=" * 70)
    print("  REAL CANCELLATION PATTERN ANALYSIS")
    print("  Based on confirmed sailing ban events from news sources")
    print("=" * 70)
    print()
    print(f"  Cancelled records:  {s['cancelled_with_features']}")
    print(f"  Sailed records:     {s['sailed_with_features']}")
    print(f"  Ban dates:          {', '.join(s['ban_dates'])}")
    print()

    # Wind buildup pattern
    print("  WIND BUILDUP PATTERN (mean wind in knots)")
    print("  " + "-" * 60)
    print(f"  {'Day':<8}", end="")
    for d in range(6):
        print(f"{'D-' + str(5-d):>8}", end="")
    print()

    print(f"  {'Cancel':<8}", end="")
    for d in range(6):
        val = report["wind_buildup"]["cancellation_days"][f"D-{5-d}"]
        print(f"{val:>8.1f}", end="")
    print()

    print(f"  {'Sailed':<8}", end="")
    for d in range(6):
        val = report["wind_buildup"]["normal_sailing_days"][f"D-{5-d}"]
        print(f"{val:>8.1f}", end="")
    print()
    print()

    # Wave buildup pattern
    print("  WAVE BUILDUP PATTERN (mean wave height in meters)")
    print("  " + "-" * 60)
    print(f"  {'Day':<8}", end="")
    for d in range(6):
        print(f"{'D-' + str(5-d):>8}", end="")
    print()

    print(f"  {'Cancel':<8}", end="")
    for d in range(6):
        val = report["wave_buildup"]["cancellation_days"][f"D-{5-d}"]
        print(f"{val:>8.2f}", end="")
    print()

    print(f"  {'Sailed':<8}", end="")
    for d in range(6):
        val = report["wave_buildup"]["normal_sailing_days"][f"D-{5-d}"]
        print(f"{val:>8.2f}", end="")
    print()
    print()

    # Early warning divergence
    print("  EARLY WARNING — when can we detect the cancellation?")
    print("  " + "-" * 60)
    for d in range(6):
        day_label = f"D-{5-d}"
        dv = report["early_warning_divergence"][day_label]
        signal = "  << DETECTABLE" if dv["detectable"] else ""
        bar = "#" * int(min(dv["ratio"], 3) * 10)
        print(f"  {day_label:<6} Cancel: {dv['cancel_wind']:>5.1f}kn  "
              f"Sail: {dv['sail_wind']:>5.1f}kn  "
              f"Ratio: {dv['ratio']:.2f}  {bar}{signal}")
    print()

    # Storm intensity
    si = report["storm_intensity"]
    print("  STORM INTENSITY (hours above threshold in 5-day window)")
    print("  " + "-" * 50)
    print(f"  Hours above Bf6:  Cancel={si['hours_above_bf6_cancel']:.0f}  "
          f"Sail={si['hours_above_bf6_sail']:.0f}")
    print(f"  Hours above Bf8:  Cancel={si['hours_above_bf8_cancel']:.0f}  "
          f"Sail={si['hours_above_bf8_sail']:.0f}")
    print()

    # Peak conditions
    pk = report["peak_conditions"]
    print("  PEAK CONDITIONS in 5-day window")
    print("  " + "-" * 50)
    print(f"  Peak wind:   Cancel={pk['peak_wind_cancel_kn']:.1f}kn  "
          f"Sail={pk['peak_wind_sail_kn']:.1f}kn")
    print(f"  Peak wave:   Cancel={pk['peak_wave_cancel_m']:.2f}m   "
          f"Sail={pk['peak_wave_sail_m']:.2f}m")
    print()

    # Trends
    tr = report["trends"]
    print("  TREND SLOPES (rate of change per day)")
    print("  " + "-" * 50)
    print(f"  Wind slope:  Cancel={tr['wind_slope_cancel']:+.2f} kn/day  "
          f"Sail={tr['wind_slope_sail']:+.2f} kn/day")
    print(f"  Wave slope:  Cancel={tr['wave_slope_cancel']:+.3f} m/day   "
          f"Sail={tr['wave_slope_sail']:+.3f} m/day")
    print()

    # Route-specific
    if report.get("route_specific"):
        print("  ROUTE-SPECIFIC ANALYSIS")
        print("  " + "-" * 50)
        print(f"  {'Route':<10} {'Cancelled':>10} {'Sailed':>10} {'Peak(C)':>10} {'Peak(S)':>10}")
        for route_id, rs in report["route_specific"].items():
            print(f"  {route_id:<10} {rs['cancelled_count']:>10} "
                  f"{rs['sailed_count']:>10} "
                  f"{rs['avg_peak_wind_cancel']:>9.1f} "
                  f"{rs['avg_peak_wind_sail']:>9.1f}")
    print()

    # Key findings
    print("  KEY FINDINGS:")
    print("  " + "-" * 60)

    # Find earliest detection day
    for d in range(6):
        day_label = f"D-{5-d}"
        dv = report["early_warning_divergence"][day_label]
        if dv["detectable"]:
            print(f"  - Storm pattern becomes detectable at {day_label}")
            print(f"    (wind ratio cancel/sail = {dv['ratio']:.2f})")
            break
    else:
        print("  - No clear early detection signal found")

    wind_slope_cancel = report["trends"]["wind_slope_cancel"]
    wind_slope_sail = report["trends"]["wind_slope_sail"]
    if wind_slope_cancel > wind_slope_sail * 1.5:
        print(f"  - Cancellation events show {wind_slope_cancel/max(wind_slope_sail,0.01):.1f}x "
              f"steeper wind increase")

    print()


def train_and_validate_real(augmented_records: list[dict]) -> dict:
    """
    Train temporal models on real data and measure accuracy.

    Uses the same TemporalPredictor architecture but trained on
    real weather patterns.
    """
    from src.models.temporal_predictor import TemporalPredictor, backtest_temporal
    from src.data_collection.temporal_dataset import load_temporal_dataset

    # First save the dataset, then load it
    save_real_temporal_dataset(augmented_records)

    # Load and train
    try:
        X, y = load_temporal_dataset(REAL_TEMPORAL_CSV)
    except FileNotFoundError as e:
        return {"error": str(e)}

    if len(X) < 50:
        return {"error": f"Need at least 50 records, got {len(X)}"}

    # Run backtest using the real dataset
    report = backtest_temporal(csv_path=REAL_TEMPORAL_CSV)
    return report


def run_full_analysis(fetch_weather: bool = True) -> dict:
    """
    Run the complete real cancellation analysis pipeline.

    Args:
        fetch_weather: If True, fetch weather from Open-Meteo (needs internet).
                      If False, use existing data for analysis only.
    """
    print("=" * 70)
    print("  REAL CANCELLATION ANALYSIS PIPELINE")
    print("  Using confirmed sailing ban events from news sources")
    print("=" * 70)
    print()

    # Step 1: Build ground truth from real ban dates + sailed dates
    print("Step 1: Building real ground truth...")
    records = build_real_ground_truth()
    cancelled = sum(1 for r in records if r["status"] == "CANCELLED")
    print(f"  Total records: {len(records)}")
    print(f"  Cancelled: {cancelled} ({100*cancelled/len(records):.1f}%)")
    print(f"  Sailed: {len(records) - cancelled}")
    print(f"  Date range: {records[0]['date']} to {records[-1]['date']}")
    print()

    # Step 2: Fetch weather (if internet available)
    if fetch_weather:
        print("Step 2: Fetching 5-day weather windows from Open-Meteo Archive...")
        augmented = fetch_weather_for_events(records)
        print(f"  Got weather for {len(augmented)} records")
        print()

        # Save temporal dataset
        print("Step 3: Saving real temporal dataset...")
        stats = save_real_temporal_dataset(augmented)
        print(f"  Saved {stats['total_records']} records to {stats['csv_path']}")
        print()

        # Analyze patterns
        print("Step 4: Analyzing weather patterns...")
        pattern_report = analyze_patterns(augmented)
        print_pattern_report(pattern_report)

        # Train and validate
        print("Step 5: Training temporal models on real data...")
        from src.models.temporal_predictor import print_temporal_report
        validation_report = train_and_validate_real(augmented)
        print_temporal_report(validation_report)

        return {
            "ground_truth": {"total": len(records), "cancelled": cancelled},
            "weather_fetched": len(augmented),
            "patterns": pattern_report,
            "validation": validation_report,
        }
    else:
        print("Step 2: Skipping weather fetch (use --fetch to enable)")
        print("  Run with internet access to fetch real weather data.")
        print()

        # Just show ground truth stats
        return {
            "ground_truth": {"total": len(records), "cancelled": cancelled},
            "message": "Run with internet access for full analysis",
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze real ferry cancellation events")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Skip weather fetch, just show ground truth stats")
    parser.add_argument("--fetch", action="store_true",
                       help="Fetch weather from Open-Meteo (requires internet)")
    args = parser.parse_args()

    run_full_analysis(fetch_weather=args.fetch)
