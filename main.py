#!/usr/bin/env python3
"""
Greek Maritime Intelligence Platform — CLI Entry Point

Usage:
    python main.py                          # Check all routes (demo mode)
    python main.py --route PIR-MYK          # Check a single route
    python main.py --vessel HIGH_SPEED      # Check for high-speed vessels
    python main.py --scenario storm         # Demo scenario: calm/storm/meltemi/auto
    python main.py --days 2                 # Forecast days (1-7)
    python main.py --live                   # Use live API (requires internet)
    python main.py --ml                     # Enable ML predictions alongside rules
    python main.py --notify                 # Send Telegram alert if risks found
    python main.py --test-api               # Test API connectivity only
    python main.py --train-ml               # Train ML models (synthetic data)
    python main.py --train-ml-gt            # Train ML from ground truth CSV
    python main.py --generate-data          # Generate sample ground truth data
    python main.py --calibrate              # Calibrate thresholds from ground truth
    python main.py --fetch-historical       # Fetch real historical weather from Open-Meteo
    python main.py --historical-days 730    # Fetch 2 years of history
    python main.py --train-ml-historical    # Train ML from real historical data
    python main.py --web                    # Launch web dashboard
"""

import argparse
import sys
from datetime import datetime

from src.config.constants import (
    ROUTES,
    PORTS,
    BEAUFORT_SCALE,
    VESSEL_TYPES,
    knots_to_beaufort,
)
from src.services.sailing_ban_checker import SailingBanChecker
from src.data_collection.demo_data import generate_demo_route_conditions, DEMO_SCENARIOS
from src.utils.logger import get_logger

logger = get_logger("maritime")

STATUS_SYMBOLS = {
    "BAN_LIKELY": "[!!]",
    "AT_RISK":    "[! ]",
    "CLEAR":      "[OK]",
}


def test_api_connection():
    """Test connectivity to Open-Meteo APIs using Piraeus coordinates."""
    from src.data_collection.weather_client import fetch_marine_forecast, fetch_weather_forecast

    print("Testing API connectivity...")
    print()

    piraeus = PORTS["PIR"]
    lat, lon = piraeus["lat"], piraeus["lon"]

    print(f"  Marine API (lat={lat}, lon={lon})...")
    try:
        marine = fetch_marine_forecast(lat, lon, forecast_days=1)
        hourly = marine.get("hourly", {})
        wave_heights = hourly.get("wave_height", [])
        valid = [w for w in wave_heights if w is not None]
        print(f"    OK -- got {len(wave_heights)} hourly records, "
              f"{len(valid)} with wave data")
        if valid:
            print(f"    Current wave height: {valid[0]:.1f}m")
    except Exception as e:
        print(f"    FAILED -- {e}")
        return False

    print(f"  Weather API (lat={lat}, lon={lon})...")
    try:
        weather = fetch_weather_forecast(lat, lon, forecast_days=1)
        hourly = weather.get("hourly", {})
        wind_speeds = hourly.get("wind_speed_10m", [])
        valid = [w for w in wind_speeds if w is not None]
        print(f"    OK -- got {len(wind_speeds)} hourly records, "
              f"{len(valid)} with wind data")
        if valid:
            bf = knots_to_beaufort(valid[0])
            desc = BEAUFORT_SCALE[bf][2]
            print(f"    Current wind: {valid[0]:.1f} knots (Beaufort {bf} -- {desc})")
    except Exception as e:
        print(f"    FAILED -- {e}")
        return False

    print()
    print("All API connections OK.")
    return True


def get_weather_data(route_id, args):
    """Get weather data — either live or demo."""
    origin_code, dest_code = route_id.split("-")
    origin_port = PORTS[origin_code]
    dest_port = PORTS[dest_code]

    if args.live:
        from src.data_collection.weather_client import fetch_route_conditions
        return fetch_route_conditions(
            origin_lat=origin_port["lat"],
            origin_lon=origin_port["lon"],
            dest_lat=dest_port["lat"],
            dest_lon=dest_port["lon"],
            forecast_days=args.days,
        )
    else:
        return generate_demo_route_conditions(
            forecast_days=args.days,
            scenario=args.scenario,
        )


def check_single_route(route_id, args):
    """Check a single route and print detailed results."""
    route_info = ROUTES.get(route_id)
    if not route_info:
        print(f"Error: Unknown route '{route_id}'")
        print(f"Available routes: {', '.join(sorted(ROUTES.keys()))}")
        return None

    mode = "LIVE" if args.live else f"DEMO ({args.scenario})"
    ml_label = " + ML" if args.ml else ""
    print(f"Route: {route_info['origin']} -> {route_info['destination']} "
          f"({route_info['distance_nm']} nm)")
    print(f"Sea area: {route_info['sea_area']} "
          f"({'exposed' if route_info.get('exposed') else 'sheltered'})")
    print(f"Vessel type: {args.vessel} -> {VESSEL_TYPES.get(args.vessel, 'conventional')}")
    print(f"Mode: {mode}{ml_label} | Forecast: {args.days} days")
    print()

    try:
        weather_data = get_weather_data(route_id, args)
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        if not args.live:
            raise
        print("Hint: try without --live to use demo data")
        return None

    checker = SailingBanChecker(use_ml=args.ml)
    result = checker.check_route(route_id, weather_data, args.vessel)

    symbol = STATUS_SYMBOLS[result["overall_status"]]
    print(f"Overall status: {symbol} {result['overall_status']}")

    # Show ML prediction if available
    if "ml_overall_status" in result:
        ml_sym = STATUS_SYMBOLS[result["ml_overall_status"]]
        print(f"ML prediction:  {ml_sym} {result['ml_overall_status']} "
              f"(max cancel prob: {result['ml_max_cancel_probability']:.1%})")

    print()
    header = f"{'Time':<18} {'Status':<14} {'Wind (kn)':<12} {'Bf':<5} {'Wave (m)':<10}"
    if args.ml:
        header += f"{'ML Prob':<8}"
    print("Hourly breakdown:")
    print("-" * (72 + (8 if args.ml else 0)))
    print(header)
    print("-" * (72 + (8 if args.ml else 0)))

    current_date = None
    for h in result["hourly"]:
        time_str = h["time"]
        day = time_str[:10]
        if day != current_date:
            current_date = day
            print(f"--- {day} ---")

        hour = time_str[11:16]
        sym = STATUS_SYMBOLS[h["status"]]
        line = (f"  {hour}         {sym} {h['status']:<11} {h['wind_speed_knots']:>6.1f}      "
                f"{h['beaufort']:<4} {h['wave_height_m']:>6.1f}")
        if args.ml and "ml_cancel_probability" in h:
            line += f"    {h['ml_cancel_probability']:.0%}"
        print(line)

    return result


def check_all_routes(args):
    """Check all configured routes and print a summary table."""
    mode = "LIVE" if args.live else f"DEMO ({args.scenario})"
    ml_label = " + ML" if args.ml else ""
    print("=" * 64)
    print("  GREEK MARITIME INTELLIGENCE PLATFORM")
    print(f"  Sailing Ban Forecast -- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Vessel: {args.vessel} | Mode: {mode}{ml_label} | Forecast: {args.days} day(s)")
    print("=" * 64)
    print()

    checker = SailingBanChecker(use_ml=args.ml)
    results = []

    for route_id, route_info in ROUTES.items():
        origin_code, dest_code = route_id.split("-")
        if origin_code not in PORTS or dest_code not in PORTS:
            continue

        label = f"  {route_id}: {route_info['origin']} -> {route_info['destination']}"
        print(f"{label}...", end=" ", flush=True)

        try:
            weather_data = get_weather_data(route_id, args)
            result = checker.check_route(route_id, weather_data, args.vessel)
            results.append(result)
            sym = STATUS_SYMBOLS[result["overall_status"]]
            ml_info = ""
            if "ml_overall_status" in result:
                ml_sym = STATUS_SYMBOLS[result["ml_overall_status"]]
                ml_info = f" | ML: {ml_sym} {result['ml_max_cancel_probability']:.0%}"
            print(f"{sym} {result['overall_status']}{ml_info}")
        except Exception as e:
            print(f"[ERR] {e}")

    # Summary table
    print()
    print("=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print()

    header = f"{'Route':<10} {'From':<14} {'To':<14} {'Status':<14} {'Max Wind':<10} {'Max Wave'}"
    if args.ml:
        header += f"  {'ML Prob'}"
    print(header)
    print("-" * (72 + (10 if args.ml else 0)))

    status_order = {"BAN_LIKELY": 0, "AT_RISK": 1, "CLEAR": 2}
    results.sort(key=lambda r: status_order.get(r["overall_status"], 3))

    for r in results:
        max_wind = max((h["wind_speed_knots"] for h in r["hourly"]), default=0)
        max_wave = max((h["wave_height_m"] for h in r["hourly"]), default=0)
        sym = STATUS_SYMBOLS[r["overall_status"]]

        route = ROUTES[r["route_id"]]
        line = (f"{r['route_id']:<10} {route['origin']:<14} {route['destination']:<14} "
                f"{sym} {r['overall_status']:<10} {max_wind:>5.1f} kn   {max_wave:>5.1f} m")
        if args.ml and "ml_max_cancel_probability" in r:
            line += f"  {r['ml_max_cancel_probability']:>5.0%}"
        print(line)

    print()
    ban_count = sum(1 for r in results if r["overall_status"] == "BAN_LIKELY")
    risk_count = sum(1 for r in results if r["overall_status"] == "AT_RISK")
    clear_count = sum(1 for r in results if r["overall_status"] == "CLEAR")
    print(f"  Total: {len(results)} routes | "
          f"[!!] Ban likely: {ban_count} | "
          f"[! ] At risk: {risk_count} | "
          f"[OK] Clear: {clear_count}")
    print()

    # Send Telegram notification if --notify
    if args.notify:
        send_notifications(results)

    return results


def send_notifications(results):
    """Send Telegram notification with route check results."""
    from src.services.notifications import TelegramNotifier

    notifier = TelegramNotifier()
    if not notifier.is_configured:
        print("  Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return

    # Add required fields for the notifier
    formatted = []
    for r in results:
        max_wind = max((h["wind_speed_knots"] for h in r["hourly"]), default=0)
        max_wave = max((h["wave_height_m"] for h in r["hourly"]), default=0)
        max_bf = knots_to_beaufort(max_wind)
        formatted.append({
            **r,
            "hourly": [
                {"wind_speed_knots": max_wind, "wave_height_m": max_wave, "beaufort": max_bf},
            ],
        })

    sent = notifier.send_alert_if_needed(formatted)
    if sent:
        print("  Telegram alert sent!")
    else:
        print("  No alerts to send (all routes clear).")


def train_ml(from_ground_truth: bool = False):
    """Train ML models."""
    try:
        from src.models.ml_predictor import MLPredictor
    except ImportError:
        print("scikit-learn required. Install with: pip install scikit-learn")
        return

    predictor = MLPredictor()

    if from_ground_truth:
        print("Training ML models from ground truth data...")
        print()
        metrics = predictor.train_from_ground_truth()
    else:
        print("Training ML models on synthetic data (5000 samples)...")
        print()
        metrics = predictor.train(n_samples=5000)

    if metrics:
        print()
        print(f"  Logistic Regression accuracy: {metrics['logistic_regression_accuracy']:.3f}")
        print(f"  Gradient Boosting accuracy:   {metrics['gradient_boosting_accuracy']:.3f}")
        print(f"  Train samples: {metrics['train_size']}")
        print(f"  Test samples:  {metrics['test_size']}")

        predictor.save()
        print()
        print("  Models saved to src/models/saved/")
    else:
        print("  Training failed. Check logs for details.")


def calibrate():
    """Analyze ground truth data and suggest threshold calibrations."""
    from src.models.ml_predictor import calibrate_thresholds

    print("Calibrating thresholds from ground truth data...")
    print()

    result = calibrate_thresholds()
    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    print(f"  Total records analyzed: {result['total_records']}")
    print()

    print("  Current thresholds:")
    for cat, bf in result["current_thresholds"].items():
        print(f"    {cat}: Beaufort {bf}")
    print()

    print("  Cancel rates by Beaufort level:")
    for category, levels in result["cancel_rates_by_beaufort"].items():
        print(f"  --- {category} ---")
        for level in levels:
            bar = "#" * int(level["cancel_rate"] * 30)
            marker = " <-- threshold" if level["beaufort"] == result["current_thresholds"].get(category) else ""
            print(f"    Bf {level['beaufort']:>2}: {level['cancel_rate']:>5.1%} "
                  f"({level['cancelled']:>3}/{level['total']:>3}) {bar}{marker}")
        print()

    if result["suggested_thresholds"]:
        print("  Suggested thresholds (>50% cancel rate):")
        for cat, bf in result["suggested_thresholds"].items():
            current = result["current_thresholds"].get(cat, "?")
            delta = ""
            if isinstance(current, int) and bf != current:
                delta = f" (currently {current}, delta {bf - current:+d})"
            print(f"    {cat}: Beaufort {bf}{delta}")
    else:
        print("  Not enough data to suggest threshold changes.")


def generate_data():
    """Generate sample ground truth data."""
    from src.data_collection.ground_truth import GroundTruthCollector

    collector = GroundTruthCollector()
    print("Generating 90 days of sample ground truth data...")
    count = collector.generate_sample_data(n_days=90)
    print(f"Generated {count} records.")
    print()

    stats = collector.get_stats()
    print(f"  Total records:   {stats['total']}")
    print(f"  Cancelled:       {stats['cancelled']}")
    print(f"  Sailed:          {stats['sailed']}")
    print(f"  Cancel rate:     {stats['cancel_rate']:.1%}")
    print(f"  Routes covered:  {stats['routes_covered']}")
    print(f"  Data saved to:   data/ground_truth/cancellation_records.csv")


def fetch_historical(days_back: int = 365):
    """Fetch real historical weather data from Open-Meteo APIs."""
    from src.data_collection.historical_weather import build_historical_dataset

    print(f"Fetching {days_back} days of historical weather data from Open-Meteo...")
    print("This will query the Marine API and Archive Weather API for all routes.")
    print()

    stats = build_historical_dataset(days_back=days_back)

    print()
    print("=" * 50)
    print("  HISTORICAL DATA COLLECTION COMPLETE")
    print("=" * 50)
    print()
    print(f"  Records:       {stats['total_records']:,}")
    print(f"  Cancelled:     {stats['cancelled']:,}")
    print(f"  Cancel rate:   {stats['cancel_rate']:.1%}")
    print(f"  Routes:        {stats['routes_fetched']}")
    print(f"  Date range:    {stats['date_range']['start']} to {stats['date_range']['end']}")
    print(f"  Saved to:      {stats['csv_path']}")
    print()
    print("Next step: train ML models with --train-ml-historical")


def train_ml_historical():
    """Train ML models from real historical weather data."""
    try:
        from src.models.ml_predictor import MLPredictor
    except ImportError:
        print("scikit-learn required. Install with: pip install scikit-learn")
        return

    predictor = MLPredictor()
    print("Training ML models from historical weather data...")
    print()

    metrics = predictor.train_from_historical()

    if metrics:
        print()
        print(f"  Logistic Regression accuracy: {metrics['logistic_regression_accuracy']:.3f}")
        print(f"  Gradient Boosting accuracy:   {metrics['gradient_boosting_accuracy']:.3f}")
        print(f"  Train samples: {metrics['train_size']}")
        print(f"  Test samples:  {metrics['test_size']}")

        predictor.save()
        print()
        print("  Models saved to src/models/saved/")
    else:
        print("  Training failed. Run --fetch-historical first to collect data.")


def launch_web():
    """Launch the web dashboard."""
    try:
        import uvicorn
    except ImportError:
        print("FastAPI and uvicorn required.")
        print("Install with: pip install fastapi uvicorn")
        return

    print("Launching web dashboard at http://localhost:8000")
    print("Press Ctrl+C to stop.")
    uvicorn.run("src.web.app:app", host="0.0.0.0", port=8000, reload=True)


def main():
    parser = argparse.ArgumentParser(
        description="Greek Maritime Intelligence Platform -- Sailing Ban Predictor",
    )
    parser.add_argument(
        "--route", type=str, default=None,
        help=f"Check a single route (e.g. PIR-MYK). Available: {', '.join(sorted(ROUTES.keys()))}",
    )
    parser.add_argument(
        "--vessel", type=str, default="CONVENTIONAL",
        choices=list(VESSEL_TYPES.keys()),
        help="Vessel type (default: CONVENTIONAL)",
    )
    parser.add_argument(
        "--days", type=int, default=2, choices=range(1, 8),
        help="Number of forecast days (default: 2)",
    )
    parser.add_argument(
        "--scenario", type=str, default="auto",
        choices=list(DEMO_SCENARIOS.keys()),
        help="Demo scenario (default: auto)",
    )
    parser.add_argument("--live", action="store_true", help="Use live API (requires internet)")
    parser.add_argument("--ml", action="store_true", help="Enable ML predictions alongside rules")
    parser.add_argument("--notify", action="store_true", help="Send Telegram alert if risks found")
    parser.add_argument("--test-api", action="store_true", help="Test API connectivity and exit")
    parser.add_argument("--train-ml", action="store_true", help="Train ML models (synthetic data)")
    parser.add_argument("--train-ml-gt", action="store_true", help="Train ML from ground truth CSV")
    parser.add_argument("--generate-data", action="store_true", help="Generate sample ground truth data")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate thresholds from ground truth")
    parser.add_argument(
        "--fetch-historical", action="store_true",
        help="Fetch real historical weather data from Open-Meteo",
    )
    parser.add_argument(
        "--historical-days", type=int, default=365,
        help="Days of history to fetch (default: 365)",
    )
    parser.add_argument(
        "--train-ml-historical", action="store_true",
        help="Train ML from real historical weather data",
    )
    parser.add_argument("--web", action="store_true", help="Launch web dashboard")

    args = parser.parse_args()

    if args.test_api:
        success = test_api_connection()
        sys.exit(0 if success else 1)

    if args.train_ml:
        train_ml(from_ground_truth=False)
        return

    if args.train_ml_gt:
        train_ml(from_ground_truth=True)
        return

    if args.generate_data:
        generate_data()
        return

    if args.calibrate:
        calibrate()
        return

    if args.fetch_historical:
        fetch_historical(days_back=args.historical_days)
        return

    if args.train_ml_historical:
        train_ml_historical()
        return

    if args.web:
        launch_web()
        return

    if args.route:
        result = check_single_route(args.route, args)
        if args.notify and result:
            send_notifications([result])
    else:
        check_all_routes(args)


if __name__ == "__main__":
    main()
