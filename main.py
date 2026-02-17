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


def check_risk_score(args):
    """Check route(s) using the probabilistic risk scoring engine."""
    from src.services.risk_scorer import score_route, compute_risk, score_to_band

    route_ids = [args.route] if args.route else list(ROUTES.keys())
    mode = "LIVE" if args.live else f"DEMO ({args.scenario})"

    print("=" * 72)
    print("  RISK SCORING ENGINE")
    print(f"  Vessel: {args.vessel} | Mode: {mode} | Forecast: {args.days} day(s)")
    if args.vessel_name:
        print(f"  Vessel: {args.vessel_name}")
    print("=" * 72)
    print()

    for route_id in route_ids:
        if route_id not in ROUTES:
            print(f"  Unknown route: {route_id}")
            continue

        weather_data = get_weather_data(route_id, args)
        result = score_route(route_id, weather_data, args.vessel, args.vessel_name)

        route = ROUTES[route_id]
        print(f"  {route_id}: {route['origin']} -> {route['destination']} "
              f"({route['distance_nm']}nm, {route['sea_area']})")
        print(f"  Band: {result['overall_band']} | "
              f"Max Risk: {result['max_risk_score']:.0f}/100 | "
              f"P(cancel): {result['max_cancel_probability']:.0%} | "
              f"P(delay): {result['max_delay_probability']:.0%}")

        if args.route:
            # Show hourly detail for single route
            print()
            header = (f"  {'Time':<18} {'Score':>6} {'Band':<10} "
                      f"{'P(cancel)':>9} {'P(delay)':>9} "
                      f"{'Wind':>6} {'Wave':>6} {'Bf':>3}")
            print(header)
            print("  " + "-" * 70)

            current_date = None
            for h in result["hourly"]:
                day = h["time"][:10]
                if day != current_date:
                    current_date = day
                    print(f"  --- {day} ---")
                hour = h["time"][11:16]
                print(f"  {hour:<18} {h['risk_score']:>5.0f} {h['band']:<10} "
                      f"{h['cancel_probability']:>8.0%} {h['delay_probability']:>8.0%} "
                      f"{h['wind_speed_knots']:>5.0f} {h['wave_height_m']:>5.1f} {h['beaufort']:>3}")
        print()

    if not args.route:
        # Summary table
        print("  " + "-" * 72)
        print(f"  {'Route':<10} {'From':<14} {'To':<14} {'Band':<10} "
              f"{'Score':>5} {'P(cancel)':>9} {'P(delay)':>9}")
        print("  " + "-" * 72)


def run_departure_optimizer(args):
    """Find optimal departure windows for a route."""
    from src.services.risk_scorer import score_route
    from src.services.departure_optimizer import optimize_route_departures

    route_id = args.route
    if route_id not in ROUTES:
        print(f"Error: Unknown route '{route_id}'")
        return

    route = ROUTES[route_id]
    print(f"  Departure optimizer: {route['origin']} -> {route['destination']}")
    print(f"  Vessel: {args.vessel}")
    print()

    weather_data = get_weather_data(route_id, args)
    scored = score_route(route_id, weather_data, args.vessel, args.vessel_name)
    results = optimize_route_departures(scored["hourly"])

    for r in results:
        print(f"  Scheduled: {r.scheduled_time}  (risk: {r.scheduled_risk:.0f}, {r.scheduled_band})")
        print(f"    -> {r.recommendation.upper()}: {r.reason}")
        if r.best_window:
            bw = r.best_window
            print(f"    Suggested: {bw.time} (risk: {bw.risk_score:.0f}, {bw.band}, "
                  f"shift: {bw.shift_minutes:+d}min)")
        print()


def run_fleet_allocation(args):
    """Run fleet allocation under current conditions."""
    from src.services.fleet_allocator import allocate_fleet

    print("=" * 60)
    print("  FLEET ALLOCATION ENGINE")
    print("=" * 60)
    print()
    print("  Using demo conditions for all routes...")
    print()

    # Generate representative conditions from demo data
    conditions = {}
    for route_id in ROUTES:
        weather_data = get_weather_data(route_id, args)
        midpoint = weather_data.get("midpoint", {})
        weather_h = midpoint.get("weather", {}).get("hourly", {})
        marine_h = midpoint.get("marine", {}).get("hourly", {})

        winds = [w for w in weather_h.get("wind_speed_10m", []) if w]
        waves = [w for w in marine_h.get("wave_height", []) if w]
        dirs_ = [d for d in weather_h.get("wind_direction_10m", []) if d is not None]

        conditions[route_id] = {
            "wind_speed_knots": max(winds) if winds else 0,
            "wave_height_m": max(waves) if waves else 0,
            "wind_direction": dirs_[0] if dirs_ else 0,
        }

    result = allocate_fleet(list(ROUTES.keys()), conditions)

    print(f"  {'Vessel':<22} {'Route':<10} {'Risk':>5} {'Band':<10} {'P(cancel)':>9} {'Primary'}")
    print("  " + "-" * 65)
    for a in result.assignments:
        primary = "yes" if a.is_primary else " "
        print(f"  {a.vessel_name:<22} {a.route_id:<10} {a.risk_score:>4.0f} "
              f"{a.band:<10} {a.cancel_probability:>8.0%} {primary:>5}")
    print()
    print(f"  Fleet avg risk: {result.fleet_risk_avg:.0f} | max: {result.fleet_risk_max:.0f}")
    if result.unassigned_routes:
        print(f"  Unassigned routes: {', '.join(result.unassigned_routes)}")
    print()


def run_backtest_cmd():
    """Run validation backtest from CLI."""
    from src.validation.backtester import run_backtest, print_backtest_report

    print("Running risk scorer backtest against ground truth data...")
    print()
    report = run_backtest()
    print_backtest_report(report)


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

    # Risk scoring engine
    parser.add_argument("--risk-score", action="store_true", help="Use probabilistic risk scoring")
    parser.add_argument("--optimize", action="store_true", help="Find optimal departure windows")
    parser.add_argument("--fleet", action="store_true", help="Run fleet allocation engine")
    parser.add_argument("--backtest", action="store_true", help="Validate risk scorer against ground truth")
    parser.add_argument("--vessel-name", type=str, default=None, help="Specific vessel name (e.g. 'Blue Star Delos')")

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

    if args.backtest:
        run_backtest_cmd()
        return

    if args.fleet:
        run_fleet_allocation(args)
        return

    if args.risk_score:
        check_risk_score(args)
        return

    if args.route and args.optimize:
        run_departure_optimizer(args)
        return

    if args.route:
        result = check_single_route(args.route, args)
        if args.notify and result:
            send_notifications([result])
    else:
        check_all_routes(args)


if __name__ == "__main__":
    main()
