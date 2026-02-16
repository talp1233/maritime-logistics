#!/usr/bin/env python3
"""
Greek Maritime Intelligence Platform — CLI Entry Point

Usage:
    python main.py                    # Check all routes (conventional vessels)
    python main.py --route PIR-MYK    # Check a single route
    python main.py --vessel HIGH_SPEED # Check for high-speed vessels
    python main.py --days 2           # Forecast days (1-7)
    python main.py --test-api         # Test API connectivity only
"""

import argparse
import sys
import json
from datetime import datetime

from src.config.constants import (
    ROUTES,
    PORTS,
    BEAUFORT_SCALE,
    VESSEL_TYPES,
    knots_to_beaufort,
)
from src.data_collection.weather_client import (
    fetch_marine_forecast,
    fetch_weather_forecast,
    fetch_route_conditions,
)
from src.services.sailing_ban_checker import SailingBanChecker


# Status display symbols (ASCII-safe)
STATUS_SYMBOLS = {
    "BAN_LIKELY": "[!!]",
    "AT_RISK":    "[! ]",
    "CLEAR":      "[OK]",
}


def test_api_connection():
    """Test connectivity to Open-Meteo APIs using Piraeus coordinates."""
    print("Testing API connectivity...")
    print()

    piraeus = PORTS["PIR"]
    lat, lon = piraeus["lat"], piraeus["lon"]

    # Test Marine API
    print(f"  Marine API (lat={lat}, lon={lon})...")
    try:
        marine = fetch_marine_forecast(lat, lon, forecast_days=1)
        hourly = marine.get("hourly", {})
        wave_heights = hourly.get("wave_height", [])
        valid = [w for w in wave_heights if w is not None]
        print(f"    OK — got {len(wave_heights)} hourly records, "
              f"{len(valid)} with wave data")
        if valid:
            print(f"    Current wave height: {valid[0]:.1f}m")
    except Exception as e:
        print(f"    FAILED — {e}")
        return False

    # Test Weather API
    print(f"  Weather API (lat={lat}, lon={lon})...")
    try:
        weather = fetch_weather_forecast(lat, lon, forecast_days=1)
        hourly = weather.get("hourly", {})
        wind_speeds = hourly.get("wind_speed_10m", [])
        valid = [w for w in wind_speeds if w is not None]
        print(f"    OK — got {len(wind_speeds)} hourly records, "
              f"{len(valid)} with wind data")
        if valid:
            bf = knots_to_beaufort(valid[0])
            desc = BEAUFORT_SCALE[bf][2]
            print(f"    Current wind: {valid[0]:.1f} knots (Beaufort {bf} — {desc})")
    except Exception as e:
        print(f"    FAILED — {e}")
        return False

    print()
    print("All API connections OK.")
    return True


def check_single_route(route_id, vessel_type, forecast_days):
    """Check a single route and print detailed results."""
    route_info = ROUTES.get(route_id)
    if not route_info:
        print(f"Error: Unknown route '{route_id}'")
        print(f"Available routes: {', '.join(sorted(ROUTES.keys()))}")
        return

    origin_code, dest_code = route_id.split("-")
    origin_port = PORTS[origin_code]
    dest_port = PORTS[dest_code]

    print(f"Route: {route_info['origin']} -> {route_info['destination']} "
          f"({route_info['distance_nm']} nm)")
    print(f"Sea area: {route_info['sea_area']} "
          f"({'exposed' if route_info.get('exposed') else 'sheltered'})")
    print(f"Vessel type: {vessel_type} -> {VESSEL_TYPES.get(vessel_type, 'conventional')}")
    print(f"Forecast: {forecast_days} days")
    print()
    print("Fetching weather data...")

    try:
        weather_data = fetch_route_conditions(
            origin_lat=origin_port["lat"],
            origin_lon=origin_port["lon"],
            dest_lat=dest_port["lat"],
            dest_lon=dest_port["lon"],
            forecast_days=forecast_days,
        )
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return

    checker = SailingBanChecker()
    result = checker.check_route(route_id, weather_data, vessel_type)

    symbol = STATUS_SYMBOLS[result["overall_status"]]
    print(f"Overall status: {symbol} {result['overall_status']}")
    print()

    # Group hourly predictions by day
    print("Hourly breakdown:")
    print("-" * 72)
    print(f"{'Time':<18} {'Status':<14} {'Wind (kn)':<12} {'Bf':<5} {'Wave (m)':<10}")
    print("-" * 72)

    current_date = None
    for h in result["hourly"]:
        time_str = h["time"]
        # Parse date for day headers
        day = time_str[:10]
        if day != current_date:
            current_date = day
            print(f"--- {day} ---")

        hour = time_str[11:16]
        sym = STATUS_SYMBOLS[h["status"]]
        wind = h["wind_speed_knots"]
        bf = h["beaufort"]
        wave = h["wave_height_m"]
        print(f"  {hour}         {sym} {h['status']:<11} {wind:>6.1f}      {bf:<4} {wave:>6.1f}")


def check_all_routes(vessel_type, forecast_days):
    """Check all configured routes and print a summary table."""
    print("=" * 64)
    print("  GREEK MARITIME INTELLIGENCE PLATFORM")
    print(f"  Sailing Ban Forecast — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Vessel type: {vessel_type} | Forecast: {forecast_days} day(s)")
    print("=" * 64)
    print()
    print("Fetching weather data for all routes...")
    print()

    checker = SailingBanChecker()
    results = []

    for route_id, route_info in ROUTES.items():
        origin_code, dest_code = route_id.split("-")
        origin_port = PORTS.get(origin_code)
        dest_port = PORTS.get(dest_code)

        if not origin_port or not dest_port:
            continue

        label = f"  {route_id}: {route_info['origin']} -> {route_info['destination']}"
        print(f"{label}...", end=" ", flush=True)

        try:
            weather_data = fetch_route_conditions(
                origin_lat=origin_port["lat"],
                origin_lon=origin_port["lon"],
                dest_lat=dest_port["lat"],
                dest_lon=dest_port["lon"],
                forecast_days=forecast_days,
            )
            result = checker.check_route(route_id, weather_data, vessel_type)
            results.append(result)
            sym = STATUS_SYMBOLS[result["overall_status"]]
            print(f"{sym} {result['overall_status']}")
        except Exception as e:
            print(f"[ERR] {e}")

    # Print summary table
    print()
    print("=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print()
    print(f"{'Route':<10} {'From':<14} {'To':<14} {'Status':<14} {'Max Wind':<10} {'Max Wave'}")
    print("-" * 72)

    # Sort by severity
    status_order = {"BAN_LIKELY": 0, "AT_RISK": 1, "CLEAR": 2}
    results.sort(key=lambda r: status_order.get(r["overall_status"], 3))

    for r in results:
        max_wind = max((h["wind_speed_knots"] for h in r["hourly"]), default=0)
        max_wave = max((h["wave_height_m"] for h in r["hourly"]), default=0)
        max_bf = knots_to_beaufort(max_wind)
        sym = STATUS_SYMBOLS[r["overall_status"]]

        route = ROUTES[r["route_id"]]
        print(f"{r['route_id']:<10} {route['origin']:<14} {route['destination']:<14} "
              f"{sym} {r['overall_status']:<10} {max_wind:>5.1f} kn   {max_wave:>5.1f} m")

    # Count by status
    print()
    ban_count = sum(1 for r in results if r["overall_status"] == "BAN_LIKELY")
    risk_count = sum(1 for r in results if r["overall_status"] == "AT_RISK")
    clear_count = sum(1 for r in results if r["overall_status"] == "CLEAR")
    print(f"  Total: {len(results)} routes | "
          f"[!!] Ban likely: {ban_count} | "
          f"[! ] At risk: {risk_count} | "
          f"[OK] Clear: {clear_count}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Greek Maritime Intelligence Platform — Sailing Ban Predictor",
    )
    parser.add_argument(
        "--route",
        type=str,
        default=None,
        help=f"Check a single route (e.g. PIR-MYK). Available: {', '.join(sorted(ROUTES.keys()))}",
    )
    parser.add_argument(
        "--vessel",
        type=str,
        default="CONVENTIONAL",
        choices=list(VESSEL_TYPES.keys()),
        help="Vessel type (default: CONVENTIONAL)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        choices=range(1, 8),
        help="Number of forecast days (default: 2)",
    )
    parser.add_argument(
        "--test-api",
        action="store_true",
        help="Test API connectivity and exit",
    )

    args = parser.parse_args()

    if args.test_api:
        success = test_api_connection()
        sys.exit(0 if success else 1)

    if args.route:
        check_single_route(args.route, args.vessel, args.days)
    else:
        check_all_routes(args.vessel, args.days)


if __name__ == "__main__":
    main()
