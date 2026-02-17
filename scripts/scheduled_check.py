#!/usr/bin/env python3
"""
Scheduled maritime weather check with Telegram alerts.

Runs periodically (default every 3 hours), checks all routes,
and sends a Telegram notification if any route is BAN_LIKELY or AT_RISK.

Usage:
    python scripts/scheduled_check.py                    # Every 3 hours
    CHECK_INTERVAL_HOURS=1 python scripts/scheduled_check.py  # Every 1 hour
"""

import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.constants import ROUTES, PORTS, knots_to_beaufort
from src.services.sailing_ban_checker import SailingBanChecker
from src.services.notifications import TelegramNotifier
from src.data_collection.demo_data import generate_demo_route_conditions
from src.utils.logger import get_logger

logger = get_logger("scheduler")

INTERVAL_HOURS = float(os.environ.get("CHECK_INTERVAL_HOURS", "3"))
USE_LIVE = os.environ.get("USE_LIVE_API", "false").lower() == "true"


def run_check():
    """Run a full route check and send alerts."""
    logger.info("Starting scheduled check at %s", datetime.now().isoformat())

    checker = SailingBanChecker(use_ml=True)
    notifier = TelegramNotifier()
    results = []

    for route_id, route_info in ROUTES.items():
        origin_code, dest_code = route_id.split("-")
        if origin_code not in PORTS or dest_code not in PORTS:
            continue

        try:
            if USE_LIVE:
                from src.data_collection.weather_client import fetch_route_conditions
                origin = PORTS[origin_code]
                dest = PORTS[dest_code]
                weather_data = fetch_route_conditions(
                    origin["lat"], origin["lon"],
                    dest["lat"], dest["lon"],
                    forecast_days=2,
                )
            else:
                weather_data = generate_demo_route_conditions(
                    forecast_days=2, scenario="auto",
                )

            result = checker.check_route(route_id, weather_data, "CONVENTIONAL")

            # Compute summary stats
            hourly = result.get("hourly", [])
            max_wind = max((h["wind_speed_knots"] for h in hourly), default=0)
            max_wave = max((h["wave_height_m"] for h in hourly), default=0)
            max_bf = knots_to_beaufort(max_wind)

            result["hourly"] = [
                {"wind_speed_knots": max_wind, "wave_height_m": max_wave, "beaufort": max_bf},
            ]
            results.append(result)

            logger.info("  %s: %s (wind %.0f kn, wave %.1f m)",
                        route_id, result["overall_status"], max_wind, max_wave)

        except Exception as e:
            logger.error("  %s: ERROR %s", route_id, e)

    # Send Telegram alert
    if notifier.is_configured:
        sent = notifier.send_alert_if_needed(results)
        if sent:
            logger.info("Telegram alert sent")
        else:
            logger.info("All routes clear — no alert sent")
    else:
        logger.warning("Telegram not configured — skipping notification")

    ban = sum(1 for r in results if r["overall_status"] == "BAN_LIKELY")
    risk = sum(1 for r in results if r["overall_status"] == "AT_RISK")
    clear = sum(1 for r in results if r["overall_status"] == "CLEAR")
    logger.info("Check complete: %d BAN, %d RISK, %d CLEAR", ban, risk, clear)

    return results


def main():
    logger.info("Maritime scheduler starting (interval: %.1f hours, live: %s)",
                INTERVAL_HOURS, USE_LIVE)

    while True:
        try:
            run_check()
        except Exception as e:
            logger.error("Check failed: %s", e)

        sleep_seconds = INTERVAL_HOURS * 3600
        logger.info("Next check in %.1f hours...", INTERVAL_HOURS)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
