"""
Route monitoring service.

Provides a high-level interface for checking all configured routes
against current/forecast weather conditions.
"""

from src.config.constants import ROUTES, PORTS
from src.data_collection.weather_client import fetch_route_conditions
from src.services.sailing_ban_checker import SailingBanChecker


class RouteMonitor:
    """Monitors all configured ferry routes for sailing ban risk."""

    def __init__(self):
        self.checker = SailingBanChecker()

    def check_all_routes(
        self,
        vessel_type: str = "CONVENTIONAL",
        forecast_days: int = 2,
    ) -> list[dict]:
        """
        Fetch weather data and check all configured routes.

        Args:
            vessel_type: Type of vessel to check thresholds for
            forecast_days: Number of forecast days

        Returns:
            List of route check results, sorted by risk level
        """
        results = []
        for route_id, route_info in ROUTES.items():
            origin_code = route_id.split("-")[0]
            dest_code = route_id.split("-")[1]

            origin_port = PORTS.get(origin_code)
            dest_port = PORTS.get(dest_code)

            if not origin_port or not dest_port:
                continue

            weather_data = fetch_route_conditions(
                origin_lat=origin_port["lat"],
                origin_lon=origin_port["lon"],
                dest_lat=dest_port["lat"],
                dest_lon=dest_port["lon"],
                forecast_days=forecast_days,
            )

            result = self.checker.check_route(
                route_id=route_id,
                weather_data=weather_data,
                vessel_type=vessel_type,
            )
            results.append(result)

        # Sort: BAN_LIKELY first, then AT_RISK, then CLEAR
        status_order = {"BAN_LIKELY": 0, "AT_RISK": 1, "CLEAR": 2}
        results.sort(key=lambda r: status_order.get(r["overall_status"], 3))

        return results

    def check_single_route(
        self,
        route_id: str,
        vessel_type: str = "CONVENTIONAL",
        forecast_days: int = 3,
    ) -> dict:
        """
        Check a single route by ID.

        Args:
            route_id: Route key (e.g. "PIR-MYK")
            vessel_type: Vessel type key
            forecast_days: Number of forecast days

        Returns:
            Route check result dict
        """
        route_info = ROUTES.get(route_id)
        if not route_info:
            raise ValueError(f"Unknown route: {route_id}")

        origin_code = route_id.split("-")[0]
        dest_code = route_id.split("-")[1]

        origin_port = PORTS[origin_code]
        dest_port = PORTS[dest_code]

        weather_data = fetch_route_conditions(
            origin_lat=origin_port["lat"],
            origin_lon=origin_port["lon"],
            dest_lat=dest_port["lat"],
            dest_lon=dest_port["lon"],
            forecast_days=forecast_days,
        )

        return self.checker.check_route(
            route_id=route_id,
            weather_data=weather_data,
            vessel_type=vessel_type,
        )
