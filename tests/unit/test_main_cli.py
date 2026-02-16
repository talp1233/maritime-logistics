"""Tests for main.py CLI logic (without live API calls)."""

import sys
from unittest.mock import patch, MagicMock

from src.config.constants import ROUTES, PORTS, VESSEL_TYPES
from src.services.sailing_ban_checker import SailingBanChecker


def _make_mock_weather(wind_speeds, wave_heights):
    """Build a mock weather_data dict matching fetch_route_conditions() output."""
    times = [f"2025-01-01T{h:02d}:00" for h in range(len(wind_speeds))]
    return {
        "midpoint": {
            "marine": {"hourly": {"wave_height": wave_heights}},
            "weather": {"hourly": {
                "time": times,
                "wind_speed_10m": wind_speeds,
            }},
        },
    }


class TestCLIIntegration:
    """Test the end-to-end flow: weather data -> ban checker -> result."""

    def test_calm_day_all_clear(self):
        checker = SailingBanChecker()
        weather = _make_mock_weather(
            wind_speeds=[8, 10, 12, 10, 8],
            wave_heights=[0.5, 0.8, 1.0, 0.8, 0.5],
        )
        result = checker.check_route("PIR-MYK", weather, "CONVENTIONAL")
        assert result["overall_status"] == "CLEAR"
        assert result["route_id"] == "PIR-MYK"

    def test_stormy_day_ban_likely(self):
        checker = SailingBanChecker()
        weather = _make_mock_weather(
            wind_speeds=[10, 20, 45, 50, 35],
            wave_heights=[1.0, 2.0, 6.0, 7.0, 4.0],
        )
        result = checker.check_route("PIR-SAN", weather, "CONVENTIONAL")
        assert result["overall_status"] == "BAN_LIKELY"

    def test_high_speed_more_sensitive(self):
        """High-speed vessels should get BAN_LIKELY at lower wind."""
        checker = SailingBanChecker()
        weather = _make_mock_weather(
            wind_speeds=[10, 15, 25, 20, 15],
            wave_heights=[1.0, 1.5, 2.0, 1.5, 1.0],
        )
        conv = checker.check_route("PIR-MYK", weather, "CONVENTIONAL")
        hs = checker.check_route("PIR-MYK", weather, "HIGH_SPEED")
        # Conventional should be fine, high-speed should flag a ban
        assert conv["overall_status"] in ("CLEAR", "AT_RISK")
        assert hs["overall_status"] == "BAN_LIKELY"

    def test_all_routes_can_be_checked(self):
        """Every configured route should be checkable without errors."""
        checker = SailingBanChecker()
        weather = _make_mock_weather(
            wind_speeds=[10, 10, 10],
            wave_heights=[1.0, 1.0, 1.0],
        )
        for route_id in ROUTES:
            result = checker.check_route(route_id, weather, "CONVENTIONAL")
            assert result["route_id"] == route_id
            assert result["overall_status"] in ("CLEAR", "AT_RISK", "BAN_LIKELY")

    def test_summary_sorted_by_severity(self):
        """Results should be sortable by status severity."""
        results = [
            {"overall_status": "CLEAR"},
            {"overall_status": "BAN_LIKELY"},
            {"overall_status": "AT_RISK"},
        ]
        order = {"BAN_LIKELY": 0, "AT_RISK": 1, "CLEAR": 2}
        results.sort(key=lambda r: order[r["overall_status"]])
        assert results[0]["overall_status"] == "BAN_LIKELY"
        assert results[1]["overall_status"] == "AT_RISK"
        assert results[2]["overall_status"] == "CLEAR"
