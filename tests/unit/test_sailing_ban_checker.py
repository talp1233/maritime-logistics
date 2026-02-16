"""Tests for the sailing ban checker service."""

from src.services.sailing_ban_checker import SailingBanChecker


class TestSailingBanChecker:
    def setup_method(self):
        self.checker = SailingBanChecker()

    # --- Conventional vessels (threshold: Beaufort 8 = 34 knots) ---

    def test_conventional_clear(self):
        result = self.checker.check_conditions(
            wind_speed_knots=15, wave_height_m=1.0, vessel_type="CONVENTIONAL"
        )
        assert result["status"] == "CLEAR"

    def test_conventional_ban_by_wind(self):
        result = self.checker.check_conditions(
            wind_speed_knots=40, wave_height_m=2.0, vessel_type="CONVENTIONAL"
        )
        assert result["status"] == "BAN_LIKELY"
        assert result["beaufort"] >= 8

    def test_conventional_ban_by_waves(self):
        result = self.checker.check_conditions(
            wind_speed_knots=10, wave_height_m=6.0, vessel_type="CONVENTIONAL"
        )
        assert result["status"] == "BAN_LIKELY"

    def test_conventional_at_risk(self):
        # Beaufort 7 (28-33 knots) is one below threshold 8
        result = self.checker.check_conditions(
            wind_speed_knots=30, wave_height_m=2.0, vessel_type="CONVENTIONAL"
        )
        assert result["status"] == "AT_RISK"

    # --- High-speed vessels (threshold: Beaufort 6 = 22 knots) ---

    def test_high_speed_clear(self):
        result = self.checker.check_conditions(
            wind_speed_knots=10, wave_height_m=1.0, vessel_type="HIGH_SPEED"
        )
        assert result["status"] == "CLEAR"

    def test_high_speed_ban_by_wind(self):
        result = self.checker.check_conditions(
            wind_speed_knots=25, wave_height_m=1.0, vessel_type="HIGH_SPEED"
        )
        assert result["status"] == "BAN_LIKELY"

    def test_high_speed_ban_by_waves(self):
        result = self.checker.check_conditions(
            wind_speed_knots=10, wave_height_m=3.0, vessel_type="HIGH_SPEED"
        )
        assert result["status"] == "BAN_LIKELY"

    # --- Catamaran follows high-speed rules ---

    def test_catamaran_uses_high_speed_rules(self):
        result = self.checker.check_conditions(
            wind_speed_knots=25, wave_height_m=1.0, vessel_type="CATAMARAN"
        )
        assert result["status"] == "BAN_LIKELY"
        assert result["vessel_category"] == "high_speed"

    # --- Result structure ---

    def test_result_contains_all_fields(self):
        result = self.checker.check_conditions(
            wind_speed_knots=10, wave_height_m=1.0
        )
        expected_keys = {
            "status", "reason", "beaufort", "wind_speed_knots",
            "wave_height_m", "vessel_category", "bf_threshold",
            "wave_threshold",
        }
        assert expected_keys.issubset(result.keys())

    # --- Route checking ---

    def test_check_route_with_mock_data(self):
        """Test route checking with synthetic weather data."""
        mock_weather = {
            "midpoint": {
                "marine": {
                    "hourly": {
                        "wave_height": [1.0, 1.5, 2.0],
                    }
                },
                "weather": {
                    "hourly": {
                        "time": ["2025-01-01T06:00", "2025-01-01T07:00", "2025-01-01T08:00"],
                        "wind_speed_10m": [10, 15, 20],
                    }
                },
            }
        }
        result = self.checker.check_route("PIR-MYK", mock_weather, "CONVENTIONAL")
        assert result["route_id"] == "PIR-MYK"
        assert result["overall_status"] == "CLEAR"
        assert len(result["hourly"]) == 3

    def test_check_route_ban_detected(self):
        """Route check should detect BAN_LIKELY when wind exceeds threshold."""
        mock_weather = {
            "midpoint": {
                "marine": {
                    "hourly": {
                        "wave_height": [1.0, 6.0],
                    }
                },
                "weather": {
                    "hourly": {
                        "time": ["2025-01-01T06:00", "2025-01-01T07:00"],
                        "wind_speed_10m": [10, 45],
                    }
                },
            }
        }
        result = self.checker.check_route("PIR-SAN", mock_weather, "HIGH_SPEED")
        assert result["overall_status"] == "BAN_LIKELY"

    def test_check_route_unknown_route(self):
        """Should raise ValueError for unknown route."""
        try:
            self.checker.check_route("XXX-YYY", {}, "CONVENTIONAL")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
