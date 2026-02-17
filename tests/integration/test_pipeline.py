"""
Integration tests — end-to-end pipeline tests.

Tests the full flow: generate weather → check route → collect results,
including web API endpoints.
"""

import json

from src.config.constants import ROUTES, PORTS, VESSELS
from src.services.sailing_ban_checker import SailingBanChecker
from src.services.route_analysis import (
    analyze_route_risk,
    get_route_sample_points,
    calculate_bearing,
    wind_angle_to_route,
)
from src.data_collection.demo_data import generate_demo_route_conditions


class TestFullPipeline:
    """Test the complete prediction pipeline from data to result."""

    def test_all_routes_can_be_checked(self):
        """Every configured route should produce a valid result."""
        checker = SailingBanChecker()

        for route_id in ROUTES:
            weather = generate_demo_route_conditions(forecast_days=1, scenario="calm_summer")
            result = checker.check_route(route_id, weather, "CONVENTIONAL")

            assert result["route_id"] == route_id
            assert result["overall_status"] in ("CLEAR", "AT_RISK", "BAN_LIKELY")
            assert len(result["hourly"]) > 0

    def test_all_vessel_types_accepted(self):
        """All vessel types should be accepted by the checker."""
        checker = SailingBanChecker()
        weather = generate_demo_route_conditions(forecast_days=1, scenario="calm_summer")

        for vessel_type in ("CONVENTIONAL", "HIGH_SPEED", "CATAMARAN", "SMALL"):
            result = checker.check_route("PIR-MYK", weather, vessel_type)
            assert result["vessel_type"] == vessel_type

    def test_storm_produces_bans(self):
        """Storm scenario should produce at least one BAN_LIKELY for high-speed."""
        checker = SailingBanChecker()
        ban_count = 0

        for _ in range(5):  # Run a few times since demo data is stochastic
            weather = generate_demo_route_conditions(forecast_days=1, scenario="storm")
            result = checker.check_route("PIR-MYK", weather, "HIGH_SPEED")
            if result["overall_status"] == "BAN_LIKELY":
                ban_count += 1

        assert ban_count >= 1, "Storm should produce at least 1 ban in 5 tries"

    def test_calm_mostly_clear(self):
        """Calm scenario should be mostly CLEAR for conventional vessels."""
        checker = SailingBanChecker()
        clear_count = 0

        for _ in range(5):
            weather = generate_demo_route_conditions(forecast_days=1, scenario="calm_summer")
            result = checker.check_route("PIR-HER", weather, "CONVENTIONAL")
            if result["overall_status"] == "CLEAR":
                clear_count += 1

        assert clear_count >= 3, "Calm should produce at least 3 clear in 5 tries"

    def test_hourly_predictions_have_required_fields(self):
        """Each hourly prediction must contain all expected fields."""
        checker = SailingBanChecker()
        weather = generate_demo_route_conditions(forecast_days=1)
        result = checker.check_route("PIR-SAN", weather, "CONVENTIONAL")

        required_fields = {"status", "reason", "beaufort", "wind_speed_knots",
                           "wave_height_m", "vessel_category", "time"}

        for hourly in result["hourly"]:
            assert required_fields.issubset(hourly.keys()), \
                f"Missing fields: {required_fields - hourly.keys()}"


class TestRouteAnalysisIntegration:
    """Integration tests for the route analysis module."""

    def test_all_routes_have_valid_analysis(self):
        """Route analysis should work for every configured route."""
        for route_id in ROUTES:
            analysis = analyze_route_risk(
                route_id, wind_speed_knots=20, wind_direction=0,
                wave_height_m=2.0, vessel_type="CONVENTIONAL",
            )
            assert analysis["route_id"] == route_id
            assert 0 <= analysis["route_bearing"] < 360
            assert analysis["wind"]["angle_to_route"] >= 0
            assert analysis["wind"]["exposure_factor"] > 0

    def test_all_routes_have_sample_points(self):
        """Every route should have at least origin + destination points."""
        for route_id in ROUTES:
            points = get_route_sample_points(route_id)
            assert len(points) >= 2
            assert "departure" in points[0]["name"]
            assert "arrival" in points[-1]["name"]

    def test_route_analysis_wind_types(self):
        """Different wind directions should produce different wind types."""
        types_seen = set()
        for direction in [0, 45, 90, 135, 180, 270]:
            analysis = analyze_route_risk(
                "PIR-MYK", wind_speed_knots=20,
                wind_direction=direction, wave_height_m=2.0,
            )
            types_seen.add(analysis["wind"]["wind_type"])

        assert len(types_seen) >= 3, f"Expected multiple wind types, got: {types_seen}"


class TestVesselData:
    """Test vessel-specific configuration."""

    def test_all_vessels_reference_valid_operators(self):
        from src.config.constants import OPERATORS
        for name, vessel in VESSELS.items():
            assert vessel["operator"] in OPERATORS, \
                f"Vessel {name} references unknown operator {vessel['operator']}"

    def test_all_vessel_routes_exist(self):
        for name, vessel in VESSELS.items():
            for route in vessel["routes"]:
                assert route in ROUTES, \
                    f"Vessel {name} references unknown route {route}"

    def test_vessel_thresholds_reasonable(self):
        for name, vessel in VESSELS.items():
            assert 3 <= vessel["bf_threshold"] <= 12, \
                f"Vessel {name} has unreasonable bf_threshold: {vessel['bf_threshold']}"
            assert 1.0 <= vessel["wave_threshold"] <= 10.0, \
                f"Vessel {name} has unreasonable wave_threshold: {vessel['wave_threshold']}"


class TestWebAPIIntegration:
    """Test web API endpoints using FastAPI TestClient."""

    def setup_method(self):
        try:
            from fastapi.testclient import TestClient
            from src.web.app import app
            self.client = TestClient(app)
            self.available = True
        except ImportError:
            self.available = False

    def test_health_endpoint(self):
        if not self.available:
            return
        resp = self.client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "routes_configured" in data
        assert "vessels_configured" in data

    def test_check_endpoint(self):
        if not self.available:
            return
        resp = self.client.get("/api/check?vessel=CONVENTIONAL&scenario=calm_summer")
        assert resp.status_code == 200
        data = resp.json()
        assert "routes" in data
        assert "summary" in data
        assert len(data["routes"]) > 0

    def test_vessels_endpoint(self):
        if not self.available:
            return
        resp = self.client.get("/api/vessels")
        assert resp.status_code == 200
        data = resp.json()
        assert "Blue Star Delos" in data

    def test_route_analysis_endpoint(self):
        if not self.available:
            return
        resp = self.client.get("/api/route-analysis/PIR-MYK?wind_speed=20&wind_direction=45")
        assert resp.status_code == 200
        data = resp.json()
        assert data["route_id"] == "PIR-MYK"
        assert "wind" in data
        assert "shelter" in data

    def test_route_analysis_invalid_route(self):
        if not self.available:
            return
        resp = self.client.get("/api/route-analysis/INVALID")
        assert resp.status_code == 404

    def test_route_points_endpoint(self):
        if not self.available:
            return
        resp = self.client.get("/api/route-points/PIR-SAN")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 2

    def test_dashboard_returns_html(self):
        if not self.available:
            return
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert "Maritime" in resp.text

    def test_ports_endpoint(self):
        if not self.available:
            return
        resp = self.client.get("/api/ports")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0

    def test_routes_endpoint(self):
        if not self.available:
            return
        resp = self.client.get("/api/routes")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
