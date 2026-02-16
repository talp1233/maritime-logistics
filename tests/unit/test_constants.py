"""Tests for configuration constants and helper functions."""

from src.config.constants import (
    BEAUFORT_SCALE,
    SAILING_BAN_THRESHOLDS,
    VESSEL_TYPES,
    ROUTES,
    PORTS,
    knots_to_beaufort,
    kmh_to_knots,
    ms_to_knots,
)


class TestBeaufortConversion:
    def test_calm(self):
        assert knots_to_beaufort(0) == 0

    def test_light_breeze(self):
        assert knots_to_beaufort(5) == 2

    def test_gale(self):
        assert knots_to_beaufort(35) == 8

    def test_hurricane(self):
        assert knots_to_beaufort(70) == 12

    def test_above_scale(self):
        assert knots_to_beaufort(200) == 12


class TestUnitConversions:
    def test_kmh_to_knots(self):
        # 1 knot = 1.852 km/h
        result = kmh_to_knots(1.852)
        assert abs(result - 1.0) < 0.01

    def test_ms_to_knots(self):
        # 1 m/s ≈ 1.94384 knots
        result = ms_to_knots(1.0)
        assert abs(result - 1.94384) < 0.001

    def test_zero_conversion(self):
        assert kmh_to_knots(0) == 0
        assert ms_to_knots(0) == 0


class TestConfigData:
    def test_beaufort_scale_complete(self):
        """Beaufort scale should cover 0-12."""
        for i in range(13):
            assert i in BEAUFORT_SCALE

    def test_all_routes_have_required_fields(self):
        for route_id, route in ROUTES.items():
            assert "origin" in route, f"Route {route_id} missing origin"
            assert "destination" in route, f"Route {route_id} missing destination"
            assert "distance_nm" in route, f"Route {route_id} missing distance_nm"
            assert "sea_area" in route, f"Route {route_id} missing sea_area"

    def test_all_ports_have_coordinates(self):
        for port_id, port in PORTS.items():
            assert "lat" in port, f"Port {port_id} missing lat"
            assert "lon" in port, f"Port {port_id} missing lon"
            assert "name" in port, f"Port {port_id} missing name"

    def test_route_endpoints_exist_in_ports(self):
        """Every route origin/destination code should exist in PORTS."""
        for route_id in ROUTES:
            origin_code, dest_code = route_id.split("-")
            assert origin_code in PORTS, f"Origin {origin_code} not in PORTS"
            assert dest_code in PORTS, f"Destination {dest_code} not in PORTS"

    def test_sailing_ban_thresholds_valid(self):
        for category, bf in SAILING_BAN_THRESHOLDS.items():
            assert 0 <= bf <= 12, f"Threshold {category}={bf} out of Beaufort range"
