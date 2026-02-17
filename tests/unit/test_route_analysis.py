"""
Unit tests for route analysis — wind angle, shelter, waypoints.
"""

import math

from src.services.route_analysis import (
    calculate_bearing,
    wind_angle_to_route,
    wind_exposure_factor,
    interpolate_route_points,
    get_route_sample_points,
    is_port_sheltered,
    port_shelter_factor,
    analyze_route_risk,
)


class TestBearingCalculation:
    def test_north_bearing(self):
        # Going due north
        bearing = calculate_bearing(37.0, 25.0, 38.0, 25.0)
        assert abs(bearing - 0) < 2 or abs(bearing - 360) < 2

    def test_east_bearing(self):
        # Going due east
        bearing = calculate_bearing(37.0, 25.0, 37.0, 26.0)
        assert 85 < bearing < 95

    def test_south_bearing(self):
        bearing = calculate_bearing(38.0, 25.0, 37.0, 25.0)
        assert 175 < bearing < 185

    def test_west_bearing(self):
        bearing = calculate_bearing(37.0, 26.0, 37.0, 25.0)
        assert 265 < bearing < 275

    def test_bearing_range(self):
        """Bearing should always be 0-360."""
        bearing = calculate_bearing(37.0, 25.0, 36.5, 24.5)
        assert 0 <= bearing < 360


class TestWindAngle:
    def test_headwind(self):
        angle = wind_angle_to_route(0, 0)
        assert angle == 0

    def test_tailwind(self):
        angle = wind_angle_to_route(180, 0)
        assert angle == 180

    def test_beam_wind(self):
        angle = wind_angle_to_route(90, 0)
        assert angle == 90

    def test_symmetry(self):
        """Wind from 90 or 270 relative to north-heading should both be 90."""
        a1 = wind_angle_to_route(90, 0)
        a2 = wind_angle_to_route(270, 0)
        assert a1 == a2

    def test_wrapping(self):
        """350-10 should be 20 degrees apart."""
        angle = wind_angle_to_route(350, 10)
        assert abs(angle - 20) < 0.1


class TestWindExposure:
    def test_beam_worst(self):
        factor = wind_exposure_factor(90)
        assert factor == 1.3

    def test_following_safest(self):
        factor = wind_exposure_factor(180)
        assert factor == 0.7

    def test_headwind_elevated(self):
        factor = wind_exposure_factor(10)
        assert factor == 1.15

    def test_quartering_neutral(self):
        factor = wind_exposure_factor(50)
        assert factor == 1.0


class TestRoutePoints:
    def test_interpolate_includes_endpoints(self):
        points = interpolate_route_points(37.0, 25.0, 38.0, 26.0, n_points=3)
        assert len(points) == 5  # 2 endpoints + 3 intermediate
        assert points[0] == (37.0, 25.0)
        assert points[-1] == (38.0, 26.0)

    def test_interpolate_monotonic(self):
        """Points should be monotonically spaced."""
        points = interpolate_route_points(37.0, 25.0, 38.0, 26.0, n_points=4)
        lats = [p[0] for p in points]
        for i in range(len(lats) - 1):
            assert lats[i] < lats[i+1]

    def test_get_route_sample_points(self):
        points = get_route_sample_points("PIR-MYK")
        assert len(points) >= 2
        assert points[0]["distance_nm"] == 0
        assert "departure" in points[0]["name"]
        assert "arrival" in points[-1]["name"]

    def test_invalid_route_raises(self):
        try:
            get_route_sample_points("INVALID")
            assert False, "Should raise"
        except ValueError:
            pass


class TestPortShelter:
    def test_piraeus_sheltered_from_south(self):
        assert is_port_sheltered("PIR", 200) is True

    def test_piraeus_not_sheltered_from_north(self):
        assert is_port_sheltered("PIR", 0) is False

    def test_santorin_no_shelter(self):
        """Santorin is marked as exposed."""
        assert is_port_sheltered("SAN", 0) is False
        assert is_port_sheltered("SAN", 180) is False

    def test_unknown_port_no_shelter(self):
        assert is_port_sheltered("UNKNOWN", 0) is False

    def test_shelter_factor(self):
        sheltered = port_shelter_factor("PIR", 200)
        exposed = port_shelter_factor("PIR", 0)
        assert sheltered < exposed

    def test_andros_wrapping(self):
        """Andros has range (270, 45) which wraps around 360."""
        assert is_port_sheltered("AND", 350) is True
        assert is_port_sheltered("AND", 10) is True
        assert is_port_sheltered("AND", 180) is False


class TestAnalyzeRouteRisk:
    def test_basic_analysis(self):
        result = analyze_route_risk("PIR-MYK", 20.0, 0.0, 2.0)
        assert result["route_id"] == "PIR-MYK"
        assert "wind" in result
        assert "shelter" in result
        assert "sample_points" in result

    def test_invalid_route(self):
        try:
            analyze_route_risk("XXX-YYY", 20.0, 0.0, 2.0)
            assert False, "Should raise"
        except ValueError:
            pass

    def test_wind_exposure_applied(self):
        result = analyze_route_risk("PIR-MYK", 20.0, 0.0, 2.0)
        # Effective wind should differ from actual due to angle factor
        assert "effective_wind_knots" in result["wind"]
        assert result["wind"]["exposure_factor"] > 0
