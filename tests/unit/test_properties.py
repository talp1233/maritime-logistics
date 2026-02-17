"""
Property-based tests using Hypothesis.

Tests invariants that should hold for ANY valid input,
not just specific test cases.
"""

import math

# Hypothesis is optional — skip gracefully if not installed
try:
    from hypothesis import given, strategies as st, settings, assume
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

import pytest

from src.config.constants import knots_to_beaufort, BEAUFORT_SCALE
from src.services.sailing_ban_checker import SailingBanChecker


# Skip entire module if hypothesis not installed
pytestmark = pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")


class TestBeaufortProperties:
    """Property-based tests for Beaufort scale conversion."""

    @given(wind=st.floats(min_value=0, max_value=200, allow_nan=False))
    def test_beaufort_always_in_valid_range(self, wind):
        """Beaufort should always be 0-12 for any non-negative wind."""
        bf = knots_to_beaufort(wind)
        assert 0 <= bf <= 12

    @given(wind=st.floats(min_value=0, max_value=200, allow_nan=False))
    def test_beaufort_monotonically_increasing(self, wind):
        """Higher wind should never give a lower Beaufort number."""
        bf1 = knots_to_beaufort(wind)
        bf2 = knots_to_beaufort(wind + 5)
        assert bf2 >= bf1

    @given(wind=st.floats(min_value=64, max_value=500, allow_nan=False))
    def test_hurricane_force(self, wind):
        """64+ knots should always be Beaufort 12."""
        assert knots_to_beaufort(wind) == 12

    @given(wind=st.floats(min_value=0, max_value=0.9, allow_nan=False))
    def test_calm(self, wind):
        """Under 1 knot should be Beaufort 0."""
        assert knots_to_beaufort(wind) == 0


class TestCheckerProperties:
    """Property-based tests for the sailing ban checker."""

    @given(
        wind=st.floats(min_value=0, max_value=100, allow_nan=False),
        wave=st.floats(min_value=0, max_value=15, allow_nan=False),
    )
    def test_status_always_valid(self, wind, wave):
        """Check result should always have a valid status."""
        checker = SailingBanChecker()
        result = checker.check_conditions(wind, wave, "CONVENTIONAL")
        assert result["status"] in ("CLEAR", "AT_RISK", "BAN_LIKELY")

    @given(
        wind=st.floats(min_value=0, max_value=100, allow_nan=False),
        wave=st.floats(min_value=0, max_value=15, allow_nan=False),
    )
    def test_high_speed_stricter_than_conventional(self, wind, wave):
        """High-speed should never be safer than conventional at same conditions."""
        checker = SailingBanChecker()
        conv = checker.check_conditions(wind, wave, "CONVENTIONAL")
        hs = checker.check_conditions(wind, wave, "HIGH_SPEED")

        severity = {"CLEAR": 0, "AT_RISK": 1, "BAN_LIKELY": 2}
        assert severity[hs["status"]] >= severity[conv["status"]]

    @given(
        wind=st.floats(min_value=60, max_value=100, allow_nan=False),
        wave=st.floats(min_value=0, max_value=15, allow_nan=False),
    )
    def test_extreme_wind_always_ban(self, wind, wave):
        """Very high wind should always result in BAN_LIKELY."""
        checker = SailingBanChecker()
        result = checker.check_conditions(wind, wave, "CONVENTIONAL")
        assert result["status"] == "BAN_LIKELY"

    @given(
        wind=st.floats(min_value=0, max_value=5, allow_nan=False),
        wave=st.floats(min_value=0, max_value=0.5, allow_nan=False),
    )
    def test_calm_conditions_always_clear(self, wind, wave):
        """Very calm conditions should always be CLEAR."""
        checker = SailingBanChecker()
        result = checker.check_conditions(wind, wave, "CONVENTIONAL")
        assert result["status"] == "CLEAR"

    @given(
        wind=st.floats(min_value=0, max_value=100, allow_nan=False),
        wave=st.floats(min_value=0, max_value=15, allow_nan=False),
    )
    def test_result_has_all_required_fields(self, wind, wave):
        """Result should always have all required fields."""
        checker = SailingBanChecker()
        result = checker.check_conditions(wind, wave)
        assert "status" in result
        assert "reason" in result
        assert "beaufort" in result
        assert "wind_speed_knots" in result
        assert "wave_height_m" in result


if HAS_HYPOTHESIS:
    from src.services.route_analysis import (
        calculate_bearing, wind_angle_to_route, wind_exposure_factor,
    )

    class TestRouteAnalysisProperties:
        """Property-based tests for geographic calculations."""

        @given(
            lat1=st.floats(min_value=-85, max_value=85, allow_nan=False),
            lon1=st.floats(min_value=-180, max_value=180, allow_nan=False),
            lat2=st.floats(min_value=-85, max_value=85, allow_nan=False),
            lon2=st.floats(min_value=-180, max_value=180, allow_nan=False),
        )
        def test_bearing_always_0_to_360(self, lat1, lon1, lat2, lon2):
            """Bearing should always be in [0, 360)."""
            assume(abs(lat1 - lat2) > 0.001 or abs(lon1 - lon2) > 0.001)
            bearing = calculate_bearing(lat1, lon1, lat2, lon2)
            assert 0 <= bearing < 360

        @given(
            wind_dir=st.floats(min_value=0, max_value=360, allow_nan=False),
            route_dir=st.floats(min_value=0, max_value=360, allow_nan=False),
        )
        def test_wind_angle_always_0_to_180(self, wind_dir, route_dir):
            """Wind angle should always be 0-180."""
            angle = wind_angle_to_route(wind_dir, route_dir)
            assert 0 <= angle <= 180

        @given(angle=st.floats(min_value=0, max_value=180, allow_nan=False))
        def test_exposure_factor_bounded(self, angle):
            """Exposure factor should always be between 0.7 and 1.3."""
            factor = wind_exposure_factor(angle)
            assert 0.7 <= factor <= 1.3
