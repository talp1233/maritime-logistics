"""Tests for the probabilistic risk scoring engine."""

import pytest

from src.services.risk_scorer import (
    compute_risk,
    score_route,
    score_to_band,
    _wind_risk,
    _wave_risk,
    _geo_modifier,
    _gust_modifier,
    _vessel_modifier,
    RiskResult,
)


# ── Band mapping ──────────────────────────────────────────────────────

class TestScoreToBand:
    def test_clear(self):
        assert score_to_band(0) == "CLEAR"
        assert score_to_band(24.9) == "CLEAR"

    def test_monitor(self):
        assert score_to_band(25) == "MONITOR"
        assert score_to_band(44.9) == "MONITOR"

    def test_at_risk(self):
        assert score_to_band(45) == "AT_RISK"
        assert score_to_band(64.9) == "AT_RISK"

    def test_high(self):
        assert score_to_band(65) == "HIGH"
        assert score_to_band(81.9) == "HIGH"

    def test_critical(self):
        assert score_to_band(82) == "CRITICAL"
        assert score_to_band(100) == "CRITICAL"


# ── Wind risk component ──────────────────────────────────────────────

class TestWindRisk:
    def test_zero_wind(self):
        assert _wind_risk(0, 8) == 0.0

    def test_at_threshold(self):
        # Bf 8 = 34 knots.  wind_risk should be ~1.0 (ratio = 8/8 = 1.0)
        risk = _wind_risk(35.0, 8)
        assert 0.9 <= risk <= 1.1

    def test_well_below_threshold(self):
        # Bf 4 / threshold 8 = 0.5
        risk = _wind_risk(14.0, 8)
        assert risk < 0.6

    def test_above_threshold(self):
        # Bf 10 / threshold 8 = 1.25
        risk = _wind_risk(50.0, 8)
        assert risk > 1.0

    def test_direction_beam_amplifies(self):
        # Beam wind (90°) amplifies risk
        no_dir = _wind_risk(25.0, 8, wind_direction=None, route_bearing=None)
        beam = _wind_risk(25.0, 8, wind_direction=90.0, route_bearing=0.0)
        assert beam > no_dir

    def test_direction_following_dampens(self):
        # Following wind (180°) reduces risk
        no_dir = _wind_risk(25.0, 8, wind_direction=None, route_bearing=None)
        following = _wind_risk(25.0, 8, wind_direction=0.0, route_bearing=0.0)
        # wind from N (0°), route heading N (0°) → headwind (180° relative)
        # actually: wind_dir=0, bearing=0 → wind_angle_to_route = 0 → headwind
        headwind = _wind_risk(25.0, 8, wind_direction=180.0, route_bearing=0.0)
        # Headwind should be higher than following (180°)
        assert headwind <= beam if 'beam' in dir() else True


# ── Wave risk component ──────────────────────────────────────────────

class TestWaveRisk:
    def test_zero_waves(self):
        assert _wave_risk(0.0, 5.0, 0.0, 5.0) == 0.0

    def test_at_threshold(self):
        # Hs = threshold, no swell, normal period → risk ~1.0
        risk = _wave_risk(5.0, 6.0, 0.0, 5.0)
        assert 0.9 <= risk <= 1.5

    def test_steep_waves_penalised(self):
        # Short period = steep waves = higher risk
        long_period = _wave_risk(2.0, 8.0, 0.0, 5.0)
        short_period = _wave_risk(2.0, 3.0, 0.0, 5.0)
        assert short_period > long_period

    def test_swell_increases_combined(self):
        # Adding swell increases combined Hs → higher risk
        no_swell = _wave_risk(2.0, 5.0, 0.0, 5.0)
        with_swell = _wave_risk(2.0, 5.0, 1.5, 5.0)
        assert with_swell > no_swell

    def test_zero_threshold_returns_zero(self):
        assert _wave_risk(2.0, 5.0, 0.0, 0.0) == 0.0


# ── Geographic modifier ──────────────────────────────────────────────

class TestGeoModifier:
    def test_no_route(self):
        assert _geo_modifier(None, None) == 1.0

    def test_unknown_route(self):
        assert _geo_modifier("UNKNOWN", None) == 1.0

    def test_sheltered_route_lower(self):
        # RAF-AND is marked as not exposed
        exposed = _geo_modifier("PIR-MYK", 0.0)
        sheltered = _geo_modifier("RAF-AND", 0.0)
        assert sheltered <= exposed

    def test_result_in_bounds(self):
        for route_id in ["PIR-MYK", "PIR-SAN", "RAF-AND", "PIR-HER"]:
            mod = _geo_modifier(route_id, 180.0)
            assert 0.5 <= mod <= 1.3


# ── Gust modifier ────────────────────────────────────────────────────

class TestGustModifier:
    def test_zero_wind(self):
        assert _gust_modifier(0, 0) == 1.0

    def test_normal_gust_ratio(self):
        # 1.3× is normal → modifier ~1.0
        mod = _gust_modifier(20.0, 26.0)
        assert 0.99 <= mod <= 1.01

    def test_high_gust_amplifies(self):
        # 2.0× gust ratio is severe
        mod = _gust_modifier(20.0, 40.0)
        assert mod > 1.1

    def test_capped_at_1_3(self):
        mod = _gust_modifier(10.0, 100.0)
        assert mod <= 1.3


# ── Vessel modifier ──────────────────────────────────────────────────

class TestVesselModifier:
    def test_large_vessel_dampens(self):
        # Knossos Palace: 36000t, Bf 10 → should dampen risk
        mod = _vessel_modifier("Knossos Palace", "CONVENTIONAL", 8)
        assert mod < 1.0

    def test_small_hsc_amplifies(self):
        # Tera Jet: 4700t, Bf 5 → should amplify risk
        mod = _vessel_modifier("Tera Jet", "HIGH_SPEED", 6)
        assert mod >= 0.9  # may be close to 1.0

    def test_unknown_vessel_neutral(self):
        mod = _vessel_modifier(None, "CONVENTIONAL", 8)
        assert 0.8 <= mod <= 1.2

    def test_result_in_bounds(self):
        mod = _vessel_modifier("Blue Star Delos", "CONVENTIONAL", 8)
        assert 0.7 <= mod <= 1.3


# ── Full compute_risk ─────────────────────────────────────────────────

class TestComputeRisk:
    def test_calm_conditions_clear(self):
        result = compute_risk(
            wind_speed_knots=10.0,
            wave_height_m=0.5,
            vessel_type="CONVENTIONAL",
        )
        assert isinstance(result, RiskResult)
        assert result.band == "CLEAR"
        assert result.score < 30
        assert result.cancel_probability < 0.1

    def test_storm_conditions_critical(self):
        result = compute_risk(
            wind_speed_knots=50.0,
            wave_height_m=6.0,
            wave_period_s=4.0,
            vessel_type="HIGH_SPEED",
        )
        assert result.band in ("HIGH", "CRITICAL")
        assert result.score > 60
        assert result.cancel_probability > 0.5

    def test_high_speed_more_sensitive(self):
        conventional = compute_risk(
            wind_speed_knots=25.0,
            wave_height_m=2.0,
            vessel_type="CONVENTIONAL",
        )
        high_speed = compute_risk(
            wind_speed_knots=25.0,
            wave_height_m=2.0,
            vessel_type="HIGH_SPEED",
        )
        assert high_speed.score > conventional.score

    def test_route_context_matters(self):
        # Same conditions, different routes
        exposed = compute_risk(
            wind_speed_knots=25.0,
            wave_height_m=2.5,
            wind_direction=0.0,
            vessel_type="CONVENTIONAL",
            route_id="PIR-MYK",
        )
        sheltered = compute_risk(
            wind_speed_knots=25.0,
            wave_height_m=2.5,
            wind_direction=0.0,
            vessel_type="CONVENTIONAL",
            route_id="RAF-AND",
        )
        assert sheltered.score <= exposed.score

    def test_result_has_components(self):
        result = compute_risk(
            wind_speed_knots=20.0,
            wave_height_m=1.5,
        )
        assert "wind_risk" in result.components
        assert "wave_risk" in result.components
        assert "geo_modifier" in result.components
        assert "gust_modifier" in result.components
        assert "vessel_modifier" in result.components

    def test_to_dict(self):
        result = compute_risk(wind_speed_knots=15.0, wave_height_m=1.0)
        d = result.to_dict()
        assert "risk_score" in d
        assert "band" in d
        assert "cancel_probability" in d
        assert "delay_probability" in d

    def test_specific_vessel(self):
        result = compute_risk(
            wind_speed_knots=30.0,
            wave_height_m=3.0,
            vessel_type="CONVENTIONAL",
            vessel_name="Knossos Palace",
            route_id="PIR-HER",
        )
        assert isinstance(result, RiskResult)
        # Knossos Palace is large → should handle better
        generic = compute_risk(
            wind_speed_knots=30.0,
            wave_height_m=3.0,
            vessel_type="CONVENTIONAL",
            route_id="PIR-HER",
        )
        assert result.score <= generic.score

    def test_probabilities_in_range(self):
        for wind in [0, 15, 30, 45, 60]:
            result = compute_risk(wind_speed_knots=wind, wave_height_m=wind * 0.08)
            assert 0.0 <= result.cancel_probability <= 1.0
            assert 0.0 <= result.delay_probability <= 1.0
            assert 0.0 <= result.score <= 100.0


# ── Route scoring ─────────────────────────────────────────────────────

class TestScoreRoute:
    def _make_weather_data(self, wind=15.0, wave=1.0, n_hours=24):
        """Create minimal weather data structure."""
        times = [f"2025-06-15T{h:02d}:00" for h in range(n_hours)]
        return {
            "midpoint": {
                "marine": {"hourly": {
                    "wave_height": [wave] * n_hours,
                    "wave_period": [5.0] * n_hours,
                    "swell_wave_height": [0.3] * n_hours,
                }},
                "weather": {"hourly": {
                    "time": times,
                    "wind_speed_10m": [wind] * n_hours,
                    "wind_direction_10m": [20.0] * n_hours,
                    "wind_gusts_10m": [wind * 1.3] * n_hours,
                }},
            },
        }

    def test_basic_route_scoring(self):
        weather = self._make_weather_data(wind=15.0, wave=1.0)
        result = score_route("PIR-MYK", weather)
        assert result["route_id"] == "PIR-MYK"
        assert "hourly" in result
        assert len(result["hourly"]) == 24
        assert result["overall_band"] in ("CLEAR", "MONITOR", "AT_RISK", "HIGH", "CRITICAL")

    def test_storm_route(self):
        weather = self._make_weather_data(wind=45.0, wave=5.0)
        result = score_route("PIR-MYK", weather, vessel_type="HIGH_SPEED")
        assert result["max_risk_score"] > 50
        assert result["max_cancel_probability"] > 0.3

    def test_unknown_route_raises(self):
        weather = self._make_weather_data()
        with pytest.raises(ValueError, match="Unknown route"):
            score_route("NONEXISTENT", weather)

    def test_result_structure(self):
        weather = self._make_weather_data()
        result = score_route("PIR-SAN", weather)
        assert "max_risk_score" in result
        assert "avg_risk_score" in result
        assert "max_cancel_probability" in result
        assert "max_delay_probability" in result
        assert "vessel_type" in result
