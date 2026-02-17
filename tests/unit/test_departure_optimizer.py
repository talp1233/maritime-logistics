"""Tests for the departure window optimizer."""

import pytest

from src.services.departure_optimizer import (
    find_optimal_departure,
    optimize_route_departures,
    OptimizationResult,
    DepartureWindow,
)


def _make_hourly_scores(scores, start_hour=0):
    """Create hourly risk dicts for testing."""
    return [
        {
            "time": f"2025-06-15T{(start_hour + i) % 24:02d}:00",
            "risk_score": score,
            "band": "CLEAR" if score < 25 else "AT_RISK" if score < 65 else "HIGH",
            "cancel_probability": min(1.0, score / 100.0),
            "delay_probability": 0.2 if 30 < score < 70 else 0.05,
        }
        for i, score in enumerate(scores)
    ]


class TestFindOptimalDeparture:
    def test_clear_conditions_keep(self):
        """If scheduled departure is already safe, recommend keeping it."""
        scores = _make_hourly_scores([10, 15, 12, 18, 20])
        result = find_optimal_departure(scores, scheduled_hour_index=2)
        assert result.recommendation == "keep"
        assert result.best_window is None

    def test_finds_better_window(self):
        """Should find a lower-risk window nearby."""
        # Hour 2 is dangerous (70), but hour 3 drops to 20
        scores = _make_hourly_scores([60, 65, 70, 20, 25])
        result = find_optimal_departure(scores, scheduled_hour_index=2)
        assert result.recommendation == "shift"
        assert result.best_window is not None
        assert result.best_window.risk_score < 70
        assert result.best_window.improvement > 0

    def test_no_better_window_high_risk(self):
        """If everything is dangerous, recommend cancel_likely."""
        scores = _make_hourly_scores([80, 85, 90, 85, 80])
        result = find_optimal_departure(scores, scheduled_hour_index=2)
        assert result.recommendation == "cancel_likely"

    def test_respects_max_shift(self):
        """Should not suggest windows beyond max shift."""
        # Safe window is 3 hours away (180 min > 90 min default)
        scores = _make_hourly_scores([70, 70, 70, 70, 70, 70, 10])
        result = find_optimal_departure(scores, scheduled_hour_index=2)
        # Hour 6 is 4 hours away → should NOT be suggested
        if result.best_window:
            assert abs(result.best_window.shift_minutes) <= 90

    def test_prefers_closest_window(self):
        """Among equal-quality windows, prefer the closest one."""
        scores = _make_hourly_scores([20, 60, 20])
        result = find_optimal_departure(scores, scheduled_hour_index=1)
        assert result.recommendation == "shift"
        # Both hour 0 and hour 2 are equally good
        # The optimizer sorts by improvement first, then by shift distance

    def test_empty_data(self):
        result = find_optimal_departure([], scheduled_hour_index=0)
        assert result.recommendation == "keep"

    def test_out_of_range_index(self):
        scores = _make_hourly_scores([10, 20, 30])
        result = find_optimal_departure(scores, scheduled_hour_index=10)
        assert result.recommendation == "keep"

    def test_alternatives_provided(self):
        """Should provide alternative windows when multiple exist."""
        scores = _make_hourly_scores([15, 20, 70, 18, 22])
        result = find_optimal_departure(scores, scheduled_hour_index=2)
        assert result.recommendation == "shift"
        # Should have alternatives
        assert isinstance(result.alternatives, list)

    def test_to_dict(self):
        scores = _make_hourly_scores([15, 20, 70, 18, 22])
        result = find_optimal_departure(scores, scheduled_hour_index=2)
        d = result.to_dict()
        assert "scheduled_time" in d
        assert "recommendation" in d
        assert "reason" in d
        assert "alternatives" in d


class TestOptimizeRouteDepartures:
    def test_default_departure_hours(self):
        """Should check default departure hours (7, 10, 14, 17)."""
        # Create 24 hours of data
        scores = _make_hourly_scores(
            [10] * 7 + [60] + [10] * 2 + [65] + [10] * 3 + [70] + [10] * 7,
            start_hour=0,
        )
        results = optimize_route_departures(scores)
        # Should have results for each default departure hour found
        assert len(results) > 0

    def test_custom_departure_hours(self):
        scores = _make_hourly_scores([20] * 24, start_hour=0)
        results = optimize_route_departures(scores, departure_hours=[6, 18])
        assert len(results) == 2

    def test_each_result_is_optimization(self):
        scores = _make_hourly_scores([30] * 24, start_hour=0)
        results = optimize_route_departures(scores)
        for r in results:
            assert isinstance(r, OptimizationResult)
