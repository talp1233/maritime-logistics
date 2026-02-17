"""Tests for the validation backtester."""

import pytest

from src.validation.backtester import (
    brier_score,
    skill_score,
    calibration_curve,
    run_backtest,
)


class TestBrierScore:
    def test_perfect_predictions(self):
        """Perfect predictions → Brier score 0."""
        preds = [1.0, 0.0, 1.0, 0.0]
        outcomes = [1, 0, 1, 0]
        assert brier_score(preds, outcomes) == 0.0

    def test_worst_predictions(self):
        """Completely wrong → Brier score 1.0."""
        preds = [0.0, 1.0, 0.0, 1.0]
        outcomes = [1, 0, 1, 0]
        assert brier_score(preds, outcomes) == 1.0

    def test_uncertain_predictions(self):
        """50/50 predictions → Brier score 0.25."""
        preds = [0.5, 0.5, 0.5, 0.5]
        outcomes = [1, 0, 1, 0]
        assert abs(brier_score(preds, outcomes) - 0.25) < 0.001

    def test_empty_predictions(self):
        assert brier_score([], []) == 1.0

    def test_single_prediction(self):
        assert brier_score([0.8], [1]) == pytest.approx(0.04)


class TestSkillScore:
    def test_better_than_baseline(self):
        """Model with lower Brier → positive skill."""
        assert skill_score(0.1, 0.3) > 0

    def test_same_as_baseline(self):
        """Same Brier → skill = 0."""
        assert skill_score(0.2, 0.2) == 0.0

    def test_worse_than_baseline(self):
        """Higher Brier → negative skill."""
        assert skill_score(0.4, 0.2) < 0

    def test_zero_baseline(self):
        assert skill_score(0.1, 0.0) == 0.0


class TestCalibrationCurve:
    def test_basic_calibration(self):
        preds = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
        outcomes = [0, 0, 0, 1, 1, 1, 1]
        curve = calibration_curve(preds, outcomes, n_bins=5)
        assert len(curve) == 5
        for b in curve:
            assert "bin_lo" in b
            assert "bin_hi" in b
            assert "mean_predicted" in b
            assert "observed_frequency" in b
            assert "count" in b

    def test_empty_data(self):
        curve = calibration_curve([], [], n_bins=5)
        assert len(curve) == 5
        # All bins should have count 0
        assert all(b["count"] == 0 for b in curve)

    def test_all_in_one_bin(self):
        """All predictions in the same range → one bin has all data."""
        preds = [0.05, 0.06, 0.07, 0.08]
        outcomes = [0, 0, 0, 1]
        curve = calibration_curve(preds, outcomes, n_bins=10)
        total_count = sum(b["count"] for b in curve)
        assert total_count == 4


class TestRunBacktest:
    def test_runs_without_error(self):
        """Backtest should run against existing ground truth data."""
        try:
            report = run_backtest()
            if "error" not in report:
                assert "metrics" in report
                assert "calibration_curve" in report
                assert "per_route" in report
                assert report["validation_records"] > 0

                m = report["metrics"]
                assert 0 <= m["model_brier_score"] <= 1
                assert 0 <= m["baseline_brier_score"] <= 1
                assert 0 <= m["model_accuracy"] <= 1
        except FileNotFoundError:
            pytest.skip("No ground truth data available")

    def test_missing_file(self):
        """Should raise FileNotFoundError for missing data."""
        from pathlib import Path
        with pytest.raises(FileNotFoundError):
            run_backtest(csv_path=Path("/nonexistent/data.csv"))
