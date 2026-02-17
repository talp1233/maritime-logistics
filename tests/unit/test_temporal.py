"""Tests for the temporal prediction system."""

import pytest

from src.data_collection.temporal_dataset import (
    TEMPORAL_FEATURE_NAMES,
    extract_temporal_features,
    _linear_slope,
    _max_consecutive,
    build_temporal_dataset_synthetic,
)
from src.models.temporal_predictor import (
    _mask_features_for_lead_time,
    TemporalPredictor,
    LEAD_TIMES,
)


# ── Helpers ───────────────────────────────────────────────────────────

class TestLinearSlope:
    def test_flat(self):
        assert _linear_slope([5, 5, 5, 5]) == 0.0

    def test_increasing(self):
        slope = _linear_slope([0, 1, 2, 3])
        assert abs(slope - 1.0) < 0.01

    def test_decreasing(self):
        slope = _linear_slope([6, 4, 2, 0])
        assert slope < 0

    def test_single_value(self):
        assert _linear_slope([5]) == 0.0

    def test_empty(self):
        assert _linear_slope([]) == 0.0


class TestMaxConsecutive:
    def test_basic(self):
        assert _max_consecutive([True, True, True, False, True]) == 3

    def test_all_true(self):
        assert _max_consecutive([True, True, True]) == 3

    def test_all_false(self):
        assert _max_consecutive([False, False, False]) == 0

    def test_empty(self):
        assert _max_consecutive([]) == 0

    def test_single(self):
        assert _max_consecutive([True]) == 1


# ── Feature extraction ────────────────────────────────────────────────

class TestExtractTemporalFeatures:
    def test_output_length(self):
        """Feature vector should match TEMPORAL_FEATURE_NAMES length."""
        features = extract_temporal_features(
            daily_wind_means=[10, 12, 14, 16, 18, 20],
            daily_wind_maxes=[15, 18, 20, 22, 25, 28],
            daily_wave_means=[0.5, 0.8, 1.0, 1.2, 1.5, 1.8],
            daily_wave_maxes=[0.8, 1.2, 1.5, 1.8, 2.0, 2.5],
            hourly_winds=[15.0] * 144,
            hourly_waves=[1.0] * 144,
            route_id="PIR-MYK",
            vessel_type="CONVENTIONAL",
            event_date="2025-07-15",
        )
        assert len(features) == len(TEMPORAL_FEATURE_NAMES)

    def test_padding_short_input(self):
        """Should pad if fewer than 6 days provided."""
        features = extract_temporal_features(
            daily_wind_means=[20, 22],
            daily_wind_maxes=[25, 28],
            daily_wave_means=[1.5, 1.8],
            daily_wave_maxes=[2.0, 2.5],
            hourly_winds=[20.0] * 48,
            hourly_waves=[1.5] * 48,
        )
        assert len(features) == len(TEMPORAL_FEATURE_NAMES)

    def test_increasing_wind_positive_slope(self):
        """Increasing daily wind means should produce positive slope."""
        features = extract_temporal_features(
            daily_wind_means=[5, 10, 15, 20, 25, 30],
            daily_wind_maxes=[8, 15, 20, 25, 30, 38],
            daily_wave_means=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            daily_wave_maxes=[0.8, 1.5, 2.0, 2.5, 3.0, 4.0],
            hourly_winds=[20.0] * 144,
            hourly_waves=[2.0] * 144,
        )
        # wind_trend_slope is at index 24
        wind_slope_idx = TEMPORAL_FEATURE_NAMES.index("wind_trend_slope")
        assert features[wind_slope_idx] > 0

    def test_storm_hours_counted(self):
        """Should count hours above Bf thresholds."""
        # 22 kn = Bf 6, create 144 hours all at 25 kn (Bf 6)
        features = extract_temporal_features(
            daily_wind_means=[25] * 6,
            daily_wind_maxes=[30] * 6,
            daily_wave_means=[2.0] * 6,
            daily_wave_maxes=[3.0] * 6,
            hourly_winds=[25.0] * 144,
            hourly_waves=[2.5] * 144,
        )
        hours_bf6_idx = TEMPORAL_FEATURE_NAMES.index("hours_above_bf6")
        assert features[hours_bf6_idx] == 144.0  # all hours above Bf 6

    def test_high_speed_flag(self):
        features_hs = extract_temporal_features(
            daily_wind_means=[10] * 6,
            daily_wind_maxes=[15] * 6,
            daily_wave_means=[1.0] * 6,
            daily_wave_maxes=[1.5] * 6,
            hourly_winds=[10.0] * 144,
            hourly_waves=[1.0] * 144,
            vessel_type="HIGH_SPEED",
        )
        features_cv = extract_temporal_features(
            daily_wind_means=[10] * 6,
            daily_wind_maxes=[15] * 6,
            daily_wave_means=[1.0] * 6,
            daily_wave_maxes=[1.5] * 6,
            hourly_winds=[10.0] * 144,
            hourly_waves=[1.0] * 144,
            vessel_type="CONVENTIONAL",
        )
        hs_idx = TEMPORAL_FEATURE_NAMES.index("vessel_is_highspeed")
        assert features_hs[hs_idx] == 1.0
        assert features_cv[hs_idx] == 0.0


# ── Feature masking for lead times ────────────────────────────────────

class TestMaskFeatures:
    def _make_features(self):
        """Create a dummy feature vector."""
        return extract_temporal_features(
            daily_wind_means=[10, 15, 20, 25, 30, 35],
            daily_wind_maxes=[15, 20, 25, 30, 35, 40],
            daily_wave_means=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            daily_wave_maxes=[0.8, 1.5, 2.0, 2.5, 3.0, 4.0],
            hourly_winds=[20.0] * 144,
            hourly_waves=[1.5] * 144,
        )

    def test_full_visibility(self):
        """6 days visible = no masking."""
        features = self._make_features()
        masked = _mask_features_for_lead_time(features, 6)
        assert masked == features

    def test_3_days_visible(self):
        """At D-3, only first 3 days are real, rest filled with D-3 values."""
        features = self._make_features()
        masked = _mask_features_for_lead_time(features, 3)
        # wind_mean D-2, D-1, D-0 (indices 3,4,5) should equal D-3 (index 2)
        assert masked[3] == masked[2]
        assert masked[4] == masked[2]
        assert masked[5] == masked[2]

    def test_1_day_visible(self):
        """At D-5, only first day is visible."""
        features = self._make_features()
        masked = _mask_features_for_lead_time(features, 1)
        # All per-day features should be D-5's value
        for i in range(1, 6):
            assert masked[i] == masked[0]

    def test_preserves_length(self):
        features = self._make_features()
        for dv in [1, 2, 3, 4, 5, 6]:
            masked = _mask_features_for_lead_time(features, dv)
            assert len(masked) == len(features)


# ── Synthetic dataset building ────────────────────────────────────────

class TestBuildTemporalDataset:
    def test_synthetic_build(self, tmp_path):
        """Should build a synthetic temporal dataset."""
        # First need ground truth data
        from src.data_collection.ground_truth import GroundTruthCollector
        collector = GroundTruthCollector(data_dir=tmp_path / "gt")
        collector.add_record("2025-01-15", "PIR-MYK", "07:00",
                             vessel_type="CONVENTIONAL", status="SAILED",
                             wind_speed_kn=15.0, wave_height_m=1.0)
        collector.add_record("2025-01-16", "PIR-MYK", "07:00",
                             vessel_type="CONVENTIONAL", status="CANCELLED",
                             wind_speed_kn=38.0, wave_height_m=4.5)

        stats = build_temporal_dataset_synthetic(
            ground_truth_path=collector.records_file,
            output_dir=tmp_path / "temporal",
        )
        assert stats["total_records"] == 2
        assert stats["cancelled"] == 1


# ── Temporal predictor ────────────────────────────────────────────────

class TestTemporalPredictor:
    def _make_training_data(self, n=200):
        """Generate quick training data."""
        import random
        rng = random.Random(42)
        X = []
        y = []
        for _ in range(n):
            # Cancellations have higher D-0 wind
            is_cancel = rng.random() < 0.3
            if is_cancel:
                base_wind = rng.uniform(25, 45)
                base_wave = rng.uniform(2.5, 5.0)
            else:
                base_wind = rng.uniform(5, 20)
                base_wave = rng.uniform(0.3, 2.0)

            wind_means = [base_wind * (0.4 + 0.6 * i / 5) + rng.gauss(0, 2)
                          for i in range(6)]
            wind_maxes = [w * 1.3 for w in wind_means]
            wave_means = [base_wave * (0.4 + 0.6 * i / 5) + rng.gauss(0, 0.2)
                          for i in range(6)]
            wave_maxes = [w * 1.4 for w in wave_means]
            hourly_winds = [base_wind + rng.gauss(0, 3) for _ in range(144)]
            hourly_waves = [base_wave + rng.gauss(0, 0.3) for _ in range(144)]

            features = extract_temporal_features(
                daily_wind_means=wind_means,
                daily_wind_maxes=wind_maxes,
                daily_wave_means=wave_means,
                daily_wave_maxes=wave_maxes,
                hourly_winds=hourly_winds,
                hourly_waves=hourly_waves,
                vessel_type=rng.choice(["CONVENTIONAL", "HIGH_SPEED"]),
            )
            X.append(features)
            y.append(1 if is_cancel else 0)
        return X, y

    def test_train_and_predict(self):
        """Should train and produce predictions."""
        X, y = self._make_training_data()
        predictor = TemporalPredictor()
        results = predictor.train(X, y)
        assert len(results) == len(LEAD_TIMES)
        for _, info in results.items():
            assert info["accuracy"] > 0.5  # better than coin flip

    def test_predict_at_different_leads(self):
        """Should produce predictions at different lead times."""
        X, y = self._make_training_data()
        predictor = TemporalPredictor()
        predictor.train(X, y)

        # Use a high-wind sample for prediction
        sample = X[0]
        for days_before in [0, 1, 3, 5]:
            result = predictor.predict(sample, days_before_departure=days_before)
            assert "cancel_probability" in result
            assert 0 <= result["cancel_probability"] <= 1
            assert result["days_before_departure"] == days_before

    def test_rolling_forecast(self):
        """Should return predictions for all lead times."""
        X, y = self._make_training_data()
        predictor = TemporalPredictor()
        predictor.train(X, y)

        rolling = predictor.predict_rolling(X[0])
        assert len(rolling) == len(LEAD_TIMES)
        for entry in rolling:
            assert "lead_time" in entry
            assert "cancel_probability" in entry
            assert 0 <= entry["cancel_probability"] <= 1

    def test_d0_more_accurate_than_d5(self):
        """D-0 should be more accurate than D-5 (more data available)."""
        X, y = self._make_training_data(n=500)
        predictor = TemporalPredictor()
        results = predictor.train(X, y)
        # D-0 should generally be more accurate
        assert results["d0"]["accuracy"] >= results["d5"]["accuracy"] - 0.05

    def test_save_and_load(self, tmp_path):
        X, y = self._make_training_data()
        predictor = TemporalPredictor()
        predictor.train(X, y)
        predictor.save(tmp_path)

        loaded = TemporalPredictor()
        loaded.load(tmp_path)
        assert loaded._trained

        # Should produce same predictions
        result1 = predictor.predict(X[0])
        result2 = loaded.predict(X[0])
        assert result1["cancel_probability"] == result2["cancel_probability"]
