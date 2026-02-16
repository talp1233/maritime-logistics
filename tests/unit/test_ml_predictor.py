"""Tests for ML predictor module."""

from src.models.ml_predictor import extract_features, generate_training_data, FEATURE_NAMES


class TestFeatureExtraction:
    def test_feature_vector_length(self):
        features = extract_features(wind_speed_knots=20, wave_height_m=2.0)
        assert len(features) == len(FEATURE_NAMES)

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == 13

    def test_high_speed_flag(self):
        hs = extract_features(wind_speed_knots=20, wave_height_m=2.0, vessel_type="HIGH_SPEED")
        conv = extract_features(wind_speed_knots=20, wave_height_m=2.0, vessel_type="CONVENTIONAL")
        # is_high_speed is at index 9
        assert hs[9] == 1.0
        assert conv[9] == 0.0

    def test_bf_ratio_increases_with_wind(self):
        low = extract_features(wind_speed_knots=10, wave_height_m=1.0)
        high = extract_features(wind_speed_knots=40, wave_height_m=1.0)
        # bf_ratio is at index 8
        assert high[8] > low[8]


class TestTrainingData:
    def test_generate_correct_count(self):
        X, y = generate_training_data(n_samples=100)
        assert len(X) == 100
        assert len(y) == 100

    def test_labels_are_binary(self):
        _, y = generate_training_data(n_samples=200)
        assert all(label in (0, 1) for label in y)

    def test_has_both_classes(self):
        _, y = generate_training_data(n_samples=1000)
        assert 0 in y
        assert 1 in y

    def test_feature_dimensions(self):
        X, _ = generate_training_data(n_samples=50)
        assert all(len(row) == len(FEATURE_NAMES) for row in X)


class TestMLPredictor:
    def test_train_and_predict(self):
        """Test full train-predict cycle (requires sklearn)."""
        try:
            from src.models.ml_predictor import MLPredictor
        except ImportError:
            return  # Skip if sklearn not installed

        predictor = MLPredictor()
        metrics = predictor.train(n_samples=500)

        if metrics is None:
            return  # sklearn not available

        assert metrics["logistic_regression_accuracy"] > 0.7
        assert metrics["gradient_boosting_accuracy"] > 0.7

        # Predict for a stormy scenario
        features = extract_features(
            wind_speed_knots=45, wave_height_m=5.0,
            vessel_type="HIGH_SPEED",
        )
        result = predictor.predict(features, model="gb")
        assert result["status"] == "BAN_LIKELY"
        assert result["cancel_probability"] > 0.5

        # Predict for calm conditions
        features = extract_features(
            wind_speed_knots=5, wave_height_m=0.3,
            vessel_type="CONVENTIONAL",
        )
        result = predictor.predict(features, model="gb")
        assert result["status"] == "CLEAR"
        assert result["cancel_probability"] < 0.3
