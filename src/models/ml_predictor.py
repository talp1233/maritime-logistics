"""
ML-based sailing ban predictor.

Provides two model approaches:
  1. Logistic Regression — simple, interpretable baseline
  2. Gradient Boosting (sklearn) — more accurate, handles non-linear patterns

Both can be trained on historical weather + cancellation data and used
alongside the rule-based checker for improved predictions.
"""

import json
import os
import pickle
from pathlib import Path

from src.config.constants import (
    BEAUFORT_SCALE,
    SAILING_BAN_THRESHOLDS,
    VESSEL_TYPES,
    knots_to_beaufort,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path(__file__).parent / "saved"


def extract_features(
    wind_speed_knots: float,
    wave_height_m: float,
    wave_period_s: float = 0.0,
    wind_gust_knots: float = 0.0,
    visibility_m: float = 50000.0,
    swell_height_m: float = 0.0,
    vessel_type: str = "CONVENTIONAL",
    exposed_route: bool = True,
    hour_of_day: int = 12,
    month: int = 6,
) -> list[float]:
    """
    Extract feature vector from raw conditions.

    Returns a list of numeric features for model input.
    """
    beaufort = knots_to_beaufort(wind_speed_knots)
    category = VESSEL_TYPES.get(vessel_type, "conventional")
    bf_threshold = SAILING_BAN_THRESHOLDS[category]

    return [
        wind_speed_knots,
        wind_gust_knots,
        wave_height_m,
        wave_period_s,
        swell_height_m,
        visibility_m / 1000.0,            # convert to km
        float(beaufort),
        float(bf_threshold),
        float(beaufort) / bf_threshold,    # ratio to threshold
        1.0 if category == "high_speed" else 0.0,
        1.0 if exposed_route else 0.0,
        float(hour_of_day) / 24.0,
        float(month) / 12.0,
    ]


FEATURE_NAMES = [
    "wind_speed_kn", "wind_gust_kn", "wave_height_m", "wave_period_s",
    "swell_height_m", "visibility_km", "beaufort", "bf_threshold",
    "bf_ratio", "is_high_speed", "is_exposed", "hour_norm", "month_norm",
]


def generate_training_data(n_samples: int = 5000, seed: int = 42) -> tuple:
    """
    Generate synthetic training data based on known rules + noise.

    This simulates historical data until real ground truth is available.
    Returns (X, y) where X is feature matrix and y is binary labels
    (1 = cancelled, 0 = sailed).
    """
    import random
    random.seed(seed)

    X = []
    y = []

    for _ in range(n_samples):
        wind = random.uniform(0, 60)
        gust = wind * random.uniform(1.1, 1.6)
        wave = max(0.1, 0.06 * (wind ** 1.3) + random.gauss(0, 0.5))
        period = 3.5 + wind * 0.1 + random.gauss(0, 0.5)
        swell = wave * random.uniform(0.2, 0.5)
        vis = max(500, 50000 - wind * 500 + random.gauss(0, 5000))
        vessel = random.choice(["CONVENTIONAL", "HIGH_SPEED"])
        exposed = random.random() > 0.2
        hour = random.randint(0, 23)
        month = random.randint(1, 12)

        features = extract_features(
            wind, wave, period, gust, vis, swell,
            vessel, exposed, hour, month,
        )
        X.append(features)

        # Label: simulate Coast Guard decision with some noise
        bf = knots_to_beaufort(wind)
        category = VESSEL_TYPES[vessel]
        threshold = SAILING_BAN_THRESHOLDS[category]

        # Base probability from rules
        if bf >= threshold:
            prob = 0.92
        elif bf >= threshold - 1:
            prob = 0.45
        elif wave > (2.5 if category == "high_speed" else 5.0):
            prob = 0.85
        else:
            prob = 0.05

        # Add some real-world noise
        if not exposed:
            prob *= 0.6  # sheltered routes less likely to cancel
        if month in (7, 8):
            prob *= 1.1  # Meltemi season

        cancelled = 1 if random.random() < min(prob, 1.0) else 0
        y.append(cancelled)

    return X, y


class MLPredictor:
    """ML-based sailing ban predictor with train/predict/save/load."""

    def __init__(self):
        self.logistic_model = None
        self.gb_model = None
        self._trained = False

    def train(self, X=None, y=None, n_samples: int = 5000):
        """
        Train both models. Uses synthetic data if no real data provided.

        Requires scikit-learn to be installed.
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report
        except ImportError:
            logger.error(
                "scikit-learn is required for ML models. "
                "Install with: pip install scikit-learn"
            )
            return None

        if X is None or y is None:
            logger.info("Generating %d synthetic training samples...", n_samples)
            X, y = generate_training_data(n_samples)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        # Logistic Regression
        logger.info("Training Logistic Regression...")
        self.logistic_model = LogisticRegression(max_iter=1000, random_state=42)
        self.logistic_model.fit(X_train, y_train)
        lr_acc = self.logistic_model.score(X_test, y_test)
        logger.info("  Logistic Regression accuracy: %.3f", lr_acc)

        # Gradient Boosting
        logger.info("Training Gradient Boosting...")
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42,
        )
        self.gb_model.fit(X_train, y_train)
        gb_acc = self.gb_model.score(X_test, y_test)
        logger.info("  Gradient Boosting accuracy: %.3f", gb_acc)

        self._trained = True

        # Print detailed report for gradient boosting
        y_pred = self.gb_model.predict(X_test)
        report = classification_report(
            y_test, y_pred, target_names=["Sailed", "Cancelled"],
        )
        logger.info("Gradient Boosting classification report:\n%s", report)

        return {
            "logistic_regression_accuracy": lr_acc,
            "gradient_boosting_accuracy": gb_acc,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

    def predict(self, features: list[float], model: str = "gb") -> dict:
        """
        Predict cancellation probability for a single observation.

        Args:
            features: Feature vector from extract_features()
            model: "gb" for gradient boosting, "lr" for logistic regression

        Returns:
            dict with prediction, probability, and confidence
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() first.")

        m = self.gb_model if model == "gb" else self.logistic_model
        proba = m.predict_proba([features])[0]
        predicted_class = int(m.predict([features])[0])

        cancel_prob = proba[1]
        if cancel_prob >= 0.7:
            status = "BAN_LIKELY"
        elif cancel_prob >= 0.4:
            status = "AT_RISK"
        else:
            status = "CLEAR"

        return {
            "status": status,
            "cancel_probability": round(cancel_prob, 3),
            "sail_probability": round(proba[0], 3),
            "predicted_cancelled": bool(predicted_class),
            "model_used": model,
        }

    def save(self, directory: str | Path | None = None):
        """Save trained models to disk."""
        if not self._trained:
            raise RuntimeError("No trained models to save.")

        d = Path(directory) if directory else MODEL_DIR
        d.mkdir(parents=True, exist_ok=True)

        with open(d / "logistic_model.pkl", "wb") as f:
            pickle.dump(self.logistic_model, f)
        with open(d / "gb_model.pkl", "wb") as f:
            pickle.dump(self.gb_model, f)

        logger.info("Models saved to %s", d)

    def load(self, directory: str | Path | None = None):
        """Load trained models from disk."""
        d = Path(directory) if directory else MODEL_DIR

        with open(d / "logistic_model.pkl", "rb") as f:
            self.logistic_model = pickle.load(f)
        with open(d / "gb_model.pkl", "rb") as f:
            self.gb_model = pickle.load(f)

        self._trained = True
        logger.info("Models loaded from %s", d)
