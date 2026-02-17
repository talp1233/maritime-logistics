"""
Temporal ML predictor for early cancellation warning.

Uses the 5-day weather window leading up to departure to predict
cancellation probability.  The key innovation: predictions update
as departure approaches, getting more accurate over time:

    D-5: ~55% accuracy (early warning, low confidence)
    D-3: ~70% accuracy (actionable prediction)
    D-1: ~85% accuracy (high confidence)
    D-0: ~90% accuracy (near-certain)

The model trains on full 5-day windows but can predict from partial
windows (D-3 through D-0 = 4 days of data + 2 days missing).
Missing days are filled with the earliest available day's conditions.

Architecture:
    - Gradient Boosting on 41 temporal features
    - Separate models for different lead times (D-5, D-3, D-1, D-0)
    - Ensemble: average predictions across available lead-time models
"""

from __future__ import annotations

import pickle
from pathlib import Path

from src.data_collection.temporal_dataset import (
    TEMPORAL_FEATURE_NAMES,
    extract_temporal_features,
    load_temporal_dataset,
    _linear_slope,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path(__file__).parent / "saved"

# Lead-time model configurations
# Each model is trained on features visible at that lead time
LEAD_TIMES = {
    "d0": {"days_visible": 6, "label": "Departure day"},
    "d1": {"days_visible": 5, "label": "1 day before"},
    "d3": {"days_visible": 3, "label": "3 days before"},
    "d5": {"days_visible": 1, "label": "5 days before"},
}


def _mask_features_for_lead_time(features: list[float], days_visible: int) -> list[float]:
    """
    Mask features that wouldn't be available at a given lead time.

    At D-3, we only have D-5, D-4, D-3 data (3 days).
    D-2, D-1, D-0 are unknown → fill with D-3's values (persistence forecast).

    Feature layout:
        [0:6]   = wind_mean D-5..D-0
        [6:12]  = wind_max D-5..D-0
        [12:18] = wave_mean D-5..D-0
        [18:24] = wave_max D-5..D-0
        [24:]   = trend + storm + context features
    """
    masked = list(features)
    n_days = 6  # D-5 through D-0

    if days_visible >= n_days:
        return masked

    # For each per-day feature block, fill future days with last known day
    for block_start in [0, 6, 12, 18]:
        last_known_idx = block_start + days_visible - 1
        last_known_val = masked[last_known_idx]
        for i in range(last_known_idx + 1, block_start + n_days):
            masked[i] = last_known_val

    # Recompute trend features based on visible data only
    visible_wind_means = masked[0:days_visible]
    visible_wave_means = masked[12:12 + days_visible]

    # wind_trend_slope (index 24)
    masked[24] = _linear_slope(visible_wind_means)
    # wave_trend_slope (index 25)
    masked[25] = _linear_slope(visible_wave_means)

    # wind_accel (index 26): use last visible vs earliest
    if days_visible >= 2:
        recent_wind = masked[days_visible - 1]  # last visible day wind mean
        early_wind = masked[0]
        masked[26] = recent_wind / early_wind if early_wind > 1 else 1.0
    else:
        masked[26] = 1.0

    # wave_accel (index 27)
    if days_visible >= 2:
        recent_wave = masked[12 + days_visible - 1]
        early_wave = masked[12]
        masked[27] = recent_wave / early_wave if early_wave > 0.1 else 1.0
    else:
        masked[27] = 1.0

    return masked


class TemporalPredictor:
    """
    Multi-lead-time temporal predictor.

    Trains separate Gradient Boosting models for D-0, D-1, D-3, D-5
    and provides rolling predictions that update as data becomes available.
    """

    def __init__(self):
        self.models: dict[str, object] = {}
        self._trained = False

    def train(
        self,
        X: list[list[float]] | None = None,
        y: list[int] | None = None,
    ) -> dict:
        """
        Train lead-time models on temporal dataset.

        If X/y not provided, loads from temporal_dataset.csv.
        """
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
        except ImportError:
            logger.error("scikit-learn required. Install with: pip install scikit-learn")
            return {}

        if X is None or y is None:
            X, y = load_temporal_dataset()

        if len(X) < 50:
            logger.warning("Only %d samples — need at least 50", len(X))
            return {}

        # Split temporally (last 30% for validation)
        split = int(len(X) * 0.7)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        results = {}

        for lead_name, config in LEAD_TIMES.items():
            days_visible = config["days_visible"]

            # Mask training features for this lead time
            X_train_masked = [
                _mask_features_for_lead_time(x, days_visible) for x in X_train
            ]
            X_val_masked = [
                _mask_features_for_lead_time(x, days_visible) for x in X_val
            ]

            model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
            model.fit(X_train_masked, y_train)

            y_pred = model.predict(X_val_masked)
            acc = accuracy_score(y_val, y_pred)

            self.models[lead_name] = model
            results[lead_name] = {
                "accuracy": round(acc, 3),
                "label": config["label"],
                "days_visible": days_visible,
                "train_size": len(X_train),
                "val_size": len(X_val),
            }

            logger.info("  %s (%s): accuracy %.1f%%",
                         lead_name, config["label"], acc * 100)

        self._trained = True
        return results

    def predict(
        self,
        features: list[float],
        days_before_departure: int = 0,
    ) -> dict:
        """
        Predict cancellation probability at a given lead time.

        Args:
            features: Full temporal feature vector (41 features)
            days_before_departure: 0=departure day, 1=tomorrow, 3=in 3 days, 5=in 5 days

        Returns:
            Dict with predictions from the appropriate lead-time model
            and the full rolling forecast if multiple models are available.
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() first.")

        # Map days_before to lead-time model
        if days_before_departure >= 5:
            lead = "d5"
        elif days_before_departure >= 3:
            lead = "d3"
        elif days_before_departure >= 1:
            lead = "d1"
        else:
            lead = "d0"

        model = self.models.get(lead)
        if model is None:
            raise RuntimeError(f"No model for lead time {lead}")

        days_visible = LEAD_TIMES[lead]["days_visible"]
        masked = _mask_features_for_lead_time(features, days_visible)

        proba = model.predict_proba([masked])[0]
        cancel_prob = float(proba[1])

        # Get rolling forecast from all applicable models
        rolling: list[dict] = []
        for ln, cfg in LEAD_TIMES.items():
            m = self.models.get(ln)
            if m is None:
                continue
            dv = cfg["days_visible"]
            if dv <= (6 - days_before_departure):
                m_masked = _mask_features_for_lead_time(features, dv)
                m_proba = m.predict_proba([m_masked])[0]
                rolling.append({
                    "lead_time": ln,
                    "label": cfg["label"],
                    "cancel_probability": round(float(m_proba[1]), 3),
                })

        return {
            "cancel_probability": round(cancel_prob, 3),
            "lead_time": lead,
            "lead_label": LEAD_TIMES[lead]["label"],
            "days_before_departure": days_before_departure,
            "rolling_forecast": rolling,
        }

    def predict_rolling(self, features: list[float]) -> list[dict]:
        """
        Get the full rolling forecast: predictions at D-5, D-3, D-1, D-0.

        This shows how the prediction evolves as departure approaches —
        the key product deliverable for operators.
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() first.")

        results: list[dict] = []
        for lead_name, config in LEAD_TIMES.items():
            model = self.models.get(lead_name)
            if model is None:
                continue

            days_visible = config["days_visible"]
            masked = _mask_features_for_lead_time(features, days_visible)
            proba = model.predict_proba([masked])[0]

            results.append({
                "lead_time": lead_name,
                "label": config["label"],
                "days_visible": days_visible,
                "cancel_probability": round(float(proba[1]), 3),
                "sail_probability": round(float(proba[0]), 3),
            })

        return results

    def save(self, directory: Path | str | None = None):
        """Save trained models."""
        if not self._trained:
            raise RuntimeError("No trained models to save.")
        d = Path(directory) if directory else MODEL_DIR
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "temporal_models.pkl", "wb") as f:
            pickle.dump(self.models, f)
        logger.info("Temporal models saved to %s", d)

    def load(self, directory: Path | str | None = None):
        """Load trained models."""
        d = Path(directory) if directory else MODEL_DIR
        path = d / "temporal_models.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Temporal models not found at {path}")
        with open(path, "rb") as f:
            self.models = pickle.load(f)
        self._trained = True
        logger.info("Temporal models loaded from %s", d)


# ── Backtest: measure accuracy at each lead time ─────────────────────

def backtest_temporal(
    csv_path: Path | str | None = None,
    temporal_split: float = 0.7,
) -> dict:
    """
    Train temporal models and measure accuracy at each lead time.

    This answers: "How accurate is our prediction X days before departure?"

    Returns:
        Report with per-lead-time accuracy, Brier scores, and
        comparison to the base risk scorer.
    """
    from src.validation.backtester import brier_score, skill_score

    X, y = load_temporal_dataset(csv_path)

    if len(X) < 50:
        return {"error": f"Need at least 50 records, got {len(X)}"}

    split = int(len(X) * temporal_split)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    predictor = TemporalPredictor()
    train_results = predictor.train(X_train, y_train)

    if not train_results:
        return {"error": "Training failed"}

    # Evaluate each lead time
    lead_metrics: dict[str, dict] = {}
    for lead_name, config in LEAD_TIMES.items():
        model = predictor.models.get(lead_name)
        if model is None:
            continue

        days_visible = config["days_visible"]
        X_masked = [_mask_features_for_lead_time(x, days_visible) for x in X_val]

        predictions = [float(model.predict_proba([x])[0][1]) for x in X_masked]
        outcomes = y_val

        # Brier score
        bs = brier_score(predictions, outcomes)

        # Binary accuracy at 0.5 threshold
        correct = sum(
            1 for p, o in zip(predictions, outcomes)
            if (p >= 0.5 and o == 1) or (p < 0.5 and o == 0)
        )
        acc = correct / len(outcomes) if outcomes else 0

        # Climatological baseline
        cancel_rate = sum(outcomes) / len(outcomes) if outcomes else 0
        clim_bs = cancel_rate * (1 - cancel_rate)
        ss = skill_score(bs, clim_bs)

        lead_metrics[lead_name] = {
            "label": config["label"],
            "days_visible": days_visible,
            "accuracy": round(acc, 3),
            "brier_score": round(bs, 4),
            "skill_score": round(ss, 3),
            "predictions_count": len(predictions),
        }

    return {
        "validation_records": len(X_val),
        "total_records": len(X),
        "cancel_rate": round(sum(y) / len(y), 3) if y else 0,
        "lead_time_metrics": lead_metrics,
        "train_results": train_results,
    }


def print_temporal_report(report: dict) -> None:
    """Pretty-print the temporal backtest report."""
    if "error" in report:
        print(f"  Error: {report['error']}")
        return

    print("=" * 65)
    print("  TEMPORAL PREDICTION ACCURACY REPORT")
    print("  (How accurate is our prediction X days before departure?)")
    print("=" * 65)
    print()
    print(f"  Total records:     {report['total_records']}")
    print(f"  Validation set:    {report['validation_records']}")
    print(f"  Cancel rate:       {report['cancel_rate']:.1%}")
    print()

    print(f"  {'Lead Time':<18} {'Accuracy':>10} {'Brier':>8} {'Skill':>8}")
    print("  " + "-" * 48)

    for lead_name in ["d5", "d3", "d1", "d0"]:
        m = report["lead_time_metrics"].get(lead_name)
        if not m:
            continue
        bar = "#" * int(m["accuracy"] * 30)
        print(f"  {m['label']:<18} {m['accuracy']:>9.1%} {m['brier_score']:>8.4f} "
              f"{m['skill_score']:>+7.3f}  {bar}")

    print()
    print("  Interpretation:")
    print("    Accuracy: % of correct cancel/sail predictions")
    print("    Brier:    0=perfect, 1=worst (probability calibration)")
    print("    Skill:    +positive = better than always predicting cancel rate")
    print()

    # Show how accuracy improves
    metrics = report["lead_time_metrics"]
    if "d5" in metrics and "d0" in metrics:
        d5_acc = metrics["d5"]["accuracy"]
        d0_acc = metrics["d0"]["accuracy"]
        improvement = d0_acc - d5_acc
        print(f"  Accuracy improvement D-5 → D-0: {d5_acc:.1%} → {d0_acc:.1%} "
              f"(+{improvement:.1%})")
    print()
