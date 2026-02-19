"""
Train temporal model on combined 2024 + 2025 real data.

This gives the model more diverse weather patterns to learn from,
improving generalization to unseen years (like 2023).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.temporal_dataset import TEMPORAL_FEATURE_NAMES
from src.models.temporal_predictor import TemporalPredictor

DATA_DIR = PROJECT_ROOT / "data" / "temporal"


def load_csv(path: Path) -> tuple[list[list[float]], list[int]]:
    """Load a temporal CSV into (X, y)."""
    X, y = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                features = [float(row[name]) for name in TEMPORAL_FEATURE_NAMES]
                label = int(float(row["label"]))
                X.append(features)
                y.append(label)
            except (KeyError, ValueError) as e:
                continue
    return X, y


def train_combined():
    """Load both datasets, combine, and train."""
    # 2025 real data
    real_2025_path = DATA_DIR / "real_temporal_dataset.csv"
    print(f"Loading 2025 real data from {real_2025_path.name}...")
    X_2025, y_2025 = load_csv(real_2025_path)
    print(f"  {len(X_2025)} records, {sum(y_2025)} cancelled "
          f"({100*sum(y_2025)/len(y_2025):.1f}%)")

    # 2024 training data
    train_2024_path = DATA_DIR / "temporal_2024_training.csv"
    print(f"Loading 2024 data from {train_2024_path.name}...")
    X_2024, y_2024 = load_csv(train_2024_path)
    print(f"  {len(X_2024)} records, {sum(y_2024)} cancelled "
          f"({100*sum(y_2024)/len(y_2024):.1f}%)")

    # Combine
    X_combined = X_2025 + X_2024
    y_combined = y_2025 + y_2024
    print(f"\nCombined: {len(X_combined)} records, {sum(y_combined)} cancelled "
          f"({100*sum(y_combined)/len(y_combined):.1f}%)")

    # Train
    print("\nTraining temporal models on combined 2024+2025 data...")
    predictor = TemporalPredictor()
    results = predictor.train(X_combined, y_combined)
    predictor.save()
    print()

    for lead, metrics in results.items():
        print(f"  {lead} ({metrics['label']}): accuracy={metrics['accuracy']:.1%}")

    print(f"\nModels saved. Ready for 2023 simulation.")


if __name__ == "__main__":
    train_combined()
