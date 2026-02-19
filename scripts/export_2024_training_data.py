"""
Export 2024 simulation data as training CSV.

Takes the 2024 ground truth + estimated temporal features and writes
them in the same format as real_temporal_dataset.csv, so they can be
combined for multi-year training.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.simulate_2024 import (
    build_2024_ground_truth,
    build_temporal_features_2024,
)
from src.data_collection.temporal_dataset import TEMPORAL_FEATURE_NAMES

DATA_DIR = PROJECT_ROOT / "data" / "temporal"


def export_2024_training_csv() -> Path:
    """Build and export 2024 data in temporal_dataset CSV format."""
    print("Building 2024 ground truth...")
    records = build_2024_ground_truth()

    # Exclude strike events — they aren't weather-related
    weather_records = [r for r in records if r.get("reason") != "strike"]
    print(f"  Weather records: {len(weather_records)} "
          f"(excluded {len(records) - len(weather_records)} strike records)")

    print("Building temporal features...")
    augmented = build_temporal_features_2024(weather_records)

    output_path = DATA_DIR / "temporal_2024_training.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = TEMPORAL_FEATURE_NAMES + ["label"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        exported = 0
        for r in augmented:
            features = r.get("temporal_features")
            if features is None or len(features) != len(TEMPORAL_FEATURE_NAMES):
                continue

            row = dict(zip(TEMPORAL_FEATURE_NAMES, features))
            row["label"] = 1 if r["status"] == "CANCELLED" else 0
            writer.writerow(row)
            exported += 1

    cancelled = sum(1 for r in augmented
                    if r.get("temporal_features") and r["status"] == "CANCELLED")
    print(f"\nExported {exported} records ({cancelled} cancelled) to {output_path}")
    return output_path


if __name__ == "__main__":
    export_2024_training_csv()
