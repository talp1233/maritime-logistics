#!/usr/bin/env python3
"""
Full pipeline runner — generates all data, trains all models, and produces
a comprehensive accuracy report.

Usage:
    python run_full_pipeline.py                   # Full run (synthetic data)
    python run_full_pipeline.py --real             # Real data from Open-Meteo (needs internet)
    python run_full_pipeline.py --report-only      # Skip training, just show report

This script runs every stage of the pipeline end-to-end:
  1. Generate ground truth data (90 days, ~3600 records)
  2. Build temporal dataset (5-day weather windows, 41 features)
  3. Train snapshot ML models (Logistic Regression + Gradient Boosting)
  4. Train temporal models (D-5, D-3, D-1, D-0 lead times)
  5. Run all backtests (risk scorer, temporal predictor)
  6. Calibrate thresholds from ground truth
  7. Produce summary report with key findings
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.constants import ROUTES, PORTS, VESSEL_TYPES, SAILING_BAN_THRESHOLDS
from src.utils.logger import get_logger

logger = get_logger("pipeline")


def step_header(step_num: int, title: str) -> None:
    print()
    print(f"{'=' * 65}")
    print(f"  STEP {step_num}: {title}")
    print(f"{'=' * 65}")
    print()


def run_pipeline(real_data: bool = False, report_only: bool = False) -> dict:
    """Run the complete pipeline and collect all metrics."""
    results = {}
    t0 = time.time()

    # ── STEP 1: Generate ground truth ─────────────────────────
    if not report_only:
        step_header(1, "GENERATE GROUND TRUTH DATA")

        from src.data_collection.ground_truth import GroundTruthCollector

        collector = GroundTruthCollector()
        count = collector.generate_sample_data(n_days=90)
        stats = collector.get_stats()

        print(f"  Generated {count} records over 90 days")
        print(f"  Cancelled: {stats['cancelled']} ({stats['cancel_rate']:.1%})")
        print(f"  Sailed:    {stats['sailed']}")
        print(f"  Routes:    {stats['routes_covered']}")

        results["ground_truth"] = {
            "total": count,
            "cancelled": stats["cancelled"],
            "sailed": stats["sailed"],
            "cancel_rate": stats["cancel_rate"],
            "routes": stats["routes_covered"],
        }

    # ── STEP 2: Build temporal dataset ────────────────────────
    if not report_only:
        step_header(2, "BUILD TEMPORAL DATASET")

        if real_data:
            from src.data_collection.temporal_dataset import (
                build_temporal_dataset_from_archive,
            )
            print("  Source: Open-Meteo Archive API (real weather)")
            temporal_stats = build_temporal_dataset_from_archive()
        else:
            from src.data_collection.temporal_dataset import (
                build_temporal_dataset_synthetic,
            )
            print("  Source: Synthetic (meteorological heuristics)")
            temporal_stats = build_temporal_dataset_synthetic()

        if "error" in temporal_stats:
            print(f"  ERROR: {temporal_stats['error']}")
            return results

        print(f"  Records:     {temporal_stats['total_records']}")
        print(f"  Cancelled:   {temporal_stats['cancelled']}")
        print(f"  Cancel rate: {temporal_stats['cancel_rate']:.1%}")
        print(f"  Features:    41 temporal features per record")

        results["temporal_dataset"] = temporal_stats

    # ── STEP 3: Train snapshot ML models ──────────────────────
    if not report_only:
        step_header(3, "TRAIN SNAPSHOT ML MODELS")

        from src.models.ml_predictor import MLPredictor

        predictor = MLPredictor()
        ml_metrics = predictor.train_from_ground_truth()

        if ml_metrics:
            print(f"  Logistic Regression: {ml_metrics['logistic_regression_accuracy']:.1%}")
            print(f"  Gradient Boosting:   {ml_metrics['gradient_boosting_accuracy']:.1%}")
            print(f"  Train/Test split:    {ml_metrics['train_size']}/{ml_metrics['test_size']}")
            predictor.save()
            print("  Saved to src/models/saved/")
            results["snapshot_ml"] = ml_metrics

    # ── STEP 4: Train temporal models ─────────────────────────
    if not report_only:
        step_header(4, "TRAIN TEMPORAL MODELS (D-5, D-3, D-1, D-0)")

        from src.models.temporal_predictor import TemporalPredictor
        from src.data_collection.temporal_dataset import load_temporal_dataset

        X, y = load_temporal_dataset()
        tp = TemporalPredictor()
        temporal_results = tp.train(X, y)

        if temporal_results:
            for lead, info in temporal_results.items():
                print(f"  {info['label']:<20} {info['accuracy']:>9.1%}")
            tp.save()
            print("  Saved to src/models/saved/temporal_models.pkl")
            results["temporal_training"] = temporal_results

    # ── STEP 5: Run temporal backtest ─────────────────────────
    step_header(5, "TEMPORAL PREDICTION BACKTEST")

    from src.models.temporal_predictor import backtest_temporal, print_temporal_report

    temporal_report = backtest_temporal()
    print_temporal_report(temporal_report)
    results["temporal_backtest"] = temporal_report

    # ── STEP 6: Run risk scorer backtest ──────────────────────
    step_header(6, "RISK SCORER BACKTEST")

    from src.validation.backtester import run_backtest, print_backtest_report

    risk_report = run_backtest()
    print_backtest_report(risk_report)
    results["risk_scorer_backtest"] = risk_report

    # ── STEP 7: Calibrate thresholds ─────────────────────────
    step_header(7, "THRESHOLD CALIBRATION")

    from src.models.ml_predictor import calibrate_thresholds

    cal_result = calibrate_thresholds()
    if "error" not in cal_result:
        print(f"  Records analyzed: {cal_result['total_records']}")
        print()
        for category, levels in cal_result["cancel_rates_by_beaufort"].items():
            threshold = cal_result["current_thresholds"].get(category)
            print(f"  --- {category} (threshold: Bf {threshold}) ---")
            for lv in levels:
                marker = " <<" if lv["beaufort"] == threshold else ""
                bar = "#" * int(lv["cancel_rate"] * 30)
                print(f"    Bf {lv['beaufort']:>2}: {lv['cancel_rate']:>5.1%} "
                      f"({lv['cancelled']:>3}/{lv['total']:>3}) {bar}{marker}")
            print()
    results["calibration"] = cal_result

    # ── SUMMARY ──────────────────────────────────────────────
    elapsed = time.time() - t0

    print()
    print("=" * 65)
    print("  PIPELINE SUMMARY")
    print("=" * 65)
    print()
    print(f"  Data source:     {'Real (Open-Meteo)' if real_data else 'Synthetic'}")
    print(f"  Total time:      {elapsed:.1f}s")
    print()

    if "temporal_backtest" in results:
        tb = results["temporal_backtest"]
        if "lead_time_metrics" in tb:
            print("  TEMPORAL MODEL ACCURACY:")
            for lead in ["d5", "d3", "d1", "d0"]:
                m = tb["lead_time_metrics"].get(lead, {})
                if m:
                    print(f"    {m['label']:<20} {m['accuracy']:>7.1%}  "
                          f"(Brier: {m['brier_score']:.4f}, Skill: {m['skill_score']:+.3f})")
            print()

    if "risk_scorer_backtest" in results:
        rb = results["risk_scorer_backtest"]
        if "metrics" in rb:
            m = rb["metrics"]
            print("  RISK SCORER vs BEAUFORT BASELINE:")
            print(f"    Risk scorer accuracy:   {m['model_accuracy']:.1%}")
            print(f"    Beaufort rule accuracy: {m['baseline_accuracy']:.1%}")
            print(f"    Skill vs Beaufort:      {m['skill_vs_beaufort']:+.3f}")
            print()

    if "snapshot_ml" in results:
        ml = results["snapshot_ml"]
        print("  SNAPSHOT ML MODELS:")
        print(f"    Logistic Regression:   {ml['logistic_regression_accuracy']:.1%}")
        print(f"    Gradient Boosting:     {ml['gradient_boosting_accuracy']:.1%}")
        print()

    # ── KEY FINDINGS (data quality analysis) ──────────────────
    print("  KEY FINDINGS:")
    print("  " + "-" * 55)

    if "temporal_backtest" in results:
        tb = results["temporal_backtest"]
        ltm = tb.get("lead_time_metrics", {})
        d0_acc = ltm.get("d0", {}).get("accuracy", 0)
        d5_acc = ltm.get("d5", {}).get("accuracy", 0)

        if d5_acc > 0.95 and not real_data:
            print("  [!] D-5 accuracy >95% on synthetic data is unrealistic.")
            print("      Real-world D-5 accuracy would be ~55-65%.")
            print("      The model learned the synthetic S-curve pattern,")
            print("      not real storm buildup dynamics.")
            print()

    if "risk_scorer_backtest" in results:
        rb = results["risk_scorer_backtest"]
        m = rb.get("metrics", {})
        if m.get("skill_vs_beaufort", 0) < 0:
            print("  [!] Risk scorer underperforms the Beaufort rule.")
            print("      This is expected with synthetic data: ground truth")
            print("      was generated FROM Beaufort thresholds, so the")
            print("      simple threshold rule captures it perfectly.")
            print("      With real data, the risk scorer should outperform")
            print("      because real cancellations depend on more factors.")
            print()

    if not real_data:
        print("  [i] TO GET REAL ACCURACY NUMBERS:")
        print("      1. Run with internet access:")
        print("         python run_full_pipeline.py --real")
        print("      2. Or fetch historical weather first:")
        print("         python main.py --fetch-historical --historical-days 180")
        print("         python main.py --build-temporal-live")
        print("         python main.py --train-temporal")
        print("         python main.py --backtest-temporal")
        print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Full maritime prediction pipeline runner",
    )
    parser.add_argument(
        "--real", action="store_true",
        help="Use real Open-Meteo data (requires internet)",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Skip training, just run backtests and show report",
    )

    args = parser.parse_args()
    run_pipeline(real_data=args.real, report_only=args.report_only)


if __name__ == "__main__":
    main()
