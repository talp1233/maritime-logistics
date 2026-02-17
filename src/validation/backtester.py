"""
Backtesting and validation framework for the risk scoring engine.

Validates the risk scorer against ground truth cancellation data using:
  1. Brier Score — measures calibration of probability forecasts
  2. Calibration Curve — predicted probability vs observed frequency
  3. Skill Score — improvement over naive Beaufort-threshold baseline
  4. Route-specific metrics — because PIR-HER ≠ RAF-AND
  5. Temporal split validation — train/validate on different periods

This answers the question: "Is our probabilistic risk score actually
better than the simple Beaufort rule the Coast Guard uses?"
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from src.config.constants import (
    VESSEL_TYPES,
    SAILING_BAN_THRESHOLDS,
    knots_to_beaufort,
)
from src.services.risk_scorer import compute_risk
from src.utils.logger import get_logger

logger = get_logger(__name__)

GROUND_TRUTH_PATH = Path("data/ground_truth/cancellation_records.csv")


def _load_ground_truth(csv_path: Path | None = None) -> list[dict]:
    """Load ground truth records from CSV."""
    path = csv_path or GROUND_TRUTH_PATH
    if not path.exists():
        raise FileNotFoundError(f"Ground truth not found: {path}")

    records = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append({
                    "date": row["date"],
                    "route_id": row["route_id"],
                    "vessel_type": row.get("vessel_type", "CONVENTIONAL"),
                    "wind_speed_kn": float(row.get("wind_speed_kn", 0)),
                    "wave_height_m": float(row.get("wave_height_m", 0)),
                    "wind_beaufort": int(row.get("wind_beaufort", 0)),
                    "wind_direction": float(row.get("wind_direction", 0)),
                    "status": row["status"],  # CANCELLED or SAILED
                })
            except (ValueError, KeyError) as e:
                logger.debug("Skipping malformed row: %s", e)

    return records


# ── Brier Score ───────────────────────────────────────────────────────

def brier_score(predictions: list[float], outcomes: list[int]) -> float:
    """
    Brier Score: mean squared error between predicted probability
    and binary outcome.

    BS = (1/N) * sum((p_i - o_i)²)

    Range: 0 (perfect) to 1 (worst).
    Climatological baseline: f*(1-f) where f = cancel frequency.
    """
    if not predictions:
        return 1.0
    return sum(
        (p - o) ** 2 for p, o in zip(predictions, outcomes)
    ) / len(predictions)


def skill_score(model_brier: float, baseline_brier: float) -> float:
    """
    Skill Score relative to baseline.

    SS = 1 - BS_model / BS_baseline

    Positive → model is better.  0 → same.  Negative → model is worse.
    """
    if baseline_brier <= 0:
        return 0.0
    return 1.0 - model_brier / baseline_brier


# ── Calibration Curve ─────────────────────────────────────────────────

def calibration_curve(
    predictions: list[float],
    outcomes: list[int],
    n_bins: int = 10,
) -> list[dict]:
    """
    Bin predictions into deciles and compute observed frequency per bin.

    Perfect calibration: predicted ≈ observed in every bin.

    Returns:
        List of {bin_lo, bin_hi, mean_predicted, observed_frequency, count}
    """
    bins: list[dict] = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        in_bin = [
            (p, o) for p, o in zip(predictions, outcomes)
            if lo <= p < hi or (i == n_bins - 1 and p == hi)
        ]
        if in_bin:
            mean_pred = sum(p for p, _ in in_bin) / len(in_bin)
            obs_freq = sum(o for _, o in in_bin) / len(in_bin)
        else:
            mean_pred = (lo + hi) / 2
            obs_freq = 0.0

        bins.append({
            "bin_lo": round(lo, 2),
            "bin_hi": round(hi, 2),
            "mean_predicted": round(mean_pred, 3),
            "observed_frequency": round(obs_freq, 3),
            "count": len(in_bin),
        })

    return bins


# ── Naive baseline (simple Beaufort threshold) ────────────────────────

def _naive_baseline_probability(
    wind_speed_knots: float,
    vessel_type: str,
) -> float:
    """
    Naive baseline: binary probability from Beaufort threshold.

    If Bf >= threshold → P(cancel) = 0.9
    If Bf = threshold-1 → P(cancel) = 0.4
    Otherwise → P(cancel) = 0.05
    """
    bf = knots_to_beaufort(wind_speed_knots)
    category = VESSEL_TYPES.get(vessel_type, "conventional")
    threshold = SAILING_BAN_THRESHOLDS[category]

    if bf >= threshold:
        return 0.90
    elif bf >= threshold - 1:
        return 0.40
    else:
        return 0.05


# ── Full backtest ─────────────────────────────────────────────────────

def run_backtest(
    csv_path: Path | None = None,
    temporal_split: float = 0.7,
) -> dict:
    """
    Run full validation of the risk scorer against ground truth.

    Args:
        csv_path: Path to ground truth CSV (default: standard location)
        temporal_split: Fraction of data to use for "training" period
                        (the rest is the validation set). We don't actually
                        train here — this just tests temporal generalization.

    Returns:
        Comprehensive validation report with metrics, calibration,
        and route-level breakdown.
    """
    records = _load_ground_truth(csv_path)
    if len(records) < 20:
        return {"error": f"Need at least 20 records, got {len(records)}"}

    # Sort by date for temporal split
    records.sort(key=lambda r: r["date"])
    split_idx = int(len(records) * temporal_split)
    validation_set = records[split_idx:]

    if len(validation_set) < 10:
        validation_set = records  # fallback: use all data

    # Generate predictions
    model_preds: list[float] = []
    baseline_preds: list[float] = []
    outcomes: list[int] = []
    route_results: dict[str, dict] = defaultdict(lambda: {
        "model_preds": [], "baseline_preds": [], "outcomes": [],
    })

    for r in validation_set:
        outcome = 1 if r["status"] == "CANCELLED" else 0
        outcomes.append(outcome)

        # Risk scorer prediction
        result = compute_risk(
            wind_speed_knots=r["wind_speed_kn"],
            wave_height_m=r["wave_height_m"],
            wind_direction=r["wind_direction"],
            vessel_type=r["vessel_type"],
            route_id=r["route_id"],
        )
        model_preds.append(result.cancel_probability)

        # Naive baseline prediction
        baseline_pred = _naive_baseline_probability(
            r["wind_speed_kn"], r["vessel_type"],
        )
        baseline_preds.append(baseline_pred)

        # Per-route tracking
        rr = route_results[r["route_id"]]
        rr["model_preds"].append(result.cancel_probability)
        rr["baseline_preds"].append(baseline_pred)
        rr["outcomes"].append(outcome)

    # ── Global metrics ────────────────────────────────────────────
    model_bs = brier_score(model_preds, outcomes)
    baseline_bs = brier_score(baseline_preds, outcomes)
    ss = skill_score(model_bs, baseline_bs)

    cancel_rate = sum(outcomes) / len(outcomes) if outcomes else 0
    climatological_bs = cancel_rate * (1 - cancel_rate)
    ss_vs_climate = skill_score(model_bs, climatological_bs)

    cal_curve = calibration_curve(model_preds, outcomes)

    # ── Binary accuracy (at P >= 0.5 threshold) ──────────────────
    model_correct = sum(
        1 for p, o in zip(model_preds, outcomes)
        if (p >= 0.5 and o == 1) or (p < 0.5 and o == 0)
    )
    baseline_correct = sum(
        1 for p, o in zip(baseline_preds, outcomes)
        if (p >= 0.5 and o == 1) or (p < 0.5 and o == 0)
    )
    model_accuracy = model_correct / len(outcomes) if outcomes else 0
    baseline_accuracy = baseline_correct / len(outcomes) if outcomes else 0

    # ── Route-level breakdown ─────────────────────────────────────
    per_route = {}
    for route_id, rr in route_results.items():
        r_model_bs = brier_score(rr["model_preds"], rr["outcomes"])
        r_baseline_bs = brier_score(rr["baseline_preds"], rr["outcomes"])
        r_cancel_rate = sum(rr["outcomes"]) / len(rr["outcomes"]) if rr["outcomes"] else 0

        per_route[route_id] = {
            "n_records": len(rr["outcomes"]),
            "cancel_rate": round(r_cancel_rate, 3),
            "model_brier": round(r_model_bs, 4),
            "baseline_brier": round(r_baseline_bs, 4),
            "skill_score": round(skill_score(r_model_bs, r_baseline_bs), 3),
        }

    return {
        "validation_records": len(validation_set),
        "total_records": len(records),
        "temporal_split": temporal_split,
        "cancel_rate": round(cancel_rate, 3),
        "metrics": {
            "model_brier_score": round(model_bs, 4),
            "baseline_brier_score": round(baseline_bs, 4),
            "climatological_brier": round(climatological_bs, 4),
            "skill_vs_beaufort": round(ss, 3),
            "skill_vs_climatology": round(ss_vs_climate, 3),
            "model_accuracy": round(model_accuracy, 3),
            "baseline_accuracy": round(baseline_accuracy, 3),
        },
        "calibration_curve": cal_curve,
        "per_route": per_route,
    }


def print_backtest_report(report: dict) -> None:
    """Pretty-print a backtest report to stdout."""
    if "error" in report:
        print(f"  Error: {report['error']}")
        return

    m = report["metrics"]

    print("=" * 60)
    print("  RISK SCORER VALIDATION REPORT")
    print("=" * 60)
    print()
    print(f"  Records:        {report['validation_records']} "
          f"(of {report['total_records']} total)")
    print(f"  Cancel rate:    {report['cancel_rate']:.1%}")
    print()
    print("  PROBABILITY CALIBRATION (Brier Score — lower is better)")
    print(f"    Risk scorer:   {m['model_brier_score']:.4f}")
    print(f"    Beaufort rule: {m['baseline_brier_score']:.4f}")
    print(f"    Climatology:   {m['climatological_brier']:.4f}")
    print()
    print("  SKILL SCORES (positive = better than baseline)")
    print(f"    vs Beaufort:    {m['skill_vs_beaufort']:+.3f}")
    print(f"    vs Climatology: {m['skill_vs_climatology']:+.3f}")
    print()
    print("  BINARY ACCURACY (P >= 0.5 threshold)")
    print(f"    Risk scorer:   {m['model_accuracy']:.1%}")
    print(f"    Beaufort rule: {m['baseline_accuracy']:.1%}")
    print()

    # Calibration curve
    print("  CALIBRATION CURVE")
    print(f"  {'Predicted':>10}  {'Observed':>10}  {'Count':>6}  {'Bar'}")
    print("  " + "-" * 50)
    for b in report["calibration_curve"]:
        bar = "#" * int(b["observed_frequency"] * 30)
        print(f"  {b['mean_predicted']:>10.2f}  {b['observed_frequency']:>10.2f}  "
              f"{b['count']:>6}  {bar}")
    print()

    # Per-route
    if report["per_route"]:
        print("  PER-ROUTE BREAKDOWN")
        print(f"  {'Route':<10} {'N':>5} {'Cancel%':>8} {'Model BS':>10} "
              f"{'Base BS':>10} {'Skill':>7}")
        print("  " + "-" * 55)
        for route_id, r in sorted(report["per_route"].items()):
            print(f"  {route_id:<10} {r['n_records']:>5} {r['cancel_rate']:>7.1%} "
                  f"{r['model_brier']:>10.4f} {r['baseline_brier']:>10.4f} "
                  f"{r['skill_score']:>+7.3f}")
    print()
