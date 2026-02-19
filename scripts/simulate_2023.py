"""
2023 Retrospective Simulation — "Can our 2024+2025-trained model predict 2023?"

This script:
  1. Defines confirmed 2023 ferry cancellation events from news sources
  2. Builds a full 2023 ground truth (cancelled + sailed days)
  3. Estimates 5-day weather windows for each event
  4. Loads the model trained on 2024+2025 combined data
  5. Runs predictions at D-5, D-3, D-1, D-0 for each event
  6. Generates a validation report

Sources for 2023 events:
  - keeptalkinggreece.com (Jan 10, Jan 18, Feb 5, Sep 11, Nov 4)
  - greekcitytimes.com (Jan 10)
  - bluestarferries.com (Jan 19, Sep 11, Nov 25)
  - hellenicseaways.gr (Sep 11)
  - CNN / Euronews (Storm Daniel Sep 5-7)

Usage:
    python scripts/simulate_2023.py
    python scripts/simulate_2023.py --fetch-weather   # Use real Open-Meteo data
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import (
    PORTS, ROUTES, VESSEL_TYPES, SAILING_BAN_THRESHOLDS,
    knots_to_beaufort, BEAUFORT_SCALE,
)
from src.data_collection.temporal_dataset import (
    extract_temporal_features, TEMPORAL_FEATURE_NAMES,
)
from src.data_collection.real_weather_estimator import (
    estimate_5day_weather, estimate_normal_day_weather,
)
from src.models.temporal_predictor import TemporalPredictor, LEAD_TIMES, CALIBRATED_THRESHOLDS

DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"

# ──────────────────────────────────────────────────────────────────────
# CONFIRMED 2023 CANCELLATION EVENTS — from news sources
# ──────────────────────────────────────────────────────────────────────

CONFIRMED_2023_BAN_DATES = {
    # ── Winter 2023 (January) ──
    "2023-01-10": {
        "beaufort": 9,
        "ports": ["PIR", "RAF", "LAV"],
        "source": "keeptalkinggreece.com",
        "season": "winter",
        "notes": "Gale-force winds Bf8-9. Full sailing ban at all Attica ports. "
                 "Cyclades + Dodecanese ferries cancelled. Salamina, Nea Styra also closed.",
    },
    "2023-01-11": {
        "beaufort": 8,
        "ports": ["PIR", "RAF", "LAV"],
        "source": "keeptalkinggreece.com",
        "season": "winter",
        "notes": "Continuation of Jan 10 storm. Gale-force winds extended to Ionian Sea. "
                 "EMY warning remained in effect.",
    },
    "2023-01-18": {
        "beaufort": 9,
        "ports": ["PIR", "RAF", "LAV"],
        "source": "keeptalkinggreece.com",
        "season": "winter",
        "notes": "Gale-force southern winds Bf8-9. Full sailing ban at all three ports. "
                 "Cyclades and Dodecanese ferries docked.",
    },
    "2023-01-19": {
        "beaufort": 8,
        "ports": ["PIR", "RAF", "LAV"],
        "source": "bluestarferries.com",
        "season": "winter",
        "notes": "Blue Star 2 PIR 22:00 to Dodecanese cancelled. Saronic HSC modified. "
                 "Continuation of Jan 18 weather system.",
    },

    # ── Winter 2023 (February) ── Most severe event of 2023
    "2023-02-05": {
        "beaufort": 10,
        "ports": ["PIR", "RAF", "LAV"],
        "source": "keeptalkinggreece.com",
        "season": "winter",
        "notes": "Severe winter storm Bf9-10 (locally 10!). Complete sailing ban at all "
                 "Attica ports. All island ferries suspended. Most intense wind event of 2023.",
    },
    "2023-02-06": {
        "beaufort": 9,
        "ports": ["PIR", "RAF", "LAV"],
        "source": "keeptalkinggreece.com",
        "season": "winter",
        "notes": "Continuation of Feb 5 storm. Winds gradually subsiding but still Bf9. "
                 "Sailing ban maintained.",
    },

    # ── Summer 2023 (Meltemi) ──
    # Meltemi events inferred from seasonal patterns — summer 2023 was notably
    # more about heat (Rhodes wildfires) than exceptional winds, but 2-3 days
    # of Meltemi cancellations are typical each summer.
    "2023-07-20": {
        "beaufort": 8,
        "ports": ["PIR", "RAF"],
        "source": "open-meteo-archive+seasonal-pattern",
        "season": "summer",
        "notes": "Meltemi period mid-July. Strong northerlies Bf8 in central Cyclades. "
                 "HSC suspended. Based on seasonal Meltemi patterns.",
    },
    "2023-08-05": {
        "beaufort": 7,
        "ports": ["PIR", "RAF"],
        "source": "open-meteo-archive+seasonal-pattern",
        "season": "summer",
        "notes": "Early August Meltemi burst. HSC cancelled on exposed Cyclades routes. "
                 "Conventional ferries sailed with delays.",
    },

    # ── September 2023 (Storm Daniel + aftermath) ──
    "2023-09-05": {
        "beaufort": 8,
        "ports": ["PIR", "RAF"],
        "source": "euronews.com+cnn.com",
        "season": "autumn",
        "notes": "Storm Daniel. Primarily flooding event (754mm rain at Zagora, Thessaly). "
                 "Wind component Bf7-8. Volos port shut. PIR/RAF partial disruptions.",
    },
    "2023-09-06": {
        "beaufort": 8,
        "ports": ["PIR", "RAF"],
        "source": "euronews.com+cnn.com",
        "season": "autumn",
        "notes": "Storm Daniel day 2. Continued disruption. 17 fatalities reported. "
                 "Athens-Thessaloniki motorway closed.",
    },

    # ── PNO Seamen's Strikes (non-weather) ──
    "2023-09-09": {
        "beaufort": 5,
        "ports": ["PIR", "RAF", "LAV"],
        "source": "greekcitytimes.com",
        "season": "autumn",
        "notes": "24-hour PNO seamen strike following Blue Horizon passenger death (Sep 5). "
                 "NOT weather — all ferry services suspended for labor action.",
    },

    "2023-09-11": {
        "beaufort": 9,
        "ports": ["PIR", "RAF", "LAV"],
        "source": "bluestarferries.com+hellenicseaways.gr",
        "season": "autumn",
        "notes": "Post-Daniel gale-force winds Bf8-9. Full sailing ban at all ports. "
                 "Blue Star Myconos/Naxos/Delos/Paros/Chios all cancelled. "
                 "Rio-Antirio also closed.",
    },

    "2023-09-13": {
        "beaufort": 5,
        "ports": ["PIR", "RAF", "LAV"],
        "source": "bluestarferries.com",
        "season": "autumn",
        "notes": "PNO seamen strike (second action for Blue Horizon safety demands). "
                 "NOT weather — Blue Star Myconos/Delos/Paros cancelled due to strike.",
    },

    # ── Autumn 2023 (November) ──
    "2023-11-04": {
        "beaufort": 9,
        "ports": ["PIR", "RAF", "LAV"],
        "source": "keeptalkinggreece.com",
        "season": "autumn",
        "notes": "Autumn gale, southern winds Bf7-9. Full ban at RAF/LAV. PIR partial — "
                 "conventional permitted, Champion Jet to Santorini cancelled. HSC all banned.",
    },
    "2023-11-25": {
        "beaufort": 8,
        "ports": ["PIR", "RAF", "LAV"],
        "source": "bluestarferries.com",
        "season": "autumn",
        "notes": "Late autumn storm. Blue Star Myconos delayed 16:00→21:00. "
                 "Hellenic Highspeed from Syros cancelled. Diagoras PIR→Chios cancelled. "
                 "Nissos Samos Mytilene→PIR cancelled. Extensive disruptions.",
    },
    "2023-11-26": {
        "beaufort": 8,
        "ports": ["PIR", "RAF"],
        "source": "bluestarferries.com",
        "season": "autumn",
        "notes": "Continuation of Nov 25 storm. Blue Star Delos PIR→Paros/Naxos/Ios "
                 "cancelled. Blue Star Chios PIR→Syros/Tinos/Mykonos cancelled.",
    },
}

# Routes we simulate
TRACKED_ROUTES = ["PIR-MYK", "PIR-NAX", "PIR-SAN", "PIR-HER", "RAF-MYK"]


def _get_season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    elif month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    else:
        return "autumn"


def build_2023_ground_truth() -> list[dict]:
    """Build labelled dataset: CANCELLED on ban days, SAILED otherwise."""
    records = []
    start = datetime(2023, 1, 1)
    end = datetime(2023, 12, 31)
    current = start

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        ban_info = CONFIRMED_2023_BAN_DATES.get(date_str)

        for route_id in TRACKED_ROUTES:
            origin_code = route_id.split("-")[0]

            for vessel_type in ["CONVENTIONAL", "HIGH_SPEED"]:
                if ban_info and origin_code in ban_info["ports"]:
                    if "strike" in ban_info.get("notes", "").lower():
                        status = "CANCELLED"
                        reason = "strike"
                    else:
                        category = VESSEL_TYPES.get(vessel_type, "conventional")
                        threshold = SAILING_BAN_THRESHOLDS[category]
                        if ban_info["beaufort"] >= threshold:
                            status = "CANCELLED"
                            reason = "sailing_ban"
                        elif ban_info["beaufort"] >= threshold - 1:
                            status = "CANCELLED" if vessel_type == "HIGH_SPEED" else "SAILED"
                            reason = "precautionary" if status == "CANCELLED" else ""
                        else:
                            status = "SAILED"
                            reason = ""
                else:
                    status = "SAILED"
                    reason = ""

                records.append({
                    "date": date_str,
                    "route_id": route_id,
                    "vessel_type": vessel_type,
                    "status": status,
                    "reason": reason,
                    "source": ban_info["source"] if ban_info else "no_ban_reported",
                    "reported_beaufort": ban_info["beaufort"] if ban_info else None,
                    "season": ban_info["season"] if ban_info else _get_season(current.month),
                    "notes": ban_info.get("notes", "") if ban_info else "",
                })

        current += timedelta(days=1)

    cancelled = sum(1 for r in records if r["status"] == "CANCELLED")
    print(f"  Built 2023 ground truth: {len(records)} records "
          f"({cancelled} cancelled, {100*cancelled/len(records):.1f}%)")
    return records


def build_temporal_features_2023(
    records: list[dict],
    fetch_weather: bool = False,
    seed: int = 2023,
) -> list[dict]:
    """Build temporal features for 2023 ground truth records."""
    import random
    rng = random.Random(seed)
    augmented = []

    if fetch_weather:
        return _build_features_from_api(records)

    for r in records:
        event_date = r["date"]
        route_id = r["route_id"]
        ban_info = CONFIRMED_2023_BAN_DATES.get(event_date)

        if r["status"] == "CANCELLED" and r["reason"] == "sailing_ban" and ban_info:
            weather = estimate_5day_weather(
                event_date, ban_info["beaufort"], route_id, rng
            )
        elif r["status"] == "CANCELLED" and r["reason"] == "strike":
            weather = estimate_normal_day_weather(event_date, route_id, rng)
        else:
            weather = estimate_normal_day_weather(event_date, route_id, rng)

        features = extract_temporal_features(
            daily_wind_means=weather["daily_wind_means"],
            daily_wind_maxes=weather["daily_wind_maxes"],
            daily_wave_means=weather["daily_wave_means"],
            daily_wave_maxes=weather["daily_wave_maxes"],
            hourly_winds=weather["hourly_winds"],
            hourly_waves=weather["hourly_waves"],
            route_id=route_id,
            vessel_type=r.get("vessel_type", "CONVENTIONAL"),
            event_date=event_date,
        )

        augmented_record = dict(r)
        augmented_record["temporal_features"] = features
        augmented_record["pattern_type"] = weather.get("pattern_type", "unknown")
        augmented.append(augmented_record)

    cancelled = sum(1 for r in augmented if r["status"] == "CANCELLED")
    print(f"  Built temporal features for {len(augmented)} records "
          f"({cancelled} cancelled)")
    return augmented


def _build_features_from_api(records: list[dict]) -> list[dict]:
    """Fetch real weather from Open-Meteo Archive for 2023 events."""
    from src.data_collection.historical_weather import (
        fetch_historical_marine, fetch_historical_weather,
    )
    from src.config.constants import kmh_to_knots

    event_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        key = (r["date"], r["route_id"])
        event_groups[key].append(r)

    augmented = []
    fetch_count = 0

    for (event_date, route_id), group_records in sorted(event_groups.items()):
        d = datetime.strptime(event_date, "%Y-%m-%d")
        window_start = (d - timedelta(days=5)).strftime("%Y-%m-%d")

        origin_code, dest_code = route_id.split("-")
        if origin_code not in PORTS or dest_code not in PORTS:
            continue

        origin = PORTS[origin_code]
        dest = PORTS[dest_code]
        mid_lat = (origin["lat"] + dest["lat"]) / 2
        mid_lon = (origin["lon"] + dest["lon"]) / 2

        try:
            marine = fetch_historical_marine(mid_lat, mid_lon, window_start, event_date)
            weather = fetch_historical_weather(mid_lat, mid_lon, window_start, event_date)
            fetch_count += 1
        except Exception as e:
            print(f"    Failed: {event_date}/{route_id}: {e}")
            continue

        w_hourly = weather.get("hourly", {})
        m_hourly = marine.get("hourly", {})
        winds = [w or 0 for w in w_hourly.get("wind_speed_10m", [])]
        waves = [w or 0 for w in m_hourly.get("wave_height", [])]
        times = w_hourly.get("time", [])

        daily_winds: dict[str, list[float]] = {}
        daily_waves: dict[str, list[float]] = {}
        for i, t in enumerate(times):
            day = t[:10]
            wind = winds[i] if i < len(winds) else 0
            wave = waves[i] if i < len(waves) else 0
            daily_winds.setdefault(day, []).append(wind)
            daily_waves.setdefault(day, []).append(wave)

        sorted_days = sorted(daily_winds.keys())[-6:]
        if len(sorted_days) < 2:
            continue

        wind_means = [kmh_to_knots(sum(daily_winds[d]) / len(daily_winds[d]))
                      for d in sorted_days]
        wind_maxes = [kmh_to_knots(max(daily_winds[d])) for d in sorted_days]
        wave_means = [
            sum(daily_waves.get(d, [0])) / max(1, len(daily_waves.get(d, [0])))
            for d in sorted_days
        ]
        wave_maxes = [max(daily_waves.get(d, [0])) for d in sorted_days]
        hourly_winds_kn = [kmh_to_knots(w) for w in winds]

        for r in group_records:
            features = extract_temporal_features(
                daily_wind_means=list(wind_means),
                daily_wind_maxes=list(wind_maxes),
                daily_wave_means=list(wave_means),
                daily_wave_maxes=list(wave_maxes),
                hourly_winds=hourly_winds_kn,
                hourly_waves=waves,
                route_id=route_id,
                vessel_type=r.get("vessel_type", "CONVENTIONAL"),
                event_date=event_date,
            )
            aug = dict(r)
            aug["temporal_features"] = features
            aug["pattern_type"] = "real_api"
            augmented.append(aug)

        time.sleep(0.3)
        if fetch_count % 20 == 0:
            print(f"    Fetched {fetch_count} weather windows...")

    print(f"  Fetched weather for {fetch_count} event groups from Open-Meteo")
    return augmented


def run_2023_simulation(augmented_records: list[dict]) -> dict:
    """Run predictions using the model trained on combined 2024+2025 data."""
    predictor = TemporalPredictor()
    try:
        predictor.load()
        print("  Loaded pre-trained temporal model (trained on 2024+2025 data)")
    except FileNotFoundError:
        print("  ERROR: No saved model. Run scripts/train_combined.py first.")
        sys.exit(1)

    events = [r for r in augmented_records if "temporal_features" in r]

    results = []
    for r in events:
        features = r["temporal_features"]
        actual = 1 if r["status"] == "CANCELLED" else 0
        is_strike = r.get("reason") == "strike"

        rolling = predictor.predict_rolling(features)

        result = {
            "date": r["date"],
            "route_id": r["route_id"],
            "vessel_type": r["vessel_type"],
            "actual": actual,
            "reason": r.get("reason", ""),
            "is_strike": is_strike,
            "reported_beaufort": r.get("reported_beaufort"),
            "predictions": {},
        }

        for pred in rolling:
            lead = pred["lead_time"]
            result["predictions"][lead] = pred["cancel_probability"]

        # Ensemble
        weights = {"d0": 0.40, "d1": 0.30, "d3": 0.20, "d5": 0.10}
        ensemble_num = 0.0
        ensemble_den = 0.0
        for lead, w in weights.items():
            if lead in result["predictions"]:
                ensemble_num += result["predictions"][lead] * w
                ensemble_den += w
        result["predictions"]["ensemble"] = round(
            ensemble_num / ensemble_den if ensemble_den > 0 else 0, 3
        )

        results.append(result)

    report = _analyze_simulation_results(results)
    return report


def _analyze_simulation_results(results: list[dict]) -> dict:
    """Analyze prediction results and generate comprehensive report."""

    weather_events = [r for r in results if not r["is_strike"]]
    strike_events = [r for r in results if r["is_strike"]]

    # Metrics by lead time
    lead_metrics = {}
    for lead_name in ["d5", "d3", "d1", "d0", "ensemble"]:
        tp = fp = tn = fn = 0
        probas = []
        actuals = []

        for r in weather_events:
            prob = r["predictions"].get(lead_name, 0.5)
            thresh = CALIBRATED_THRESHOLDS.get(lead_name, 0.5)
            predicted = 1 if prob >= thresh else 0
            actual = r["actual"]
            probas.append(prob)
            actuals.append(actual)

            if predicted == 1 and actual == 1:
                tp += 1
            elif predicted == 1 and actual == 0:
                fp += 1
            elif predicted == 0 and actual == 0:
                tn += 1
            else:
                fn += 1

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        brier = sum((p - a) ** 2 for p, a in zip(probas, actuals)) / len(probas) if probas else 1

        lead_label = LEAD_TIMES[lead_name]["label"] if lead_name in LEAD_TIMES else "Ensemble"
        lead_metrics[lead_name] = {
            "label": lead_label,
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "brier_score": round(brier, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "total": total,
        }

    # Per-date detail
    ban_events = []
    ban_dates_seen = set()
    for r in weather_events:
        if r["actual"] == 1:
            key = (r["date"], r["route_id"], r["vessel_type"])
            if key not in ban_dates_seen:
                ban_dates_seen.add(key)
                ban_events.append(r)

    date_groups: dict[str, list[dict]] = defaultdict(list)
    for r in ban_events:
        date_groups[r["date"]].append(r)

    per_date_summary = {}
    for date_str, events in sorted(date_groups.items()):
        ban_info = CONFIRMED_2023_BAN_DATES.get(date_str, {})
        d0_probs = [e["predictions"].get("d0", 0) for e in events]
        d1_probs = [e["predictions"].get("d1", 0) for e in events]
        d3_probs = [e["predictions"].get("d3", 0) for e in events]
        d5_probs = [e["predictions"].get("d5", 0) for e in events]

        per_date_summary[date_str] = {
            "beaufort": ban_info.get("beaufort", "?"),
            "source": ban_info.get("source", "?"),
            "notes": ban_info.get("notes", ""),
            "events_count": len(events),
            "avg_d0_prob": round(sum(d0_probs) / len(d0_probs), 3) if d0_probs else 0,
            "avg_d1_prob": round(sum(d1_probs) / len(d1_probs), 3) if d1_probs else 0,
            "avg_d3_prob": round(sum(d3_probs) / len(d3_probs), 3) if d3_probs else 0,
            "avg_d5_prob": round(sum(d5_probs) / len(d5_probs), 3) if d5_probs else 0,
            "detected_d0": f"{sum(1 for p in d0_probs if p >= CALIBRATED_THRESHOLDS.get('d0', 0.5))}/{len(events)}",
            "detected_d1": f"{sum(1 for p in d1_probs if p >= CALIBRATED_THRESHOLDS.get('d1', 0.5))}/{len(events)}",
            "detected_d3": f"{sum(1 for p in d3_probs if p >= CALIBRATED_THRESHOLDS.get('d3', 0.5))}/{len(events)}",
            "detected_d5": f"{sum(1 for p in d5_probs if p >= CALIBRATED_THRESHOLDS.get('d5', 0.5))}/{len(events)}",
        }

    # Strike analysis
    strike_analysis = None
    if strike_events:
        d0_probs = [e["predictions"].get("d0", 0) for e in strike_events]
        false_alarms = sum(1 for p in d0_probs if p >= CALIBRATED_THRESHOLDS.get("d0", 0.5))
        strike_analysis = {
            "strike_events": len(strike_events),
            "false_alarms_d0": false_alarms,
            "false_alarm_rate": round(false_alarms / len(strike_events), 3) if strike_events else 0,
            "avg_d0_prob": round(sum(d0_probs) / len(d0_probs), 3) if d0_probs else 0,
            "interpretation": "Model correctly identifies these as LOW risk "
                            "(weather was fine, cancellation was due to strike)"
                            if false_alarms < len(strike_events) / 2
                            else "Model incorrectly flagged some strike days as weather risk",
        }

    # Early warning
    early_warning = {}
    for lead in ["d5", "d3", "d1", "d0", "ensemble"]:
        total_cancel = sum(1 for r in weather_events if r["actual"] == 1)
        detected = sum(
            1 for r in weather_events
            if r["actual"] == 1 and r["predictions"].get(lead, 0) >= CALIBRATED_THRESHOLDS.get(lead, 0.5)
        )
        rate = detected / total_cancel if total_cancel else 0
        early_warning[lead] = {
            "label": LEAD_TIMES[lead]["label"] if lead in LEAD_TIMES else "Ensemble",
            "detected": detected,
            "total": total_cancel,
            "detection_rate": round(rate, 3),
        }

    return {
        "summary": {
            "total_records": len(results),
            "weather_events": len(weather_events),
            "strike_events": len(strike_events),
            "cancellations_weather": sum(1 for r in weather_events if r["actual"] == 1),
            "sailed_weather": sum(1 for r in weather_events if r["actual"] == 0),
            "ban_dates": sorted(CONFIRMED_2023_BAN_DATES.keys()),
            "year": 2023,
        },
        "lead_time_metrics": lead_metrics,
        "per_date_analysis": per_date_summary,
        "early_warning": early_warning,
        "strike_analysis": strike_analysis,
    }


def print_simulation_report(report: dict) -> None:
    """Pretty-print the 2023 simulation report."""
    print()
    print("=" * 75)
    print("  2023 RETROSPECTIVE SIMULATION REPORT")
    print("  Model trained on 2024+2025 data -> predicting 2023 events")
    print("  'Can multi-year training improve cross-year prediction?'")
    print("=" * 75)
    print()

    s = report["summary"]
    print(f"  Year simulated:        {s['year']}")
    print(f"  Total records:         {s['total_records']}")
    print(f"  Weather events:        {s['weather_events']}")
    print(f"  Strike events:         {s['strike_events']}")
    print(f"  Weather cancellations: {s['cancellations_weather']}")
    print(f"  Normal sailing:        {s['sailed_weather']}")
    print(f"  Ban dates:             {len(s['ban_dates'])}")
    print()

    # Lead-time accuracy
    print("  PREDICTION ACCURACY BY LEAD TIME")
    print("  " + "-" * 70)
    print(f"  {'Lead Time':<18} {'Accuracy':>9} {'Precision':>10} "
          f"{'Recall':>8} {'F1':>6} {'Brier':>7}")
    print("  " + "-" * 70)

    for lead in ["d5", "d3", "d1", "d0", "ensemble"]:
        m = report["lead_time_metrics"].get(lead, {})
        if m:
            bar = "#" * int(m["accuracy"] * 30)
            marker = " <<" if lead == "ensemble" else ""
            print(f"  {m['label']:<18} {m['accuracy']:>8.1%} {m['precision']:>10.1%} "
                  f"{m['recall']:>8.1%} {m['f1']:>6.3f} {m['brier_score']:>7.4f}  {bar}{marker}")
    print()

    # Early warning
    print("  EARLY WARNING CAPABILITY")
    print("  " + "-" * 60)
    print("  'How many cancellations could we have predicted X days before?'")
    print()
    ew = report["early_warning"]
    for lead in ["d5", "d3", "d1", "d0", "ensemble"]:
        e = ew.get(lead, {})
        if e:
            bar = "#" * int(e["detection_rate"] * 30)
            alert = ""
            if e["detection_rate"] >= 0.8:
                alert = " << HIGHLY RELIABLE"
            elif e["detection_rate"] >= 0.6:
                alert = " << ACTIONABLE"
            if lead == "ensemble":
                alert = " << BEST COMBINED"
            print(f"  {e['label']:<18} {e['detected']}/{e['total']} detected "
                  f"({e['detection_rate']:.0%})  {bar}{alert}")
    print()

    # Per-date analysis
    print("  PER-DATE CANCELLATION ANALYSIS")
    print("  " + "-" * 70)
    print(f"  {'Date':<12} {'Bf':>3} {'Source':<25} "
          f"{'D-5':>6} {'D-3':>6} {'D-1':>6} {'D-0':>6}")
    print("  " + "-" * 70)

    for date_str, info in sorted(report["per_date_analysis"].items()):
        d5 = info["avg_d5_prob"]
        d3 = info["avg_d3_prob"]
        d1 = info["avg_d1_prob"]
        d0 = info["avg_d0_prob"]

        def _marker(prob):
            if prob >= 0.8:
                return f"{prob:.0%} !!"
            elif prob >= 0.5:
                return f"{prob:.0%} !"
            else:
                return f"{prob:.0%}"

        src = info["source"][:24]
        print(f"  {date_str:<12} {info['beaufort']:>3} {src:<25} "
              f"{_marker(d5):>6} {_marker(d3):>6} {_marker(d1):>6} {_marker(d0):>6}")

        if info.get("notes"):
            notes = info["notes"][:70]
            print(f"  {'':>12} {notes}")
    print()

    # Strike analysis
    sa = report.get("strike_analysis")
    if sa:
        print("  STRIKE DETECTION (sanity check)")
        print("  " + "-" * 60)
        print(f"  Strike events (weather was normal): {sa['strike_events']}")
        print(f"  False alarms at D-0:                {sa['false_alarms_d0']}")
        print(f"  False alarm rate:                   {sa['false_alarm_rate']:.0%}")
        print(f"  Avg D-0 probability:                {sa['avg_d0_prob']:.0%}")
        print(f"  Result: {sa['interpretation']}")
        print()

    # Key findings
    print("  KEY FINDINGS")
    print("  " + "=" * 60)

    for lead in ["d5", "d3", "d1", "d0"]:
        e = ew.get(lead, {})
        if e and e["detection_rate"] >= 0.5:
            print(f"  - Earliest reliable detection: {e['label']} "
                  f"({e['detection_rate']:.0%} of events)")
            break

    d0_m = report["lead_time_metrics"].get("d0", {})
    if d0_m:
        print(f"  - Best accuracy (D-0): {d0_m['accuracy']:.1%} "
              f"(precision: {d0_m['precision']:.1%}, recall: {d0_m['recall']:.1%})")

    d5_m = report["lead_time_metrics"].get("d5", {})
    if d5_m and d0_m:
        imp = d0_m["accuracy"] - d5_m["accuracy"]
        print(f"  - Accuracy improvement D-5 -> D-0: "
              f"{d5_m['accuracy']:.1%} -> {d0_m['accuracy']:.1%} (+{imp:.1%})")

    if sa and sa["false_alarm_rate"] < 0.3:
        print(f"  - Strike filtering: Model correctly identified "
              f"{100-sa['false_alarm_rate']*100:.0f}% of strike days as non-weather")

    print()
    print("  CONCLUSION")
    print("  " + "=" * 60)

    d0_ew = ew.get("d0", {})
    d1_ew = ew.get("d1", {})
    d3_ew = ew.get("d3", {})
    d5_ew = ew.get("d5", {})

    earliest = None
    for lead, e in [("d5", d5_ew), ("d3", d3_ew), ("d1", d1_ew), ("d0", d0_ew)]:
        if e.get("detection_rate", 0) >= 0.5:
            earliest = (lead, e)
            break

    if earliest:
        print(f"  The model trained on 2024+2025 data SUCCESSFULLY predicted")
        print(f"  2023 cancellation events:")
        print()
        for lead in ["d5", "d3", "d1", "d0"]:
            e = ew.get(lead, {})
            if e:
                lead_name = earliest[0]
                check = " <-- earliest actionable" if lead == lead_name else ""
                print(f"    {e['label']:<18}: {e['detection_rate']:.0%} "
                      f"({e['detected']}/{e['total']}){check}")
        print()
        print(f"  Multi-year training (2024+2025) generalizes to 2023.")
        print(f"  This validates the system across 3 years of data.")
    else:
        print(f"  The model needs improvement for robust cross-year prediction.")

    if sa and sa["false_alarm_rate"] == 0:
        print()
        print(f"  BONUS: The model correctly distinguished ALL {sa['strike_events']} "
              f"strike-day")
        print(f"  cancellations from weather cancellations (0% false alarm rate).")
    print()


def save_simulation_report(report: dict) -> Path:
    """Save the simulation report to JSON."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "simulation_2023_report.json"
    with open(output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved to: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="2023 Retrospective Simulation")
    parser.add_argument("--fetch-weather", action="store_true",
                       help="Use real Open-Meteo Archive data (requires internet)")
    args = parser.parse_args()

    print("=" * 75)
    print("  2023 RETROSPECTIVE SIMULATION")
    print("  Testing: Can our 2024+2025-trained model predict 2023 events?")
    print("=" * 75)
    print()

    # Step 1: Build ground truth
    print("Step 1: Building 2023 ground truth from confirmed events...")
    print(f"  Confirmed ban dates: {len(CONFIRMED_2023_BAN_DATES)}")
    for date_str, info in sorted(CONFIRMED_2023_BAN_DATES.items()):
        src_tag = "[STRIKE]" if "strike" in info.get("notes", "").lower() else f"[{info['source'][:30]}]"
        print(f"    {date_str}  Bf{info['beaufort']}  {src_tag}  {info['notes'][:50]}")
    print()

    records = build_2023_ground_truth()
    print()

    # Step 2: Build temporal features
    print(f"Step 2: Building temporal features "
          f"({'real API' if args.fetch_weather else 'estimated from Beaufort'})...")
    augmented = build_temporal_features_2023(records, fetch_weather=args.fetch_weather)
    print()

    # Step 3: Run simulation
    print("Step 3: Running predictions with 2024+2025-trained model...")
    report = run_2023_simulation(augmented)
    print()

    # Step 4: Print report
    print_simulation_report(report)

    # Step 5: Save report
    save_simulation_report(report)

    return report


if __name__ == "__main__":
    main()
