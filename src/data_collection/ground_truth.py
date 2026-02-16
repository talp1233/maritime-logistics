"""
Ground truth data collector for ferry cancellations.

Collects real cancellation/sailing data from public sources:
  1. Ferry schedule comparison (scheduled vs actual)
  2. Coast Guard sailing ban announcements
  3. Historical weather at time of cancellation

This data is essential for training and validating ML models.
"""

import json
import os
import csv
from datetime import datetime, timedelta
from pathlib import Path

from src.config.constants import ROUTES, PORTS, knots_to_beaufort
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "ground_truth"


class GroundTruthCollector:
    """Collects and stores ground truth cancellation records."""

    def __init__(self, data_dir: Path | str | None = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.records_file = self.data_dir / "cancellation_records.csv"
        self._ensure_csv()

    def _ensure_csv(self):
        """Create CSV file with headers if it doesn't exist."""
        if not self.records_file.exists():
            with open(self.records_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "date", "route_id", "scheduled_departure", "vessel_name",
                    "vessel_type", "status", "reason", "source",
                    "wind_speed_kn", "wind_beaufort", "wave_height_m",
                    "wind_direction", "recorded_at",
                ])
            logger.info("Created ground truth CSV: %s", self.records_file)

    def add_record(
        self,
        date: str,
        route_id: str,
        scheduled_departure: str,
        vessel_name: str = "",
        vessel_type: str = "CONVENTIONAL",
        status: str = "CANCELLED",
        reason: str = "",
        source: str = "manual",
        wind_speed_kn: float = 0.0,
        wave_height_m: float = 0.0,
        wind_direction: float = 0.0,
    ):
        """
        Add a cancellation/sailing record.

        Args:
            date: Date string (YYYY-MM-DD)
            route_id: Route key (e.g. "PIR-MYK")
            scheduled_departure: Scheduled departure time (HH:MM)
            vessel_name: Name of the vessel
            vessel_type: CONVENTIONAL, HIGH_SPEED, etc.
            status: CANCELLED, SAILED, DELAYED
            reason: Reason text (e.g. "sailing ban", "weather", "mechanical")
            source: Data source ("manual", "coast_guard", "ais", "schedule_check")
            wind_speed_kn: Wind speed at time of event (knots)
            wave_height_m: Wave height at time of event (meters)
            wind_direction: Wind direction in degrees
        """
        beaufort = knots_to_beaufort(wind_speed_kn)

        with open(self.records_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                date, route_id, scheduled_departure, vessel_name,
                vessel_type, status, reason, source,
                f"{wind_speed_kn:.1f}", beaufort, f"{wave_height_m:.1f}",
                f"{wind_direction:.0f}", datetime.now().isoformat(),
            ])

        logger.info(
            "Recorded: %s %s %s %s (Bf %d, %.1fm waves)",
            date, route_id, status, vessel_name, beaufort, wave_height_m,
        )

    def load_records(self) -> list[dict]:
        """Load all records from CSV as list of dicts."""
        if not self.records_file.exists():
            return []

        records = []
        with open(self.records_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                row["wind_speed_kn"] = float(row.get("wind_speed_kn", 0))
                row["wind_beaufort"] = int(row.get("wind_beaufort", 0))
                row["wave_height_m"] = float(row.get("wave_height_m", 0))
                row["wind_direction"] = float(row.get("wind_direction", 0))
                records.append(row)
        return records

    def get_stats(self) -> dict:
        """Get summary statistics of collected data."""
        records = self.load_records()
        if not records:
            return {"total": 0}

        cancelled = [r for r in records if r["status"] == "CANCELLED"]
        sailed = [r for r in records if r["status"] == "SAILED"]

        routes = set(r["route_id"] for r in records)
        dates = set(r["date"] for r in records)

        return {
            "total": len(records),
            "cancelled": len(cancelled),
            "sailed": len(sailed),
            "cancel_rate": len(cancelled) / len(records) if records else 0,
            "routes_covered": len(routes),
            "date_range": {
                "earliest": min(dates) if dates else None,
                "latest": max(dates) if dates else None,
            },
            "sources": list(set(r.get("source", "") for r in records)),
        }

    def generate_sample_data(self, n_days: int = 90):
        """
        Generate sample ground truth data for development/testing.
        Simulates 90 days of ferry operations with weather-correlated cancellations.
        """
        import random
        random.seed(42)

        start_date = datetime.now() - timedelta(days=n_days)
        vessels = {
            "PIR-MYK": [("Blue Star Delos", "CONVENTIONAL"), ("Champion Jet 1", "HIGH_SPEED")],
            "PIR-SAN": [("Blue Star Naxos", "CONVENTIONAL"), ("WorldChampion Jet", "HIGH_SPEED")],
            "PIR-HER": [("Knossos Palace", "CONVENTIONAL"), ("Festos Palace", "CONVENTIONAL")],
            "RAF-MYK": [("Superferry II", "CONVENTIONAL"), ("Tera Jet", "HIGH_SPEED")],
            "PIR-NAX": [("Blue Star Paros", "CONVENTIONAL"), ("Naxos Jet", "HIGH_SPEED")],
        }

        count = 0
        for day in range(n_days):
            date = start_date + timedelta(days=day)
            date_str = date.strftime("%Y-%m-%d")
            month = date.month

            # Seasonal wind pattern
            base_wind = {1: 18, 2: 17, 3: 15, 4: 12, 5: 14, 6: 18,
                         7: 25, 8: 26, 9: 20, 10: 16, 11: 17, 12: 19}[month]
            daily_wind = max(0, base_wind + random.gauss(0, 8))
            daily_wave = max(0.1, 0.06 * (daily_wind ** 1.3) + random.gauss(0, 0.3))
            wind_dir = random.gauss(20, 40) % 360

            for route_id, route_vessels in vessels.items():
                for vessel_name, vessel_type in route_vessels:
                    for dep_hour in ["07:00", "15:00"]:
                        # Vary wind slightly per departure
                        wind = max(0, daily_wind + random.gauss(0, 3))
                        wave = max(0.1, daily_wave + random.gauss(0, 0.2))
                        bf = knots_to_beaufort(wind)

                        # Determine if cancelled
                        from src.config.constants import SAILING_BAN_THRESHOLDS, VESSEL_TYPES
                        cat = VESSEL_TYPES[vessel_type]
                        threshold = SAILING_BAN_THRESHOLDS[cat]

                        if bf >= threshold:
                            status = "CANCELLED"
                            reason = "sailing_ban"
                        elif bf >= threshold - 1 and random.random() < 0.3:
                            status = "CANCELLED"
                            reason = "precautionary"
                        elif random.random() < 0.02:
                            status = "CANCELLED"
                            reason = "mechanical"
                        else:
                            status = "SAILED"
                            reason = ""

                        self.add_record(
                            date=date_str,
                            route_id=route_id,
                            scheduled_departure=dep_hour,
                            vessel_name=vessel_name,
                            vessel_type=vessel_type,
                            status=status,
                            reason=reason,
                            source="synthetic",
                            wind_speed_kn=wind,
                            wave_height_m=wave,
                            wind_direction=wind_dir,
                        )
                        count += 1

        logger.info("Generated %d sample records over %d days", count, n_days)
        return count
