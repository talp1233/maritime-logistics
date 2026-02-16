"""Tests for ground truth data collector."""

import tempfile
from pathlib import Path
from src.data_collection.ground_truth import GroundTruthCollector


class TestGroundTruthCollector:
    def test_create_and_add_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = GroundTruthCollector(data_dir=tmpdir)
            collector.add_record(
                date="2025-01-15",
                route_id="PIR-MYK",
                scheduled_departure="07:00",
                vessel_name="Blue Star Delos",
                vessel_type="CONVENTIONAL",
                status="CANCELLED",
                reason="sailing_ban",
                wind_speed_kn=38.0,
                wave_height_m=4.5,
            )
            records = collector.load_records()
            assert len(records) == 1
            assert records[0]["route_id"] == "PIR-MYK"
            assert records[0]["status"] == "CANCELLED"

    def test_stats_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = GroundTruthCollector(data_dir=tmpdir)
            stats = collector.get_stats()
            assert stats["total"] == 0

    def test_stats_with_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = GroundTruthCollector(data_dir=tmpdir)
            collector.add_record(
                date="2025-01-15", route_id="PIR-MYK",
                scheduled_departure="07:00", status="CANCELLED",
                wind_speed_kn=40.0, wave_height_m=5.0,
            )
            collector.add_record(
                date="2025-01-15", route_id="PIR-SAN",
                scheduled_departure="08:00", status="SAILED",
                wind_speed_kn=10.0, wave_height_m=1.0,
            )
            stats = collector.get_stats()
            assert stats["total"] == 2
            assert stats["cancelled"] == 1
            assert stats["sailed"] == 1
            assert stats["routes_covered"] == 2

    def test_generate_sample_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = GroundTruthCollector(data_dir=tmpdir)
            count = collector.generate_sample_data(n_days=3)
            assert count > 0
            records = collector.load_records()
            assert len(records) == count
            # Check we have both cancelled and sailed
            statuses = set(r["status"] for r in records)
            assert "SAILED" in statuses
