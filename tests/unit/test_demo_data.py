"""Tests for demo data generation."""

from src.data_collection.demo_data import (
    generate_hourly_forecast,
    generate_demo_route_conditions,
    DEMO_SCENARIOS,
)


class TestDemoData:
    def test_generate_hourly_forecast_structure(self):
        data = generate_hourly_forecast(base_wind_knots=15, hours=24)
        assert "weather" in data
        assert "marine" in data
        assert "hourly" in data["weather"]
        assert "hourly" in data["marine"]
        assert len(data["weather"]["hourly"]["time"]) == 24
        assert len(data["weather"]["hourly"]["wind_speed_10m"]) == 24
        assert len(data["marine"]["hourly"]["wave_height"]) == 24

    def test_generate_hourly_forecast_no_negative_wind(self):
        data = generate_hourly_forecast(base_wind_knots=5, hours=48)
        for w in data["weather"]["hourly"]["wind_speed_10m"]:
            assert w >= 0

    def test_generate_hourly_forecast_no_negative_waves(self):
        data = generate_hourly_forecast(base_wind_knots=20, hours=24)
        for w in data["marine"]["hourly"]["wave_height"]:
            assert w >= 0.1

    def test_storm_scenario_has_high_winds(self):
        data = generate_hourly_forecast(
            base_wind_knots=25, hours=48, storm_probability=1.0,
        )
        max_wind = max(data["weather"]["hourly"]["wind_speed_10m"])
        assert max_wind > 30  # Storm should produce high winds

    def test_calm_scenario_low_winds(self):
        data = generate_demo_route_conditions(forecast_days=1, scenario="calm")
        winds = data["midpoint"]["weather"]["hourly"]["wind_speed_10m"]
        avg_wind = sum(winds) / len(winds)
        assert avg_wind < 25  # Calm scenario should have moderate winds

    def test_route_conditions_structure(self):
        data = generate_demo_route_conditions(forecast_days=1, scenario="auto")
        assert "origin" in data
        assert "midpoint" in data
        assert "destination" in data
        for point in ["origin", "midpoint", "destination"]:
            assert "marine" in data[point]
            assert "weather" in data[point]

    def test_all_scenarios_valid(self):
        for name in DEMO_SCENARIOS:
            data = generate_demo_route_conditions(forecast_days=1, scenario=name)
            assert "midpoint" in data
