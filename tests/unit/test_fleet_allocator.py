"""Tests for the fleet allocation engine."""

import pytest

from src.services.fleet_allocator import (
    assess_vessel_fitness,
    allocate_fleet,
    VesselFitness,
    AllocationResult,
)


class TestAssessVesselFitness:
    def test_calm_conditions_high_fitness(self):
        fitness = assess_vessel_fitness(
            vessel_name="Blue Star Delos",
            route_id="PIR-MYK",
            wind_speed_knots=10.0,
            wave_height_m=0.5,
        )
        assert isinstance(fitness, VesselFitness)
        assert fitness.fitness > 0.5
        assert fitness.risk_score < 50

    def test_storm_conditions_low_fitness(self):
        fitness = assess_vessel_fitness(
            vessel_name="Tera Jet",
            route_id="PIR-MYK",
            wind_speed_knots=40.0,
            wave_height_m=4.0,
        )
        assert fitness.fitness < 0.5
        assert fitness.risk_score > 50

    def test_larger_vessel_better_fitness(self):
        """Knossos Palace (36000t) should fit better than Tera Jet (4700t)."""
        large = assess_vessel_fitness(
            vessel_name="Knossos Palace",
            route_id="PIR-HER",
            wind_speed_knots=30.0,
            wave_height_m=3.0,
        )
        small = assess_vessel_fitness(
            vessel_name="Tera Jet",
            route_id="PIR-HER",  # not its normal route, but that's ok for testing
            wind_speed_knots=30.0,
            wave_height_m=3.0,
        )
        assert large.fitness > small.fitness

    def test_unknown_vessel_raises(self):
        with pytest.raises(ValueError, match="Unknown vessel"):
            assess_vessel_fitness(
                vessel_name="Nonexistent Ship",
                route_id="PIR-MYK",
                wind_speed_knots=10.0,
                wave_height_m=0.5,
            )

    def test_fitness_in_bounds(self):
        fitness = assess_vessel_fitness(
            vessel_name="Blue Star Delos",
            route_id="PIR-MYK",
            wind_speed_knots=20.0,
            wave_height_m=1.5,
        )
        assert 0.0 <= fitness.fitness <= 1.0


class TestAllocateFleet:
    def _calm_conditions(self):
        return {
            r: {"wind_speed_knots": 10.0, "wave_height_m": 0.5}
            for r in ["PIR-MYK", "PIR-SAN", "PIR-HER"]
        }

    def _storm_conditions(self):
        return {
            r: {"wind_speed_knots": 40.0, "wave_height_m": 4.0}
            for r in ["PIR-MYK", "PIR-SAN", "PIR-HER"]
        }

    def test_basic_allocation(self):
        routes = ["PIR-MYK", "PIR-SAN", "PIR-HER"]
        result = allocate_fleet(routes, self._calm_conditions())
        assert isinstance(result, AllocationResult)
        assert len(result.assignments) > 0

    def test_no_duplicate_assignments(self):
        """Each vessel should be assigned to at most one route."""
        routes = ["PIR-MYK", "PIR-SAN", "PIR-HER"]
        result = allocate_fleet(routes, self._calm_conditions())
        vessels = [a.vessel_name for a in result.assignments]
        assert len(vessels) == len(set(vessels))

    def test_no_duplicate_routes(self):
        """Each route should have at most one vessel."""
        routes = ["PIR-MYK", "PIR-SAN"]
        result = allocate_fleet(routes, self._calm_conditions())
        assigned_routes = [a.route_id for a in result.assignments]
        assert len(assigned_routes) == len(set(assigned_routes))

    def test_storm_assigns_larger_vessels(self):
        """In storms, larger vessels should be preferred."""
        routes = ["PIR-HER"]
        result = allocate_fleet(routes, self._storm_conditions())
        if result.assignments:
            # The best fit for PIR-HER in a storm should be a large conventional
            assert result.assignments[0].vessel_name in [
                "Knossos Palace", "Festos Palace", "Blue Star Delos",
                "Blue Star Naxos", "Blue Star Paros",
            ]

    def test_specific_vessels(self):
        """Can limit to specific available vessels."""
        routes = ["PIR-MYK"]
        result = allocate_fleet(
            routes,
            self._calm_conditions(),
            available_vessels=["Blue Star Delos", "Champion Jet 1"],
        )
        assigned = [a.vessel_name for a in result.assignments]
        assert all(v in ["Blue Star Delos", "Champion Jet 1"] for v in assigned)

    def test_unassigned_routes_tracked(self):
        """Routes that can't be covered should be listed."""
        routes = ["PIR-MYK", "PIR-SAN", "PIR-HER", "PIR-CHN", "RAF-MYK"]
        result = allocate_fleet(
            routes,
            self._calm_conditions(),
            available_vessels=["Blue Star Delos"],  # only 1 vessel for 5 routes
        )
        assert len(result.unassigned_routes) >= 4

    def test_to_dict(self):
        routes = ["PIR-MYK"]
        result = allocate_fleet(routes, self._calm_conditions())
        d = result.to_dict()
        assert "assignments" in d
        assert "unassigned_routes" in d
        assert "fleet_risk_avg" in d
        assert "fleet_risk_max" in d
