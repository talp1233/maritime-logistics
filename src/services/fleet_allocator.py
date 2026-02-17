"""
Fleet allocation engine for Greek ferry operations.

Given current sea-state conditions across all routes, recommends the
optimal vessel-to-route assignment that minimises total fleet risk.

The allocator solves a greedy assignment problem:
  - Each vessel has a fitness score for each route it can serve
  - Fitness = inverse of risk score (lower risk → higher fitness)
  - Vessels with higher thresholds are preferred for riskier routes
  - The result is a recommended assignment that a fleet manager can review

This is NOT meant to replace human judgment — it is a decision support
tool that surfaces the safest configuration under current conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config.constants import ROUTES, VESSELS, VESSEL_TYPES, SAILING_BAN_THRESHOLDS
from src.services.risk_scorer import compute_risk, score_to_band
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VesselFitness:
    """Risk assessment of a specific vessel on a specific route."""
    vessel_name: str
    route_id: str
    risk_score: float
    band: str
    cancel_probability: float
    fitness: float  # 0-1, higher = better fit


@dataclass
class FleetAssignment:
    """One vessel-route pairing in the allocation."""
    vessel_name: str
    route_id: str
    risk_score: float
    band: str
    cancel_probability: float
    is_primary: bool   # True if this is the vessel's normal route


@dataclass
class AllocationResult:
    """Output of the fleet allocation engine."""
    assignments: list[FleetAssignment]
    unassigned_routes: list[str]
    unassigned_vessels: list[str]
    fleet_risk_avg: float
    fleet_risk_max: float

    def to_dict(self) -> dict:
        return {
            "assignments": [
                {
                    "vessel": a.vessel_name,
                    "route": a.route_id,
                    "risk_score": round(a.risk_score, 1),
                    "band": a.band,
                    "cancel_probability": round(a.cancel_probability, 3),
                    "is_primary_route": a.is_primary,
                }
                for a in self.assignments
            ],
            "unassigned_routes": self.unassigned_routes,
            "unassigned_vessels": self.unassigned_vessels,
            "fleet_risk_avg": round(self.fleet_risk_avg, 1),
            "fleet_risk_max": round(self.fleet_risk_max, 1),
        }


def assess_vessel_fitness(
    vessel_name: str,
    route_id: str,
    wind_speed_knots: float,
    wave_height_m: float,
    wave_period_s: float = 5.0,
    wind_gust_knots: float | None = None,
    swell_height_m: float = 0.0,
    wind_direction: float | None = None,
) -> VesselFitness:
    """
    Compute how well a specific vessel fits a route under current conditions.

    Returns VesselFitness with risk_score and normalised fitness (0-1).
    """
    vessel = VESSELS.get(vessel_name)
    if not vessel:
        raise ValueError(f"Unknown vessel: {vessel_name}")

    result = compute_risk(
        wind_speed_knots=wind_speed_knots,
        wave_height_m=wave_height_m,
        wave_period_s=wave_period_s,
        wind_gust_knots=wind_gust_knots,
        swell_height_m=swell_height_m,
        wind_direction=wind_direction,
        vessel_type=vessel["type"],
        vessel_name=vessel_name,
        route_id=route_id,
    )

    # Fitness: inverse-linear mapping from risk score
    # score 0 → fitness 1.0,  score 100 → fitness 0.0
    fitness = max(0.0, 1.0 - result.score / 100.0)

    return VesselFitness(
        vessel_name=vessel_name,
        route_id=route_id,
        risk_score=result.score,
        band=result.band,
        cancel_probability=result.cancel_probability,
        fitness=round(fitness, 3),
    )


def allocate_fleet(
    routes_to_cover: list[str],
    conditions: dict[str, dict],
    available_vessels: list[str] | None = None,
) -> AllocationResult:
    """
    Greedy fleet allocation: assign vessels to routes minimising total risk.

    Args:
        routes_to_cover: List of route IDs that need coverage
        conditions: Per-route weather conditions, keyed by route_id.
                    Each value: {"wind_speed_knots", "wave_height_m",
                    "wave_period_s", "wind_gust_knots", "swell_height_m",
                    "wind_direction"}
        available_vessels: List of vessel names (default: all in VESSELS)

    Returns:
        AllocationResult with recommended assignments.
    """
    if available_vessels is None:
        available_vessels = list(VESSELS.keys())

    # Build fitness matrix: every vessel × every route it can serve
    fitness_matrix: list[VesselFitness] = []
    for vessel_name in available_vessels:
        vessel = VESSELS.get(vessel_name)
        if not vessel:
            continue
        vessel_routes = vessel.get("routes", [])
        for route_id in routes_to_cover:
            # Vessel must be capable of this route (in its route list)
            # or it's a flexible assignment
            can_serve = route_id in vessel_routes or route_id in ROUTES
            if not can_serve:
                continue

            cond = conditions.get(route_id, {})
            try:
                fitness = assess_vessel_fitness(
                    vessel_name=vessel_name,
                    route_id=route_id,
                    wind_speed_knots=cond.get("wind_speed_knots", 0),
                    wave_height_m=cond.get("wave_height_m", 0),
                    wave_period_s=cond.get("wave_period_s", 5.0),
                    wind_gust_knots=cond.get("wind_gust_knots"),
                    swell_height_m=cond.get("swell_height_m", 0),
                    wind_direction=cond.get("wind_direction"),
                )
                fitness_matrix.append(fitness)
            except Exception as e:
                logger.debug("Fitness calc failed for %s on %s: %s",
                             vessel_name, route_id, e)

    # Sort by fitness descending (best fits first)
    fitness_matrix.sort(key=lambda f: -f.fitness)

    # Greedy assignment: take the best fit, remove vessel and route
    assigned_vessels: set[str] = set()
    assigned_routes: set[str] = set()
    assignments: list[FleetAssignment] = []

    for fit in fitness_matrix:
        if fit.vessel_name in assigned_vessels:
            continue
        if fit.route_id in assigned_routes:
            continue

        vessel_data = VESSELS.get(fit.vessel_name, {})
        is_primary = fit.route_id in vessel_data.get("routes", [])

        assignments.append(FleetAssignment(
            vessel_name=fit.vessel_name,
            route_id=fit.route_id,
            risk_score=fit.risk_score,
            band=fit.band,
            cancel_probability=fit.cancel_probability,
            is_primary=is_primary,
        ))
        assigned_vessels.add(fit.vessel_name)
        assigned_routes.add(fit.route_id)

    unassigned_routes = [r for r in routes_to_cover if r not in assigned_routes]
    unassigned_vessels = [v for v in available_vessels if v not in assigned_vessels]

    scores = [a.risk_score for a in assignments]
    avg_risk = sum(scores) / len(scores) if scores else 0
    max_risk = max(scores) if scores else 0

    return AllocationResult(
        assignments=assignments,
        unassigned_routes=unassigned_routes,
        unassigned_vessels=unassigned_vessels,
        fleet_risk_avg=avg_risk,
        fleet_risk_max=max_risk,
    )
