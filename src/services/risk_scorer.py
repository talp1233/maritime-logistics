"""
Probabilistic risk scoring engine for Greek ferry operations.

Computes a composite risk score (0-100) from sea-state conditions,
route geometry, and vessel characteristics.  Unlike the binary
rule-based checker, this produces a continuous risk surface that
maps to:

    - Cancellation probability  P(cancel)
    - Delay probability         P(delay)
    - Operational band          CLEAR / MONITOR / AT_RISK / HIGH / CRITICAL

The physics:
    RawRisk  = SeaState(wind, waves) * GeoModifier * GustModifier
    Score    = clamp(RawRisk * VesselModifier, 0, 100)

where SeaState blends wind risk (the APAGEN driver) and wave risk
(the comfort / delay driver), and the modifiers capture route-specific
exposure, gust instability, and vessel fitness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.config.constants import (
    ROUTES,
    PORTS,
    VESSELS,
    VESSEL_TYPES,
    SAILING_BAN_THRESHOLDS,
    knots_to_beaufort,
)
from src.services.route_analysis import (
    calculate_bearing,
    wind_angle_to_route,
    wind_exposure_factor,
    port_shelter_factor,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Gravity constant for wave steepness calculation ────────────────────
GRAVITY = 9.81  # m/s²

# ── Wave thresholds (meters) by vessel category ───────────────────────
WAVE_THRESHOLDS: dict[str, float] = {
    "high_speed": 2.5,
    "conventional": 5.0,
    "small_craft": 2.0,
}


# ── Result dataclass ──────────────────────────────────────────────────

@dataclass
class RiskResult:
    """Output of a single risk assessment."""

    score: float                       # 0-100
    band: str                          # CLEAR / MONITOR / AT_RISK / HIGH / CRITICAL
    cancel_probability: float          # 0.0-1.0
    delay_probability: float           # 0.0-1.0
    components: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "risk_score": round(self.score, 1),
            "band": self.band,
            "cancel_probability": round(self.cancel_probability, 3),
            "delay_probability": round(self.delay_probability, 3),
            "components": self.components,
        }


# ── Score-to-band mapping ────────────────────────────────────────────

def score_to_band(score: float) -> str:
    """Map a 0-100 risk score to an operational band."""
    if score < 25:
        return "CLEAR"
    if score < 45:
        return "MONITOR"
    if score < 65:
        return "AT_RISK"
    if score < 82:
        return "HIGH"
    return "CRITICAL"


# ── Sigmoid helper ────────────────────────────────────────────────────

def _sigmoid(x: float, midpoint: float, k: float) -> float:
    """Standard logistic sigmoid scaled to [0, 1]."""
    z = -k * (x - midpoint)
    # Clamp to avoid overflow
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(z))


# ── Core component functions ─────────────────────────────────────────

def _wind_risk(
    wind_speed_knots: float,
    bf_threshold: int,
    wind_direction: float | None = None,
    route_bearing: float | None = None,
) -> float:
    """
    Wind risk component.

    Normalised so that 1.0 ≈ "at Beaufort threshold".
    Direction factor amplifies beam winds (1.3×) and dampens
    following winds (0.7×).
    """
    beaufort = knots_to_beaufort(wind_speed_knots)
    bf_ratio = beaufort / bf_threshold if bf_threshold > 0 else 0.0

    direction_factor = 1.0
    if wind_direction is not None and route_bearing is not None:
        angle = wind_angle_to_route(wind_direction, route_bearing)
        direction_factor = wind_exposure_factor(angle)

    return bf_ratio * direction_factor


def _wave_risk(
    wave_height_m: float,
    wave_period_s: float,
    swell_height_m: float,
    wave_threshold: float,
) -> float:
    """
    Wave risk component.

    Accounts for:
      1. Height relative to vessel threshold
      2. Wave steepness (short-period chop is worse than long swell)
      3. Combined sea state (wind waves + swell, ISO 19901-1)
    """
    if wave_threshold <= 0:
        return 0.0

    # Combined significant wave height  (wind-sea + swell)
    combined_hs = math.sqrt(wave_height_m ** 2 + swell_height_m ** 2)
    wave_ratio = combined_hs / wave_threshold

    # Wave steepness:  S = 2π·Hs / (g·Tp²)
    # S > ~0.04  → steep, uncomfortable
    # S > ~0.07  → risk of green water / breaking
    steepness_penalty = 1.0
    if wave_period_s > 0:
        steepness = (2.0 * math.pi * wave_height_m) / (GRAVITY * wave_period_s ** 2)
        steepness_penalty = 1.0 + 3.0 * max(0.0, steepness - 0.04)

    return wave_ratio * steepness_penalty


def _geo_modifier(
    route_id: str | None,
    wind_direction: float | None,
) -> float:
    """
    Geographic modifier (0.6–1.3).

    Exposed open-sea routes get amplified;
    sheltered routes / ports get dampened.
    """
    if route_id is None:
        return 1.0

    route = ROUTES.get(route_id)
    if not route:
        return 1.0

    # Base exposure
    exposure = 1.0 if route.get("exposed", True) else 0.7

    # Port shelter (average of origin and destination)
    if wind_direction is not None:
        origin_code, dest_code = route_id.split("-")
        shelter_o = port_shelter_factor(origin_code, wind_direction)
        shelter_d = port_shelter_factor(dest_code, wind_direction)
        avg_shelter = (shelter_o + shelter_d) / 2.0
    else:
        avg_shelter = 1.0

    # Longer routes cross more open water → slightly higher risk
    distance_nm = route.get("distance_nm", 80)
    distance_factor = 1.0 + 0.1 * max(0, distance_nm - 80) / 100.0  # up to +0.1

    modifier = exposure * avg_shelter * distance_factor
    # Clamp to [0.5, 1.3]
    return max(0.5, min(1.3, modifier))


def _gust_modifier(wind_speed_knots: float, gust_knots: float) -> float:
    """
    Gust instability modifier (1.0–1.3).

    High gust-to-sustained ratio means turbulent, unpredictable
    conditions — more dangerous than a steady breeze at the
    same sustained speed.
    """
    if wind_speed_knots <= 0:
        return 1.0
    ratio = gust_knots / wind_speed_knots
    # ratio ~1.3 is normal;  >1.5 is gusty;  >2.0 is severe
    penalty = max(0.0, ratio - 1.3) * 0.75
    return min(1.3, 1.0 + penalty)


def _vessel_modifier(
    vessel_name: str | None,
    vessel_type: str,
    bf_threshold: int,
) -> float:
    """
    Vessel fitness modifier (0.7–1.3).

    Larger, more capable vessels get dampened risk.
    Smaller vessels with lower thresholds get amplified.
    """
    # If we have vessel-specific data, use its actual threshold
    if vessel_name and vessel_name in VESSELS:
        vessel_data = VESSELS[vessel_name]
        actual_threshold = vessel_data.get("bf_threshold", bf_threshold)
        tonnage = vessel_data.get("gross_tonnage", 10000)
    else:
        actual_threshold = bf_threshold
        tonnage = 10000  # assume mid-range

    # Tonnage effect: larger ships are more stable
    # 5000t → 1.15 (worse),  15000t → 1.0 (baseline),  35000t → 0.8 (better)
    tonnage_factor = 1.0 - 0.15 * (tonnage - 15000) / 20000
    tonnage_factor = max(0.7, min(1.3, tonnage_factor))

    # Threshold headroom: vessel with Bf 10 threshold has more margin than Bf 6
    # Relative to category default
    category = VESSEL_TYPES.get(vessel_type, "conventional")
    default_threshold = SAILING_BAN_THRESHOLDS.get(category, 8)
    headroom = actual_threshold - default_threshold
    threshold_factor = 1.0 - headroom * 0.08  # +1 Bf headroom → ~8% risk reduction

    modifier = tonnage_factor * threshold_factor
    return max(0.7, min(1.3, modifier))


# ── Main scoring function ────────────────────────────────────────────

def compute_risk(
    wind_speed_knots: float,
    wave_height_m: float,
    wave_period_s: float = 5.0,
    wind_gust_knots: float | None = None,
    swell_height_m: float = 0.0,
    wind_direction: float | None = None,
    vessel_type: str = "CONVENTIONAL",
    vessel_name: str | None = None,
    route_id: str | None = None,
) -> RiskResult:
    """
    Compute composite risk score for a single set of conditions.

    Args:
        wind_speed_knots: Sustained wind speed (knots)
        wave_height_m:    Significant wave height (m)
        wave_period_s:    Dominant wave period (s)
        wind_gust_knots:  Gust speed (knots); defaults to 1.3× sustained
        swell_height_m:   Swell component height (m)
        wind_direction:   Meteorological wind direction (degrees, 0=N)
        vessel_type:      CONVENTIONAL / HIGH_SPEED / CATAMARAN / SMALL
        vessel_name:      Specific vessel name (optional, for fitness lookup)
        route_id:         Route ID (optional, for geographic modifiers)

    Returns:
        RiskResult with score, band, probabilities, and component breakdown.
    """
    if wind_gust_knots is None:
        wind_gust_knots = wind_speed_knots * 1.3

    # Resolve vessel category and thresholds
    category = VESSEL_TYPES.get(vessel_type, "conventional")
    bf_threshold = SAILING_BAN_THRESHOLDS[category]
    wave_threshold = WAVE_THRESHOLDS[category]

    # If a specific vessel has override thresholds, use them
    if vessel_name and vessel_name in VESSELS:
        bf_threshold = VESSELS[vessel_name].get("bf_threshold", bf_threshold)
        wave_threshold = VESSELS[vessel_name].get("wave_threshold", wave_threshold)

    # Route bearing for direction-dependent calculations
    route_bearing = None
    if route_id and route_id in ROUTES:
        origin_code, dest_code = route_id.split("-")
        if origin_code in PORTS and dest_code in PORTS:
            origin = PORTS[origin_code]
            dest = PORTS[dest_code]
            route_bearing = calculate_bearing(
                origin["lat"], origin["lon"],
                dest["lat"], dest["lon"],
            )

    # ── Component calculations ────────────────────────────────────
    wind_comp = _wind_risk(wind_speed_knots, bf_threshold, wind_direction, route_bearing)
    wave_comp = _wave_risk(wave_height_m, wave_period_s, swell_height_m, wave_threshold)

    # Base sea-state risk: blend wind (APAGEN driver) and waves (comfort/delay driver)
    sea_state = 0.55 * wind_comp + 0.45 * wave_comp

    geo_mod = _geo_modifier(route_id, wind_direction)
    gust_mod = _gust_modifier(wind_speed_knots, wind_gust_knots)
    vessel_mod = _vessel_modifier(vessel_name, vessel_type, bf_threshold)

    # Combine
    raw = sea_state * geo_mod * gust_mod * vessel_mod

    # Scale to 0-100 via piecewise linear mapping
    # raw ~0.0 → score 0,  raw ~0.5 → score 30,  raw ~1.0 → score 70,  raw ~1.3 → score 100
    score = min(100.0, max(0.0, raw * 77.0))

    band = score_to_band(score)

    # ── Probability mapping ───────────────────────────────────────
    # Cancellation: steep sigmoid centred at score 68
    cancel_prob = _sigmoid(score, midpoint=68.0, k=0.12)

    # Delay: peaks in the 35-65 band, then absorbed by cancellation
    delay_raw = _sigmoid(score, midpoint=42.0, k=0.09)
    delay_prob = delay_raw * (1.0 - cancel_prob)

    components = {
        "wind_risk": round(wind_comp, 3),
        "wave_risk": round(wave_comp, 3),
        "sea_state": round(sea_state, 3),
        "geo_modifier": round(geo_mod, 3),
        "gust_modifier": round(gust_mod, 3),
        "vessel_modifier": round(vessel_mod, 3),
        "raw_risk": round(raw, 3),
        "beaufort": knots_to_beaufort(wind_speed_knots),
        "bf_threshold": bf_threshold,
        "wave_threshold": wave_threshold,
    }

    return RiskResult(
        score=round(score, 1),
        band=band,
        cancel_probability=round(cancel_prob, 3),
        delay_probability=round(delay_prob, 3),
        components=components,
    )


# ── Route-level scoring (hourly) ─────────────────────────────────────

def score_route(
    route_id: str,
    weather_data: dict,
    vessel_type: str = "CONVENTIONAL",
    vessel_name: str | None = None,
) -> dict:
    """
    Score an entire route across all forecast hours.

    Args:
        route_id:     Route key (e.g. "PIR-MYK")
        weather_data: Output from fetch_route_conditions() or demo generator
        vessel_type:  Vessel type key
        vessel_name:  Optional specific vessel name

    Returns:
        dict with hourly risk scores, overall summary, and component breakdown.
    """
    route = ROUTES.get(route_id)
    if not route:
        raise ValueError(f"Unknown route: {route_id}")

    # Use midpoint data — most representative of open-sea conditions
    midpoint = weather_data.get("midpoint", {})
    marine = midpoint.get("marine", {}).get("hourly", {})
    weather = midpoint.get("weather", {}).get("hourly", {})

    times = weather.get("time", [])
    wind_speeds = weather.get("wind_speed_10m", [])
    wind_directions = weather.get("wind_direction_10m", [])
    wind_gusts = weather.get("wind_gusts_10m", [])
    wave_heights = marine.get("wave_height", [])
    wave_periods = marine.get("wave_period", [])
    swell_heights = marine.get("swell_wave_height", [])

    hourly = []
    for i, time_str in enumerate(times):
        wind = (wind_speeds[i] if i < len(wind_speeds) else 0) or 0
        direction = (wind_directions[i] if i < len(wind_directions) else None)
        gust = (wind_gusts[i] if i < len(wind_gusts) else None)
        wave = (wave_heights[i] if i < len(wave_heights) else 0) or 0
        period = (wave_periods[i] if i < len(wave_periods) else 5.0) or 5.0
        swell = (swell_heights[i] if i < len(swell_heights) else 0) or 0

        result = compute_risk(
            wind_speed_knots=wind,
            wave_height_m=wave,
            wave_period_s=period,
            wind_gust_knots=gust,
            swell_height_m=swell,
            wind_direction=direction,
            vessel_type=vessel_type,
            vessel_name=vessel_name,
            route_id=route_id,
        )

        hourly.append({
            "time": time_str,
            **result.to_dict(),
            "wind_speed_knots": round(wind, 1),
            "wave_height_m": round(wave, 1),
            "beaufort": knots_to_beaufort(wind),
        })

    # Aggregates
    scores = [h["risk_score"] for h in hourly]
    max_score = max(scores) if scores else 0
    avg_score = sum(scores) / len(scores) if scores else 0
    max_cancel = max(h["cancel_probability"] for h in hourly) if hourly else 0
    max_delay = max(h["delay_probability"] for h in hourly) if hourly else 0

    return {
        "route_id": route_id,
        "route_name": f"{route['origin']} \u2192 {route['destination']}",
        "distance_nm": route["distance_nm"],
        "sea_area": route["sea_area"],
        "vessel_type": vessel_type,
        "vessel_name": vessel_name,
        "overall_band": score_to_band(max_score),
        "max_risk_score": round(max_score, 1),
        "avg_risk_score": round(avg_score, 1),
        "max_cancel_probability": round(max_cancel, 3),
        "max_delay_probability": round(max_delay, 3),
        "hourly": hourly,
    }
