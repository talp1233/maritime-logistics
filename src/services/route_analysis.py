"""
Advanced route analysis — wind angle, multi-point sampling, port sheltering.

Adds geographic intelligence to sailing ban predictions beyond simple
wind speed / wave height thresholds.
"""

import math

from src.config.constants import ROUTES, PORTS, knots_to_beaufort
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Route heading calculation
# ---------------------------------------------------------------------------

def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the initial bearing (heading) from point 1 to point 2.

    Returns degrees (0-360), where 0 = North, 90 = East.
    """
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)

    x = math.sin(dlon) * math.cos(lat2_r)
    y = (math.cos(lat1_r) * math.sin(lat2_r) -
         math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon))

    bearing = math.degrees(math.atan2(x, y))
    return bearing % 360


def wind_angle_to_route(wind_direction: float, route_bearing: float) -> float:
    """
    Calculate the relative angle between wind and route heading.

    Returns 0-180 degrees:
      0   = headwind (worst for waves hitting bow)
      90  = beam wind (worst for rolling)
      180 = tailwind (safest for following seas)
    """
    diff = abs(wind_direction - route_bearing) % 360
    if diff > 180:
        diff = 360 - diff
    return diff


def wind_exposure_factor(wind_angle: float) -> float:
    """
    Calculate a risk multiplier based on wind angle relative to route.

    Beam winds (80-100 degrees) are most dangerous for rolling.
    Headwinds reduce effective speed and increase wave impact.
    Tailwinds are safest.

    Returns a factor 0.7 (safest) to 1.3 (most dangerous).
    """
    if 70 <= wind_angle <= 110:
        return 1.3   # Beam seas — worst for rolling
    elif wind_angle <= 30:
        return 1.15  # Headwind — wave slamming
    elif wind_angle >= 150:
        return 0.7   # Following seas — safest
    else:
        return 1.0   # Quartering


# ---------------------------------------------------------------------------
# Multi-point route sampling
# ---------------------------------------------------------------------------

def interpolate_route_points(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    n_points: int = 5,
) -> list[tuple[float, float]]:
    """
    Generate evenly spaced points along a great-circle route.

    Args:
        lat1, lon1: Origin coordinates
        lat2, lon2: Destination coordinates
        n_points: Number of intermediate points (excluding endpoints)

    Returns:
        List of (lat, lon) tuples including both endpoints
    """
    points = [(lat1, lon1)]
    for i in range(1, n_points + 1):
        fraction = i / (n_points + 1)
        lat = lat1 + fraction * (lat2 - lat1)
        lon = lon1 + fraction * (lon2 - lon1)
        points.append((lat, lon))
    points.append((lat2, lon2))
    return points


def get_route_sample_points(route_id: str, interval_nm: float = 20.0) -> list[dict]:
    """
    Generate sample points along a route based on distance.

    Args:
        route_id: Route key (e.g. "PIR-MYK")
        interval_nm: Spacing in nautical miles

    Returns:
        List of dicts with lat, lon, name, and distance_from_origin
    """
    route = ROUTES.get(route_id)
    if not route:
        raise ValueError(f"Unknown route: {route_id}")

    origin_code, dest_code = route_id.split("-")
    origin = PORTS[origin_code]
    dest = PORTS[dest_code]

    total_nm = route["distance_nm"]
    n_intermediate = max(1, int(total_nm / interval_nm) - 1)

    raw_points = interpolate_route_points(
        origin["lat"], origin["lon"],
        dest["lat"], dest["lon"],
        n_points=n_intermediate,
    )

    result = []
    for i, (lat, lon) in enumerate(raw_points):
        if i == 0:
            name = f"{origin['name']} (departure)"
            dist = 0
        elif i == len(raw_points) - 1:
            name = f"{dest['name']} (arrival)"
            dist = total_nm
        else:
            dist = round(total_nm * i / (len(raw_points) - 1), 1)
            name = f"Waypoint {i} ({dist}nm)"

        result.append({
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "name": name,
            "distance_nm": dist,
        })

    return result


# ---------------------------------------------------------------------------
# Port sheltering analysis
# ---------------------------------------------------------------------------

# Ports with known wind shelter from specific directions
# angle_range: (min_degrees, max_degrees) of winds that the port is sheltered from
PORT_SHELTER = {
    "PIR": {"sheltered_from": (180, 300), "description": "Sheltered from S/SW/W by Salamis & mainland"},
    "RAF": {"sheltered_from": (270, 360), "description": "Sheltered from W/NW by Attica coast"},
    "HER": {"sheltered_from": (150, 270), "description": "Sheltered from S/SW by Crete coast"},
    "CHN": {"sheltered_from": (150, 300), "description": "Souda Bay sheltered from S/SW/W"},
    "AND": {"sheltered_from": (270, 45), "description": "Gavrio sheltered from NW/N winds"},
    "SAN": {"sheltered_from": (0, 0), "description": "Athinios exposed — minimal shelter"},
    "MYK": {"sheltered_from": (90, 180), "description": "Partly sheltered from E/SE"},
    "NAX": {"sheltered_from": (180, 270), "description": "Naxos town sheltered from S/SW"},
    "SYR": {"sheltered_from": (180, 300), "description": "Ermoupoli sheltered from S/W"},
    "TIN": {"sheltered_from": (180, 300), "description": "Partly sheltered from S/W"},
    "CHI": {"sheltered_from": (270, 360), "description": "Partly sheltered from W/NW"},
    "MIT": {"sheltered_from": (0, 90), "description": "Adamas sheltered from N/NE"},
    "LAV": {"sheltered_from": (270, 360), "description": "Sheltered from W/NW"},
}


def is_port_sheltered(port_code: str, wind_direction: float) -> bool:
    """
    Check if a port is sheltered from the given wind direction.

    Args:
        port_code: Port code (e.g. "PIR")
        wind_direction: Wind direction in degrees (0=N, 90=E)

    Returns:
        True if port is sheltered from this wind direction
    """
    shelter = PORT_SHELTER.get(port_code)
    if not shelter:
        return False

    lo, hi = shelter["sheltered_from"]
    if lo == hi == 0:
        return False  # No shelter

    if lo <= hi:
        return lo <= wind_direction <= hi
    else:
        # Wraps around 360 (e.g. 270-45)
        return wind_direction >= lo or wind_direction <= hi


def port_shelter_factor(port_code: str, wind_direction: float) -> float:
    """
    Return a risk reduction factor based on port shelter.

    0.6 = well sheltered (40% risk reduction)
    1.0 = no shelter effect
    """
    if is_port_sheltered(port_code, wind_direction):
        return 0.6
    return 1.0


# ---------------------------------------------------------------------------
# Route risk analysis (combines all factors)
# ---------------------------------------------------------------------------

def analyze_route_risk(
    route_id: str,
    wind_speed_knots: float,
    wind_direction: float,
    wave_height_m: float,
    vessel_type: str = "CONVENTIONAL",
) -> dict:
    """
    Comprehensive route risk analysis combining all geographic factors.

    Returns detailed risk breakdown including wind angle, shelter,
    and adjusted risk level.
    """
    route = ROUTES.get(route_id)
    if not route:
        raise ValueError(f"Unknown route: {route_id}")

    origin_code, dest_code = route_id.split("-")
    origin = PORTS[origin_code]
    dest = PORTS[dest_code]

    # Route bearing
    bearing = calculate_bearing(
        origin["lat"], origin["lon"],
        dest["lat"], dest["lon"],
    )

    # Wind angle analysis
    wind_angle = wind_angle_to_route(wind_direction, bearing)
    exposure = wind_exposure_factor(wind_angle)

    # Classify wind angle
    if wind_angle <= 30:
        wind_type = "headwind"
    elif wind_angle <= 60:
        wind_type = "bow_quarter"
    elif wind_angle <= 120:
        wind_type = "beam"
    elif wind_angle <= 150:
        wind_type = "stern_quarter"
    else:
        wind_type = "following"

    # Port shelter
    origin_shelter = port_shelter_factor(origin_code, wind_direction)
    dest_shelter = port_shelter_factor(dest_code, wind_direction)

    # Adjusted effective wind (accounting for angle and shelter)
    effective_wind = wind_speed_knots * exposure
    effective_beaufort = knots_to_beaufort(effective_wind)

    # Sample points
    sample_points = get_route_sample_points(route_id)

    return {
        "route_id": route_id,
        "route_bearing": round(bearing, 1),
        "distance_nm": route["distance_nm"],
        "sea_area": route["sea_area"],
        "exposed": route.get("exposed", True),

        "wind": {
            "speed_knots": wind_speed_knots,
            "direction": wind_direction,
            "angle_to_route": round(wind_angle, 1),
            "wind_type": wind_type,
            "exposure_factor": exposure,
            "effective_wind_knots": round(effective_wind, 1),
            "effective_beaufort": effective_beaufort,
        },

        "shelter": {
            "origin": {
                "port": origin_code,
                "sheltered": origin_shelter < 1.0,
                "factor": origin_shelter,
                "info": PORT_SHELTER.get(origin_code, {}).get("description", ""),
            },
            "destination": {
                "port": dest_code,
                "sheltered": dest_shelter < 1.0,
                "factor": dest_shelter,
                "info": PORT_SHELTER.get(dest_code, {}).get("description", ""),
            },
        },

        "sample_points": sample_points,
        "wave_height_m": wave_height_m,
    }
