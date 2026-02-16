"""
Sailing ban prediction service.

Uses weather forecast data + Greek Coast Guard thresholds to predict
whether a route is likely to face a sailing ban (απαγορευτικό απόπλου).
"""

from src.config.constants import (
    BEAUFORT_SCALE,
    SAILING_BAN_THRESHOLDS,
    VESSEL_TYPES,
    ROUTES,
    PORTS,
    knots_to_beaufort,
    kmh_to_knots,
)


# Wave height thresholds (meters) — supplementary to wind-based Beaufort rules
WAVE_THRESHOLDS = {
    "high_speed": 2.5,    # high-speed craft very sensitive to waves
    "conventional": 5.0,  # conventional ferries handle more
    "small_craft": 2.0,   # small vessels
}


class SailingBanChecker:
    """
    Checks whether sailing conditions exceed ban thresholds for a given
    vessel type and route.
    """

    def check_conditions(
        self,
        wind_speed_knots: float,
        wave_height_m: float,
        vessel_type: str = "CONVENTIONAL",
    ) -> dict:
        """
        Evaluate sailing conditions against ban thresholds.

        Args:
            wind_speed_knots: Wind speed in knots
            wave_height_m: Significant wave height in meters
            vessel_type: One of VESSEL_TYPES keys

        Returns:
            dict with prediction status and details
        """
        category = VESSEL_TYPES.get(vessel_type, "conventional")
        beaufort = knots_to_beaufort(wind_speed_knots)
        bf_threshold = SAILING_BAN_THRESHOLDS[category]
        wave_threshold = WAVE_THRESHOLDS[category]

        # Determine status
        if beaufort >= bf_threshold:
            status = "BAN_LIKELY"
            reason = (
                f"Wind Beaufort {beaufort} >= threshold {bf_threshold} "
                f"for {category} vessels"
            )
        elif wave_height_m >= wave_threshold:
            status = "BAN_LIKELY"
            reason = (
                f"Wave height {wave_height_m:.1f}m >= threshold "
                f"{wave_threshold:.1f}m for {category} vessels"
            )
        elif beaufort >= bf_threshold - 1 or wave_height_m >= wave_threshold * 0.8:
            status = "AT_RISK"
            reason = "Conditions approaching ban thresholds"
        else:
            status = "CLEAR"
            reason = "Conditions within safe limits"

        return {
            "status": status,
            "reason": reason,
            "beaufort": beaufort,
            "wind_speed_knots": wind_speed_knots,
            "wave_height_m": wave_height_m,
            "vessel_category": category,
            "bf_threshold": bf_threshold,
            "wave_threshold": wave_threshold,
        }

    def check_route(
        self,
        route_id: str,
        weather_data: dict,
        vessel_type: str = "CONVENTIONAL",
    ) -> dict:
        """
        Check conditions for a specific route using forecast data.

        Args:
            route_id: Route key from ROUTES (e.g. "PIR-MYK")
            weather_data: Output from fetch_route_conditions()
            vessel_type: Vessel type key

        Returns:
            dict with hourly predictions for the route
        """
        route = ROUTES.get(route_id)
        if not route:
            raise ValueError(f"Unknown route: {route_id}")

        # Use midpoint data as most representative of open-sea conditions
        midpoint = weather_data.get("midpoint", {})
        marine = midpoint.get("marine", {})
        weather = midpoint.get("weather", {})

        marine_hourly = marine.get("hourly", {})
        weather_hourly = weather.get("hourly", {})

        times = weather_hourly.get("time", [])
        wind_speeds = weather_hourly.get("wind_speed_10m", [])
        wave_heights = marine_hourly.get("wave_height", [])

        hourly_predictions = []
        for i, time_str in enumerate(times):
            wind = wind_speeds[i] if i < len(wind_speeds) else 0
            wave = wave_heights[i] if i < len(wave_heights) else 0

            # Handle None values from API
            wind = wind if wind is not None else 0
            wave = wave if wave is not None else 0

            prediction = self.check_conditions(wind, wave, vessel_type)
            prediction["time"] = time_str
            hourly_predictions.append(prediction)

        # Summary: worst status across all hours
        statuses = [p["status"] for p in hourly_predictions]
        if "BAN_LIKELY" in statuses:
            overall = "BAN_LIKELY"
        elif "AT_RISK" in statuses:
            overall = "AT_RISK"
        else:
            overall = "CLEAR"

        return {
            "route_id": route_id,
            "route_name": f"{route['origin']} → {route['destination']}",
            "vessel_type": vessel_type,
            "overall_status": overall,
            "hourly": hourly_predictions,
        }
