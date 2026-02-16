"""
Sailing ban prediction service.

Uses weather forecast data + Greek Coast Guard thresholds to predict
whether a route is likely to face a sailing ban.

Supports two prediction modes:
  - Rule-based: deterministic Beaufort/wave threshold checks (always available)
  - ML-enhanced: probabilistic prediction using trained models (optional)
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
from src.utils.logger import get_logger

logger = get_logger(__name__)


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

    Optionally uses ML models for probabilistic predictions alongside
    rule-based checks.
    """

    def __init__(self, use_ml: bool = False):
        self.ml_predictor = None
        if use_ml:
            self._load_ml()

    def _load_ml(self):
        """Try to load trained ML models."""
        try:
            from src.models.ml_predictor import MLPredictor
            predictor = MLPredictor()
            predictor.load()
            self.ml_predictor = predictor
            logger.info("ML models loaded successfully")
        except Exception as e:
            logger.info("ML models not available: %s", e)
            self.ml_predictor = None

    def check_conditions(
        self,
        wind_speed_knots: float,
        wave_height_m: float,
        vessel_type: str = "CONVENTIONAL",
        hour_of_day: int = 12,
        month: int = 6,
    ) -> dict:
        """
        Evaluate sailing conditions against ban thresholds.

        Args:
            wind_speed_knots: Wind speed in knots
            wave_height_m: Significant wave height in meters
            vessel_type: One of VESSEL_TYPES keys
            hour_of_day: Hour (0-23) for ML prediction
            month: Month (1-12) for ML prediction

        Returns:
            dict with prediction status and details
        """
        category = VESSEL_TYPES.get(vessel_type, "conventional")
        beaufort = knots_to_beaufort(wind_speed_knots)
        bf_threshold = SAILING_BAN_THRESHOLDS[category]
        wave_threshold = WAVE_THRESHOLDS[category]

        # Rule-based determination
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

        result = {
            "status": status,
            "reason": reason,
            "beaufort": beaufort,
            "wind_speed_knots": wind_speed_knots,
            "wave_height_m": wave_height_m,
            "vessel_category": category,
            "bf_threshold": bf_threshold,
            "wave_threshold": wave_threshold,
        }

        # ML enhancement (if available)
        if self.ml_predictor is not None:
            try:
                from src.models.ml_predictor import extract_features
                features = extract_features(
                    wind_speed_knots=wind_speed_knots,
                    wave_height_m=wave_height_m,
                    vessel_type=vessel_type,
                    hour_of_day=hour_of_day,
                    month=month,
                )
                ml_result = self.ml_predictor.predict(features)
                result["ml_status"] = ml_result["status"]
                result["ml_cancel_probability"] = ml_result["cancel_probability"]
            except Exception as e:
                logger.debug("ML prediction failed: %s", e)

        return result

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

            # Extract temporal context for ML
            hour = 12
            month = 6
            try:
                hour = int(time_str[11:13])
                month = int(time_str[5:7])
            except (ValueError, IndexError):
                pass

            prediction = self.check_conditions(
                wind, wave, vessel_type, hour_of_day=hour, month=month,
            )
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

        result = {
            "route_id": route_id,
            "route_name": f"{route['origin']} \u2192 {route['destination']}",
            "vessel_type": vessel_type,
            "overall_status": overall,
            "hourly": hourly_predictions,
        }

        # Add ML summary if available
        ml_statuses = [p.get("ml_status") for p in hourly_predictions if "ml_status" in p]
        if ml_statuses:
            if "BAN_LIKELY" in ml_statuses:
                result["ml_overall_status"] = "BAN_LIKELY"
            elif "AT_RISK" in ml_statuses:
                result["ml_overall_status"] = "AT_RISK"
            else:
                result["ml_overall_status"] = "CLEAR"

            probas = [p["ml_cancel_probability"] for p in hourly_predictions if "ml_cancel_probability" in p]
            result["ml_max_cancel_probability"] = max(probas) if probas else 0
            result["ml_avg_cancel_probability"] = round(sum(probas) / len(probas), 3) if probas else 0

        return result
