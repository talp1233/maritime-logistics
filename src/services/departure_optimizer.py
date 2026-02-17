"""
Departure window optimizer for Greek ferry operations.

Given a scheduled departure and hourly risk scores, finds the nearest
safer window within a ±90-minute range.  The goal is to prevent
cancellations by shifting departures into lower-risk slots.

Operational constraints:
  - Maximum shift: 90 minutes (configurable)
  - Shift must reduce risk score by at least a meaningful margin
  - We prefer the closest safe window (minimise passenger disruption)
  - Night departures (00:00-05:00) are penalised unless already scheduled
"""

from __future__ import annotations

from dataclasses import dataclass

from src.services.risk_scorer import RiskResult, compute_risk, score_to_band
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum risk reduction (in score points) to justify a departure shift
MIN_IMPROVEMENT = 8.0

# Maximum departure shift in minutes
MAX_SHIFT_MINUTES = 90

# Night penalty: departures shifted into 00:00-05:00 get penalised
NIGHT_HOURS = range(0, 5)
NIGHT_PENALTY = 10.0


@dataclass
class DepartureWindow:
    """A candidate departure window."""
    time: str              # ISO timestamp
    hour_index: int        # index into hourly array
    risk_score: float
    band: str
    cancel_probability: float
    delay_probability: float
    shift_minutes: int     # positive = later, negative = earlier
    improvement: float     # risk score reduction vs scheduled


@dataclass
class OptimizationResult:
    """Output of departure optimization."""
    scheduled_time: str
    scheduled_risk: float
    scheduled_band: str
    recommendation: str       # "keep" | "shift" | "cancel_likely"
    best_window: DepartureWindow | None
    alternatives: list[DepartureWindow]
    reason: str

    def to_dict(self) -> dict:
        result = {
            "scheduled_time": self.scheduled_time,
            "scheduled_risk_score": round(self.scheduled_risk, 1),
            "scheduled_band": self.scheduled_band,
            "recommendation": self.recommendation,
            "reason": self.reason,
        }
        if self.best_window:
            result["suggested_departure"] = {
                "time": self.best_window.time,
                "risk_score": round(self.best_window.risk_score, 1),
                "band": self.best_window.band,
                "cancel_probability": round(self.best_window.cancel_probability, 3),
                "shift_minutes": self.best_window.shift_minutes,
                "improvement": round(self.best_window.improvement, 1),
            }
        result["alternatives"] = [
            {
                "time": w.time,
                "risk_score": round(w.risk_score, 1),
                "band": w.band,
                "shift_minutes": w.shift_minutes,
                "improvement": round(w.improvement, 1),
            }
            for w in self.alternatives
        ]
        return result


def find_optimal_departure(
    hourly_scores: list[dict],
    scheduled_hour_index: int,
    max_shift_minutes: int = MAX_SHIFT_MINUTES,
) -> OptimizationResult:
    """
    Find the best departure window near the scheduled time.

    Args:
        hourly_scores: List of hourly risk dicts from score_route()["hourly"]
        scheduled_hour_index: Index of the scheduled departure hour
        max_shift_minutes: Maximum shift in minutes (default 90)

    Returns:
        OptimizationResult with recommendation and alternatives.
    """
    if not hourly_scores or scheduled_hour_index >= len(hourly_scores):
        return OptimizationResult(
            scheduled_time="",
            scheduled_risk=0,
            scheduled_band="CLEAR",
            recommendation="keep",
            best_window=None,
            alternatives=[],
            reason="No forecast data available",
        )

    scheduled = hourly_scores[scheduled_hour_index]
    scheduled_risk = scheduled["risk_score"]
    scheduled_band = scheduled["band"]
    scheduled_time = scheduled["time"]

    # If already clear, no need to shift
    if scheduled_risk < 25:
        return OptimizationResult(
            scheduled_time=scheduled_time,
            scheduled_risk=scheduled_risk,
            scheduled_band=scheduled_band,
            recommendation="keep",
            best_window=None,
            alternatives=[],
            reason="Scheduled departure is within safe conditions",
        )

    # Search window: ±max_shift hours (each index = 1 hour)
    max_shift_hours = max_shift_minutes // 60 + 1
    lo = max(0, scheduled_hour_index - max_shift_hours)
    hi = min(len(hourly_scores), scheduled_hour_index + max_shift_hours + 1)

    candidates: list[DepartureWindow] = []
    for i in range(lo, hi):
        if i == scheduled_hour_index:
            continue

        h = hourly_scores[i]
        shift_minutes = (i - scheduled_hour_index) * 60
        if abs(shift_minutes) > max_shift_minutes:
            continue

        score = h["risk_score"]
        improvement = scheduled_risk - score

        # Apply night penalty
        try:
            hour_of_day = int(h["time"][11:13])
        except (ValueError, IndexError):
            hour_of_day = 12
        if hour_of_day in NIGHT_HOURS:
            score += NIGHT_PENALTY
            improvement = scheduled_risk - score

        if improvement >= MIN_IMPROVEMENT:
            candidates.append(DepartureWindow(
                time=h["time"],
                hour_index=i,
                risk_score=score,
                band=score_to_band(score),
                cancel_probability=h["cancel_probability"],
                delay_probability=h["delay_probability"],
                shift_minutes=shift_minutes,
                improvement=improvement,
            ))

    # Sort by: best improvement first, then smallest shift
    candidates.sort(key=lambda w: (-w.improvement, abs(w.shift_minutes)))

    if not candidates:
        # No better window found
        if scheduled_risk >= 75:
            return OptimizationResult(
                scheduled_time=scheduled_time,
                scheduled_risk=scheduled_risk,
                scheduled_band=scheduled_band,
                recommendation="cancel_likely",
                best_window=None,
                alternatives=[],
                reason=f"No safer window within ±{max_shift_minutes}min. "
                       f"Risk score {scheduled_risk:.0f} — cancellation likely.",
            )
        return OptimizationResult(
            scheduled_time=scheduled_time,
            scheduled_risk=scheduled_risk,
            scheduled_band=scheduled_band,
            recommendation="keep",
            best_window=None,
            alternatives=[],
            reason=f"No significantly safer window found within ±{max_shift_minutes}min",
        )

    best = candidates[0]
    return OptimizationResult(
        scheduled_time=scheduled_time,
        scheduled_risk=scheduled_risk,
        scheduled_band=scheduled_band,
        recommendation="shift",
        best_window=best,
        alternatives=candidates[1:4],  # top 3 alternatives
        reason=f"Shifting {best.shift_minutes:+d}min reduces risk by "
               f"{best.improvement:.0f} points ({scheduled_band} → {best.band})",
    )


def optimize_route_departures(
    hourly_scores: list[dict],
    departure_hours: list[int] | None = None,
) -> list[OptimizationResult]:
    """
    Optimize all scheduled departures for a route.

    Args:
        hourly_scores: Full hourly risk breakdown from score_route()
        departure_hours: Hours of day with scheduled departures
                         (default: [7, 10, 14, 17] — typical Greek ferry schedule)

    Returns:
        List of OptimizationResult for each scheduled departure.
    """
    if departure_hours is None:
        departure_hours = [7, 10, 14, 17]

    results = []
    for target_hour in departure_hours:
        # Find the first hourly entry matching this hour
        matched_index = None
        for i, h in enumerate(hourly_scores):
            try:
                h_hour = int(h["time"][11:13])
            except (ValueError, IndexError):
                continue
            if h_hour == target_hour:
                matched_index = i
                break

        if matched_index is not None:
            results.append(find_optimal_departure(hourly_scores, matched_index))

    return results
