"""
Simple thread-safe rate limiter for API calls.

Prevents exceeding API rate limits (Open-Meteo allows ~10,000/day free).
"""

import time
import threading

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter.

    Args:
        max_calls: Maximum calls per period
        period_seconds: Length of the rate-limit window
    """

    def __init__(self, max_calls: int = 600, period_seconds: float = 60.0):
        self.max_calls = max_calls
        self.period = period_seconds
        self._lock = threading.Lock()
        self._calls: list[float] = []

    def acquire(self, timeout: float = 30.0) -> bool:
        """
        Wait until a call is allowed, or timeout.

        Returns True if acquired, False if timed out.
        """
        deadline = time.time() + timeout

        while True:
            with self._lock:
                now = time.time()
                # Remove calls outside the current window
                self._calls = [t for t in self._calls if now - t < self.period]

                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return True

            if time.time() >= deadline:
                logger.warning("Rate limiter timeout after %.1fs", timeout)
                return False

            # Wait a fraction of the period before retrying
            time.sleep(self.period / self.max_calls)

    @property
    def remaining(self) -> int:
        """Number of calls remaining in the current window."""
        with self._lock:
            now = time.time()
            active = sum(1 for t in self._calls if now - t < self.period)
            return max(0, self.max_calls - active)


# Global rate limiter: 600 calls per minute (conservative for Open-Meteo)
api_rate_limiter = RateLimiter(max_calls=600, period_seconds=60.0)
