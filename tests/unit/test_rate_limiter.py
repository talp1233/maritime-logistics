"""
Unit tests for rate limiter.
"""

import time
import threading
from src.utils.rate_limiter import RateLimiter


class TestRateLimiter:
    def test_acquire_within_limit(self):
        limiter = RateLimiter(max_calls=10, period_seconds=60)
        assert limiter.acquire() is True
        assert limiter.remaining == 9

    def test_exhaust_limit(self):
        limiter = RateLimiter(max_calls=3, period_seconds=60)
        assert limiter.acquire() is True
        assert limiter.acquire() is True
        assert limiter.acquire() is True
        # 4th call should timeout quickly
        assert limiter.acquire(timeout=0.1) is False

    def test_remaining_count(self):
        limiter = RateLimiter(max_calls=5, period_seconds=60)
        assert limiter.remaining == 5
        limiter.acquire()
        assert limiter.remaining == 4
        limiter.acquire()
        assert limiter.remaining == 3

    def test_window_expires(self):
        limiter = RateLimiter(max_calls=2, period_seconds=0.1)
        limiter.acquire()
        limiter.acquire()
        # Wait for window to expire
        time.sleep(0.15)
        assert limiter.acquire() is True

    def test_thread_safety(self):
        limiter = RateLimiter(max_calls=50, period_seconds=60)
        results = []

        def worker():
            r = limiter.acquire(timeout=5)
            results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(results) == 50  # All should succeed
        assert limiter.remaining == 0
