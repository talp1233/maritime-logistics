"""Tests for TTL cache."""

import time
from src.utils.cache import TTLCache


class TestTTLCache:
    def test_set_and_get(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_missing_key_returns_none(self):
        cache = TTLCache(ttl_seconds=60)
        assert cache.get("nonexistent") is None

    def test_expired_entry_returns_none(self):
        cache = TTLCache(ttl_seconds=0)  # Instant expiry
        cache.set("key1", "value1")
        time.sleep(0.01)
        assert cache.get("key1") is None

    def test_clear(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.size == 2
        cache.clear()
        assert cache.size == 0

    def test_cleanup_removes_expired(self):
        cache = TTLCache(ttl_seconds=0)
        cache.set("a", 1)
        cache.set("b", 2)
        time.sleep(0.01)
        cache.cleanup()
        assert cache.size == 0

    def test_size(self):
        cache = TTLCache(ttl_seconds=60)
        assert cache.size == 0
        cache.set("a", 1)
        assert cache.size == 1
