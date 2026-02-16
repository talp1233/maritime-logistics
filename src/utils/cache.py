"""
Simple in-memory TTL cache for API responses.

Prevents duplicate API calls for the same location within a short window.
"""

import time
import hashlib
import json


class TTLCache:
    """Time-based cache with configurable TTL (default 10 minutes)."""

    def __init__(self, ttl_seconds: int = 600):
        self.ttl = ttl_seconds
        self._store: dict[str, tuple[float, any]] = {}

    def _make_key(self, *args, **kwargs) -> str:
        raw = json.dumps({"a": args, "k": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, key: str):
        """Get value if present and not expired. Returns None otherwise."""
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.time() - ts > self.ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value):
        """Store a value with current timestamp."""
        self._store[key] = (time.time(), value)

    def clear(self):
        """Clear all cached entries."""
        self._store.clear()

    def cleanup(self):
        """Remove all expired entries."""
        now = time.time()
        expired = [k for k, (ts, _) in self._store.items() if now - ts > self.ttl]
        for k in expired:
            del self._store[k]

    @property
    def size(self) -> int:
        return len(self._store)


# Global cache instance shared across the application
api_cache = TTLCache(ttl_seconds=600)
