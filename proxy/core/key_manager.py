"""
API key rotation manager with cooldown/blacklisting.

Supports multiple upstream API keys with round-robin rotation.
When a key hits rate limits or quota exhaustion, it gets blacklisted
for a configurable cooldown period, then automatically re-enabled.

If ALL keys are blacklisted, the proxy passes the error through
to the client so they know the upstream is exhausted.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger("proxy.key_manager")

# ── Patterns that indicate a key is exhausted / rate-limited ────────
# These match common error messages from OpenAI-compatible APIs.
_QUOTA_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"rate.?limit",
        r"quota.*exceeded",
        r"too.?many.?requests",
        r"insufficient.?quota",
        r"billing.*limit",
        r"exceeded.*limit",
        r"tokens?.?per.?min",
        r"requests?.?per.?min",
        r"capacity",
        r"overloaded",
        r"resource.?exhausted",
        r"limit.*reached",
        r"try.?again.?later",
        r"throttl",
    ]
]

# HTTP status codes that indicate rate limiting / quota issues
_QUOTA_STATUS_CODES: set[int] = {429, 503}


@dataclass
class _KeyState:
    """Internal state for a single API key."""

    key: str
    blacklisted_until: float = 0.0  # unix timestamp when cooldown expires
    total_requests: int = 0
    total_errors: int = 0
    last_used: float = 0.0
    last_error: str = ""
    last_error_time: float = 0.0

    @property
    def is_blacklisted(self) -> bool:
        return time.time() < self.blacklisted_until

    @property
    def masked_key(self) -> str:
        """Show first 12 and last 4 chars for logging."""
        if len(self.key) <= 20:
            return self.key[:8] + "..."
        return self.key[:12] + "..." + self.key[-4:]


class KeyManager:
    """Thread-safe API key rotation manager.

    Usage:
        km = KeyManager(["sk-key1", "sk-key2"], cooldown_seconds=7200)

        # Get next available key
        key = km.get_key()  # returns key or None if all blacklisted

        # Report an error (will auto-detect if it's a quota error)
        km.report_error(key, status_code=429, error_body="rate limit exceeded")

        # Report success
        km.report_success(key)
    """

    def __init__(
        self,
        keys: list[str],
        cooldown_seconds: int = 7200,
    ) -> None:
        if not keys:
            raise ValueError("At least one API key is required")

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_keys: list[str] = []
        for k in keys:
            k = k.strip()
            if k and k not in seen:
                seen.add(k)
                unique_keys.append(k)

        if not unique_keys:
            raise ValueError("No valid API keys provided (all empty or duplicates)")

        self._keys: list[_KeyState] = [_KeyState(key=k) for k in unique_keys]
        self._cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()
        self._round_robin_index = 0

        logger.info(
            "KeyManager initialized with %d key(s), cooldown=%ds",
            len(self._keys),
            cooldown_seconds,
        )
        for ks in self._keys:
            logger.info("  Key: %s", ks.masked_key)

    @property
    def key_count(self) -> int:
        """Total number of registered keys."""
        return len(self._keys)

    @property
    def active_key_count(self) -> int:
        """Number of keys not currently blacklisted."""
        with self._lock:
            return sum(1 for ks in self._keys if not ks.is_blacklisted)

    def get_key(self) -> str | None:
        """Get the next available API key using round-robin.

        Returns None if ALL keys are currently blacklisted.
        The caller should then pass through the upstream error.
        """
        with self._lock:
            n = len(self._keys)
            # Try each key starting from the round-robin position
            for _ in range(n):
                ks = self._keys[self._round_robin_index]
                self._round_robin_index = (self._round_robin_index + 1) % n

                if not ks.is_blacklisted:
                    ks.total_requests += 1
                    ks.last_used = time.time()
                    logger.debug(
                        "KeyManager: using key %s (request #%d)",
                        ks.masked_key,
                        ks.total_requests,
                    )
                    return ks.key

            # All keys are blacklisted
            soonest = min(ks.blacklisted_until for ks in self._keys)
            remaining = max(0, soonest - time.time())
            logger.warning(
                "KeyManager: ALL %d keys are blacklisted! Soonest recovery in %.0f seconds",
                n,
                remaining,
            )
            return None

    def report_success(self, key: str) -> None:
        """Report a successful request for a key."""
        with self._lock:
            ks = self._find_key(key)
            if ks is not None:
                ks.last_used = time.time()
                logger.debug("KeyManager: success for key %s", ks.masked_key)

    def report_error(
        self,
        key: str,
        status_code: int,
        error_body: str = "",
        error_message: str = "",
    ) -> None:
        """Report an error for a key. Auto-detects quota/rate-limit errors.

        If the error looks like a quota/rate-limit issue, the key gets
        blacklisted for the configured cooldown period.
        """
        with self._lock:
            ks = self._find_key(key)
            if ks is None:
                logger.warning("KeyManager: report_error for unknown key %s", key[:12] + "...")
                return

            ks.total_errors += 1
            ks.last_error_time = time.time()

            # Combine all text to check against patterns
            check_text = f"{error_body} {error_message}"
            ks.last_error = check_text[:200]

            is_quota_error = self._is_quota_error(status_code, check_text)

            if is_quota_error:
                ks.blacklisted_until = time.time() + self._cooldown_seconds
                remaining_active = sum(1 for k in self._keys if not k.is_blacklisted)
                logger.warning(
                    "KeyManager: BLACKLISTED key %s for %d seconds "
                    "(status=%d, reason=%s). Active keys remaining: %d/%d",
                    ks.masked_key,
                    self._cooldown_seconds,
                    status_code,
                    check_text[:100],
                    remaining_active,
                    len(self._keys),
                )
            else:
                logger.debug(
                    "KeyManager: non-quota error for key %s (status=%d), not blacklisting",
                    ks.masked_key,
                    status_code,
                )

    def force_blacklist(self, key: str, seconds: int | None = None) -> None:
        """Manually blacklist a key for the given duration (or default cooldown)."""
        with self._lock:
            ks = self._find_key(key)
            if ks is None:
                return
            duration = seconds if seconds is not None else self._cooldown_seconds
            ks.blacklisted_until = time.time() + duration
            logger.warning(
                "KeyManager: manually blacklisted key %s for %d seconds",
                ks.masked_key,
                duration,
            )

    def force_unblacklist(self, key: str) -> None:
        """Manually remove a key from the blacklist."""
        with self._lock:
            ks = self._find_key(key)
            if ks is None:
                return
            ks.blacklisted_until = 0.0
            logger.info("KeyManager: manually unblacklisted key %s", ks.masked_key)

    def get_status(self) -> list[dict]:
        """Get status of all keys (for health endpoint / debugging)."""
        with self._lock:
            now = time.time()
            result = []
            for ks in self._keys:
                info: dict = {
                    "key": ks.masked_key,
                    "active": not ks.is_blacklisted,
                    "total_requests": ks.total_requests,
                    "total_errors": ks.total_errors,
                }
                if ks.is_blacklisted:
                    remaining = max(0, ks.blacklisted_until - now)
                    info["blacklisted"] = True
                    info["cooldown_remaining_seconds"] = round(remaining)
                    info["last_error"] = ks.last_error[:100] if ks.last_error else ""
                else:
                    info["blacklisted"] = False

                result.append(info)
            return result

    def _find_key(self, key: str) -> _KeyState | None:
        """Find a key state by its raw key string. Must be called with lock held."""
        for ks in self._keys:
            if ks.key == key:
                return ks
        return None

    @staticmethod
    def _is_quota_error(status_code: int, text: str) -> bool:
        """Determine if an error response indicates rate limiting or quota exhaustion."""
        if status_code in _QUOTA_STATUS_CODES:
            return True

        for pattern in _QUOTA_PATTERNS:
            if pattern.search(text):
                return True

        return False
