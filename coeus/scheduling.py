"""One paced request lane per provider host, shared by all workers in a run."""

import asyncio
import math
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime


def retry_delay(header: str | None, attempt: int) -> float:
    if header:
        try:
            value = float(header)
            if math.isfinite(value):
                return max(0, value)
        except ValueError:
            try:
                stamp = parsedate_to_datetime(header)
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=UTC)
                return max(0, (stamp - datetime.now(UTC)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
    return min(60, 5 * 2**attempt)


class RequestLane:
    def __init__(self, interval: float):
        self.lock = asyncio.Lock()
        self.interval = interval
        self.last_start = 0.0
        self.penalty = 0.0

    async def pace(self):
        delay = self.last_start + max(self.interval, self.penalty) - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        self.last_start = time.monotonic()

    async def cooldown(self, response, attempt, emit):
        delay = retry_delay(response.headers.get("retry-after"), attempt)
        self.penalty = min(30, max(2, self.penalty * 2))
        emit(f"Rate limit: waiting {delay:.0f}s; all workers sharing this provider are paused")
        # Deliberately hold the lane lock. No worker bypasses Retry-After.
        await asyncio.sleep(delay)
        emit("Rate-limit pause finished; resuming provider requests")
