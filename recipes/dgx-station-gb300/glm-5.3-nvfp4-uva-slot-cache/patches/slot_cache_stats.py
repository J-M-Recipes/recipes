"""Pure helpers for slot-cache telemetry.

The cache hook counts misses and routed expert uses on device. Hit rate must use
observed routed uses as its denominator: speculative verification can route more
than one token in a single engine step, so a fixed top-k denominator is wrong.
"""

from __future__ import annotations


def summarize_window(
    *, delta_misses: int, delta_routes: int, delta_steps: int
) -> dict[str, float] | None:
    """Summarize one monotonic-counter window, or reject an empty/reset window."""
    if delta_misses < 0 or delta_routes <= 0 or delta_steps <= 0:
        return None
    return {
        "hit_rate": 1.0 - (delta_misses / delta_routes),
        "misses_per_step": delta_misses / delta_steps,
        "routes_per_step": delta_routes / delta_steps,
    }
