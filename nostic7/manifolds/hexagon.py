"""
Hexagon Manifold — State Orchestration (M_⬡)

The hexagon tessellates perfectly — every face shared, no gaps. This
manifold acts as the Epinoetic Dashboard: it tracks all sibling
manifolds, computes system-wide coherence, and broadcasts events.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any


class HexagonManifold:
    """
    State Orchestration Manifold — Epinoetic Dashboard.

    Registers sibling manifolds, polls their `.state` property to build
    a unified snapshot, computes a coherence score, and maintains a
    rolling event log.
    """

    MAX_EVENT_LOG: int = 100

    def __init__(self) -> None:
        self._manifolds: dict[str, Any] = {}
        self._event_log: deque[dict[str, Any]] = deque(maxlen=self.MAX_EVENT_LOG)
        self._last_snapshot: dict[str, Any] = {}
        self._sync_count: int = 0

    # ------------------------------------------------------------------
    # Manifold registration
    # ------------------------------------------------------------------

    def register_manifold(self, name: str, manifold: Any) -> None:
        """Register a manifold under the given *name* for tracking."""
        self._manifolds[name] = manifold
        self._log_event("register", {"manifold": name})

    # ------------------------------------------------------------------
    # State synchronisation
    # ------------------------------------------------------------------

    def sync(self) -> dict[str, Any]:
        """
        Poll `.state` on all registered manifolds.

        Returns a unified snapshot keyed by manifold name. Any manifold
        lacking a `.state` attribute gets a default entry.
        """
        snapshot: dict[str, Any] = {}
        for name, manifold in self._manifolds.items():
            try:
                snapshot[name] = manifold.state
            except Exception as exc:  # noqa: BLE001
                snapshot[name] = {"error": str(exc), "healthy": False}

        self._last_snapshot = snapshot
        self._sync_count += 1
        self._log_event("sync", {"manifold_count": len(snapshot)})
        return snapshot

    # ------------------------------------------------------------------
    # Coherence
    # ------------------------------------------------------------------

    def coherence_score(self) -> float:
        """
        Return a 0.0–1.0 coherence score.

        A manifold is considered 'healthy' if its state dict contains
        ``"healthy": True``. Score = healthy_count / total_count.
        """
        if not self._manifolds:
            return 1.0  # vacuously coherent

        snapshot = self._last_snapshot or self.sync()
        total = len(snapshot)
        healthy = sum(
            1 for s in snapshot.values() if isinstance(s, dict) and s.get("healthy", False)
        )
        return round(healthy / total, 4)

    # ------------------------------------------------------------------
    # Event broadcasting
    # ------------------------------------------------------------------

    def broadcast(self, event: str, data: dict[str, Any]) -> None:
        """Log *event* with *data* to the internal event log."""
        self._log_event(event, data)

    def _log_event(self, event: str, data: dict[str, Any]) -> None:
        entry = {
            "timestamp": time.time(),
            "event": event,
            "data": data,
        }
        self._event_log.append(entry)

    def get_event_log(self, n: int = 20) -> list[dict[str, Any]]:
        """Return the last *n* events."""
        events = list(self._event_log)
        return events[-n:]

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        return {
            "manifold": "HexagonManifold",
            "registered_manifolds": list(self._manifolds.keys()),
            "sync_count": self._sync_count,
            "coherence_score": self.coherence_score(),
            "event_log_size": len(self._event_log),
            "healthy": True,
        }

    def __repr__(self) -> str:  # pragma: no cover
        s = self.state
        return (
            f"<HexagonManifold manifolds={len(s['registered_manifolds'])} "
            f"coherence={s['coherence_score']:.3f}>"
        )
