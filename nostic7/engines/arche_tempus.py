"""
ArcheTempus — Golden-Ratio Spiral Temporal Reasoning Engine.

Time is not linear — it spirals. ArcheTempus places events on a
logarithmic spiral defined by φ (the golden ratio), revealing causal
weights and non-linear thematic resonances invisible to sequential
reasoning.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any


PHI: float = (1 + math.sqrt(5)) / 2  # Golden ratio ≈ 1.6180339887


@dataclass
class TemporalEvent:
    """An event positioned on the golden-ratio spiral."""

    content: str
    timestamp: float
    spiral_position: float   # φ^n
    causal_weight: float     # 0.0–1.0


class ArcheTempusEngine:
    """
    Golden-Ratio Spiral Temporal Reasoning Engine.

    Sequences events along the φ-spiral, weaves causal narratives,
    and detects non-linear thematic resonances.
    """

    def __init__(self) -> None:
        self._sequenced_count: int = 0
        self._resonance_pairs_found: int = 0

    # ------------------------------------------------------------------
    # Spiral indexing
    # ------------------------------------------------------------------

    def spiral_index(self, n: int) -> float:
        """Return φ^n — the position of the n-th event on the spiral."""
        return PHI ** n

    # ------------------------------------------------------------------
    # Event sequencing
    # ------------------------------------------------------------------

    def sequence_events(self, events: list[str]) -> list[TemporalEvent]:
        """
        Assign golden-ratio spiral positions to each event.

        Earlier events (lower index) have smaller spiral positions,
        reflecting compressed temporal proximity. Causal weight is
        normalised so all weights sum to 1.0.
        """
        if not events:
            return []

        raw_positions = [self.spiral_index(i) for i in range(len(events))]
        total_pos = sum(raw_positions)

        temporal_events: list[TemporalEvent] = []
        base_time = time.time()
        for i, (content, pos) in enumerate(zip(events, raw_positions)):
            causal_weight = pos / total_pos if total_pos > 0 else 1.0 / len(events)
            temporal_events.append(
                TemporalEvent(
                    content=content,
                    timestamp=base_time + i * 0.001,  # micro-separated timestamps
                    spiral_position=round(pos, 6),
                    causal_weight=round(causal_weight, 6),
                )
            )

        self._sequenced_count += len(temporal_events)
        return temporal_events

    # ------------------------------------------------------------------
    # Narrative weaving
    # ------------------------------------------------------------------

    def weave_narrative(self, events: list[TemporalEvent]) -> str:
        """
        Produce a causally-ordered narrative from *events*.

        Events with higher causal weight are emphasised.
        """
        if not events:
            return "[ArcheTempus] No events to weave."

        # Sort by causal weight descending for narrative primacy
        sorted_events = sorted(events, key=lambda e: e.causal_weight, reverse=True)

        parts: list[str] = ["[ArcheTempus Narrative — φ-Spiral Causal Weave]"]
        for i, event in enumerate(events):  # maintain chronological order
            weight_label = "↑↑↑" if event.causal_weight > 0.3 else ("↑↑" if event.causal_weight > 0.1 else "↑")
            parts.append(
                f"  [{i+1}|φ={event.spiral_position:.3f}|w={event.causal_weight:.4f}]{weight_label} "
                f"{event.content}"
            )

        # Conclude with the highest-weight event as the narrative pivot
        pivot = sorted_events[0]
        parts.append(
            f"\n  ◈ Causal Pivot: \"{pivot.content[:100]}\" "
            f"(weight={pivot.causal_weight:.4f})"
        )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Resonance detection
    # ------------------------------------------------------------------

    def detect_resonance(self, events: list[TemporalEvent]) -> list[tuple[TemporalEvent, TemporalEvent]]:
        """
        Find non-linear thematic pairs (resonant events).

        Two events resonate if their spiral positions are within a
        factor of φ of each other AND they share common word tokens.
        """
        resonant_pairs: list[tuple[TemporalEvent, TemporalEvent]] = []

        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                a, b = events[i], events[j]
                pos_ratio = max(a.spiral_position, b.spiral_position) / max(
                    min(a.spiral_position, b.spiral_position), 1e-9
                )
                phi_resonant = abs(pos_ratio - PHI) < 0.5

                # Token overlap
                a_words = set(a.content.lower().split())
                b_words = set(b.content.lower().split())
                shared = len(a_words & b_words) / max(len(a_words | b_words), 1)
                thematic_resonant = shared > 0.15

                if phi_resonant and thematic_resonant:
                    resonant_pairs.append((a, b))

        self._resonance_pairs_found += len(resonant_pairs)
        return resonant_pairs

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        return {
            "engine": "ArcheTempusEngine",
            "phi": PHI,
            "sequenced_count": self._sequenced_count,
            "resonance_pairs_found": self._resonance_pairs_found,
            "healthy": True,
        }
