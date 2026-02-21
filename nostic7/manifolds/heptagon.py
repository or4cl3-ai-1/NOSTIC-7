"""
Heptagon Manifold — Affective Foresight (M_⬡⁺)

Seven sides, seven values. The heptagon encodes the emotional and
ethical field that colours every cognitive act. It forecasts value
trajectories and translates system state into human-readable affect.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np


class HeptagonManifold:
    """
    Affective Foresight Manifold.

    Maintains a 7-dimensional ValueField and provides tools for
    estimating affective state, computing risk/reward, forecasting
    value trajectories, and generating emotional context labels.
    """

    # Value dimension indices
    VALUE_DIMS: list[str] = [
        "safety", "ethics", "creativity", "curiosity",
        "empathy", "resilience", "transcendence",
    ]

    # Keyword maps for each dimension
    _KEYWORD_MAP: dict[str, list[str]] = {
        "safety":       ["safe", "protect", "secure", "stable", "guard", "prevent", "risk"],
        "ethics":       ["ethical", "moral", "just", "fair", "right", "wrong", "virtue", "integrity"],
        "creativity":   ["creative", "novel", "imagine", "invent", "innovate", "original", "design"],
        "curiosity":    ["curious", "explore", "discover", "wonder", "learn", "question", "investigate"],
        "empathy":      ["empathy", "feel", "understand", "compassion", "care", "relate", "listen"],
        "resilience":   ["resilient", "adapt", "overcome", "persist", "endure", "recover", "robust"],
        "transcendence":["transcend", "beyond", "infinite", "consciousness", "awareness", "unity", "sublime"],
    }

    # Emotional labels (keyed by dominant dimension index)
    _EMOTIONAL_LABELS: list[str] = [
        "Carefully Protective",       # safety dominant
        "Ethically Grounded",         # ethics dominant
        "Expansively Creative",       # creativity dominant
        "Curiously Exploratory",      # curiosity dominant
        "Deeply Empathic",            # empathy dominant
        "Tenaciously Resilient",      # resilience dominant
        "Transcendently Aware",       # transcendence dominant
    ]

    def __init__(self) -> None:
        # Initialise value field with mild positive bias
        self._value_field: np.ndarray = np.array(
            [0.6, 0.7, 0.5, 0.6, 0.6, 0.5, 0.4], dtype=np.float64
        )
        self._state_history: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Affective estimation
    # ------------------------------------------------------------------

    def estimate_affective_state(self, context: str) -> dict[str, float]:
        """
        Score each value dimension based on keyword frequency in *context*.

        Returns a dict mapping dimension name → score (0.0–1.0).
        """
        lower = context.lower()
        words = re.findall(r"\b\w+\b", lower)
        word_set = set(words)

        scores: dict[str, float] = {}
        for dim, keywords in self._KEYWORD_MAP.items():
            hits = sum(1 for kw in keywords if kw in word_set)
            # Blend keyword hit rate with baseline value field
            raw = hits / max(len(keywords), 1)
            idx = self.VALUE_DIMS.index(dim)
            blended = 0.4 * raw + 0.6 * self._value_field[idx]
            scores[dim] = round(float(np.clip(blended, 0.0, 1.0)), 4)

        self._state_history.append(scores)
        return scores

    # ------------------------------------------------------------------
    # Risk / reward
    # ------------------------------------------------------------------

    def compute_risk_reward(self, action: str) -> tuple[float, float]:
        """
        Return (risk_score, reward_score) ∈ [0.0, 1.0] for *action*.

        Risk proxied by inverse safety/ethics scores; reward by
        creativity + curiosity.
        """
        state = self.estimate_affective_state(action)
        risk = float(np.clip(1.0 - 0.5 * (state["safety"] + state["ethics"]), 0.0, 1.0))
        reward = float(np.clip(0.5 * (state["creativity"] + state["curiosity"]), 0.0, 1.0))
        return round(risk, 4), round(reward, 4)

    # ------------------------------------------------------------------
    # Trajectory forecasting
    # ------------------------------------------------------------------

    def trajectory_forecast(
        self, current_state: dict[str, float], steps: int = 5
    ) -> list[dict[str, float]]:
        """
        Project value field evolution over *steps* time steps.

        Uses a simple mean-reversion dynamics: each dimension decays
        toward the long-run baseline (self._value_field) at rate 0.1.
        """
        REVERSION_RATE = 0.1
        trajectory: list[dict[str, float]] = []
        state_vec = np.array([current_state.get(d, self._value_field[i])
                               for i, d in enumerate(self.VALUE_DIMS)], dtype=np.float64)

        for _ in range(steps):
            delta = REVERSION_RATE * (self._value_field - state_vec)
            state_vec = np.clip(state_vec + delta, 0.0, 1.0)
            trajectory.append({
                dim: round(float(state_vec[i]), 4)
                for i, dim in enumerate(self.VALUE_DIMS)
            })

        return trajectory

    # ------------------------------------------------------------------
    # Emotional context
    # ------------------------------------------------------------------

    def emotional_context(self, state: dict[str, float]) -> str:
        """
        Return a human-readable emotional label for *state*.

        Picks the dominant value dimension, then qualifies it based
        on overall valence.
        """
        if not state:
            return "Neutrally Balanced"

        dominant_dim = max(state, key=lambda d: state[d])
        idx = self.VALUE_DIMS.index(dominant_dim)
        base_label = self._EMOTIONAL_LABELS[idx]

        # Compute overall valence
        mean_val = float(np.mean(list(state.values())))
        if mean_val > 0.75:
            qualifier = "Confidently"
        elif mean_val > 0.55:
            qualifier = "Cautiously"
        else:
            qualifier = "Tentatively"

        return f"{qualifier} {base_label}"

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        return {
            "manifold": "HeptagonManifold",
            "value_field": {
                dim: round(float(self._value_field[i]), 4)
                for i, dim in enumerate(self.VALUE_DIMS)
            },
            "history_depth": len(self._state_history),
            "healthy": True,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<HeptagonManifold history={len(self._state_history)}>"
