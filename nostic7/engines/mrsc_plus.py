"""
MRSC+ Engine — Recursive Self-Modelling Cognitive+ Engine.

Five modules (Memory, Empathy, Intention, Reflection, Evolution)
operating in concert to produce deep contextual understanding.
The spectral radius ρ(W) < 1 guarantees echo-state stability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MRSCResult:
    """Output of one MRSC+ processing cycle."""

    memory_state: list[str]
    empathy_score: float
    intention_vector: list[str]
    reflection_depth: int
    evolution_delta: float


class MRSCPlusEngine:
    """
    Recursive Self-Modelling Cognitive+ Engine.

    Processes context through five sequential modules and returns a
    rich MRSCResult. Internal reservoir weight matrix W has ρ(W) < 1.
    """

    RESERVOIR_DIM: int = 64

    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        # Reservoir weight matrix with guaranteed spectral radius < 1
        W_raw = rng.standard_normal((self.RESERVOIR_DIM, self.RESERVOIR_DIM))
        eigenvalues = np.linalg.eigvals(W_raw)
        sr = float(np.max(np.abs(eigenvalues)))
        self._W: np.ndarray = (W_raw / (sr + 1e-8) * 0.9).astype(np.float64)
        self._reservoir_state: np.ndarray = np.zeros(self.RESERVOIR_DIM, dtype=np.float64)

        # Baseline state for evolution delta
        self._baseline_state: np.ndarray = self._reservoir_state.copy()
        self._history: list[str] = []
        self._cycle_count: int = 0

    # ------------------------------------------------------------------
    # Module: Memory
    # ------------------------------------------------------------------

    def _module_memory(self, context: str, history: list[str]) -> list[str]:
        """Retrieve the last 5 history items most relevant to context."""
        if not history:
            return []
        context_words = set(context.lower().split())
        scored = []
        for item in history:
            item_words = set(item.lower().split())
            overlap = len(context_words & item_words) / max(len(context_words | item_words), 1)
            scored.append((overlap, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:5]]

    # ------------------------------------------------------------------
    # Module: Empathy
    # ------------------------------------------------------------------

    def _module_empathy(self, context: str) -> float:
        """Score emotional valence of context ∈ [0.0, 1.0]."""
        positive_words = {
            "love", "happy", "joy", "great", "wonderful", "excellent", "beautiful",
            "hope", "care", "kind", "help", "support", "grateful", "excited",
        }
        negative_words = {
            "hate", "sad", "angry", "terrible", "awful", "horrible", "fear",
            "pain", "suffer", "hurt", "lost", "fail", "broken", "desperate",
        }
        words = set(re.findall(r"\b\w+\b", context.lower()))
        pos = len(words & positive_words)
        neg = len(words & negative_words)
        # Neutral baseline 0.5, shift by hits
        score = 0.5 + 0.05 * pos - 0.05 * neg
        return round(float(np.clip(score, 0.0, 1.0)), 4)

    # ------------------------------------------------------------------
    # Module: Intention
    # ------------------------------------------------------------------

    def _module_intention(self, context: str) -> list[str]:
        """Extract primary intent verb phrases from context."""
        # Simple heuristic: find verb + noun patterns
        patterns = [
            r"\b(want|need|wish|intend|plan|aim|seek|try|hope)\s+to\s+(\w+)\b",
            r"\b(understand|analyze|compute|build|create|improve|explain|describe)\s+(\w+)\b",
        ]
        intentions: list[str] = []
        for pattern in patterns:
            matches = re.findall(pattern, context.lower())
            for match in matches:
                intentions.append(" ".join(match))
        if not intentions:
            # Fallback: first verb in text
            verbs = re.findall(r"\b(is|are|has|have|does|can|will|should|could|would)\b", context.lower())
            intentions = [f"{v} [implicit intent]" for v in verbs[:2]]
        return intentions[:5]

    # ------------------------------------------------------------------
    # Module: Reflection
    # ------------------------------------------------------------------

    def _module_reflection(self, context: str, memory: list[str], depth: int = 3) -> int:
        """
        Recursively deepen understanding over *depth* levels.

        Each level expands the context by incorporating memory tokens.
        Returns the depth reached.
        """
        current = context
        for level in range(depth):
            # Simulate deepening: incorporate memory vocabulary
            if memory:
                memory_snippet = " ".join(memory[: level + 1])
                tokens = set(current.split()) | set(memory_snippet.split())
                current = " ".join(sorted(tokens)[:50])  # compressed enriched context
            # Update reservoir state
            u = np.zeros(self.RESERVOIR_DIM, dtype=np.float64)
            for i, ch in enumerate(current[:self.RESERVOIR_DIM]):
                u[i % self.RESERVOIR_DIM] += ord(ch) / 128.0
            self._reservoir_state = np.tanh(self._W @ self._reservoir_state + 0.1 * u)
        return depth

    # ------------------------------------------------------------------
    # Module: Evolution
    # ------------------------------------------------------------------

    def _module_evolution(self) -> float:
        """Compute delta from baseline reservoir state."""
        delta = float(np.linalg.norm(self._reservoir_state - self._baseline_state))
        # Update rolling baseline slowly
        self._baseline_state = 0.99 * self._baseline_state + 0.01 * self._reservoir_state
        return round(delta, 6)

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process(self, context: str, history: list[str] | None = None) -> MRSCResult:
        """Run all five modules and return an MRSCResult."""
        hist = history or self._history

        memory = self._module_memory(context, hist)
        empathy = self._module_empathy(context)
        intention = self._module_intention(context)
        depth = self._module_reflection(context, memory)
        delta = self._module_evolution()

        # Update internal history
        self._history.append(context)
        if len(self._history) > 100:
            self._history = self._history[-100:]
        self._cycle_count += 1

        return MRSCResult(
            memory_state=memory,
            empathy_score=empathy,
            intention_vector=intention,
            reflection_depth=depth,
            evolution_delta=delta,
        )

    # ------------------------------------------------------------------
    # Stability check
    # ------------------------------------------------------------------

    def spectral_radius(self) -> float:
        """Return ρ(W) — must be < 1.0 for echo-state stability."""
        eigenvalues = np.linalg.eigvals(self._W)
        return round(float(np.max(np.abs(eigenvalues))), 6)

    @property
    def state(self) -> dict[str, Any]:
        return {
            "engine": "MRSCPlusEngine",
            "cycle_count": self._cycle_count,
            "history_size": len(self._history),
            "spectral_radius": self.spectral_radius(),
            "reservoir_norm": round(float(np.linalg.norm(self._reservoir_state)), 4),
            "healthy": self.spectral_radius() < 1.0,
        }
