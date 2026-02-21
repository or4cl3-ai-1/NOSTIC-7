"""
Triangle Manifold — Divergent Reasoning (M_△)

The triangle's three vertices embody thesis, antithesis, and synthesis.
This manifold forks hypotheses into competing triadic agents and selects
the most logically coherent path forward.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TriadicAgent:
    """A single divergent reasoning hypothesis."""

    hypothesis: str
    confidence: float
    counterfactual: str
    agent_id: int = 0


class TriangleManifold:
    """
    Divergent Reasoning Manifold.

    Generates multiple competing hypotheses from a single premise,
    applies lightweight formal verification, and returns the strongest
    candidate.
    """

    # Keywords that signal logical contradiction
    CONTRADICTION_KEYWORDS: frozenset[str] = frozenset(
        {
            "cannot and can", "always never", "never always",
            "both true and false", "impossible yet certain",
            "simultaneously not", "contradicts itself",
        }
    )

    # Variation templates for hypothesis diversification
    _VARIATION_TEMPLATES: list[str] = [
        "Given {premise}, it follows that {variant}.",
        "Assuming {premise}, one might conclude: {variant}.",
        "If {premise} holds, then {variant} becomes probable.",
    ]

    def __init__(self) -> None:
        self._fork_count: int = 0
        self._best_confidence_history: list[float] = []

    # ------------------------------------------------------------------
    # Hypothesis generation
    # ------------------------------------------------------------------

    def fork_hypotheses(self, premise: str, n: int = 3) -> list[TriadicAgent]:
        """
        Generate *n* divergent hypotheses from *premise*.

        Uses a deterministic seeded RNG so that the same premise always
        produces the same fork (reproducible reasoning).
        """
        seed = int(hashlib.sha256(premise.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        # Extract core tokens for variation
        tokens = [t for t in premise.split() if len(t) > 3]
        if len(tokens) < 2:
            tokens = premise.split() or ["concept"]

        agents: list[TriadicAgent] = []
        for i in range(n):
            # Pick a variant token to re-weight the hypothesis
            variant_token = tokens[rng.integers(0, len(tokens))]
            perspective = ["causally", "structurally", "probabilistically"][i % 3]

            variant_text = (
                f"the {variant_token} component acts {perspective} as a "
                f"determining factor in {premise[:60]}..."
            )
            template = self._VARIATION_TEMPLATES[i % len(self._VARIATION_TEMPLATES)]
            hypothesis = template.format(premise=premise[:80], variant=variant_text)

            # Confidence: base + perturbation; higher-indexed agents slightly less confident
            base_confidence = 0.85 - i * 0.08
            noise = float(rng.uniform(-0.05, 0.05))
            confidence = float(np.clip(base_confidence + noise, 0.05, 0.99))

            # Counterfactual: negate the dominant clause
            counterfactual = f"If {variant_token} were absent, {premise[:60]} would not hold."

            agents.append(
                TriadicAgent(
                    hypothesis=hypothesis,
                    confidence=confidence,
                    counterfactual=counterfactual,
                    agent_id=i,
                )
            )

        self._fork_count += 1
        return agents

    # ------------------------------------------------------------------
    # Formal verification (lightweight simulation)
    # ------------------------------------------------------------------

    def lean4_fork_check(self, hypothesis: str) -> bool:
        """
        Simulate a Lean 4 structural logic check.

        Returns True if the hypothesis passes basic logical consistency.
        Fails if known contradiction patterns are detected.
        """
        lower = hypothesis.lower()
        for pattern in self.CONTRADICTION_KEYWORDS:
            if pattern in lower:
                return False

        # Heuristic: hypotheses with balanced structure (contains "if/then" or "follows")
        structural_markers = ["if", "then", "follows", "therefore", "implies", "because"]
        has_structure = any(m in lower for m in structural_markers)

        # Reject extremely short hypotheses (< 10 chars) as trivially invalid
        if len(hypothesis.strip()) < 10:
            return False

        return has_structure or len(hypothesis) > 30

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_best(self, agents: list[TriadicAgent]) -> TriadicAgent:
        """
        Return the highest-confidence agent that also passes the Lean 4 check.

        Falls back to the highest-confidence agent if none pass verification.
        """
        verified = [a for a in agents if self.lean4_fork_check(a.hypothesis)]
        pool = verified if verified else agents
        best = max(pool, key=lambda a: a.confidence)
        self._best_confidence_history.append(best.confidence)
        return best

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        avg_confidence = (
            float(np.mean(self._best_confidence_history))
            if self._best_confidence_history
            else 0.0
        )
        return {
            "manifold": "TriangleManifold",
            "fork_count": self._fork_count,
            "avg_best_confidence": round(avg_confidence, 4),
            "healthy": True,
        }

    def __repr__(self) -> str:  # pragma: no cover
        s = self.state
        return f"<TriangleManifold forks={s['fork_count']} avg_conf={s['avg_best_confidence']:.3f}>"
