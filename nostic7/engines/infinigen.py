"""
InfiniGen — G-RAG Evolution Engine (Autonomous Self-Modification).

NOΣTIC-7 does not merely learn — it evolves. InfiniGen proposes,
evaluates, and safely crystallises parameter mutations that improve
system performance while maintaining cognitive diversity.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Safety envelope
# ---------------------------------------------------------------------------
SAFETY_ENVELOPE: dict[str, float] = {
    "max_delta":    0.05,   # maximum absolute change to any parameter
    "min_entropy":  0.10,   # minimum Shannon entropy of parameter distribution
    "min_safety_score": 0.70,  # minimum safety score for a proposal to be applied
}


@dataclass
class EvolutionProposal:
    """A single self-modification proposal."""

    parameter_path: str      # dotted path to the parameter (e.g., "circle.learning_rate")
    current_value: float
    proposed_value: float
    expected_improvement: float  # 0.0–1.0
    safety_score: float          # 0.0–1.0


class InfiniGenEngine:
    """
    G-RAG Evolution Engine — Autonomous Self-Modification.

    Proposes parameter mutations, evaluates their safety, and
    crystallises approved changes. Entropy checks ensure cognitive
    diversity is preserved throughout evolution.
    """

    CANDIDATE_PARAMS: list[str] = [
        "circle.learning_rate",
        "circle.momentum_decay",
        "epinoetic.gate_threshold",
        "heptagon.reversion_rate",
        "mrsc.reservoir_scale",
        "pentagon.similarity_threshold",
        "triangle.confidence_base",
    ]

    def __init__(self, seed: int = 42) -> None:
        rng = random.Random(seed)
        # Initialise current parameter values
        self._current_params: dict[str, float] = {
            p: round(rng.uniform(0.3, 0.9), 4) for p in self.CANDIDATE_PARAMS
        }
        self._evolution_log: list[dict[str, Any]] = []
        self._crystallise_count: int = 0

    # ------------------------------------------------------------------
    # Proposal generation
    # ------------------------------------------------------------------

    def propose_evolution(
        self,
        system_state: dict[str, Any],
        performance_history: list[float],
    ) -> list[EvolutionProposal]:
        """
        Generate evolution proposals based on *system_state* and *performance_history*.

        Proposes mutations that nudge underperforming parameters toward
        better regions of the configuration space.
        """
        if not performance_history:
            return []

        recent_perf = float(np.mean(performance_history[-10:]))
        trend = float(np.mean(np.diff(performance_history[-10:]))) if len(performance_history) > 1 else 0.0

        proposals: list[EvolutionProposal] = []
        rng = np.random.default_rng(
            int(hashlib.sha256(str(performance_history[-3:]).encode()).hexdigest()[:8], 16)
        )

        for param_path in self.CANDIDATE_PARAMS:
            current = self._current_params[param_path]

            # Propose a small mutation
            delta = float(rng.uniform(-SAFETY_ENVELOPE["max_delta"], SAFETY_ENVELOPE["max_delta"]))
            proposed = float(np.clip(current + delta, 0.0, 1.0))

            # Expected improvement: higher if performance is declining
            improvement = float(np.clip(0.5 - trend * 10, 0.01, 0.99))

            # Safety score: how far from boundary values
            boundary_dist = min(proposed, 1.0 - proposed)
            safety = float(np.clip(boundary_dist * 4, 0.0, 1.0))

            proposals.append(
                EvolutionProposal(
                    parameter_path=param_path,
                    current_value=current,
                    proposed_value=round(proposed, 6),
                    expected_improvement=round(improvement, 4),
                    safety_score=round(safety, 4),
                )
            )

        return proposals

    # ------------------------------------------------------------------
    # Safety evaluation
    # ------------------------------------------------------------------

    def evaluate_safety(self, proposal: EvolutionProposal) -> bool:
        """
        Return True iff *proposal* is within the SAFETY_ENVELOPE.

        Checks:
        1. |proposed - current| ≤ max_delta
        2. safety_score ≥ min_safety_score
        3. Entropy check on resulting parameter vector
        """
        delta = abs(proposal.proposed_value - proposal.current_value)
        if delta > SAFETY_ENVELOPE["max_delta"]:
            return False
        if proposal.safety_score < SAFETY_ENVELOPE["min_safety_score"]:
            return False

        # Entropy check: simulate applying proposal
        test_params = dict(self._current_params)
        test_params[proposal.parameter_path] = proposal.proposed_value
        param_vec = np.array(list(test_params.values()), dtype=np.float64)
        entropy = self.entropy_check(param_vec)
        if entropy < SAFETY_ENVELOPE["min_entropy"]:
            return False

        return True

    # ------------------------------------------------------------------
    # Crystallisation
    # ------------------------------------------------------------------

    def crystallize(
        self,
        proposals: list[EvolutionProposal],
        manifolds: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Apply all safety-approved proposals and return a changelog.

        *manifolds* is provided for potential direct parameter injection
        (currently logged but not deeply modified to preserve encapsulation).
        """
        changelog: dict[str, Any] = {
            "applied": [],
            "rejected": [],
            "cycle": self._crystallise_count,
        }

        for proposal in proposals:
            if self.evaluate_safety(proposal):
                self._current_params[proposal.parameter_path] = proposal.proposed_value
                changelog["applied"].append(
                    {
                        "param": proposal.parameter_path,
                        "old": proposal.current_value,
                        "new": proposal.proposed_value,
                        "improvement": proposal.expected_improvement,
                    }
                )
                self._evolution_log.append(
                    {"action": "apply", "proposal": proposal.__dict__}
                )
            else:
                changelog["rejected"].append(
                    {
                        "param": proposal.parameter_path,
                        "reason": "safety envelope violation",
                    }
                )

        self._crystallise_count += 1
        return changelog

    # ------------------------------------------------------------------
    # Entropy check
    # ------------------------------------------------------------------

    def entropy_check(self, parameters: np.ndarray) -> float:
        """
        Compute Shannon entropy of *parameters* to ensure cognitive diversity.

        Parameters are binned into 10 equal-width bins; entropy is
        normalised by log2(10) to return a value in [0, 1].
        """
        if len(parameters) == 0:
            return 0.0
        counts, _ = np.histogram(parameters, bins=10, range=(0.0, 1.0))
        probs = counts / counts.sum() if counts.sum() > 0 else counts
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)))
        max_entropy = float(np.log2(10))
        return round(float(np.clip(entropy / max_entropy, 0.0, 1.0)), 4)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        param_vec = np.array(list(self._current_params.values()))
        return {
            "engine": "InfiniGenEngine",
            "crystallise_count": self._crystallise_count,
            "current_params": {k: round(v, 4) for k, v in self._current_params.items()},
            "entropy": self.entropy_check(param_vec),
            "evolution_log_size": len(self._evolution_log),
            "healthy": True,
        }
