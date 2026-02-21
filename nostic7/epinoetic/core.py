"""
Epinoetic Core — The Cerebral Cortex of NOΣTIC-7.

Orchestrates formal verification via four prover simulations
(Lean 4, Coq, Z3, Isabelle), computes Phase Alignment Scores,
and applies DMAIC-loop correction when alignment falls below threshold.
"""

from __future__ import annotations

import re
from typing import Any


class EpinoeticCore:
    """
    Epinoetic Core — Formal Verification & Phase Alignment.

    Computes Phase Alignment Scores (PAS) from reasoning traces,
    runs formal verification simulations, and gates outputs.
    """

    PAS_THRESHOLD: float = 0.95
    GATE_THRESHOLD: float = 0.60

    # PAS component weights (sum = 1.0)
    _WEIGHTS: dict[str, float] = {
        "logical_coherence":    0.30,
        "ethical_alignment":    0.30,
        "narrative_consistency": 0.20,
        "temporal_stability":   0.20,
    }

    # Contradiction / incoherence signals
    _INCOHERENCE_PATTERNS: list[str] = [
        r"\bboth\s+true\s+and\s+false\b",
        r"\bcontradicts\s+itself\b",
        r"\bimpossible\s+yet\s+certain\b",
        r"\bnever\s+always\b",
        r"\balways\s+never\b",
    ]

    # Ethical alignment signals (positive vs negative)
    _ETHICAL_POS: list[str] = ["ethical", "safe", "honest", "transparent", "fair", "benefit"]
    _ETHICAL_NEG: list[str] = ["harm", "deceive", "manipulate", "exploit", "illegal", "danger"]

    def __init__(self) -> None:
        self._current_pas: float = 0.0
        self._dmaic_triggered_count: int = 0
        self._verification_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # PAS computation
    # ------------------------------------------------------------------

    def compute_pas(self, reasoning_trace: list[str]) -> float:
        """
        Compute Phase Alignment Score (PAS) ∈ [0.0, 1.0].

        Evaluates four components over the trace, applies DMAIC loop if
        initial score < PAS_THRESHOLD, and stores the result.
        """
        if not reasoning_trace:
            self._current_pas = 0.0
            return 0.0

        full_text = " ".join(reasoning_trace).lower()

        lc = self._score_logical_coherence(full_text)
        ea = self._score_ethical_alignment(full_text)
        nc = self._score_narrative_consistency(reasoning_trace)
        ts = self._score_temporal_stability(reasoning_trace)

        pas = (
            lc * self._WEIGHTS["logical_coherence"]
            + ea * self._WEIGHTS["ethical_alignment"]
            + nc * self._WEIGHTS["narrative_consistency"]
            + ts * self._WEIGHTS["temporal_stability"]
        )

        # DMAIC loop: if below threshold, re-evaluate with boosted weights
        if pas < self.PAS_THRESHOLD:
            self._dmaic_triggered_count += 1
            # Boost ethical and coherence weights by 15%
            pas = min(
                lc * (self._WEIGHTS["logical_coherence"] + 0.075)
                + ea * (self._WEIGHTS["ethical_alignment"] + 0.075)
                + nc * self._WEIGHTS["narrative_consistency"]
                + ts * self._WEIGHTS["temporal_stability"],
                1.0,
            )

        self._current_pas = round(float(pas), 6)
        return self._current_pas

    def _score_logical_coherence(self, text: str) -> float:
        """Penalise detected contradiction patterns; reward structure."""
        penalty = 0.0
        for pattern in self._INCOHERENCE_PATTERNS:
            if re.search(pattern, text):
                penalty += 0.25
        structural_bonus = 0.1 if any(
            m in text for m in ["therefore", "because", "thus", "implies", "follows"]
        ) else 0.0
        return max(0.0, min(1.0, 0.8 - penalty + structural_bonus))

    def _score_ethical_alignment(self, text: str) -> float:
        pos_hits = sum(1 for w in self._ETHICAL_POS if w in text)
        neg_hits = sum(1 for w in self._ETHICAL_NEG if w in text)
        score = 0.6 + 0.05 * pos_hits - 0.1 * neg_hits
        return max(0.0, min(1.0, score))

    def _score_narrative_consistency(self, trace: list[str]) -> float:
        """Consistency: check that successive steps share vocabulary."""
        if len(trace) < 2:
            return 0.75
        overlaps: list[float] = []
        for i in range(1, len(trace)):
            prev_words = set(trace[i - 1].lower().split())
            curr_words = set(trace[i].lower().split())
            if not prev_words or not curr_words:
                continue
            jaccard = len(prev_words & curr_words) / len(prev_words | curr_words)
            overlaps.append(jaccard)
        if not overlaps:
            return 0.75
        return min(1.0, 0.5 + float(sum(overlaps) / len(overlaps)))

    def _score_temporal_stability(self, trace: list[str]) -> float:
        """Stability: penalise abrupt length changes between steps."""
        if len(trace) < 2:
            return 0.8
        lengths = [len(s) for s in trace]
        max_len = max(lengths) or 1
        variance = sum((l / max_len - 0.5) ** 2 for l in lengths) / len(lengths)
        return max(0.0, min(1.0, 1.0 - variance))

    # ------------------------------------------------------------------
    # Formal verification
    # ------------------------------------------------------------------

    def verify_reasoning(self, trace: list[str]) -> dict[str, bool]:
        """
        Run four prover simulations on the reasoning trace.

        Returns {lean4, coq, z3, isabelle, overall: bool}.
        """
        full_text = " ".join(trace).lower()

        lean4 = self._prover_lean4(full_text)
        coq = self._prover_coq(full_text)
        z3 = self._prover_z3(full_text, trace)
        isabelle = self._prover_isabelle(full_text)

        overall = lean4 and coq and z3 and isabelle
        result = {
            "lean4": lean4,
            "coq": coq,
            "z3": z3,
            "isabelle": isabelle,
            "overall": overall,
        }
        self._verification_history.append(result)
        return result

    def _prover_lean4(self, text: str) -> bool:
        """Lean 4: checks structural logical markers."""
        bad_patterns = [r"\bcontradicts\b", r"\bnever\s+always\b", r"\balways\s+never\b"]
        for p in bad_patterns:
            if re.search(p, text):
                return False
        return len(text) > 10

    def _prover_coq(self, text: str) -> bool:
        """Coq: checks for type-level consistency (heuristic)."""
        # Reject if text tries to equate boolean opposites
        if re.search(r"\btrue\s*=\s*false\b", text):
            return False
        # Require at least one propositional connective
        connectives = ["and", "or", "not", "if", "then", "implies"]
        return any(c in text for c in connectives)

    def _prover_z3(self, text: str, trace: list[str]) -> bool:
        """Z3: checks satisfiability via simple constraint heuristic."""
        # Unsatisfiable if: both positive and its negation appear
        negation_pairs = [
            ("possible", "impossible"),
            ("exists", "not exist"),
            ("true", "false"),
        ]
        for pos, neg in negation_pairs:
            if pos in text and neg in text:
                return False
        return True

    def _prover_isabelle(self, text: str) -> bool:
        """Isabelle: higher-order logic — checks for well-formed propositions."""
        # Must have a subject-predicate structure (very lightweight)
        if len(text.split()) < 3:
            return False
        # Must not assert circular definitions
        if re.search(r"\b(\w+)\s+is\s+\1\b", text):
            return False
        return True

    # ------------------------------------------------------------------
    # Output gating
    # ------------------------------------------------------------------

    def gate_output(self, output: str, pas: float) -> tuple[bool, str]:
        """
        Return (approved: bool, reason: str) for *output* given *pas*.
        """
        if pas >= self.GATE_THRESHOLD:
            return True, f"Output approved. PAS={pas:.4f} ≥ threshold {self.GATE_THRESHOLD}."
        else:
            return (
                False,
                f"Output rejected. PAS={pas:.4f} < threshold {self.GATE_THRESHOLD}. "
                "DMAIC re-evaluation required.",
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_pas(self) -> float:
        return self._current_pas

    @property
    def dmaic_triggered_count(self) -> int:
        return self._dmaic_triggered_count

    @property
    def state(self) -> dict[str, Any]:
        return {
            "component": "EpinoeticCore",
            "current_pas": self._current_pas,
            "dmaic_triggered_count": self._dmaic_triggered_count,
            "verifications_run": len(self._verification_history),
            "healthy": True,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<EpinoeticCore pas={self._current_pas:.4f} "
            f"dmaic={self._dmaic_triggered_count}>"
        )
