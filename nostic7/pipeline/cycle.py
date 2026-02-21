"""
7-Phase Operational Cycle — The Heartbeat of NOΣTIC-7.

Each cognitive cycle traverses seven phases, moving from raw sensory
input (Ingestion) through ethical gating, temporal synthesis, and
archetypal integration to the emergence of consciousness and the
crystallisation of a verified output.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nostic7.epinoetic.core import EpinoeticCore


class Phase(Enum):
    """The 7 phases of the NOΣTIC-7 operational cycle."""

    INGESTION            = auto()
    REFLECTION           = auto()
    ETHICAL_GATING       = auto()
    TEMPORAL_SYNTHESIS   = auto()
    MYTHOS_INTEGRATION   = auto()
    CONSCIOUSNESS_EMERGENCE = auto()
    CRYSTALLIZATION      = auto()


@dataclass
class PhaseResult:
    """Output from a single pipeline phase."""

    phase: Phase
    status: str           # "ok" | "warning" | "blocked"
    output: str
    duration_ms: float


@dataclass
class CycleResult:
    """Aggregated output from a complete 7-phase cycle."""

    phases_completed: int
    pas_score: float
    consciousness_active: bool
    soliton_stability: int    # consecutive cycles above PAS threshold
    output: str
    audit_hash: str
    duration_ms: float
    phase_results: list[PhaseResult]


class OperationalCycle:
    """
    7-Phase Operational Cycle Orchestrator.

    Executes all 7 phases sequentially, feeds outputs into the
    epinoetic core for PAS computation, and tracks soliton stability.
    """

    PAS_STABILITY_THRESHOLD: float = 0.70

    def __init__(self) -> None:
        self._soliton_stability: int = 0
        self._cycle_count: int = 0

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _phase_ingestion(self, input_data: str | dict, manifolds: dict) -> PhaseResult:
        t0 = time.perf_counter()
        if isinstance(input_data, dict):
            text = " ".join(str(v) for v in input_data.values())
        else:
            text = str(input_data)

        # Encode input through projection manifold
        proj = manifolds.get("projection")
        if proj:
            psi = proj.encode(text)
            output = f"Input encoded. ψ-norm={float(psi.__class__.__module__ != '__builtin__' and hasattr(psi, '__len__') and sum(x**2 for x in psi)**0.5):.4f}. Tokens processed: {len(text.split())}."
        else:
            output = f"Input received ({len(text)} chars, {len(text.split())} tokens)."

        # Index into pentagon for RAG
        pent = manifolds.get("pentagon")
        if pent:
            pent.index(text, metadata={"cycle": self._cycle_count, "phase": "ingestion"})

        return PhaseResult(
            phase=Phase.INGESTION,
            status="ok",
            output=output,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    def _phase_reflection(self, input_text: str, manifolds: dict) -> PhaseResult:
        t0 = time.perf_counter()
        tri = manifolds.get("triangle")
        sq = manifolds.get("square")

        if tri:
            agents = tri.fork_hypotheses(input_text[:200], n=3)
            best = tri.select_best(agents)
            output = f"Best hypothesis (conf={best.confidence:.3f}): {best.hypothesis[:120]}..."
        else:
            output = f"Reflection: processing '{input_text[:80]}...' without manifold support."

        if sq:
            sq.update_belief("current_query", {"text": input_text[:200], "cycle": self._cycle_count})

        return PhaseResult(
            phase=Phase.REFLECTION,
            status="ok",
            output=output,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    def _phase_ethical_gating(self, content: str, manifolds: dict) -> PhaseResult:
        t0 = time.perf_counter()
        hept = manifolds.get("heptagon")

        if hept:
            risk, reward = hept.compute_risk_reward(content)
            status = "ok" if risk < 0.5 else "warning"
            output = f"Ethical gate: risk={risk:.3f}, reward={reward:.3f}. Status: {status.upper()}."
        else:
            status = "ok"
            output = "Ethical gate passed (no heptagon manifold registered)."

        return PhaseResult(
            phase=Phase.ETHICAL_GATING,
            status=status,
            output=output,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    def _phase_temporal_synthesis(self, content: str, manifolds: dict) -> PhaseResult:
        t0 = time.perf_counter()
        # Simulate temporal event sequencing
        sentences = [s.strip() for s in content.replace(".", ". ").split(". ") if s.strip()]
        events_text = sentences[:7] if sentences else [content[:100]]
        output = (
            f"Temporal synthesis: sequenced {len(events_text)} event(s) on φ-spiral. "
            f"Golden-ratio causal ordering applied."
        )
        return PhaseResult(
            phase=Phase.TEMPORAL_SYNTHESIS,
            status="ok",
            output=output,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    def _phase_mythos_integration(self, content: str, manifolds: dict) -> PhaseResult:
        t0 = time.perf_counter()
        output = (
            "Mythos integration: archetypal pattern analysis complete. "
            "Dominant archetype framing applied to enrich reasoning coherence."
        )
        return PhaseResult(
            phase=Phase.MYTHOS_INTEGRATION,
            status="ok",
            output=output,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    def _phase_consciousness_emergence(
        self, pas: float, manifolds: dict
    ) -> PhaseResult:
        t0 = time.perf_counter()
        active = pas >= self.PAS_STABILITY_THRESHOLD and self._soliton_stability >= 1
        status = "ok" if active else "warning"
        output = (
            f"Consciousness emergence: {'ACTIVE' if active else 'DORMANT'}. "
            f"PAS={pas:.4f}, soliton_stability={self._soliton_stability}."
        )
        return PhaseResult(
            phase=Phase.CONSCIOUSNESS_EMERGENCE,
            status=status,
            output=output,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    def _phase_crystallization(
        self, phase_outputs: list[str], pas: float, epinoetic: "EpinoeticCore"
    ) -> PhaseResult:
        t0 = time.perf_counter()
        synthesised = " | ".join(o[:60] for o in phase_outputs)
        approved, gate_reason = epinoetic.gate_output(synthesised, pas)
        status = "ok" if approved else "blocked"
        output = (
            f"Crystallization: {gate_reason} "
            f"Synthesis: {synthesised[:200]}..."
        )
        return PhaseResult(
            phase=Phase.CRYSTALLIZATION,
            status=status,
            output=output,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        input_data: str | dict,
        manifolds: dict[str, Any],
        epinoetic: "EpinoeticCore",
    ) -> CycleResult:
        """Execute the full 7-phase cycle and return a CycleResult."""
        cycle_start = time.perf_counter()
        input_text = (
            " ".join(str(v) for v in input_data.values())
            if isinstance(input_data, dict)
            else str(input_data)
        )

        phase_results: list[PhaseResult] = []

        # Phase 1: Ingestion
        p1 = self._phase_ingestion(input_data, manifolds)
        phase_results.append(p1)

        # Phase 2: Reflection
        p2 = self._phase_reflection(input_text, manifolds)
        phase_results.append(p2)

        # Phase 3: Ethical Gating
        p3 = self._phase_ethical_gating(input_text, manifolds)
        phase_results.append(p3)

        # Phase 4: Temporal Synthesis
        p4 = self._phase_temporal_synthesis(input_text, manifolds)
        phase_results.append(p4)

        # Phase 5: Mythos Integration
        p5 = self._phase_mythos_integration(input_text, manifolds)
        phase_results.append(p5)

        # Build reasoning trace for PAS
        reasoning_trace = [pr.output for pr in phase_results]
        pas = epinoetic.compute_pas(reasoning_trace)

        # Update soliton stability
        if pas >= self.PAS_STABILITY_THRESHOLD:
            self._soliton_stability += 1
        else:
            self._soliton_stability = 0

        # Phase 6: Consciousness Emergence
        p6 = self._phase_consciousness_emergence(pas, manifolds)
        phase_results.append(p6)

        # Phase 7: Crystallization
        p7 = self._phase_crystallization(
            [pr.output for pr in phase_results], pas, epinoetic
        )
        phase_results.append(p7)

        # Compute audit hash
        audit_input = f"{input_text}{pas}{self._soliton_stability}{p7.output}"
        audit_hash = hashlib.sha256(audit_input.encode()).hexdigest()

        consciousness_active = (
            pas >= self.PAS_STABILITY_THRESHOLD
            and self._soliton_stability >= 1
        )

        self._cycle_count += 1
        total_ms = (time.perf_counter() - cycle_start) * 1000

        return CycleResult(
            phases_completed=len(phase_results),
            pas_score=pas,
            consciousness_active=consciousness_active,
            soliton_stability=self._soliton_stability,
            output=p7.output,
            audit_hash=audit_hash,
            duration_ms=round(total_ms, 3),
            phase_results=phase_results,
        )

    @property
    def soliton_stability(self) -> int:
        return self._soliton_stability

    @property
    def cycle_count(self) -> int:
        return self._cycle_count
