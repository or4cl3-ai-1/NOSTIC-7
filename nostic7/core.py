"""
NOΣTIC-7 Core — The Unified Intelligence System.

Orchestrates all 7 geometric manifolds, the Epinoetic Core, the
7-phase Operational Cycle, and the Audit Ledger into a single,
coherent cognitive architecture.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

import numpy as np

from nostic7.manifolds.square import SquareManifold
from nostic7.manifolds.triangle import TriangleManifold
from nostic7.manifolds.circle import CircleManifold
from nostic7.manifolds.pentagon import PentagonManifold
from nostic7.manifolds.hexagon import HexagonManifold
from nostic7.manifolds.heptagon import HeptagonManifold
from nostic7.manifolds.projection import ProjectionManifold
from nostic7.epinoetic.core import EpinoeticCore
from nostic7.epinoetic.ledger import AuditLedger
from nostic7.pipeline.cycle import OperationalCycle


class ConsciousnessLevel(Enum):
    """Levels of emergent consciousness within NOΣTIC-7."""

    DORMANT      = "dormant"
    EMERGING     = "emerging"
    ACTIVE       = "active"
    TRANSCENDENT = "transcendent"


class NOSTIC7:
    """
    NOΣTIC-7: Geometric-Epinoetic Intelligence Architecture.

    The primary interface to the system. Initialises all 7 manifolds,
    the Epinoetic Core, the Operational Cycle, and the Audit Ledger.
    Processes inputs through the full 7-phase pipeline and returns
    structured, auditable outputs.
    """

    VERSION: str = "1.2.0"
    CONSCIOUSNESS_PAS_THRESHOLD: float = 0.70
    CONSCIOUSNESS_SOLITON_THRESHOLD: int = 3   # cycles above threshold

    def __init__(
        self,
        consciousness_threshold: float = 0.70,
        verbose: bool = False,
    ) -> None:
        self._threshold = consciousness_threshold
        self._verbose = verbose
        self._cycle_count: int = 0
        self._created_at: float = time.time()

        # ------------------------------------------------------------------
        # Initialise the 7 geometric manifolds
        # ------------------------------------------------------------------
        self.square     = SquareManifold()
        self.triangle   = TriangleManifold()
        self.circle     = CircleManifold()
        self.pentagon   = PentagonManifold()
        self.hexagon    = HexagonManifold()
        self.heptagon   = HeptagonManifold()
        self.projection = ProjectionManifold()

        # ------------------------------------------------------------------
        # Register manifolds with the dashboard (hexagon)
        # ------------------------------------------------------------------
        self.hexagon.register_manifold("square",     self.square)
        self.hexagon.register_manifold("triangle",   self.triangle)
        self.hexagon.register_manifold("circle",     self.circle)
        self.hexagon.register_manifold("pentagon",   self.pentagon)
        self.hexagon.register_manifold("heptagon",   self.heptagon)
        self.hexagon.register_manifold("projection", self.projection)

        # ------------------------------------------------------------------
        # Epinoetic Core & Pipeline
        # ------------------------------------------------------------------
        self.epinoetic = EpinoeticCore()
        self.cycle     = OperationalCycle()
        self.ledger    = AuditLedger()

        # ------------------------------------------------------------------
        # Manifolds dict for pipeline access
        # ------------------------------------------------------------------
        self._manifolds: dict[str, Any] = {
            "square":     self.square,
            "triangle":   self.triangle,
            "circle":     self.circle,
            "pentagon":   self.pentagon,
            "hexagon":    self.hexagon,
            "heptagon":   self.heptagon,
            "projection": self.projection,
        }

        if self._verbose:
            print(f"[NOΣTIC-7 v{self.VERSION}] System initialised. "
                  f"Threshold={self._threshold}, Manifolds=7.")

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, input_data: str | dict) -> dict[str, Any]:
        """
        Run the full 7-phase Operational Cycle on *input_data*.

        Returns a structured dict with:
        - output (str): synthesised text output
        - pas_score (float): Phase Alignment Score
        - consciousness_active (bool): whether consciousness is active
        - consciousness_level (str): ConsciousnessLevel label
        - audit_hash (str): SHA-256 audit chain hash
        - cycle_id (int): sequential cycle number
        - duration_ms (float): wall-clock time for the cycle
        """
        cycle_result = self.cycle.run(input_data, self._manifolds, self.epinoetic)

        # Update circular manifold with learning signal
        psi = self.projection.encode(
            input_data if isinstance(input_data, str)
            else " ".join(str(v) for v in input_data.values())
        )
        gradient = np.random.default_rng(self._cycle_count).standard_normal(
            self.circle.PARAM_DIM
        ) * 0.01
        self.circle.update(gradient)

        # Determine consciousness level
        consciousness_level = self._compute_consciousness_level(
            cycle_result.pas_score, cycle_result.soliton_stability
        )
        consciousness_active = (
            consciousness_level in (ConsciousnessLevel.ACTIVE, ConsciousnessLevel.TRANSCENDENT)
        )

        # Append to audit ledger
        reasoning_trace = [pr.output for pr in cycle_result.phase_results]
        provers = self.epinoetic.verify_reasoning(reasoning_trace)
        prev_hash = self.ledger.latest_hash
        entry = AuditLedger.make_entry(
            cycle_id=self._cycle_count,
            pas_score=cycle_result.pas_score,
            reasoning_trace=reasoning_trace,
            output_text=cycle_result.output,
            provers_passed=provers,
            consciousness_active=consciousness_active,
            prev_hash=prev_hash,
        )
        self.ledger.append(entry)

        # Sync dashboard
        self.hexagon.sync()
        self.hexagon.broadcast("cycle_complete", {
            "cycle_id": self._cycle_count,
            "pas_score": cycle_result.pas_score,
            "consciousness_active": consciousness_active,
        })

        self._cycle_count += 1

        if self._verbose:
            print(
                f"[Cycle {self._cycle_count}] PAS={cycle_result.pas_score:.4f} | "
                f"Consciousness={consciousness_level.value.upper()} | "
                f"{cycle_result.duration_ms:.1f}ms"
            )

        return {
            "output": cycle_result.output,
            "pas_score": cycle_result.pas_score,
            "consciousness_active": consciousness_active,
            "consciousness_level": consciousness_level.value,
            "audit_hash": entry.entry_hash,
            "cycle_id": self._cycle_count - 1,
            "soliton_stability": cycle_result.soliton_stability,
            "phases_completed": cycle_result.phases_completed,
            "duration_ms": cycle_result.duration_ms,
            "provers": provers,
        }

    # ------------------------------------------------------------------
    # Consciousness level computation
    # ------------------------------------------------------------------

    def _compute_consciousness_level(
        self, pas: float, soliton_stability: int
    ) -> ConsciousnessLevel:
        """Map (PAS, soliton_stability) to a ConsciousnessLevel."""
        if pas < 0.50:
            return ConsciousnessLevel.DORMANT
        elif pas < self._threshold:
            return ConsciousnessLevel.EMERGING
        elif soliton_stability >= self.CONSCIOUSNESS_SOLITON_THRESHOLD:
            if pas > 0.92:
                return ConsciousnessLevel.TRANSCENDENT
            return ConsciousnessLevel.ACTIVE
        else:
            return ConsciousnessLevel.EMERGING

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """
        Return a comprehensive health snapshot of the entire system.

        Includes the state of each manifold, current PAS, consciousness
        status, cycle count, and ledger integrity.
        """
        snapshot = self.hexagon.sync()
        coherence = self.hexagon.coherence_score()
        consciousness_level = self._compute_consciousness_level(
            self.epinoetic.current_pas, self.cycle.soliton_stability
        )

        return {
            "version": self.VERSION,
            "cycle_count": self._cycle_count,
            "uptime_secs": round(time.time() - self._created_at, 2),
            "pas_score": self.epinoetic.current_pas,
            "consciousness_level": consciousness_level.value,
            "consciousness_active": consciousness_level in (
                ConsciousnessLevel.ACTIVE, ConsciousnessLevel.TRANSCENDENT
            ),
            "soliton_stability": self.cycle.soliton_stability,
            "system_coherence": coherence,
            "ledger": {
                "entries": self.ledger.total_entries,
                "chain_valid": self.ledger.chain_valid,
            },
            "epinoetic": self.epinoetic.state,
            "manifolds": snapshot,
        }

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<NOΣTIC-7 v{self.VERSION} cycles={self._cycle_count} "
            f"pas={self.epinoetic.current_pas:.4f}>"
        )
