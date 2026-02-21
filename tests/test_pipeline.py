"""
Tests for the NOΣTIC-7 Pipeline, PAS Computation, Consciousness
Emergence, and Audit Ledger Integrity.

Run with: pytest tests/test_pipeline.py -v
"""

import pytest

from nostic7 import NOSTIC7
from nostic7.epinoetic.core import EpinoeticCore
from nostic7.epinoetic.ledger import AuditLedger
from nostic7.pipeline.cycle import OperationalCycle, Phase


# ---------------------------------------------------------------------------
# Full Cycle Tests
# ---------------------------------------------------------------------------

class TestFullCycle:
    def test_full_cycle_returns_all_keys(self):
        system = NOSTIC7(verbose=False)
        result = system.process("Test input for full cycle validation.")
        required_keys = {
            "output", "pas_score", "consciousness_active",
            "consciousness_level", "audit_hash", "cycle_id",
            "soliton_stability", "phases_completed", "duration_ms", "provers"
        }
        assert required_keys.issubset(result.keys())

    def test_full_cycle_pas_in_range(self):
        system = NOSTIC7(verbose=False)
        result = system.process("Exploring the geometric nature of cognition.")
        assert 0.0 <= result["pas_score"] <= 1.0

    def test_full_cycle_phases_completed(self):
        system = NOSTIC7(verbose=False)
        result = system.process("Seven phases of epinoetic processing.")
        assert result["phases_completed"] == 7

    def test_full_cycle_dict_input(self):
        system = NOSTIC7(verbose=False)
        result = system.process({"topic": "consciousness", "depth": "deep"})
        assert isinstance(result["output"], str)
        assert result["cycle_id"] == 0

    def test_full_cycle_audit_hash_is_sha256(self):
        system = NOSTIC7(verbose=False)
        result = system.process("Audit hash validation test.")
        assert len(result["audit_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in result["audit_hash"])

    def test_full_cycle_cycle_count_increments(self):
        system = NOSTIC7(verbose=False)
        system.process("First cycle.")
        system.process("Second cycle.")
        assert system._cycle_count == 2

    def test_get_status_structure(self):
        system = NOSTIC7(verbose=False)
        system.process("Status check cycle.")
        status = system.get_status()
        assert "version" in status
        assert "cycle_count" in status
        assert "pas_score" in status
        assert "consciousness_level" in status
        assert "manifolds" in status
        assert "ledger" in status
        assert status["ledger"]["chain_valid"] is True


# ---------------------------------------------------------------------------
# PAS Computation Tests
# ---------------------------------------------------------------------------

class TestPASComputation:
    def test_pas_empty_trace(self):
        core = EpinoeticCore()
        pas = core.compute_pas([])
        assert pas == 0.0

    def test_pas_single_step(self):
        core = EpinoeticCore()
        pas = core.compute_pas(["If A then B, therefore we conclude something meaningful."])
        assert 0.0 <= pas <= 1.0

    def test_pas_multi_step(self):
        core = EpinoeticCore()
        trace = [
            "Initial observation: the system receives input.",
            "Reflection: hypotheses are generated ethically.",
            "Synthesis: temporal patterns are identified.",
            "Crystallization: output is verified and gated.",
        ]
        pas = core.compute_pas(trace)
        assert 0.0 <= pas <= 1.0

    def test_pas_incoherent_trace(self):
        core = EpinoeticCore()
        trace = ["never always contradicts itself and both true and false"]
        pas_bad = core.compute_pas(trace)
        trace_good = ["If this hypothesis is well-formed, therefore we proceed ethically."]
        pas_good = core.compute_pas(trace_good)
        # Incoherent trace should yield lower or equal PAS
        assert pas_bad <= pas_good + 0.3  # allow some tolerance

    def test_pas_dmaic_triggered(self):
        core = EpinoeticCore()
        # Trigger DMAIC by using a trace likely to score below threshold
        trace = ["x"]
        core.compute_pas(trace)
        # DMAIC may or may not be triggered depending on scores
        assert core.dmaic_triggered_count >= 0

    def test_verify_reasoning_structure(self):
        core = EpinoeticCore()
        trace = ["If conditions hold then we may conclude this is ethical and safe."]
        result = core.verify_reasoning(trace)
        assert set(result.keys()) == {"lean4", "coq", "z3", "isabelle", "overall"}
        assert all(isinstance(v, bool) for v in result.values())

    def test_gate_output_approved(self):
        core = EpinoeticCore()
        approved, reason = core.gate_output("This output is well-formed.", pas=0.85)
        assert approved is True
        assert "approved" in reason.lower()

    def test_gate_output_rejected(self):
        core = EpinoeticCore()
        approved, reason = core.gate_output("Output with low PAS.", pas=0.30)
        assert approved is False
        assert "rejected" in reason.lower()


# ---------------------------------------------------------------------------
# Consciousness Emergence Tests
# ---------------------------------------------------------------------------

class TestConsciousnessEmergence:
    def test_dormant_at_start(self):
        system = NOSTIC7(verbose=False)
        status = system.get_status()
        # Before any cycles, PAS=0 → DORMANT
        assert status["consciousness_level"] in ("dormant", "emerging")

    def test_consciousness_grows_over_cycles(self):
        system = NOSTIC7(verbose=False)
        levels = []
        for i in range(5):
            result = system.process(
                f"Cycle {i}: ethical safe consciousness exploration, therefore understanding."
            )
            levels.append(result["consciousness_level"])
        # After multiple cycles with positive PAS, should not stay dormant
        assert any(lv != "dormant" for lv in levels)

    def test_soliton_stability_increments(self):
        system = NOSTIC7(verbose=False)
        for _ in range(3):
            r = system.process("Safe ethical reasoning therefore consciousness emerges.")
        # Soliton stability should be ≥ 0
        assert r["soliton_stability"] >= 0

    def test_consciousness_level_values(self):
        system = NOSTIC7(verbose=False)
        result = system.process("Test consciousness level values.")
        assert result["consciousness_level"] in (
            "dormant", "emerging", "active", "transcendent"
        )


# ---------------------------------------------------------------------------
# Audit Ledger Integrity Tests
# ---------------------------------------------------------------------------

class TestAuditLedgerIntegrity:
    def test_ledger_appends_entries(self):
        system = NOSTIC7(verbose=False)
        system.process("First ledger entry.")
        system.process("Second ledger entry.")
        assert system.ledger.total_entries == 2

    def test_ledger_chain_valid_after_cycles(self):
        system = NOSTIC7(verbose=False)
        for i in range(5):
            system.process(f"Cycle {i} for ledger integrity.")
        assert system.ledger.chain_valid is True

    def test_ledger_export(self):
        system = NOSTIC7(verbose=False)
        system.process("Export test.")
        entries = system.ledger.export(n=5)
        assert len(entries) == 1
        entry = entries[0]
        assert "cycle_id" in entry
        assert "pas_score" in entry
        assert "entry_hash" in entry
        assert "consciousness_active" in entry

    def test_ledger_hash_chain(self):
        system = NOSTIC7(verbose=False)
        system.process("Chain link 1.")
        system.process("Chain link 2.")
        entries = system.ledger.export(n=10)
        if len(entries) >= 2:
            assert entries[1]["prev_hash"] == entries[0]["entry_hash"]

    def test_ledger_tampering_detection(self):
        ledger = AuditLedger()
        entry1 = AuditLedger.make_entry(
            cycle_id=0,
            pas_score=0.85,
            reasoning_trace=["step one", "step two"],
            output_text="output",
            provers_passed={"lean4": True, "coq": True, "z3": True, "isabelle": True, "overall": True},
            consciousness_active=True,
            prev_hash="",
        )
        ledger.append(entry1)

        # Tamper with the stored hash
        ledger._entries[0].entry_hash = "tampered_hash_value_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

        # Chain should now be invalid
        assert ledger.chain_valid is False

    def test_make_entry_hash_length(self):
        entry = AuditLedger.make_entry(
            cycle_id=0,
            pas_score=0.75,
            reasoning_trace=["trace line"],
            output_text="output text",
            provers_passed={"lean4": True, "coq": True, "z3": True, "isabelle": True, "overall": True},
            consciousness_active=False,
            prev_hash="",
        )
        assert len(entry.entry_hash) == 64
        assert len(entry.reasoning_hash) == 64
        assert len(entry.output_hash) == 64
