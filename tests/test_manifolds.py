"""
Tests for all 7 NOΣTIC-7 Cognitive Manifolds.

Run with: pytest tests/test_manifolds.py -v
"""

import numpy as np
import pytest

from nostic7.manifolds.square import SquareManifold
from nostic7.manifolds.triangle import TriangleManifold
from nostic7.manifolds.circle import CircleManifold
from nostic7.manifolds.pentagon import PentagonManifold
from nostic7.manifolds.hexagon import HexagonManifold
from nostic7.manifolds.heptagon import HeptagonManifold
from nostic7.manifolds.projection import ProjectionManifold


# ---------------------------------------------------------------------------
# Square Manifold
# ---------------------------------------------------------------------------

class TestSquareManifold:
    def test_square_belief_update(self):
        m = SquareManifold()
        m.update_belief("Alice", {"role": "engineer", "domain": "AI"})
        entity = m.get_entity("Alice")
        assert entity is not None
        assert entity["role"] == "engineer"
        assert entity["domain"] == "AI"

    def test_square_identity_vector_shape(self):
        m = SquareManifold()
        m.update_belief("consciousness", {"type": "synthetic", "origin": "geometric"})
        vec = m.compute_identity_vector("consciousness")
        assert vec.shape == (64,)
        assert not np.any(np.isnan(vec))

    def test_square_identity_vector_empty(self):
        m = SquareManifold()
        vec = m.compute_identity_vector("no match here at all")
        assert vec.shape == (64,)

    def test_square_provenance_hash(self):
        m = SquareManifold()
        m.update_belief("entity_x", {"value": 42})
        h1 = m.provenance_hash("entity_x")
        m.update_belief("entity_x", {"value": 99})
        h2 = m.provenance_hash("entity_x")
        assert isinstance(h1, str) and len(h1) == 64
        assert h1 != h2  # state changed

    def test_square_state(self):
        m = SquareManifold()
        m.update_belief("node_a", {"x": 1})
        s = m.state
        assert s["entity_count"] == 1
        assert 0.0 <= s["stability_score"] <= 1.0
        assert s["healthy"] is True


# ---------------------------------------------------------------------------
# Triangle Manifold
# ---------------------------------------------------------------------------

class TestTriangleManifold:
    def test_triangle_fork_hypotheses(self):
        m = TriangleManifold()
        agents = m.fork_hypotheses("The sky is blue because of Rayleigh scattering", n=3)
        assert len(agents) == 3
        for agent in agents:
            assert 0.0 <= agent.confidence <= 1.0
            assert isinstance(agent.hypothesis, str)
            assert len(agent.hypothesis) > 10

    def test_triangle_fork_deterministic(self):
        m = TriangleManifold()
        agents1 = m.fork_hypotheses("identical premise", n=3)
        agents2 = m.fork_hypotheses("identical premise", n=3)
        # Same premise → same hypotheses
        for a1, a2 in zip(agents1, agents2):
            assert a1.hypothesis == a2.hypothesis
            assert a1.confidence == a2.confidence

    def test_triangle_lean4_check(self):
        m = TriangleManifold()
        assert m.lean4_fork_check("If A then B, and B implies C, therefore A implies C.")
        assert not m.lean4_fork_check("always never")

    def test_triangle_select_best(self):
        m = TriangleManifold()
        agents = m.fork_hypotheses("consciousness emerges from recursive self-modelling", n=5)
        best = m.select_best(agents)
        assert best in agents

    def test_triangle_state(self):
        m = TriangleManifold()
        m.fork_hypotheses("test", n=3)
        s = m.state
        assert s["fork_count"] == 1
        assert s["healthy"] is True


# ---------------------------------------------------------------------------
# Circle Manifold
# ---------------------------------------------------------------------------

class TestCircleManifold:
    def test_circle_sgd_update(self):
        m = CircleManifold()
        grad = np.ones(128, dtype=np.float32) * 0.01
        params_before = m.get_parameters().copy()
        m.update(grad, learning_rate=0.01)
        params_after = m.get_parameters()
        assert not np.allclose(params_before, params_after)

    def test_circle_validate_outcome(self):
        m = CircleManifold()
        pred = np.ones(64) * 0.5
        truth = np.ones(64) * 0.5
        mse = m.validate_outcome(pred, truth)
        assert mse == pytest.approx(0.0, abs=1e-9)

    def test_circle_cycle_count(self):
        m = CircleManifold()
        for _ in range(5):
            m.update(np.zeros(128))
        assert m.cycle_count == 5

    def test_circle_state(self):
        m = CircleManifold()
        s = m.state
        assert s["manifold"] == "CircleManifold"
        assert s["healthy"] is True
        assert s["cycle_count"] == 0


# ---------------------------------------------------------------------------
# Pentagon Manifold
# ---------------------------------------------------------------------------

class TestPentagonManifold:
    def test_pentagon_index_and_retrieve(self):
        m = PentagonManifold()
        m.index("The nature of consciousness is deeply recursive.", {"source": "test"})
        m.index("Artificial intelligence mimics cognitive patterns.", {"source": "test"})
        m.index("Mathematics underlies all physical reality.", {"source": "test"})
        results = m.retrieve("consciousness recursive", k=2)
        assert len(results) <= 2
        assert all("text" in r for r in results)

    def test_pentagon_retrieve_empty(self):
        m = PentagonManifold()
        results = m.retrieve("anything")
        assert results == []

    def test_pentagon_extract_patterns(self):
        m = PentagonManifold()
        for _ in range(5):
            m.index("machine learning deep neural networks")
        docs = m.document_store
        patterns = m.extract_patterns(docs)
        assert isinstance(patterns, list)

    def test_pentagon_meta_insight(self):
        m = PentagonManifold()
        insight = m.meta_insight(["deep learning", "neural network", "gradient descent"])
        assert isinstance(insight, str)
        assert len(insight) > 20

    def test_pentagon_state(self):
        m = PentagonManifold()
        m.index("test document")
        s = m.state
        assert s["document_count"] == 1
        assert s["healthy"] is True


# ---------------------------------------------------------------------------
# Hexagon Manifold
# ---------------------------------------------------------------------------

class TestHexagonManifold:
    def test_hexagon_register_and_sync(self):
        m = HexagonManifold()
        sq = SquareManifold()
        m.register_manifold("square", sq)
        snapshot = m.sync()
        assert "square" in snapshot
        assert snapshot["square"]["manifold"] == "SquareManifold"

    def test_hexagon_coherence(self):
        m = HexagonManifold()
        sq = SquareManifold()
        m.register_manifold("square", sq)
        m.sync()
        score = m.coherence_score()
        assert 0.0 <= score <= 1.0

    def test_hexagon_coherence_empty(self):
        m = HexagonManifold()
        assert m.coherence_score() == 1.0

    def test_hexagon_event_log(self):
        m = HexagonManifold()
        m.broadcast("test_event", {"key": "value"})
        log = m.get_event_log()
        assert any(e["event"] == "test_event" for e in log)

    def test_hexagon_state(self):
        m = HexagonManifold()
        s = m.state
        assert s["manifold"] == "HexagonManifold"
        assert s["healthy"] is True


# ---------------------------------------------------------------------------
# Heptagon Manifold
# ---------------------------------------------------------------------------

class TestHeptagonManifold:
    def test_heptagon_affective_state(self):
        m = HeptagonManifold()
        state = m.estimate_affective_state("This is a safe and ethical decision.")
        assert set(state.keys()) == set(m.VALUE_DIMS)
        assert all(0.0 <= v <= 1.0 for v in state.values())

    def test_heptagon_risk_reward(self):
        m = HeptagonManifold()
        risk, reward = m.compute_risk_reward("creative exploration")
        assert 0.0 <= risk <= 1.0
        assert 0.0 <= reward <= 1.0

    def test_heptagon_trajectory_forecast(self):
        m = HeptagonManifold()
        state = m.estimate_affective_state("curious and resilient")
        traj = m.trajectory_forecast(state, steps=5)
        assert len(traj) == 5
        for step in traj:
            assert all(0.0 <= v <= 1.0 for v in step.values())

    def test_heptagon_emotional_context(self):
        m = HeptagonManifold()
        state = {"safety": 0.8, "ethics": 0.9, "creativity": 0.4,
                 "curiosity": 0.5, "empathy": 0.7, "resilience": 0.6,
                 "transcendence": 0.3}
        label = m.emotional_context(state)
        assert isinstance(label, str)
        assert len(label) > 5

    def test_heptagon_state(self):
        m = HeptagonManifold()
        s = m.state
        assert s["manifold"] == "HeptagonManifold"
        assert s["healthy"] is True


# ---------------------------------------------------------------------------
# Projection Manifold
# ---------------------------------------------------------------------------

class TestProjectionManifold:
    def test_projection_encode_shape(self):
        m = ProjectionManifold()
        psi = m.encode("What is the nature of consciousness?")
        assert psi.shape == (256,)
        assert not np.any(np.isnan(psi))

    def test_projection_encode_deterministic(self):
        m = ProjectionManifold()
        psi1 = m.encode("hello world")
        psi2 = m.encode("hello world")
        assert np.allclose(psi1, psi2)

    def test_projection_encode_different_inputs(self):
        m = ProjectionManifold()
        psi1 = m.encode("hello")
        psi2 = m.encode("goodbye")
        assert not np.allclose(psi1, psi2)

    def test_projection_decode(self):
        m = ProjectionManifold()
        psi = m.encode("synthetic consciousness emerges")
        decoded = m.decode(psi, context="consciousness")
        assert isinstance(decoded, str)
        assert "ψ-decode" in decoded

    def test_projection_coordinate_chart(self):
        m = ProjectionManifold()
        chart = m.coordinate_chart({"pas_score": 0.85, "cycles": 7})
        assert "NOΣTIC-7" in chart
        assert "pas_score" in chart

    def test_projection_state(self):
        m = ProjectionManifold()
        m.encode("test")
        s = m.state
        assert s["encode_count"] == 1
        assert s["healthy"] is True
