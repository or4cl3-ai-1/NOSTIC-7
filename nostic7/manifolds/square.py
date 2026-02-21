"""
Square Manifold — Persistence & Identity (M_KG)

The square's four corners represent the cardinal anchors of identity:
past, present, future, and the eternal now. This manifold maintains
the knowledge graph that grounds NOΣTIC-7's sense of self.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import numpy as np


class SquareManifold:
    """
    Belief Manifold — Persistence & Identity.

    Maintains an entity-centric knowledge graph (M_KG) where each entity
    is a node with typed attributes. Provides identity vectors for
    context-sensitive reasoning and cryptographic provenance for trust.
    """

    EMBEDDING_DIM: int = 64

    def __init__(self) -> None:
        self._knowledge_graph: dict[str, dict[str, Any]] = {}
        self._provenance_history: dict[str, list[str]] = {}
        self._created_at: float = time.time()
        self._update_count: int = 0

    # ------------------------------------------------------------------
    # Core knowledge-graph operations
    # ------------------------------------------------------------------

    def update_belief(self, entity: str, attributes: dict[str, Any]) -> None:
        """Add or merge *entity* into the knowledge graph with given attributes."""
        if entity not in self._knowledge_graph:
            self._knowledge_graph[entity] = {}
            self._provenance_history[entity] = []

        self._knowledge_graph[entity].update(attributes)
        self._knowledge_graph[entity]["__last_updated"] = time.time()

        # Record provenance snapshot
        snapshot = self.provenance_hash(entity)
        self._provenance_history[entity].append(snapshot)
        self._update_count += 1

    def get_entity(self, entity: str) -> dict[str, Any] | None:
        """Return the full attribute dict for *entity*, or None if absent."""
        return self._knowledge_graph.get(entity)

    # ------------------------------------------------------------------
    # Embedding & identity
    # ------------------------------------------------------------------

    def compute_identity_vector(self, context: str) -> np.ndarray:
        """
        Generate a 64-dimensional belief embedding from entities relevant
        to *context*.

        Relevance is determined by substring/token overlap between
        entity names/attributes and the context string. Each matching
        entity contributes a hash-derived row to the embedding.
        """
        context_tokens = set(context.lower().split())
        relevant_entities: list[str] = []

        for entity, attrs in self._knowledge_graph.items():
            entity_tokens = set(entity.lower().split())
            attr_text = " ".join(str(v) for v in attrs.values()).lower()
            attr_tokens = set(attr_text.split())
            if context_tokens & (entity_tokens | attr_tokens):
                relevant_entities.append(entity)

        if not relevant_entities:
            # Return a deterministic zero-ish vector seeded by context
            rng = np.random.default_rng(
                int(hashlib.sha256(context.encode()).hexdigest()[:16], 16)
            )
            return rng.standard_normal(self.EMBEDDING_DIM).astype(np.float32) * 0.01

        accumulator = np.zeros(self.EMBEDDING_DIM, dtype=np.float64)
        for entity in relevant_entities:
            seed = int(hashlib.sha256(entity.encode()).hexdigest()[:16], 16) % (2**32)
            rng = np.random.default_rng(seed)
            row = rng.standard_normal(self.EMBEDDING_DIM)
            # Weight by recency
            attrs = self._knowledge_graph[entity]
            ts = attrs.get("__last_updated", self._created_at)
            age = max(time.time() - ts, 1.0)
            weight = 1.0 / np.log1p(age)
            accumulator += row * weight

        norm = np.linalg.norm(accumulator)
        if norm > 0:
            accumulator /= norm
        return accumulator.astype(np.float32)

    def provenance_hash(self, entity: str) -> str:
        """Return a SHA-256 digest of the current state of *entity*."""
        data = self._knowledge_graph.get(entity, {})
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    # ------------------------------------------------------------------
    # State reporting
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        """Return a summary of the manifold's current state."""
        entity_count = len(self._knowledge_graph)
        # Stability score: fraction of entities with >1 provenance entry
        if entity_count == 0:
            stability = 1.0
        else:
            stable = sum(
                1 for h in self._provenance_history.values() if len(h) > 1
            )
            stability = stable / entity_count

        return {
            "manifold": "SquareManifold",
            "entity_count": entity_count,
            "update_count": self._update_count,
            "stability_score": round(stability, 4),
            "healthy": entity_count >= 0,  # always healthy; extend as needed
        }

    def __repr__(self) -> str:  # pragma: no cover
        s = self.state
        return (
            f"<SquareManifold entities={s['entity_count']} "
            f"stability={s['stability_score']:.3f}>"
        )
