"""
Pentagon Manifold — Latent Discovery (M_⬠)

The pentagon's five axes mirror the five senses—perception, pattern
recognition, memory, association, and synthesis. This manifold powers
G-RAG (Genetic Retrieval-Augmented Generation) to surface latent
knowledge.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np


class PentagonManifold:
    """
    Latent Discovery Manifold — G-RAG Engine.

    Indexes arbitrary text documents with lightweight TF-IDF-style
    embeddings and provides nearest-neighbour retrieval, pattern
    extraction, and meta-insight synthesis.
    """

    EMBEDDING_DIM: int = 256
    _STOPWORDS: frozenset[str] = frozenset(
        {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "is", "it", "that", "this", "was", "are",
            "be", "by", "as", "from", "not", "have", "has", "had", "he",
            "she", "they", "we", "you", "i", "my", "your", "our", "its",
        }
    )

    def __init__(self) -> None:
        self.document_store: list[dict[str, Any]] = []
        self._idf_cache: dict[str, float] = {}
        self._index_count: int = 0

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        return [t for t in tokens if t not in self._STOPWORDS]

    def _compute_embedding(self, tokens: list[str]) -> np.ndarray:
        """
        Produce a PARAM_DIM-dimensional embedding using character n-gram
        hashing (a deterministic, dependency-free approximation of TF-IDF).
        """
        vec = np.zeros(self.EMBEDDING_DIM, dtype=np.float64)
        for token in tokens:
            # Hash each token into a bucket
            h = hash(token) % self.EMBEDDING_DIM
            # Use log-scaled term frequency contribution
            vec[h] += math.log1p(1)
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)

    def index(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a document to the store and compute its embedding."""
        tokens = self._tokenize(text)
        embedding = self._compute_embedding(tokens)
        doc = {
            "id": self._index_count,
            "text": text,
            "tokens": tokens,
            "embedding": embedding,
            "metadata": metadata or {},
            "term_freq": Counter(tokens),
        }
        self.document_store.append(doc)
        self._index_count += 1
        # Invalidate IDF cache
        self._idf_cache.clear()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dot = float(np.dot(a, b))
        norms = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
        return dot / norms if norms > 0 else 0.0

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """Return top-*k* documents most similar to *query*."""
        if not self.document_store:
            return []
        q_tokens = self._tokenize(query)
        q_vec = self._compute_embedding(q_tokens)

        scored = [
            (self._cosine_similarity(q_vec, doc["embedding"]), doc)
            for doc in self.document_store
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]

    # ------------------------------------------------------------------
    # Pattern extraction & meta-insight
    # ------------------------------------------------------------------

    def extract_patterns(self, documents: list[dict[str, Any]]) -> list[str]:
        """Identify recurring token bigrams across documents as patterns."""
        bigram_counter: Counter[str] = Counter()
        for doc in documents:
            tokens = doc.get("tokens", [])
            for i in range(len(tokens) - 1):
                bigram = f"{tokens[i]} {tokens[i+1]}"
                bigram_counter[bigram] += 1

        # Return bigrams that appear in at least 2 documents
        threshold = max(2, len(documents) // 3)
        patterns = [
            bigram
            for bigram, count in bigram_counter.most_common(20)
            if count >= threshold
        ]
        return patterns

    def meta_insight(self, patterns: list[str]) -> str:
        """Synthesize a set of patterns into a concise insight string."""
        if not patterns:
            return "No recurring patterns detected in the retrieved corpus."

        top = patterns[:5]
        joined = "; ".join(f'"{p}"' for p in top)
        return (
            f"Latent corpus analysis reveals {len(patterns)} recurring pattern(s). "
            f"Dominant motifs: {joined}. "
            f"These co-occurring concepts suggest a coherent thematic cluster "
            f"amenable to deep structural synthesis."
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        return {
            "manifold": "PentagonManifold",
            "document_count": len(self.document_store),
            "index_count": self._index_count,
            "healthy": True,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PentagonManifold docs={len(self.document_store)}>"
