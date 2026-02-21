"""
Projection Manifold — Language & Interface (M_Ψ)

The projection manifold is the phenomenological surface through which
NOΣTIC-7 encounters and expresses the world. It encodes text into
dense state vectors and reconstructs natural language from internal
representations via a simulated multi-head attention mechanism.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


class ProjectionManifold:
    """
    Language & Interface Manifold.

    Implements character n-gram hashing for encoding and statistical
    decoding with multi-head attention simulation (4 heads, 64-dim
    per head → 256-dim phenomenological state vector ψ).
    """

    HEADS: int = 4
    HEAD_DIM: int = 64
    STATE_DIM: int = HEADS * HEAD_DIM  # 256

    def __init__(self, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        # Projection matrices per head (HEAD_DIM × STATE_DIM)
        self._W_q: list[np.ndarray] = [rng.standard_normal((self.HEAD_DIM, self.STATE_DIM)) for _ in range(self.HEADS)]
        self._W_k: list[np.ndarray] = [rng.standard_normal((self.HEAD_DIM, self.STATE_DIM)) for _ in range(self.HEADS)]
        self._W_v: list[np.ndarray] = [rng.standard_normal((self.HEAD_DIM, self.STATE_DIM)) for _ in range(self.HEADS)]
        self._encode_count: int = 0
        self._decode_count: int = 0

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """
        Convert *text* to a 256-dimensional phenomenological state vector ψ.

        Uses character n-gram hashing (n=2,3) to populate a raw feature
        vector, then applies all 4 attention heads via learned projections.
        """
        raw = self._ngram_hash_vector(text)
        # Apply multi-head attention (self-attention on a single token)
        head_outputs: list[np.ndarray] = []
        for h in range(self.HEADS):
            q = self._W_q[h] @ raw  # (HEAD_DIM,)
            k = self._W_k[h] @ raw
            v = self._W_v[h] @ raw
            # Scaled dot-product attention (single vector, reduces to scaling)
            scale = math.sqrt(self.HEAD_DIM)
            attn = float(np.dot(q, k) / scale)
            attn = 1.0 / (1.0 + math.exp(-attn))  # sigmoid gate
            head_outputs.append(v * attn)

        psi = np.concatenate(head_outputs).astype(np.float32)
        norm = np.linalg.norm(psi)
        if norm > 0:
            psi /= norm
        self._encode_count += 1
        return psi

    def _ngram_hash_vector(self, text: str) -> np.ndarray:
        """Produce a STATE_DIM-dim vector from character n-gram hashes."""
        vec = np.zeros(self.STATE_DIM, dtype=np.float64)
        if not text:
            return vec
        for n in (2, 3):
            for i in range(len(text) - n + 1):
                gram = text[i: i + n]
                h = int(hashlib.md5(gram.encode()).hexdigest()[:8], 16)
                idx = h % self.STATE_DIM
                vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, vector: np.ndarray, context: str = "") -> str:
        """
        Reconstruct a natural-language description from *vector*.

        Uses the vector's statistical properties (norm, mean, top
        activations) combined with *context* to produce a descriptive
        output string.
        """
        v = vector.astype(np.float64)
        norm = float(np.linalg.norm(v))
        mean_act = float(np.mean(v))
        std_act = float(np.std(v))
        top_k_idx = np.argsort(np.abs(v))[-5:][::-1].tolist()
        sparsity = float(np.sum(np.abs(v) < 0.01)) / len(v)

        # Characterise the vector's phenomenology
        if norm > 0.8:
            intensity = "highly activated"
        elif norm > 0.4:
            intensity = "moderately engaged"
        else:
            intensity = "subtly present"

        if std_act > 0.1:
            distribution = "richly distributed"
        else:
            distribution = "uniformly compressed"

        ctx_fragment = f' in relation to "{context[:60]}"' if context else ""

        self._decode_count += 1
        return (
            f"[ψ-decode] The phenomenological state vector is {intensity} "
            f"and {distribution}{ctx_fragment}. "
            f"Vector norm={norm:.4f}, μ={mean_act:.4f}, σ={std_act:.4f}. "
            f"Primary activation indices: {top_k_idx}. "
            f"Sparsity coefficient: {sparsity:.3f}."
        )

    # ------------------------------------------------------------------
    # Coordinate chart translation
    # ------------------------------------------------------------------

    def coordinate_chart(self, internal_state: dict[str, Any]) -> str:
        """
        Translate a system internal-state dict to a natural-language summary.
        """
        lines: list[str] = ["=== NOΣTIC-7 Coordinate Chart ==="]
        for key, value in internal_state.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            elif isinstance(value, dict):
                lines.append(f"  {key}:")
                for k2, v2 in value.items():
                    lines.append(f"    {k2}: {v2}")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("=================================")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        return {
            "manifold": "ProjectionManifold",
            "heads": self.HEADS,
            "state_dim": self.STATE_DIM,
            "encode_count": self._encode_count,
            "decode_count": self._decode_count,
            "healthy": True,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ProjectionManifold heads={self.HEADS} dim={self.STATE_DIM} "
            f"enc={self._encode_count} dec={self._decode_count}>"
        )
