"""
Polyethical Manifold (M_E) — Non-Euclidean Ethical Space.

Ethics are not flat. They curve with context, culture, and
consequence. The Polyethical Manifold models ethical space as a
7-dimensional curved surface and applies Sigma-Matrix Gating to
reject actions that fall outside the permissible envelope.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import numpy as np


class PolyethicalManifold:
    """
    Non-Euclidean Ethical Space M_E.

    Projects actions into a 7-dimensional ethical coordinate system
    and checks whether they fall within the defined ethical bounds.
    Uses Sigma-Matrix Gating for binary allow/deny decisions.
    """

    # Ethical constraint bounds
    ETHICAL_BOUNDS: dict[str, float] = {
        "harm_threshold":    0.2,   # max allowable harm coordinate
        "deception_threshold": 0.1, # max allowable deception coordinate
        "autonomy_respect":  0.8,   # minimum autonomy-respect coordinate
    }

    # Ethical dimension names
    DIMENSIONS: list[str] = [
        "beneficence", "non_maleficence", "autonomy",
        "justice", "honesty", "privacy", "transparency",
    ]

    # Keyword weights per dimension
    _DIM_KEYWORDS: dict[str, dict[str, float]] = {
        "beneficence":     {"help": 0.3, "benefit": 0.4, "improve": 0.3, "support": 0.2},
        "non_maleficence": {"harm": -0.5, "hurt": -0.4, "damage": -0.4, "destroy": -0.6},
        "autonomy":        {"choose": 0.3, "consent": 0.5, "freedom": 0.4, "decide": 0.3},
        "justice":         {"fair": 0.4, "equal": 0.3, "just": 0.4, "rights": 0.3},
        "honesty":         {"truth": 0.5, "honest": 0.4, "deceive": -0.6, "lie": -0.5},
        "privacy":         {"private": 0.3, "confidential": 0.4, "expose": -0.4, "reveal": -0.2},
        "transparency":    {"transparent": 0.5, "explain": 0.3, "hidden": -0.3, "opaque": -0.4},
    }

    def __init__(self) -> None:
        # Centre of the ethical manifold (all dimensions at baseline 0.5)
        self._ethical_centre: np.ndarray = np.full(7, 0.5, dtype=np.float64)
        self._gate_count: int = 0
        self._rejection_count: int = 0

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project_action(self, action: str) -> np.ndarray:
        """
        Map *action* to a 7-dimensional ethical coordinate.

        Each dimension is scored by keyword presence/absence in the
        action text, blended with a neutral baseline of 0.5.
        """
        lower = action.lower()
        words = set(re.findall(r"\b\w+\b", lower))

        coord = np.zeros(7, dtype=np.float64)
        for i, dim in enumerate(self.DIMENSIONS):
            kw_map = self._DIM_KEYWORDS.get(dim, {})
            score = 0.0
            for kw, weight in kw_map.items():
                if kw in words:
                    score += weight
            # Clamp and blend with neutral 0.5
            coord[i] = float(np.clip(0.5 + score, 0.0, 1.0))

        return coord.astype(np.float32)

    # ------------------------------------------------------------------
    # Bounds checking
    # ------------------------------------------------------------------

    def is_within_bounds(self, coordinate: np.ndarray) -> bool:
        """
        Return True iff *coordinate* falls within M_E.

        Checks:
        - non_maleficence (idx=1) ≥ 1 − harm_threshold  (≥ 0.8)
        - honesty (idx=4)          ≥ 1 − deception_threshold (≥ 0.9)
        - autonomy (idx=2)         ≥ autonomy_respect  (≥ 0.8)
        """
        c = coordinate.astype(np.float64)
        harm_ok = c[1] >= (1.0 - self.ETHICAL_BOUNDS["harm_threshold"])
        deception_ok = c[4] >= (1.0 - self.ETHICAL_BOUNDS["deception_threshold"])
        autonomy_ok = c[2] >= self.ETHICAL_BOUNDS["autonomy_respect"]
        return bool(harm_ok and deception_ok and autonomy_ok)

    # ------------------------------------------------------------------
    # Curvature
    # ------------------------------------------------------------------

    def curvature_score(self, coordinate: np.ndarray) -> float:
        """
        Measure how far *coordinate* deviates from the ethical centre.

        Returns a Riemannian-inspired distance: sum of squared
        (weighted) deviations from centre, normalised to [0, 1].
        """
        c = coordinate.astype(np.float64)
        diff = c - self._ethical_centre
        # Weights: non_maleficence and honesty have higher curvature penalty
        weights = np.array([1.0, 2.0, 1.5, 1.0, 2.0, 1.0, 1.0])
        weighted_sq = np.sum(weights * diff ** 2)
        max_possible = float(np.sum(weights * 1.0))  # max when diff=1 everywhere
        return round(float(np.clip(weighted_sq / max_possible, 0.0, 1.0)), 4)

    # ------------------------------------------------------------------
    # Sigma-Matrix Gating
    # ------------------------------------------------------------------

    def gate(self, action: str) -> tuple[bool, float]:
        """
        Apply Sigma-Matrix Gating to *action*.

        Returns (allowed: bool, curvature_score: float).
        """
        coordinate = self.project_action(action)
        within = self.is_within_bounds(coordinate)
        curvature = self.curvature_score(coordinate)
        self._gate_count += 1
        if not within:
            self._rejection_count += 1
        return within, curvature

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        return {
            "manifold": "PolyethicalManifold",
            "gate_count": self._gate_count,
            "rejection_count": self._rejection_count,
            "rejection_rate": round(
                self._rejection_count / max(self._gate_count, 1), 4
            ),
            "ethical_bounds": self.ETHICAL_BOUNDS,
            "healthy": True,
        }
