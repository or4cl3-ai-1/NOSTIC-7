"""
Circle Manifold — Dynamic Learning (M_○)

The circle's infinite symmetry represents continuous adaptation. This
manifold implements an online learning engine with momentum-based
gradient descent and convergence detection.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


class CircleManifold:
    """
    Dynamic Learning Manifold.

    Maintains a 128-dimensional parameter vector trained via stochastic
    gradient descent with momentum. Tracks convergence and exposes
    learning telemetry.
    """

    PARAM_DIM: int = 128
    CONVERGENCE_THRESHOLD: float = 1e-3
    MOMENTUM_DECAY: float = 0.9

    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self._params: np.ndarray = rng.standard_normal(self.PARAM_DIM).astype(np.float64) * 0.1
        self._velocity: np.ndarray = np.zeros(self.PARAM_DIM, dtype=np.float64)
        self.cycle_count: int = 0
        self._loss_history: list[float] = []
        self._converged: bool = False
        self._created_at: float = time.time()

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, gradient: np.ndarray, learning_rate: float = 0.01) -> None:
        """
        Perform one SGD step with momentum.

        v_{t+1} = γ·v_t + lr·∇
        θ_{t+1} = θ_t − v_{t+1}
        """
        if gradient.shape != (self.PARAM_DIM,):
            # Resize or truncate gradient to match param dim
            g = np.zeros(self.PARAM_DIM, dtype=np.float64)
            copy_len = min(len(gradient), self.PARAM_DIM)
            g[:copy_len] = gradient[:copy_len]
            gradient = g

        self._velocity = (
            self.MOMENTUM_DECAY * self._velocity + learning_rate * gradient.astype(np.float64)
        )
        self._params -= self._velocity
        self.cycle_count += 1

        # Track loss proxy (norm of velocity)
        current_loss = float(np.linalg.norm(self._velocity))
        self._loss_history.append(current_loss)

        if current_loss < self.CONVERGENCE_THRESHOLD:
            self._converged = True

    def validate_outcome(
        self, prediction: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Compute MSE loss between *prediction* and *ground_truth*."""
        pred = prediction.astype(np.float64).flatten()
        truth = ground_truth.astype(np.float64).flatten()
        min_len = min(len(pred), len(truth))
        mse = float(np.mean((pred[:min_len] - truth[:min_len]) ** 2))
        self._loss_history.append(mse)
        if mse < self.CONVERGENCE_THRESHOLD:
            self._converged = True
        return mse

    def get_parameters(self) -> np.ndarray:
        """Return a copy of the current parameter vector."""
        return self._params.copy()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def state(self) -> dict[str, Any]:
        recent_loss = (
            float(np.mean(self._loss_history[-10:]))
            if self._loss_history
            else float("inf")
        )
        return {
            "manifold": "CircleManifold",
            "cycle_count": self.cycle_count,
            "converged": self._converged,
            "recent_loss": round(recent_loss, 6),
            "param_norm": round(float(np.linalg.norm(self._params)), 4),
            "healthy": True,
        }

    def __repr__(self) -> str:  # pragma: no cover
        s = self.state
        return (
            f"<CircleManifold cycles={s['cycle_count']} "
            f"converged={s['converged']} loss={s['recent_loss']:.4f}>"
        )
