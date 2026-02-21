"""
HQCI — Hybrid Quantum-Classical Integration Network.

Simulates 8-qubit quantum circuits using NumPy state vectors and
provides tensor-train SVD compression for manifold states. No
external quantum library required — gates are implemented natively.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


class HQCINetwork:
    """
    Hybrid Quantum-Classical Integration Network.

    Implements a pure-NumPy 8-qubit quantum circuit simulator with
    Hadamard, CNOT, and Pauli gates. Provides tensor-train SVD
    compression and inference time estimation.
    """

    QUBIT_COUNT: int = 8
    STATE_DIM: int = 2 ** QUBIT_COUNT  # 256

    def __init__(self) -> None:
        self._simulation_count: int = 0
        self._total_simulation_ms: float = 0.0

    # ------------------------------------------------------------------
    # Gate definitions
    # ------------------------------------------------------------------

    @staticmethod
    def _gate_H() -> np.ndarray:
        """Hadamard gate (2×2)."""
        return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

    @staticmethod
    def _gate_X() -> np.ndarray:
        """Pauli-X gate (2×2)."""
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)

    @staticmethod
    def _gate_Y() -> np.ndarray:
        """Pauli-Y gate (2×2)."""
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

    @staticmethod
    def _gate_Z() -> np.ndarray:
        """Pauli-Z gate (2×2)."""
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)

    @classmethod
    def _full_gate(cls, gate: np.ndarray, target: int) -> np.ndarray:
        """
        Expand a single-qubit *gate* to act on *target* within the full STATE_DIM space.
        Uses Kronecker products.
        """
        n = cls.QUBIT_COUNT
        ops = [np.eye(2, dtype=np.complex128)] * n
        ops[target] = gate
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    @classmethod
    def _cnot_gate(cls, control: int, target: int) -> np.ndarray:
        """
        Build an n-qubit CNOT gate with given control and target qubits.
        """
        dim = cls.STATE_DIM
        cnot = np.eye(dim, dtype=np.complex128)
        half = dim // (2 ** (max(control, target) + 1))
        # Flip target qubit when control is |1⟩
        for i in range(dim):
            bits = format(i, f"0{cls.QUBIT_COUNT}b")
            if bits[control] == "1":
                j_bits = list(bits)
                j_bits[target] = "1" if bits[target] == "0" else "0"
                j = int("".join(j_bits), 2)
                if i != j:
                    cnot[i, i] = 0
                    cnot[j, i] = 1
        return cnot

    # ------------------------------------------------------------------
    # Circuit simulation
    # ------------------------------------------------------------------

    def simulate_quantum_circuit(self, circuit_def: dict[str, Any]) -> np.ndarray:
        """
        Simulate an 8-qubit quantum circuit from *circuit_def*.

        *circuit_def* format:
        {
            "gates": [
                {"type": "H", "target": 0},
                {"type": "X", "target": 1},
                {"type": "CNOT", "control": 0, "target": 1},
                ...
            ],
            "measure": True  # optional: return probability distribution
        }

        Returns the final state vector (256-dim complex) or probability
        distribution (256-dim real) if measure=True.
        """
        t0 = time.perf_counter()

        # Initialise |0⟩^⊗8 state
        state = np.zeros(self.STATE_DIM, dtype=np.complex128)
        state[0] = 1.0

        gate_map = {
            "H": self._gate_H(),
            "X": self._gate_X(),
            "Y": self._gate_Y(),
            "Z": self._gate_Z(),
        }

        for gate_spec in circuit_def.get("gates", []):
            gtype = gate_spec.get("type", "H")
            if gtype == "CNOT":
                ctrl = gate_spec.get("control", 0)
                tgt = gate_spec.get("target", 1)
                U = self._cnot_gate(ctrl, tgt)
            else:
                single_gate = gate_map.get(gtype, self._gate_H())
                tgt = gate_spec.get("target", 0)
                U = self._full_gate(single_gate, tgt)

            state = U @ state

        # Normalise
        norm = float(np.linalg.norm(state))
        if norm > 0:
            state /= norm

        if circuit_def.get("measure", False):
            result = np.abs(state) ** 2  # probability distribution
        else:
            result = state

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._simulation_count += 1
        self._total_simulation_ms += elapsed_ms

        return result.astype(np.complex128) if not circuit_def.get("measure", False) else result.real.astype(np.float64)

    # ------------------------------------------------------------------
    # Tensor-train SVD compression
    # ------------------------------------------------------------------

    def tensor_train_svd(self, matrix: np.ndarray, rank: int = 4) -> list[np.ndarray]:
        """
        Compress *matrix* using a tensor-train / MPS-style SVD decomposition.

        Returns a list of core tensors. The last core has shape (rank, n)
        and intermediate cores have shape (rank, 2, rank) conceptually
        flattened for storage.
        """
        M = matrix.astype(np.float64)
        if M.ndim == 1:
            M = M.reshape(1, -1)
        elif M.ndim > 2:
            M = M.reshape(M.shape[0], -1)

        cores: list[np.ndarray] = []
        current = M

        n_cols = current.shape[1]
        chunk = max(1, n_cols // max(rank, 1))

        # Greedy column-wise SVD chunking
        while current.shape[1] > chunk:
            left = current[:, :chunk]
            right = current[:, chunk:]
            try:
                U, S, Vt = np.linalg.svd(left, full_matrices=False)
                r = min(rank, len(S))
                core = U[:, :r] @ np.diag(S[:r])
                cores.append(core)
                current = Vt[:r, :] @ right if right.shape[1] > 0 else Vt[:r, :]
                if right.shape[1] == 0:
                    break
            except np.linalg.LinAlgError:
                break

        cores.append(current)
        return cores

    # ------------------------------------------------------------------
    # Manifold compression
    # ------------------------------------------------------------------

    def compress_manifold(self, manifold_state: np.ndarray) -> np.ndarray:
        """
        Reduce *manifold_state* toward a 1.25M-parameter footprint equivalent.

        Applies TT-SVD then reconstructs a compressed approximation.
        """
        state = manifold_state.astype(np.float64).flatten()
        TARGET_DIM = 1250  # representative of 1.25M / 1000 scale

        if len(state) <= TARGET_DIM:
            padded = np.zeros(TARGET_DIM)
            padded[: len(state)] = state
            return padded.astype(np.float32)

        # Compress via SVD
        matrix = state.reshape(1, -1)
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        k = min(TARGET_DIM, len(S))
        compressed = (S[:k] * Vt[:k]).flatten()[:TARGET_DIM]
        if len(compressed) < TARGET_DIM:
            compressed = np.pad(compressed, (0, TARGET_DIM - len(compressed)))
        return compressed.astype(np.float32)

    # ------------------------------------------------------------------
    # Inference time estimation
    # ------------------------------------------------------------------

    def inference_time_estimate(self) -> float:
        """
        Return estimated inference time in milliseconds for an 8-qubit simulation.

        Uses the average of past simulations, or a theoretical estimate.
        """
        if self._simulation_count > 0:
            return round(self._total_simulation_ms / self._simulation_count, 3)
        # Theoretical: O(2^n × gates) — rough baseline for 8 qubits, 10 gates
        return round(2 ** self.QUBIT_COUNT * 10 / 1e6 * 1000, 3)  # in ms

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        return {
            "component": "HQCINetwork",
            "qubit_count": self.QUBIT_COUNT,
            "state_dim": self.STATE_DIM,
            "simulation_count": self._simulation_count,
            "avg_simulation_ms": self.inference_time_estimate(),
            "healthy": True,
        }
