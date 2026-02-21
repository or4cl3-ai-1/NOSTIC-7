"""
EchoNode — Distributed Deployment Infrastructure.

EchoNodes form the physical substrate of a NOΣTIC-7 deployment.
Each node registers its capabilities, and the EchoNodeNetwork routes
computation optimally. AgentPackets provide signed inter-node messaging.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# EchoNode
# ---------------------------------------------------------------------------

@dataclass
class EchoNode:
    """A single compute node in the EchoNode network."""

    node_id: str
    ip_address: str
    capacity: float           # 0.0–1.0 normalised compute capacity
    active_manifolds: list[str]
    last_heartbeat: float = field(default_factory=time.time)

    @property
    def load(self) -> float:
        """Estimate load from active manifold count vs capacity."""
        manifold_load = len(self.active_manifolds) * 0.1
        return min(manifold_load / max(self.capacity, 0.01), 1.0)

    def heartbeat(self) -> None:
        """Update the heartbeat timestamp."""
        self.last_heartbeat = time.time()

    def is_alive(self, timeout_secs: float = 60.0) -> bool:
        """Return True iff the node sent a heartbeat within *timeout_secs*."""
        return (time.time() - self.last_heartbeat) < timeout_secs


# ---------------------------------------------------------------------------
# AgentPacket
# ---------------------------------------------------------------------------

@dataclass
class AgentPacket:
    """
    A signed inter-agent message packet for Universal TCP/IP communication.

    Uses HMAC-SHA256 for signature verification.
    """

    packet_id: str
    source_node: str
    destination_node: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    signature: str = field(default="", init=False)

    _SECRET_KEY: bytes = b"nostic7-echo-key-v1"  # shared secret (replace in production)

    def sign(self) -> None:
        """Compute and store the HMAC signature of this packet."""
        canonical = json.dumps(
            {
                "packet_id": self.packet_id,
                "source_node": self.source_node,
                "destination_node": self.destination_node,
                "payload": self.payload,
                "timestamp": self.timestamp,
            },
            sort_keys=True,
        )
        self.signature = hmac.new(
            self._SECRET_KEY,
            canonical.encode(),
            hashlib.sha256,
        ).hexdigest()

    def verify(self) -> bool:
        """Return True iff the stored signature is valid."""
        expected = self.__class__(
            packet_id=self.packet_id,
            source_node=self.source_node,
            destination_node=self.destination_node,
            payload=self.payload,
            timestamp=self.timestamp,
        )
        expected.sign()
        return hmac.compare_digest(self.signature, expected.signature)

    @classmethod
    def create(
        cls,
        source: str,
        destination: str,
        payload: dict[str, Any],
    ) -> "AgentPacket":
        """Factory: create and sign a new AgentPacket."""
        pkt = cls(
            packet_id=str(uuid.uuid4()),
            source_node=source,
            destination_node=destination,
            payload=payload,
        )
        pkt.sign()
        return pkt


# ---------------------------------------------------------------------------
# EchoNodeNetwork
# ---------------------------------------------------------------------------

class EchoNodeNetwork:
    """
    Orchestrates a collection of EchoNodes.

    Provides node registration, intelligent computation routing,
    state broadcast, and health monitoring.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, EchoNode] = {}
        self._routed_tasks: int = 0
        self._broadcast_count: int = 0
        self._packet_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def register_node(self, node: EchoNode) -> None:
        """Add *node* to the network."""
        node.heartbeat()
        self._nodes[node.node_id] = node

    def deregister_node(self, node_id: str) -> None:
        """Remove a node from the network."""
        self._nodes.pop(node_id, None)

    # ------------------------------------------------------------------
    # Computation routing
    # ------------------------------------------------------------------

    def route_computation(self, task: dict[str, Any]) -> EchoNode:
        """
        Select the optimal node for *task* — the one with lowest load
        that is alive.

        Raises RuntimeError if no live nodes are available.
        """
        alive_nodes = [n for n in self._nodes.values() if n.is_alive()]
        if not alive_nodes:
            raise RuntimeError("EchoNodeNetwork: no alive nodes available for routing.")

        best = min(alive_nodes, key=lambda n: n.load)
        self._routed_tasks += 1
        return best

    # ------------------------------------------------------------------
    # State broadcast
    # ------------------------------------------------------------------

    def broadcast_state(self, state: dict[str, Any]) -> None:
        """Push a state update to all registered nodes (logged)."""
        entry = {
            "timestamp": time.time(),
            "state_keys": list(state.keys()),
            "node_count": len(self._nodes),
        }
        self._packet_log.append(entry)
        if len(self._packet_log) > 200:
            self._packet_log = self._packet_log[-200:]
        self._broadcast_count += 1

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Return network health metrics."""
        total = len(self._nodes)
        alive = sum(1 for n in self._nodes.values() if n.is_alive())
        avg_load = (
            sum(n.load for n in self._nodes.values()) / total
            if total > 0 else 0.0
        )
        return {
            "total_nodes": total,
            "alive_nodes": alive,
            "dead_nodes": total - alive,
            "avg_load": round(avg_load, 4),
            "routed_tasks": self._routed_tasks,
            "broadcast_count": self._broadcast_count,
            "healthy": alive > 0 or total == 0,
        }

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        return {
            "component": "EchoNodeNetwork",
            **self.health_check(),
        }

    def __repr__(self) -> str:  # pragma: no cover
        h = self.health_check()
        return (
            f"<EchoNodeNetwork nodes={h['total_nodes']} "
            f"alive={h['alive_nodes']} load={h['avg_load']:.3f}>"
        )
