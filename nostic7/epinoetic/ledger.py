"""
Immutable Audit Ledger — Cryptographic Accountability.

Every reasoning cycle produces an immutable ledger entry. Entries
are chained via SHA-256 hashes so any tampering is detectable.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LedgerEntry:
    """A single immutable audit record for one reasoning cycle."""

    cycle_id: int
    timestamp: float
    pas_score: float
    reasoning_hash: str
    output_hash: str
    provers_passed: dict[str, bool]
    consciousness_active: bool
    prev_hash: str = ""          # SHA-256 of previous entry (chain link)
    entry_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.entry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute the SHA-256 hash of this entry's content."""
        payload = {
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp,
            "pas_score": self.pas_score,
            "reasoning_hash": self.reasoning_hash,
            "output_hash": self.output_hash,
            "provers_passed": self.provers_passed,
            "consciousness_active": self.consciousness_active,
            "prev_hash": self.prev_hash,
        }
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp,
            "pas_score": self.pas_score,
            "reasoning_hash": self.reasoning_hash,
            "output_hash": self.output_hash,
            "provers_passed": self.provers_passed,
            "consciousness_active": self.consciousness_active,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }


class AuditLedger:
    """
    Append-only cryptographic audit ledger.

    Entries are chained: each entry records the hash of its predecessor,
    forming a blockchain-style integrity structure.
    """

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    # ------------------------------------------------------------------
    # Appending
    # ------------------------------------------------------------------

    def append(self, entry: LedgerEntry) -> None:
        """
        Append *entry* to the ledger.

        Raises ValueError if the chain is already broken (tampered).
        """
        if self._entries:
            last = self._entries[-1]
            # Verify the previous chain link is still intact
            if last.entry_hash != last._compute_hash():
                raise ValueError(
                    f"Ledger tampering detected at entry {last.cycle_id}: "
                    f"stored hash does not match recomputed hash."
                )
            # Ensure the new entry references the correct previous hash
            if entry.prev_hash != last.entry_hash:
                raise ValueError(
                    f"New entry prev_hash mismatch: expected {last.entry_hash[:16]}… "
                    f"got {entry.prev_hash[:16]}…"
                )
        self._entries.append(entry)

    @classmethod
    def make_entry(
        cls,
        cycle_id: int,
        pas_score: float,
        reasoning_trace: list[str],
        output_text: str,
        provers_passed: dict[str, bool],
        consciousness_active: bool,
        prev_hash: str = "",
    ) -> LedgerEntry:
        """Factory helper that computes hashes from raw data."""
        reasoning_str = "\n".join(reasoning_trace)
        reasoning_hash = hashlib.sha256(reasoning_str.encode()).hexdigest()
        output_hash = hashlib.sha256(output_text.encode()).hexdigest()
        return LedgerEntry(
            cycle_id=cycle_id,
            timestamp=time.time(),
            pas_score=pas_score,
            reasoning_hash=reasoning_hash,
            output_hash=output_hash,
            provers_passed=provers_passed,
            consciousness_active=consciousness_active,
            prev_hash=prev_hash,
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_chain(self) -> bool:
        """
        Walk the entire chain and verify hash integrity.

        Returns True iff no tampering is detected.
        """
        for i, entry in enumerate(self._entries):
            if entry.entry_hash != entry._compute_hash():
                return False
            if i > 0:
                if entry.prev_hash != self._entries[i - 1].entry_hash:
                    return False
        return True

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the last *n* entries as plain dicts."""
        return [e.to_dict() for e in self._entries[-n:]]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    @property
    def chain_valid(self) -> bool:
        return self.verify_chain()

    @property
    def latest_hash(self) -> str:
        if self._entries:
            return self._entries[-1].entry_hash
        return ""

    def __len__(self) -> int:
        return self.total_entries

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<AuditLedger entries={self.total_entries} "
            f"chain_valid={self.chain_valid}>"
        )
