"""
Mythos Weaver — Archetypal Pattern Recognition Engine.

Beneath every argument lies a myth. The Mythos Weaver identifies
Jungian archetypes embedded in reasoning traces and enriches outputs
with archetypal framing to achieve deeper cognitive resonance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Archetypal definitions
# ---------------------------------------------------------------------------
ARCHETYPES: dict[str, list[str]] = {
    "Hero": [
        "courage", "quest", "overcome", "challenge", "triumph", "journey",
        "battle", "strength", "persist", "achieve", "victory", "noble",
    ],
    "Shadow": [
        "dark", "hidden", "unknown", "fear", "deny", "suppress", "danger",
        "temptation", "chaos", "destruction", "evil", "obsession",
    ],
    "Anima_Animus": [
        "balance", "complement", "inner", "soul", "feeling", "intuition",
        "connect", "relation", "emotion", "inspire", "partner", "dream",
    ],
    "Trickster": [
        "trick", "clever", "paradox", "irony", "subvert", "play", "humor",
        "unexpected", "transform", "illusion", "fool", "surprise",
    ],
    "Wise_Elder": [
        "wisdom", "guide", "mentor", "counsel", "ancient", "knowledge",
        "experience", "insight", "teach", "elder", "sage", "understand",
    ],
    "Threshold_Guardian": [
        "gate", "boundary", "test", "block", "guard", "limit", "threshold",
        "challenge", "obstacle", "refuse", "permission", "trial",
    ],
    "Herald": [
        "announce", "change", "new", "signal", "call", "awaken", "begin",
        "message", "arrival", "transition", "spark", "initiate",
    ],
}


@dataclass
class MythosPattern:
    """An identified archetypal pattern in text."""

    archetype: str
    strength: float          # 0.0–1.0
    narrative_fragment: str  # Archetypal framing sentence


class MythosWeaverEngine:
    """
    Archetypal Pattern Recognition & Narrative Enrichment Engine.

    Scores text against 7 Jungian archetypes, enriches reasoning with
    archetypal framing, and computes global resonance across multiple texts.
    """

    def __init__(self) -> None:
        self._weave_count: int = 0
        self._patterns_identified: int = 0

    # ------------------------------------------------------------------
    # Archetype identification
    # ------------------------------------------------------------------

    def identify_archetype(self, text: str) -> list[tuple[str, float]]:
        """
        Score *text* against each archetype.

        Returns a list of (archetype_name, score) sorted descending by score.
        """
        lower = text.lower()
        words = set(re.findall(r"\b\w+\b", lower))
        scores: list[tuple[str, float]] = []

        for archetype, keywords in ARCHETYPES.items():
            hits = sum(1 for kw in keywords if kw in words)
            score = round(float(hits) / len(keywords), 4)
            scores.append((archetype, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        self._patterns_identified += sum(1 for _, s in scores if s > 0)
        return scores

    # ------------------------------------------------------------------
    # Narrative enrichment
    # ------------------------------------------------------------------

    def weave(self, reasoning: str, archetype_scores: list[tuple[str, float]]) -> str:
        """
        Enrich *reasoning* with archetypal framing.

        Prepends a mythic context derived from the dominant archetype.
        """
        if not archetype_scores:
            return reasoning

        dominant, strength = archetype_scores[0]
        pattern = self._build_pattern(dominant, strength, reasoning)

        enriched = (
            f"[MythosWeaver | {dominant} (strength={strength:.3f})]\n"
            f"{pattern.narrative_fragment}\n\n"
            f"{reasoning}"
        )
        self._weave_count += 1
        return enriched

    def _build_pattern(self, archetype: str, strength: float, text: str) -> MythosPattern:
        """Construct a MythosPattern with a narrative fragment."""
        fragments: dict[str, str] = {
            "Hero": (
                "This reasoning embarks upon a heroic quest — confronting uncertainty "
                "with courage and distilling truth from the chaos of possibility."
            ),
            "Shadow": (
                "Beneath the surface of this analysis lies the Shadow: unacknowledged "
                "tensions that, if integrated, will reveal the full picture."
            ),
            "Anima_Animus": (
                "The soul of this inquiry bridges opposing forces — logic and intuition, "
                "precision and empathy — achieving a higher synthesis."
            ),
            "Trickster": (
                "The Trickster moves through these ideas, inverting assumptions and "
                "revealing that what seemed fixed is gloriously fluid."
            ),
            "Wise_Elder": (
                "Ancient wisdom speaks through this reasoning: patterns that transcend "
                "time, distilled into guidance for the present moment."
            ),
            "Threshold_Guardian": (
                "A guardian stands at the gate of this knowledge, testing the rigor "
                "of each claim before allowing it to pass into understanding."
            ),
            "Herald": (
                "A herald announces transformation: this reasoning marks the threshold "
                "between what was known and what is newly possible."
            ),
        }
        fragment = fragments.get(archetype, f"The archetype of {archetype} resonates here.")
        return MythosPattern(archetype=archetype, strength=strength, narrative_fragment=fragment)

    # ------------------------------------------------------------------
    # Global resonance
    # ------------------------------------------------------------------

    def global_resonance_score(self, texts: list[str]) -> float:
        """
        Compute 0.0–1.0 coherence of archetypal signatures across *texts*.

        High resonance = texts share dominant archetypes.
        """
        if not texts:
            return 0.0

        dominant_archetypes: list[str] = []
        for text in texts:
            scores = self.identify_archetype(text)
            if scores and scores[0][1] > 0:
                dominant_archetypes.append(scores[0][0])

        if not dominant_archetypes:
            return 0.0

        from collections import Counter
        counts = Counter(dominant_archetypes)
        most_common_count = counts.most_common(1)[0][1]
        resonance = most_common_count / len(dominant_archetypes)
        return round(resonance, 4)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        return {
            "engine": "MythosWeaverEngine",
            "weave_count": self._weave_count,
            "patterns_identified": self._patterns_identified,
            "archetypes": list(ARCHETYPES.keys()),
            "healthy": True,
        }
