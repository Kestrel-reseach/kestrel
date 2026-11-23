"""Step 7 scaffold: router over the components that survived their gates.

Decided components so far:
- Verifier head (Step 3, passed) supplies correctness scores.
- SSD knowledge store (Step 5, passed) supplies retrieval when confidence is low.
- Halting: the Step 4 gate FAILED, so the product path keeps fixed two-pass
  (the plan's kill rule). The gated policy stays available for ablations only.
- Memory (Step 6) is not decided yet: the router accepts an optional memory
  reader and leaves that slot empty until A's result arrives.

The router is stdlib-only and tested with fakes. Real model/store wiring plugs
in through the same protocols used by features.py, halting.py and
knowledge_store.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "Decision",
    "Memory",
    "Retriever",
    "Router",
    "Verifier",
]


@dataclass(frozen=True)
class Decision:
    """One routing decision for a query."""

    route: str
    verifier_score: float
    used_store: bool
    used_memory: bool


class Verifier(Protocol):
    """Scores an answer's correctness probability."""

    def score(self, query: str, answer: str) -> float:
        """Return a score in [0, 1]."""
        ...


class Retriever(Protocol):
    """Retrieves context from the SSD store, reporting bytes read."""

    def retrieve(self, query: str) -> tuple[str, int]:
        """Return (context, bytes_read)."""
        ...


class Memory(Protocol):
    """Reads verified entries for a query."""

    def recall(self, query: str) -> str:
        """Return relevant memory text (empty when none)."""
        ...


class Router:
    """Chooses a path from verifier score and optional components."""

    def __init__(
        self,
        verifier: Verifier,
        retriever: Retriever | None = None,
        memory: Memory | None = None,
        *,
        threshold: float = 0.5,
        abstain_below: float = 0.1,
    ) -> None:
        if not 0 <= abstain_below <= threshold <= 1:
            raise ValueError("require 0 <= abstain_below <= threshold <= 1")
        self.verifier = verifier
        self.retriever = retriever
        self.memory = memory
        self.threshold = threshold
        self.abstain_below = abstain_below

    def decide(self, query: str, answer: str) -> Decision:
        """Route on the verifier score: abstain, retrieve, or answer directly."""
        score = self.verifier.score(query, answer)
        if score < self.abstain_below:
            return Decision("abstain", score, False, False)

        used_store = False
        if score < self.threshold and self.retriever is not None:
            self.retriever.retrieve(query)
            used_store = True

        used_memory = self.memory is not None and bool(self.memory.recall(query))
        if used_memory:
            route = "memory"
        elif used_store:
            route = "retrieve"
        else:
            route = "answer"
        return Decision(route, score, used_store, used_memory)
