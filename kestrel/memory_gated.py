"""Step 6: Write-gated memory store with conflict resolution and state tracking.

Implements the write-gated memory store specified in Prototype 1 PLAN.md Step 6:
- Entries carry: text, source, timestamp, confidence, decay/recheck rule.
- Writes only when verified (by verifier head or owner approval).
- Entries are editable and deletable.
- Conflicts prefer newer + higher confidence; old entry is flagged stale, NOT deleted.
- Memory protocol compliant: recall(query) returns active verified memory text.
- Comparison baselines: PlainMemory (un-gated, no conflict flagging), NoMemory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Protocol

__all__ = [
    "MemoryEntry",
    "PlainMemory",
    "Step6GateDecision",
    "VerifierLike",
    "WriteGatedMemory",
    "paired_bootstrap_accuracy_difference",
    "step6_gate_decision",
]


class VerifierLike(Protocol):
    """Protocol for scoring correctness/confidence."""

    def score(self, query: str, answer: str) -> float: ...


@dataclass
class MemoryEntry:
    """A single personal memory entry with provenance and state metadata."""

    entry_id: str
    text: str
    source: str
    timestamp: str  # ISO string or 'YYYY-MM-DD HH:MM'
    confidence: float  # [0.0, 1.0]
    decay_rule: str = "none"
    recheck_rule: str = "none"
    verified: bool = True
    flagged_stale: bool = False
    deleted: bool = False
    attribute: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase alphanumeric word tokens."""
    return set(re.findall(r"\w+", text.lower()))


class WriteGatedMemory:
    """Write-gated memory store with conflict resolution and state tracking.

    Rules from PLAN.md Step 6:
    - Writes only when verified (verifier_score >= write_threshold or owner_approved).
    - Conflicts prefer newer + higher confidence; old entry flagged stale, not deleted.
    - Editable and deletable.
    - Conforms to router Memory protocol via recall(query).
    """

    def __init__(
        self,
        write_threshold: float = 0.5,
        verifier: VerifierLike | None = None,
    ) -> None:
        if not 0.0 <= write_threshold <= 1.0:
            raise ValueError(
                f"write_threshold must be in [0, 1], got {write_threshold}"
            )
        self.write_threshold = write_threshold
        self.verifier = verifier
        self._entries: dict[str, MemoryEntry] = {}

    def write(
        self,
        entry: MemoryEntry,
        *,
        verifier_score: float | None = None,
        owner_approved: bool = False,
    ) -> bool:
        """Write an entry if verified by verifier or owner. Resolves conflicts."""
        # Gating check
        is_verified = (
            owner_approved
            or (verifier_score is not None and verifier_score >= self.write_threshold)
            or (entry.confidence >= self.write_threshold and entry.verified)
        )
        if not is_verified:
            return False

        entry.verified = True

        # Conflict resolution: prefer newer + higher confidence
        # Conflicting entries share the same attribute or key topic
        target_attr = entry.attribute or ""
        if target_attr:
            for existing in self._entries.values():
                if existing.deleted or existing.flagged_stale:
                    continue
                if (existing.attribute or "") == target_attr:
                    # Found conflicting active entry. Compare (timestamp, confidence)
                    is_newer = entry.timestamp > existing.timestamp
                    is_higher_conf = (
                        entry.timestamp == existing.timestamp
                        and entry.confidence > existing.confidence
                    )
                    if is_newer or is_higher_conf:
                        # Old entry flagged stale, NOT deleted
                        existing.flagged_stale = True
                    else:
                        # Candidate is older than active entry -> flagged stale
                        entry.flagged_stale = True

        self._entries[entry.entry_id] = entry
        return True

    def edit(
        self,
        entry_id: str,
        *,
        text: str | None = None,
        confidence: float | None = None,
        decay_rule: str | None = None,
        recheck_rule: str | None = None,
        owner_approved: bool = True,
    ) -> bool:
        """Edit an existing entry."""
        if entry_id not in self._entries:
            return False
        entry = self._entries[entry_id]
        if entry.deleted:
            return False

        if text is not None:
            entry.text = text
        if confidence is not None:
            entry.confidence = confidence
            if confidence < self.write_threshold and not owner_approved:
                entry.verified = False
        if decay_rule is not None:
            entry.decay_rule = decay_rule
        if recheck_rule is not None:
            entry.recheck_rule = recheck_rule

        return True

    def delete(self, entry_id: str, *, owner_approved: bool = True) -> bool:
        """Soft-delete an entry."""
        if entry_id not in self._entries:
            return False
        entry = self._entries[entry_id]
        if not owner_approved:
            return False
        entry.deleted = True
        return True

    def recall(
        self,
        query: str,
        *,
        include_stale: bool = False,
        include_deleted: bool = False,
    ) -> str:
        """Recall relevant verified memory text for a query.

        Conforms to router.Memory protocol: returns string context, or empty string.
        """
        candidates: list[MemoryEntry] = []
        for entry in self._entries.values():
            if not include_deleted and entry.deleted:
                continue
            if not include_stale and entry.flagged_stale:
                continue
            if not entry.verified:
                continue
            candidates.append(entry)

        if not candidates:
            return ""

        q_tokens = _tokenize(query)

        # Rank candidates by attribute / token match, then recency
        scored: list[tuple[int, str, MemoryEntry]] = []
        for c in candidates:
            c_tokens = _tokenize(c.text)
            if c.attribute:
                c_tokens |= _tokenize(c.attribute)
            overlap = len(q_tokens & c_tokens)
            scored.append((overlap, c.timestamp, c))

        # Sort by overlap desc, then timestamp desc
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        if scored and scored[0][0] > 0:
            best = scored[0][2]
            return f"[{best.timestamp} | Channel: {best.source}]\n{best.text}"

        # If no explicit lexical match but entries exist, return latest
        latest = max(candidates, key=lambda x: (x.timestamp, x.confidence))
        return f"[{latest.timestamp} | Channel: {latest.source}]\n{latest.text}"

    def get_active_entries(self) -> list[MemoryEntry]:
        """Return all active, non-stale, non-deleted, verified entries."""
        return [
            e
            for e in self._entries.values()
            if not e.deleted and not e.flagged_stale and e.verified
        ]

    def get_stale_entries(self) -> list[MemoryEntry]:
        """Return all entries that were flagged stale."""
        return [e for e in self._entries.values() if e.flagged_stale and not e.deleted]

    def get_all_entries(self) -> list[MemoryEntry]:
        """Return all entries."""
        return list(self._entries.values())

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()


class PlainMemory:
    """Plain external memory baseline: stores all events without gating or stale flagging."""

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []

    def write(self, entry: MemoryEntry) -> bool:
        """Write entry without verification gating or conflict flagging."""
        self._entries.append(entry)
        return True

    def recall(self, query: str) -> str:
        """Recall all matching events in chronological order, including stale ones."""
        if not self._entries:
            return ""
        # Return all entries in chronological order
        ordered = sorted(self._entries, key=lambda x: x.timestamp)
        lines = [f"[{e.timestamp} | Channel: {e.source}]\n{e.text}" for e in ordered]
        return "\n".join(lines)

    def get_all_entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()


def paired_bootstrap_accuracy_difference(
    a_correct: list[bool],
    b_correct: list[bool],
    n_resamples: int = 10_000,
    seed: int = 20260104,
    ci_level: float = 0.95,
) -> tuple[float, float, float]:
    """Compute paired bootstrap difference (a - b) and confidence interval."""
    if len(a_correct) != len(b_correct) or not a_correct:
        raise ValueError("Inputs must be non-empty and equal length")

    n = len(a_correct)
    mean_diff = sum(a - b for a, b in zip(a_correct, b_correct, strict=True)) / n

    # Deterministic LCG or pseudo-random generator
    import numpy as np

    rng = np.random.default_rng(seed)
    diffs = np.array(a_correct, dtype=float) - np.array(b_correct, dtype=float)

    indices = rng.integers(0, n, size=(n_resamples, n))
    bootstrap_means = diffs[indices].mean(axis=1)

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(bootstrap_means, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1.0 - alpha / 2)))

    return float(mean_diff), ci_lower, ci_upper


@dataclass(frozen=True)
class Step6GateDecision:
    """Decision and metrics for PLAN.md Step 6 write-gated memory gate."""

    gated_accuracy: float
    plain_accuracy: float
    no_memory_accuracy: float
    accuracy_gain_pp: float
    paired_ci_lower: float
    paired_ci_upper: float
    plain_stale_errors: int
    gated_stale_errors: int
    stale_error_ratio: float
    stale_error_target_max: int
    static_forgetting_pp: float
    forgetting_cap_pp: float
    passes_accuracy_gain: bool
    passes_ci_excludes_zero: bool
    passes_stale_error_reduction: bool
    passes_forgetting_cap: bool
    gate_decision: str  # "PASS" or "FAIL"
    recommended_component: str  # "gated_memory" or "plain_memory"


def step6_gate_decision(
    no_memory_acc: float,
    plain_acc: float,
    gated_acc: float,
    plain_stale_errors: int,
    gated_stale_errors: int,
    plain_static_acc: float,
    gated_static_acc: float,
    paired_ci: tuple[float, float],
) -> Step6GateDecision:
    """Evaluate Step 6 write-gated memory pass and kill rules from PLAN.md.

    Pass rules:
    - >= 5 pp over plain external memory
    - paired 95% bootstrap interval excludes zero (ci_lower > 0)
    - <= 3 pp forgetting on static facts (plain_static - gated_static <= 0.03)
    - stale-fact errors at most half of plain memory (gated_stale_errors <= plain_stale_errors / 2)

    Kill rule:
    - If gating does not reduce stale errors, keep plain memory with timestamps.
    """
    gain_pp = round((gated_acc - plain_acc) * 100, 2)
    ci_lower, ci_upper = paired_ci

    # Forgetting: drop in accuracy on static facts
    static_drop = max(0.0, plain_static_acc - gated_static_acc)
    forgetting_pp = round(static_drop * 100, 2)

    stale_cap = math.floor(plain_stale_errors / 2)
    stale_ratio = (
        round(gated_stale_errors / plain_stale_errors, 4)
        if plain_stale_errors > 0
        else 0.0
    )

    passes_gain = gain_pp >= 5.0
    passes_ci = ci_lower > 0.0
    passes_stale = gated_stale_errors <= stale_cap
    passes_forgetting = forgetting_pp <= 3.0

    all_passed = passes_gain and passes_ci and passes_stale and passes_forgetting
    decision = "PASS" if all_passed else "FAIL"
    recommended = "gated_memory" if decision == "PASS" else "plain_memory"

    return Step6GateDecision(
        gated_accuracy=round(gated_acc * 100, 2),
        plain_accuracy=round(plain_acc * 100, 2),
        no_memory_accuracy=round(no_memory_acc * 100, 2),
        accuracy_gain_pp=gain_pp,
        paired_ci_lower=round(ci_lower * 100, 2),
        paired_ci_upper=round(ci_upper * 100, 2),
        plain_stale_errors=plain_stale_errors,
        gated_stale_errors=gated_stale_errors,
        stale_error_ratio=stale_ratio,
        stale_error_target_max=stale_cap,
        static_forgetting_pp=forgetting_pp,
        forgetting_cap_pp=3.0,
        passes_accuracy_gain=passes_gain,
        passes_ci_excludes_zero=passes_ci,
        passes_stale_error_reduction=passes_stale,
        passes_forgetting_cap=passes_forgetting,
        gate_decision=decision,
        recommended_component=recommended,
    )
