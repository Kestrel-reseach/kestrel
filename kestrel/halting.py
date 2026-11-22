"""Step 4: verifier-gated halting policy and baseline comparisons.

The policy generates, scores, and returns as soon as the verifier score clears
a threshold; otherwise it resamples up to a pass budget and abstains. The
threshold is tuned on validation only. Cloud escalation is off.

Energy is tracked per attempted pass, so every strategy is compared on the
same budget. These helpers are stdlib-only; the real model plugs in through an
AttemptSource backed by QwenBackend and the trained verifier head.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from collections.abc import Callable
from typing import Protocol

__all__ = [
    "Attempt",
    "AttemptSource",
    "HaltingResult",
    "best_of_three",
    "compare_strategies",
    "evaluate_policy",
    "fixed_two_pass",
    "gated",
    "self_consistency_at_five",
    "single_pass",
    "tune_threshold",
]


@dataclass(frozen=True)
class Attempt:
    """One generated answer with its verifier score and correctness."""

    answer: str
    score: float
    correct: bool


class AttemptSource(Protocol):
    """Produces independent attempts for an item; each pass costs energy."""

    @property
    def energy_per_pass(self) -> float:
        """Energy in joules for one generation pass."""
        ...

    def sample(self, item_id: str) -> Attempt:
        """Generate one attempt for the item."""
        ...


@dataclass(frozen=True)
class HaltingResult:
    """Outcome for one item under a policy."""

    correct: bool
    abstained: bool
    passes: int
    energy: float


Policy = Callable[[AttemptSource, str], HaltingResult]


def single_pass(source: AttemptSource, item_id: str) -> HaltingResult:
    """One pass, always return it."""
    attempt = source.sample(item_id)
    return HaltingResult(attempt.correct, False, 1, source.energy_per_pass)


def _best_of(source: AttemptSource, item_id: str, count: int) -> HaltingResult:
    attempts = [source.sample(item_id) for _ in range(count)]
    best = max(attempts, key=lambda attempt: attempt.score)
    return HaltingResult(best.correct, False, count, source.energy_per_pass * count)


def fixed_two_pass(source: AttemptSource, item_id: str) -> HaltingResult:
    """Two fixed passes, return the higher-scored answer."""
    return _best_of(source, item_id, 2)


def best_of_three(source: AttemptSource, item_id: str) -> HaltingResult:
    """Three fixed passes, return the higher-scored answer."""
    return _best_of(source, item_id, 3)


def self_consistency_at_five(source: AttemptSource, item_id: str) -> HaltingResult:
    """Five passes, return the majority answer (ties broken by verifier score)."""
    attempts = [source.sample(item_id) for _ in range(5)]
    counts = Counter(attempt.answer for attempt in attempts)
    majority = counts.most_common(1)[0][0]
    candidates = [attempt for attempt in attempts if attempt.answer == majority]
    best = max(candidates, key=lambda attempt: attempt.score)
    return HaltingResult(best.correct, False, 5, source.energy_per_pass * 5)


def gated(threshold: float, max_passes: int) -> Policy:
    """Return a policy that stops at the first score >= threshold, then abstains."""
    if max_passes < 1:
        raise ValueError("max_passes must be positive")

    def policy(source: AttemptSource, item_id: str) -> HaltingResult:
        for index in range(1, max_passes + 1):
            attempt = source.sample(item_id)
            if attempt.score >= threshold:
                return HaltingResult(
                    attempt.correct, False, index, source.energy_per_pass * index
                )
        return HaltingResult(
            False, True, max_passes, source.energy_per_pass * max_passes
        )

    return policy


def evaluate_policy(
    source: AttemptSource, item_ids: list[str], policy: Policy
) -> dict[str, float]:
    """Accuracy (abstention counts as wrong), energy per correct, and pass stats."""
    results = [policy(source, item_id) for item_id in item_ids]
    n = len(results)
    if n == 0:
        return {
            "accuracy": 0.0,
            "energy_per_correct": 0.0,
            "passes": 0.0,
            "abstain_rate": 0.0,
        }
    correct = sum(result.correct for result in results)
    energy = sum(result.energy for result in results)
    passes = sum(result.passes for result in results)
    abstentions = sum(result.abstained for result in results)
    return {
        "accuracy": correct / n,
        "energy_per_correct": energy / correct if correct else 0.0,
        "passes": passes / n,
        "abstain_rate": abstentions / n,
    }


def tune_threshold(
    source: AttemptSource,
    item_ids: list[str],
    *,
    max_passes: int = 3,
    energy_cap_ratio: float = 1.5,
    candidates: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> float:
    """Pick the threshold with best validation accuracy within the energy cap.

    The cap is relative to single-pass energy per correct task. Ties go to the
    cheaper threshold. This must run on validation only, never on locked test.
    """
    baseline = evaluate_policy(source, item_ids, single_pass)["energy_per_correct"]
    cap = baseline * energy_cap_ratio
    best_threshold = 1.0
    best_accuracy = -1.0
    best_energy = float("inf")
    for threshold in candidates:
        metrics = evaluate_policy(source, item_ids, gated(threshold, max_passes))
        if metrics["energy_per_correct"] > cap:
            continue
        if metrics["accuracy"] > best_accuracy or (
            metrics["accuracy"] == best_accuracy
            and metrics["energy_per_correct"] < best_energy
        ):
            best_threshold = threshold
            best_accuracy = metrics["accuracy"]
            best_energy = metrics["energy_per_correct"]
    return best_threshold


def compare_strategies(
    source: AttemptSource,
    item_ids: list[str],
    *,
    threshold: float = 0.7,
    max_passes: int = 3,
) -> dict[str, dict[str, float]]:
    """Evaluate the gate and all required baselines on the same items."""
    policies = {
        "single_pass": single_pass,
        "fixed_two_pass": fixed_two_pass,
        "best_of_three": best_of_three,
        "self_consistency_at_five": self_consistency_at_five,
        "gated": gated(threshold, max_passes),
    }
    return {
        name: evaluate_policy(source, item_ids, policy)
        for name, policy in policies.items()
    }
