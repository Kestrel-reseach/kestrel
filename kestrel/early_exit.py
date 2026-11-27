"""Token-level verifier early-exit policy (hardware-free).

The pass-level gate was rejected earlier, so this module only decides when to
stop generating: it never starts a retry. A calibrated verifier score and an
extractable answer are both required before stopping early. All thresholds are
supplied by the caller and must be chosen on validation data only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EarlyExitDecision:
    stop: bool
    reason: str
    tokens_seen: int


class TokenEarlyExit:
    """Decide whether generation can stop before its token cap.

    Args:
        threshold: calibrated verifier score required to consider stopping.
        min_tokens: never stop before this many tokens have been generated.
        check_every: evaluate the score at most once every N tokens.
        max_tokens: force a stop at this token count (None disables).
        consecutive_hits: consecutive qualifying checks needed to stop.
    """

    def __init__(
        self,
        threshold: float,
        *,
        min_tokens: int = 32,
        check_every: int = 8,
        max_tokens: int | None = None,
        consecutive_hits: int = 1,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if min_tokens < 0 or check_every < 1 or consecutive_hits < 1:
            raise ValueError("invalid early-exit configuration")
        self.threshold = threshold
        self.min_tokens = min_tokens
        self.check_every = check_every
        self.max_tokens = max_tokens
        self.consecutive_hits = consecutive_hits
        self._hits = 0
        self._last_checked = -1

    def update(
        self,
        *,
        tokens_generated: int,
        verifier_score: float,
        answer_extractable: bool,
    ) -> EarlyExitDecision:
        """Update state with one token's signal and return the stop decision."""
        if tokens_generated < 0:
            raise ValueError("tokens_generated must be non-negative")

        if self.max_tokens is not None and tokens_generated >= self.max_tokens:
            return EarlyExitDecision(True, "max_tokens", tokens_generated)
        if tokens_generated < self.min_tokens:
            return EarlyExitDecision(False, "min_tokens", tokens_generated)

        due = tokens_generated - self._last_checked >= self.check_every
        if not due:
            return EarlyExitDecision(False, "waiting", tokens_generated)
        self._last_checked = tokens_generated

        qualifies = (
            answer_extractable
            and 0.0 <= verifier_score <= 1.0
            and verifier_score >= self.threshold
        )
        self._hits = self._hits + 1 if qualifies else 0
        if self._hits >= self.consecutive_hits:
            return EarlyExitDecision(True, "verifier", tokens_generated)
        return EarlyExitDecision(False, "not_confident", tokens_generated)


def select_threshold(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    min_precision: float = 0.95,
) -> float:
    """Pick the lowest threshold whose accepted set meets target precision.

    Callers must supply validation data only. Returns 1.0 when no threshold
    meets the target, which disables early stopping without changing generation.
    """
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be non-empty and equal in length")
    if not 0.0 < min_precision <= 1.0:
        raise ValueError("min_precision must be in (0, 1]")

    eligible: list[float] = []
    for candidate in sorted(set(float(s) for s in scores)):
        accepted = [
            bool(label) for score, label in zip(scores, labels) if score >= candidate
        ]
        if not accepted:
            continue
        precision = sum(accepted) / len(accepted)
        if precision >= min_precision:
            eligible.append(candidate)
    return min(eligible) if eligible else 1.0
