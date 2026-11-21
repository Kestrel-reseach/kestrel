"""Step 3 evaluation helpers: gaming checks, baseline comparison and gates.

These depend only on the verifier's stdlib metrics, so they run in CI and on
the target hardware with the same code. Scores are correctness probabilities
in [0, 1]; labels are per-item correctness.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

from kestrel.verifier import auc_roc, expected_calibration_error

__all__ = [
    "bootstrap_auroc_delta",
    "calibration_table",
    "compare_baselines",
    "gate_decision",
    "gaming_check",
]

ECE_LIMIT = 0.10
AUROC_DELTA_MIN = 0.03


def bootstrap_auroc_delta(
    head_scores: Sequence[float],
    baseline_scores: Sequence[float],
    labels: Sequence[bool],
    *,
    seed: int = 20260103,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
) -> tuple[float, float]:
    """Paired-bootstrap 95% CI on AUROC(head) - AUROC(baseline).

    Items are resampled together, pairing head and baseline on the same items.
    """
    if not (len(head_scores) == len(baseline_scores) == len(labels)):
        raise ValueError("scores and labels must have the same length")
    if not head_scores or n_bootstrap < 1:
        raise ValueError("scores must be non-empty and n_bootstrap positive")

    rng = random.Random(seed)
    n = len(labels)
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_head = [head_scores[i] for i in indices]
        sample_base = [baseline_scores[i] for i in indices]
        sample_labels = [labels[i] for i in indices]
        deltas.append(
            auc_roc(sample_head, sample_labels) - auc_roc(sample_base, sample_labels)
        )
    deltas.sort()
    alpha = (1 - ci_level) / 2
    return deltas[int(math.floor(alpha * n_bootstrap))], deltas[
        int(math.floor((1 - alpha) * n_bootstrap)) - 1
    ]


def compare_baselines(
    head_scores: Sequence[float],
    labels: Sequence[bool],
    baselines: Mapping[str, Sequence[float]],
    *,
    seed: int = 20260103,
    n_bootstrap: int = 2000,
) -> dict[str, dict[str, float]]:
    """AUROC delta and paired interval against every simple baseline."""
    result: dict[str, dict[str, float]] = {}
    head_auroc = auc_roc(head_scores, labels)
    for name, scores in baselines.items():
        low, high = bootstrap_auroc_delta(
            head_scores, scores, labels, seed=seed, n_bootstrap=n_bootstrap
        )
        result[name] = {
            "head_auroc": head_auroc,
            "baseline_auroc": auc_roc(scores, labels),
            "delta": head_auroc - auc_roc(scores, labels),
            "ci_low": low,
            "ci_high": high,
        }
    return result


def gate_decision(
    ece: float,
    comparison: Mapping[str, Mapping[str, float]],
    *,
    ece_limit: float = ECE_LIMIT,
    delta_min: float = AUROC_DELTA_MIN,
) -> dict[str, object]:
    """Apply the Step 3 proposed gate.

    Pass requires ECE <= 0.10 and AUROC at least 0.03 above the best simple
    baseline, with the paired interval excluding zero. If no baselines are
    given, only calibration is checked.
    """
    calibration_ok = ece <= ece_limit
    best_name = None
    best_delta = float("-inf")
    for name, entry in comparison.items():
        if entry["delta"] > best_delta:
            best_name = name
            best_delta = entry["delta"]
    beats_best = best_name is not None and best_delta >= delta_min
    ci_excludes_zero = best_name is not None and comparison[best_name]["ci_low"] > 0
    return {
        "ece": ece,
        "ece_limit": ece_limit,
        "calibration_ok": calibration_ok,
        "best_baseline": best_name,
        "best_delta": best_delta if best_name is not None else None,
        "delta_min": delta_min,
        "beats_best_baseline": beats_best,
        "ci_excludes_zero": ci_excludes_zero,
        "passes": calibration_ok and beats_best and ci_excludes_zero,
    }


def calibration_table(
    scores: Sequence[float], labels: Sequence[bool], bins: int = 10
) -> list[dict[str, float]]:
    """Reliability table: mean score, observed accuracy and count per bin."""
    if bins < 1:
        raise ValueError("bins must be positive")
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    rows = []
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        members = [
            i
            for i, score in enumerate(scores)
            if low <= score < high or (index == bins - 1 and score == 1.0)
        ]
        if not members:
            continue
        rows.append(
            {
                "bin": index,
                "count": len(members),
                "mean_score": sum(scores[i] for i in members) / len(members),
                "accuracy": sum(1 for i in members if labels[i]) / len(members),
            }
        )
    return rows


def gaming_check(
    scores: Sequence[float], labels: Sequence[bool], conditions: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Calibration and accuracy per condition for verifier-gaming tests.

    Conditions are held-out task types, paraphrases and adversarial items.
    A large positive confidence gap (mean score minus accuracy) means the
    verifier looks confident without being right on that condition.
    """
    if not (len(scores) == len(labels) == len(conditions)):
        raise ValueError("scores, labels and conditions must have the same length")
    grouped: dict[str, list[int]] = {}
    for index, condition in enumerate(conditions):
        grouped.setdefault(condition, []).append(index)

    result: dict[str, dict[str, float]] = {}
    for condition, indices in grouped.items():
        condition_scores = [scores[i] for i in indices]
        condition_labels = [labels[i] for i in indices]
        accuracy = sum(condition_labels) / len(condition_labels)
        mean_score = sum(condition_scores) / len(condition_scores)
        result[condition] = {
            "count": len(indices),
            "accuracy": accuracy,
            "mean_score": mean_score,
            "confidence_gap": mean_score - accuracy,
            "ece": expected_calibration_error(condition_scores, condition_labels),
        }
    return result
