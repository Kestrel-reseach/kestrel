"""Step 3 scaffold: verifier signals, metrics and a small reference head.

The contracts here are real; the model-specific parts are not. A base-model
feature extractor and a trained head plug into ``FeatureExtractor`` and
``VerifierHead`` once Member A has delivered Step 1 (locked suite and splits)
and Step 2 (chosen base model). Nothing in this module loads a model or data.

Proposed gate (PLAN.md Step 3): calibration error at most 0.10 on held-out
data, and AUROC at least 0.03 above the best simple baseline with the interval
excluding zero. Kill gate: if the head does not beat token probability or
self-consistency, use the simpler signal and drop the word "learned".
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "LogisticVerifierHead",
    "VerifierExample",
    "VerifierFeatures",
    "VerifierHead",
    "auc_roc",
    "evaluate",
    "expected_calibration_error",
    "selective_accuracy",
    "self_consistency_baseline",
    "sigmoid",
    "stated_confidence_baseline",
    "temperature_scale",
    "token_probability_baseline",
]


@dataclass(frozen=True)
class VerifierFeatures:
    """Per-answer signals available to the verifier head.

    ``last_token_hidden`` is the feature vector the reference head uses.
    ``token_logprobs`` feeds the token-probability baseline. ``sampled_answers``
    feeds self-consistency. ``stated_confidence`` is the model's own claim, if
    the base model exposes one.
    """

    last_token_hidden: tuple[float, ...]
    token_logprobs: tuple[float, ...] = ()
    sampled_answers: tuple[str, ...] = ()
    stated_confidence: float | None = None

    @property
    def sample_agreement(self) -> float:
        """Fraction of sampled answers matching the most common answer."""
        if not self.sampled_answers:
            return 0.0
        counts = Counter(self.sampled_answers)
        return max(counts.values()) / len(self.sampled_answers)


@dataclass(frozen=True)
class VerifierExample:
    """One labelled answer used to fit or evaluate a verifier."""

    item_id: str
    split: str
    features: VerifierFeatures
    correct: bool


class FeatureExtractor(Protocol):
    """Seam for the base-model feature extractor (implemented later)."""

    def extract(self, item_id: str, prompt: str, answer: str) -> VerifierFeatures:
        """Return the verifier signals for one generated answer."""
        ...


class VerifierHead(Protocol):
    """Seam for a trained head; the reference implementation is below."""

    def fit(self, examples: Sequence[VerifierExample]) -> None:
        """Fit the head on labelled examples."""
        ...

    def predict(self, features: VerifierFeatures) -> float:
        """Return a calibrated-style correctness score in [0, 1]."""
        ...


def sigmoid(value: float) -> float:
    """Numerically stable logistic function."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def token_probability_baseline(features: VerifierFeatures) -> float:
    """Mean token probability, i.e. exp of the mean token log-probability."""
    if not features.token_logprobs:
        return 0.0
    mean_logprob = sum(features.token_logprobs) / len(features.token_logprobs)
    return math.exp(min(mean_logprob, 0.0))


def self_consistency_baseline(features: VerifierFeatures) -> float:
    """Agreement across sampled answers."""
    return features.sample_agreement


def stated_confidence_baseline(features: VerifierFeatures) -> float:
    """The base model's stated confidence, or 0.5 when unavailable."""
    if features.stated_confidence is None:
        return 0.5
    return min(max(features.stated_confidence, 0.0), 1.0)


def auc_roc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve via average ranks (handles ties)."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5

    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[start]]:
            end += 1
        average_rank = (start + end) / 2 + 1
        for position in range(start, end + 1):
            ranks[order[position]] = average_rank
        start = end + 1

    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def expected_calibration_error(
    scores: Sequence[float], labels: Sequence[bool], bins: int = 10
) -> float:
    """Equal-width binning calibration error over scores clipped to [0, 1]."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if not scores:
        return 0.0

    total = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            position
            for position, score in enumerate(scores)
            if lower <= score < upper or (index == bins - 1 and score == 1.0)
        ]
        if not members:
            continue
        mean_score = sum(scores[position] for position in members) / len(members)
        mean_label = sum(1 for position in members if labels[position]) / len(members)
        total += len(members) / len(scores) * abs(mean_score - mean_label)
    return total


def selective_accuracy(
    scores: Sequence[float], labels: Sequence[bool], coverage: float
) -> float | None:
    """Accuracy on the highest-scored ``coverage`` fraction of answers."""
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        return None

    count = max(1, math.ceil(coverage * len(scores)))
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[
        :count
    ]
    return sum(1 for index in order if labels[index]) / len(order)


def temperature_scale(scores: Sequence[float], temperature: float) -> tuple[float, ...]:
    """Convert probabilities to logits, divide by temperature, then squash."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    epsilon = 1e-12
    scaled = []
    for score in scores:
        probability = min(max(score, epsilon), 1.0 - epsilon)
        logit = math.log(probability / (1.0 - probability)) / temperature
        scaled.append(sigmoid(logit))
    return tuple(scaled)


def evaluate(
    scores: Sequence[float],
    labels: Sequence[bool],
    baselines: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, float]:
    """Gate-facing summary: AUROC, ECE, selective accuracy and baseline deltas."""
    result = {
        "auroc": auc_roc(scores, labels),
        "ece": expected_calibration_error(scores, labels),
    }
    coverage_accuracy = selective_accuracy(scores, labels, coverage=0.8)
    if coverage_accuracy is not None:
        result["selective_accuracy_at_0.8"] = coverage_accuracy

    for name, baseline_scores in (baselines or {}).items():
        result[f"auroc_delta_vs_{name}"] = result["auroc"] - auc_roc(
            baseline_scores, labels
        )
    return result


class LogisticVerifierHead:
    """Reference head over ``last_token_hidden``.

    Pure stdlib so tests and CI need no model dependencies. Replace with a
    trained torch or scikit-learn head when real features exist; keep the
    ``fit``/``predict`` interface and re-run the same ``evaluate`` checks.
    """

    def __init__(self, learning_rate: float = 0.1, epochs: int = 200) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: list[float] | None = None
        self.bias = 0.0

    def fit(self, examples: Sequence[VerifierExample]) -> None:
        """Fit by batch gradient descent on the hidden-state features."""
        if not examples:
            raise ValueError("cannot fit a verifier head without examples")
        dimensions = {len(example.features.last_token_hidden) for example in examples}
        if len(dimensions) != 1:
            raise ValueError("all hidden states must have the same length")

        weights = [0.0] * dimensions.pop()
        self.bias = 0.0
        inputs = [list(example.features.last_token_hidden) for example in examples]
        targets = [1.0 if example.correct else 0.0 for example in examples]

        for _ in range(self.epochs):
            gradients = [0.0] * len(weights)
            bias_gradient = 0.0
            for values, target in zip(inputs, targets):
                score = sum(weight * value for weight, value in zip(weights, values))
                error = sigmoid(score + self.bias) - target
                for index, value in enumerate(values):
                    gradients[index] += error * value
                bias_gradient += error
            count = len(inputs)
            weights = [
                weight - self.learning_rate * gradient / count
                for weight, gradient in zip(weights, gradients)
            ]
            self.bias -= self.learning_rate * bias_gradient / count
        self.weights = weights

    def predict(self, features: VerifierFeatures) -> float:
        """Return the head's correctness probability for one answer."""
        if self.weights is None:
            raise RuntimeError("fit the head before calling predict")
        if len(features.last_token_hidden) != len(self.weights):
            raise ValueError("hidden state length does not match the fitted head")
        score = sum(
            weight * value
            for weight, value in zip(self.weights, features.last_token_hidden)
        )
        return sigmoid(score + self.bias)
