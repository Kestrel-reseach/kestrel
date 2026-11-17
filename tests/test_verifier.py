"""Step 3 scaffold tests: signals, metrics and the reference head.

These use synthetic in-memory data only. Real training items arrive with
Member A's Step 1 suite and Step 2 base model.
"""

import math

import pytest

from kestrel.verifier import (
    LogisticVerifierHead,
    VerifierExample,
    VerifierFeatures,
    auc_roc,
    evaluate,
    expected_calibration_error,
    selective_accuracy,
    self_consistency_baseline,
    stated_confidence_baseline,
    temperature_scale,
    token_probability_baseline,
)


def test_baseline_signals() -> None:
    features = VerifierFeatures(
        last_token_hidden=(1.0, 0.0),
        token_logprobs=(-1.0, -1.0),
        sampled_answers=("a", "a", "b"),
        stated_confidence=0.8,
    )
    assert math.isclose(token_probability_baseline(features), math.exp(-1.0))
    assert math.isclose(self_consistency_baseline(features), 2 / 3)
    assert stated_confidence_baseline(features) == 0.8

    without_extras = VerifierFeatures(last_token_hidden=(1.0,))
    assert token_probability_baseline(without_extras) == 0.0
    assert self_consistency_baseline(without_extras) == 0.0
    assert stated_confidence_baseline(without_extras) == 0.5


def test_auc_roc_with_ties_and_degenerate_labels() -> None:
    assert auc_roc([0.1, 0.9], [False, True]) == 1.0
    assert auc_roc([0.9, 0.1], [False, True]) == 0.0
    assert auc_roc([0.5, 0.5], [False, True]) == 0.5
    assert auc_roc([0.1], [True]) == 0.5


def test_expected_calibration_error() -> None:
    assert expected_calibration_error([0.0, 1.0], [False, True]) == 0.0
    assert expected_calibration_error([1.0, 0.0], [False, True], bins=2) == 1.0
    assert expected_calibration_error([], []) == 0.0


def test_selective_accuracy_prefers_high_scores() -> None:
    scores = [0.9, 0.8, 0.1]
    labels = [True, False, False]
    assert selective_accuracy(scores, labels, 1 / 3) == 1.0
    assert selective_accuracy(scores, labels, 1.0) == 1 / 3


def test_temperature_scale_softens_confidence() -> None:
    scaled = temperature_scale([0.2, 0.8], 2.0)
    assert scaled[0] > 0.2
    assert scaled[1] < 0.8
    assert math.isclose(temperature_scale([0.2], 1.0)[0], 0.2)


def test_head_learns_separable_synthetic_data() -> None:
    examples = []
    for index in range(20):
        value = index / 20
        examples.append(
            VerifierExample(
                item_id=str(index),
                split="train",
                features=VerifierFeatures(last_token_hidden=(value, value)),
                correct=index >= 10,
            )
        )

    head = LogisticVerifierHead(learning_rate=0.5, epochs=300)
    head.fit(examples)
    low = head.predict(VerifierFeatures(last_token_hidden=(0.05, 0.05)))
    high = head.predict(VerifierFeatures(last_token_hidden=(0.95, 0.95)))
    assert low < 0.3
    assert high > 0.7


def test_head_guards_against_unfitted_or_mismatched_features() -> None:
    head = LogisticVerifierHead()
    with pytest.raises(RuntimeError):
        head.predict(VerifierFeatures(last_token_hidden=(1.0,)))
    with pytest.raises(ValueError):
        head.fit([])

    example = VerifierExample(
        item_id="one",
        split="train",
        features=VerifierFeatures(last_token_hidden=(1.0, 2.0)),
        correct=True,
    )
    head.fit([example])
    with pytest.raises(ValueError):
        head.predict(VerifierFeatures(last_token_hidden=(1.0,)))


def test_evaluate_reports_baseline_deltas() -> None:
    result = evaluate(
        [0.9, 0.1],
        [True, False],
        baselines={"token_probability": [0.1, 0.9]},
    )
    assert result["auroc"] == 1.0
    assert result["auroc_delta_vs_token_probability"] == 1.0
    assert result["selective_accuracy_at_0.8"] == 0.5
