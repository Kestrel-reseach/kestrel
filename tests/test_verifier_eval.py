"""Step 3 evaluation helper tests: bootstrap intervals, gate and gaming checks."""

import pytest

from kestrel.verifier_eval import (
    bootstrap_auroc_delta,
    calibration_table,
    compare_baselines,
    gate_decision,
    gaming_check,
)


def test_bootstrap_delta_detects_real_gain() -> None:
    head = [0.9, 0.8, 0.85, 0.7, 0.2, 0.1, 0.3, 0.15]
    baseline = [0.2, 0.1, 0.3, 0.25, 0.9, 0.8, 0.85, 0.7]
    labels = [True, True, True, True, False, False, False, False]
    low, high = bootstrap_auroc_delta(head, baseline, labels, n_bootstrap=200)
    assert low > 0
    assert high >= low


def test_bootstrap_delta_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        bootstrap_auroc_delta([0.5], [0.5, 0.4], [True])


def test_compare_baselines_and_gate_pass() -> None:
    head = [0.9, 0.8, 0.3, 0.2, 0.85, 0.1]
    labels = [True, True, False, False, True, False]
    baseline = [0.55] * len(labels)
    comparison = compare_baselines(
        head, labels, {"constant": baseline}, n_bootstrap=200
    )
    assert comparison["constant"]["delta"] > 0.03
    decision = gate_decision(0.05, comparison)
    assert decision["best_baseline"] == "constant"
    assert decision["passes"] is True


def test_gate_fails_on_poor_calibration() -> None:
    decision = gate_decision(0.2, {})
    assert decision["calibration_ok"] is False
    assert decision["passes"] is False


def test_calibration_table_counts_and_bins() -> None:
    rows = calibration_table([0.1, 0.2, 0.8, 0.9], [False, False, True, True], bins=2)
    assert len(rows) == 2
    assert rows[0]["count"] == 2
    assert rows[0]["accuracy"] == 0.0
    assert rows[1]["accuracy"] == 1.0


def test_gaming_check_finds_overconfidence() -> None:
    scores = [0.9, 0.9, 0.2, 0.2]
    labels = [True, False, False, True]
    conditions = ["held_out", "held_out", "adversarial", "adversarial"]
    result = gaming_check(scores, labels, conditions)
    assert result["held_out"]["confidence_gap"] == pytest.approx(0.4)
    assert result["adversarial"]["confidence_gap"] == pytest.approx(-0.3)


def test_gaming_check_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        gaming_check([0.5], [True], ["a", "b"])
