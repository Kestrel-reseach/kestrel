"""Step 4 tests: gated halting policies and baseline comparisons with fake sources."""

import pytest

from kestrel.halting import (
    Attempt,
    best_of_three,
    compare_strategies,
    evaluate_policy,
    fixed_two_pass,
    gated,
    self_consistency_at_five,
    single_pass,
    tune_threshold,
)


class FakeSource:
    """Deterministic attempts: early samples are weak, later ones strong."""

    energy_per_pass = 10.0

    def __init__(self, quality: dict[str, bool] | None = None) -> None:
        self.quality = quality or {}
        self.calls: dict[str, int] = {}

    def sample(self, item_id: str) -> Attempt:
        count = self.calls.get(item_id, 0)
        self.calls[item_id] = count + 1
        correct = self.quality.get(item_id, False)
        return Attempt(
            answer="good" if correct else "bad",
            score=0.5 + 0.1 * count,
            correct=correct,
        )


class ConstantSource:
    """One answer, fixed score and correctness."""

    energy_per_pass = 5.0

    def __init__(self, score: float, correct: bool, answer: str = "a") -> None:
        self.score = score
        self.correct = correct
        self.answer = answer

    def sample(self, item_id: str) -> Attempt:
        return Attempt(self.answer, self.score, self.correct)


def test_single_pass_and_fixed_policies() -> None:
    source = ConstantSource(0.6, True)
    assert single_pass(source, "x").passes == 1
    assert fixed_two_pass(source, "x").passes == 2
    assert best_of_three(source, "x").passes == 3
    assert self_consistency_at_five(source, "x").passes == 5


def test_gated_stops_early_and_abstains() -> None:
    source = FakeSource({"good": True})
    policy = gated(0.6, 3)
    early = policy(source, "good")
    assert early.correct is True
    assert early.abstained is False
    assert early.passes == 2

    low = ConstantSource(0.1, False)
    abstain = policy(low, "bad")
    assert abstain.abstained is True
    assert abstain.correct is False
    assert abstain.passes == 3


def test_gated_rejects_zero_budget() -> None:
    with pytest.raises(ValueError):
        gated(0.7, 0)


def test_evaluate_policy_energy_and_abstention() -> None:
    class MixedSource:
        energy_per_pass = 10.0

        def sample(self, item_id: str) -> Attempt:
            if item_id == "a":
                return Attempt("good", 0.9, True)
            return Attempt("bad", 0.1, False)

    metrics = evaluate_policy(MixedSource(), ["a", "b"], gated(0.6, 2))
    assert metrics["accuracy"] == 0.5
    assert metrics["abstain_rate"] == 0.5
    assert metrics["passes"] == 1.5
    assert metrics["energy_per_correct"] == 30.0


def test_tune_threshold_uses_validation_only() -> None:
    source = FakeSource({item: True for item in ("a", "b")})
    assert tune_threshold(source, ["a", "b"]) in (0.5, 0.6, 0.7, 0.8, 0.9)


def test_compare_strategies_covers_all_policies() -> None:
    source = ConstantSource(0.8, True)
    result = compare_strategies(source, ["x"])
    assert set(result) == {
        "single_pass",
        "fixed_two_pass",
        "best_of_three",
        "self_consistency_at_five",
        "gated",
    }
    assert result["single_pass"]["accuracy"] == 1.0
