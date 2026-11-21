"""Step 3 tests: feature mapping, numpy head, calibration and training run.

Fake backends and scorers keep these fast and dependency-free. Real hardware
runs happen separately on the RTX 3050 machine.
"""

import json

import pytest

from kestrel.features import GeneratedAnswer, to_features
from kestrel.train_verifier import build_examples, load_items, run
from kestrel.verifier import VerifierExample, VerifierFeatures, fit_temperature
from kestrel.verifier_numpy import NumpyLogisticHead


class FakeBackend:
    """Signals that separate correct from wrong answers on one dimension."""

    model_name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt, n_samples=1, max_new_tokens=256):
        self.calls += 1
        strength = float(prompt)
        return GeneratedAnswer(
            text="answer",
            hidden=(strength, strength * 0.1),
            token_logprobs=(-0.1, -0.2),
            sampled_answers=("answer",),
        )


def test_to_features_carries_signals() -> None:
    answer = GeneratedAnswer(
        text="x",
        hidden=(1.0, 2.0),
        token_logprobs=(-0.5, -0.5),
        sampled_answers=("x", "y"),
    )
    features = to_features(answer)
    assert features.last_token_hidden == (1.0, 2.0)
    assert features.token_logprobs == (-0.5, -0.5)
    assert features.sample_agreement == 0.5
    assert features.stated_confidence is None


def test_numpy_head_learns_separable_features() -> None:
    examples = [
        VerifierExample(
            item_id=str(i),
            split="train",
            features=VerifierFeatures(last_token_hidden=(i / 20, 0.0)),
            correct=i >= 10,
        )
        for i in range(20)
    ]
    head = NumpyLogisticHead(epochs=200)
    head.fit(examples)
    assert head.predict(VerifierFeatures(last_token_hidden=(0.05, 0.0))) < 0.3
    assert head.predict(VerifierFeatures(last_token_hidden=(0.95, 0.0))) > 0.7


def test_fit_temperature_reduces_overconfidence() -> None:
    scores = [0.9, 0.9, 0.1, 0.1]
    labels = [True, False, True, False]
    assert fit_temperature(scores, labels) >= 1.0


def _write_partition(path, correct_value):
    path.mkdir(parents=True, exist_ok=True)
    for group in ("math", "science_mcq", "code", "knowledge_mcq"):
        with (path / f"{group}.jsonl").open("w", encoding="utf-8") as handle:
            for index in range(6):
                item = {
                    "item_id": f"{group}_{index}",
                    "group": group,
                    "prompt": str(correct_value(index)),
                    "target": "1",
                    "answer_check": "answer_key",
                }
                handle.write(json.dumps(item) + "\n")


def test_end_to_end_with_fake_backend(tmp_path) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "validation"
    _write_partition(train_dir, lambda i: -1.0 if i % 2 == 0 else 1.0)
    _write_partition(val_dir, lambda i: -1.0 if i % 2 == 0 else 1.0)
    backend = FakeBackend()

    def scorer(item, text):
        return float(item["prompt"]) > 0

    output = tmp_path / "reports" / "verifier.json"

    results = run(train_dir, val_dir, output, backend, scorer)

    assert backend.calls == 48
    assert results["n_train"] == 24
    assert results["n_validation"] == 24
    assert results["metrics"]["auroc"] == 1.0
    assert output.is_file()
    assert json.loads(output.read_text())["step"] == 3


def test_build_examples_uses_frozen_scorer() -> None:
    item = {"item_id": "one", "split": "train", "prompt": "1.0"}
    examples = build_examples([item], FakeBackend(), lambda i, text: True)
    assert examples[0].correct is True
    assert examples[0].features.last_token_hidden == (1.0, 0.1)


def test_load_items_rejects_missing_group(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_items(tmp_path, ("math",))
