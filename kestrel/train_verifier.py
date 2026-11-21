"""Step 3: train and calibrate the verifier head on the locked train split.

Run on the RTX 3050 machine where torch and the 4-bit model are available:

    KESTREL_EVAL_DIR=/path/to/kestrel-eval python -m kestrel.train_verifier \
        --train /path/to/kestrel-eval/suite/train \
        --validation /path/to/kestrel-eval/suite/validation \
        --out reports/verifier_step3.json

Correctness is scored with A's frozen extractors via KESTREL_EVAL_DIR, so the
training labels match the gate evaluation exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kestrel.features import ModelBackend, QwenBackend, to_features
from kestrel.verifier import (
    VerifierExample,
    evaluate,
    fit_temperature,
    self_consistency_baseline,
    temperature_scale,
    token_probability_baseline,
)
from kestrel.verifier_numpy import NumpyLogisticHead

CHECKABLE = ("math", "science_mcq", "code", "knowledge_mcq")
Scorer = Callable[[dict[str, Any], str], bool]


def load_items(directory: Path, groups: tuple[str, ...]) -> list[dict[str, Any]]:
    """Load checkable items from JSONL files in a suite partition."""
    items: list[dict[str, Any]] = []
    for group in groups:
        path = directory / f"{group}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    items.append(json.loads(line))
    return items


def load_scorer(eval_dir: str) -> Scorer:
    """Import A's frozen extractor for item-level correctness."""
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from evaluation.extractors import extract_answer_for_group

    def score(item: dict[str, Any], text: str) -> bool:
        return bool(extract_answer_for_group(item, text)[1])

    return score


def build_examples(
    items: list[dict[str, Any]], backend: ModelBackend, scorer: Scorer
) -> list[VerifierExample]:
    """Generate answers and label them for verifier training."""
    examples = []
    for item in items:
        answer = backend.generate(item["prompt"])
        examples.append(
            VerifierExample(
                item_id=item["item_id"],
                split=item.get("split", "train"),
                features=to_features(answer),
                correct=scorer(item, answer.text),
            )
        )
    return examples


def run(
    train_dir: Path,
    validation_dir: Path,
    output: Path,
    backend: ModelBackend,
    scorer: Scorer,
    seed: int = 20260103,
) -> dict[str, Any]:
    """Train, calibrate and evaluate; write a JSON summary."""
    train_items = load_items(train_dir, CHECKABLE)
    validation_items = load_items(validation_dir, CHECKABLE)
    train = build_examples(train_items, backend, scorer)
    validation = build_examples(validation_items, backend, scorer)

    head = NumpyLogisticHead()
    head.fit(train)
    raw_scores = [head.predict(example.features) for example in validation]
    labels = [example.correct for example in validation]
    temperature = fit_temperature(raw_scores, labels)
    calibrated = temperature_scale(raw_scores, temperature)

    baselines = {
        "token_probability": [
            token_probability_baseline(example.features) for example in validation
        ],
        "self_consistency": [
            self_consistency_baseline(example.features) for example in validation
        ],
    }
    results = evaluate(calibrated, labels, baselines=baselines)
    payload = {
        "step": 3,
        "seed": seed,
        "n_train": len(train),
        "n_validation": len(validation),
        "temperature": temperature,
        "base_model": getattr(backend, "model_name", "fake"),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "config_hash": hashlib.sha256(
            json.dumps(
                {"groups": CHECKABLE, "seed": seed, "temperature": temperature},
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "metrics": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    """CLI entry point for the target-hardware run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--eval-dir", default=os.environ.get("KESTREL_EVAL_DIR", ""))
    args = parser.parse_args()
    if not args.eval_dir:
        parser.error("set --eval-dir or KESTREL_EVAL_DIR to the kestrel-eval repo")

    backend: ModelBackend = QwenBackend()
    results = run(
        args.train, args.validation, args.out, backend, load_scorer(args.eval_dir)
    )
    print(json.dumps(results["metrics"], indent=2))


if __name__ == "__main__":
    main()
