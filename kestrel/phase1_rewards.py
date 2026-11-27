"""Hardware-free reward helpers for verifiable-reward training.

Kept dependency-free so it can be unit-tested without torch. The heavy training
loop lives in scripts/train_phase1_grpo.py and imports this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

ExtractFn = Callable[[dict[str, Any], str], tuple[Any, bool]]

CORRECT_REWARD = 1.0
FORMAT_REWARD = 0.1
CHECKABLE_GROUPS = ("math", "science_mcq", "knowledge_mcq", "code")


def default_extractor() -> ExtractFn:
    """Load the shared frozen extractor from the companion evaluation repo."""
    from evaluation.extractors import extract_answer_for_group

    return extract_answer_for_group


def format_ok(item: dict[str, Any], completion: str) -> bool:
    """Cheap structural check; never larger than the correctness reward."""
    group = str(item.get("group", ""))
    text = completion.strip()
    if group == "math":
        return "final answer:" in text.lower()
    if group in ("science_mcq", "knowledge_mcq"):
        return bool(text) and text[0].upper() in "ABCD"
    if group == "code":
        return "def " in text or "```" in text
    return bool(text)


def reward_completion(
    item: dict[str, Any],
    completion: str,
    *,
    extractor: ExtractFn | None = None,
    format_bonus: bool = False,
) -> float:
    """Return the verifiable reward for one completion."""
    extract = extractor or default_extractor()
    _, is_correct = extract(item, completion)
    score = CORRECT_REWARD if is_correct else 0.0
    if score and format_bonus and format_ok(item, completion):
        score += FORMAT_REWARD
    return score


def reward_batch(
    items: list[dict[str, Any]],
    completions: list[str],
    *,
    extractor: ExtractFn | None = None,
    format_bonus: bool = False,
) -> list[float]:
    """Return one reward per (item, completion) pair."""
    if len(items) != len(completions):
        raise ValueError("items and completions must have the same length")
    return [
        reward_completion(item, text, extractor=extractor, format_bonus=format_bonus)
        for item, text in zip(items, completions)
    ]


def load_train_items(
    path: Path | str,
    *,
    groups: tuple[str, ...] = CHECKABLE_GROUPS,
) -> list[dict[str, Any]]:
    """Load checkable train items from a JSONL file without printing item text."""
    selected = set(groups)
    items: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("group") in selected:
                items.append(record)
    return items
