"""Hardware-free tests for the Phase 1 reward and early-exit helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.train_phase1_grpo import build_rows, make_reward_fn, sha256_file

from kestrel.early_exit import EarlyExitDecision, TokenEarlyExit, select_threshold
from kestrel.phase1_rewards import (
    CORRECT_REWARD,
    FORMAT_REWARD,
    format_ok,
    load_train_items,
    reward_batch,
)


def _fake_extractor(results: dict[str, bool]):
    def extract(item: dict, completion: str):
        return completion, results.get(item["item_id"], False)

    return extract


def test_reward_batch_uses_only_correctness_by_default() -> None:
    items = [{"item_id": "a", "group": "math"}, {"item_id": "b", "group": "math"}]
    rewards = reward_batch(
        items, ["answer A", "answer B"], extractor=_fake_extractor({"a": True})
    )
    assert rewards == [CORRECT_REWARD, 0.0]


def test_format_bonus_applies_only_to_correct_completions() -> None:
    item = {"item_id": "a", "group": "math"}
    extractor = _fake_extractor({"a": True})
    assert reward_batch(
        [item], ["Final Answer: 5"], extractor=extractor, format_bonus=True
    ) == [CORRECT_REWARD + FORMAT_REWARD]
    assert reward_batch([item], ["5"], extractor=extractor, format_bonus=True) == [
        CORRECT_REWARD
    ]
    wrong = {"item_id": "b", "group": "math"}
    assert reward_batch(
        [wrong], ["Final Answer: 5"], extractor=_fake_extractor({}), format_bonus=True
    ) == [0.0]


def test_format_ok_by_group() -> None:
    assert format_ok({"group": "math"}, "reasoning\nFinal Answer: 5")
    assert format_ok({"group": "science_mcq"}, "B")
    assert format_ok({"group": "code"}, "```python\ndef f(): pass\n```")
    assert not format_ok({"group": "math"}, "no marker")


def test_reward_batch_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        reward_batch(
            [{"item_id": "a", "group": "math"}], [], extractor=_fake_extractor({})
        )


def test_load_train_items_filters_to_checkable_groups(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    rows = [
        {"item_id": "m1", "group": "math", "target": "5"},
        {"item_id": "k1", "group": "knowledge_qa", "target": "Paris"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    items = load_train_items(path)
    assert [item["item_id"] for item in items] == ["m1"]


def test_early_exit_respects_min_tokens_and_threshold() -> None:
    policy = TokenEarlyExit(0.8, min_tokens=16, check_every=4)
    assert policy.update(
        tokens_generated=4, verifier_score=0.99, answer_extractable=True
    ) == (EarlyExitDecision(False, "min_tokens", 4))
    assert policy.update(
        tokens_generated=16, verifier_score=0.5, answer_extractable=True
    ).reason == ("not_confident")
    assert policy.update(
        tokens_generated=20, verifier_score=0.9, answer_extractable=True
    ).stop


def test_early_exit_requires_extractable_answer_and_honors_max_tokens() -> None:
    policy = TokenEarlyExit(0.5, min_tokens=0, check_every=1, max_tokens=10)
    assert not policy.update(
        tokens_generated=4, verifier_score=0.9, answer_extractable=False
    ).stop
    assert policy.update(
        tokens_generated=10, verifier_score=0.0, answer_extractable=False
    ) == (EarlyExitDecision(True, "max_tokens", 10))


def test_early_exit_consecutive_hits_and_check_interval() -> None:
    policy = TokenEarlyExit(0.5, min_tokens=0, check_every=2, consecutive_hits=2)
    assert policy.update(
        tokens_generated=2, verifier_score=0.9, answer_extractable=True
    ).reason == ("not_confident")
    assert not policy.update(
        tokens_generated=3, verifier_score=0.9, answer_extractable=True
    ).stop  # not due yet
    assert policy.update(
        tokens_generated=4, verifier_score=0.9, answer_extractable=True
    ).stop


def test_select_threshold_prefers_coverage_at_target_precision() -> None:
    scores = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    labels = [False, False, True, True, True, True]
    threshold = select_threshold(scores, labels, min_precision=0.75)
    assert threshold == 0.5
    assert select_threshold([0.2, 0.9], [False, True], min_precision=0.95) == 0.9
    assert select_threshold([0.2, 0.9], [False, False], min_precision=0.95) == 1.0


def test_build_rows_keeps_reward_metadata(tmp_path: Path) -> None:
    items = [
        {
            "item_id": "math_train_1",
            "group": "math",
            "prompt": "Solve this.",
            "target": "5",
            "metadata": {"source": "synthetic"},
        }
    ]
    rows = build_rows(items)
    assert rows[0]["item_id"] == "math_train_1"
    assert rows[0]["group"] == "math"
    assert rows[0]["target"] == "5"
    assert json.loads(rows[0]["metadata"]) == {"source": "synthetic"}
    assert rows[0]["prompt_sha256"]


def test_trl_reward_function_reads_columns() -> None:
    extractor = _fake_extractor({"math_train_1": True})
    reward_fn = make_reward_fn(extractor)
    rewards = reward_fn(
        ["a correct completion", "a wrong completion"],
        item_id=["math_train_1", "math_train_2"],
        group=["math", "math"],
        target=["5", "7"],
        metadata=['{"source": "synthetic"}', "{}"],
    )
    assert rewards == [CORRECT_REWARD, 0.0]


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n', encoding="utf-8")
    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64
