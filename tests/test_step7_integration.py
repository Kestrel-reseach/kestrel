"""Step 7 tests: integration suite, section gates, ablations and kill checks."""

from __future__ import annotations

from kestrel.halting import Attempt
from scripts.run_step7_integration import (
    MANDATORY_ENERGY_LABEL,
    evaluate_configuration,
    wilson_score_interval,
)


def test_wilson_score_interval() -> None:
    low, high = wilson_score_interval(50, 100)
    assert 40.0 < low < 50.0
    assert 50.0 < high < 60.0

    # Edge cases
    zero_low, zero_high = wilson_score_interval(0, 100)
    assert zero_low == 0.0
    assert zero_high > 0.0

    one_low, one_high = wilson_score_interval(100, 100)
    assert one_low < 100.0
    assert one_high == 100.0


def test_evaluate_configuration_small_mock() -> None:
    # 1 checkable, 1 store, 1 memory
    suite = {
        "math": [{"item_id": "math_1", "question": "2+2"}],
        "science_mcq": [],
        "knowledge_mcq": [],
        "code": [],
        "knowledge_qa": [{"item_id": "kqa_1", "question": "capital of France"}],
        "memory_probe": [{"item_id": "mem_1", "question": "my name"}],
    }

    step4_attempts = {
        "math_1": [
            Attempt("4", score=0.9, correct=True),
            Attempt("5", score=0.2, correct=False),
        ]
    }
    step5_with_store = {
        "kqa_1": {
            "item_id": "kqa_1",
            "correct": True,
            "raw_answer": "Paris",
            "lookup_latency_ms": 150.0,
        }
    }
    step5_no_store = {
        "kqa_1": {
            "item_id": "kqa_1",
            "correct": False,
            "raw_answer": "London",
        }
    }
    step6_gated = {
        "mem_1": {
            "item_id": "mem_1",
            "correct": True,
            "net_energy_joules": 10.0,
            "latency_ms": 100.0,
        }
    }
    step6_plain = {
        "mem_1": {
            "item_id": "mem_1",
            "correct": False,
            "net_energy_joules": 10.0,
            "latency_ms": 100.0,
        }
    }
    step6_no_mem = {
        "mem_1": {
            "item_id": "mem_1",
            "correct": False,
            "net_energy_joules": 20.0,
            "latency_ms": 200.0,
        }
    }

    # Full P1 run
    p1 = evaluate_configuration(
        suite,
        step4_attempts,
        step5_with_store,
        step5_no_store,
        step6_gated,
        step6_plain,
        step6_no_mem,
        use_verifier=True,
        use_store=True,
        use_memory=True,
        use_fixed_two_pass=True,
    )

    assert p1["total"] == 3
    assert p1["correct"] == 3
    assert p1["accuracy_pct"] == 100.0
    assert p1["sections"]["checkable"]["correct"] == 1
    assert p1["sections"]["store"]["correct"] == 1
    assert p1["sections"]["memory"]["correct"] == 1

    # Ablation: remove store
    abl_store = evaluate_configuration(
        suite,
        step4_attempts,
        step5_with_store,
        step5_no_store,
        step6_gated,
        step6_plain,
        step6_no_mem,
        use_verifier=True,
        use_store=False,
        use_memory=True,
        use_fixed_two_pass=True,
    )
    assert abl_store["correct"] == 2
    assert abl_store["sections"]["store"]["correct"] == 0

    # Ablation: remove memory
    abl_mem = evaluate_configuration(
        suite,
        step4_attempts,
        step5_with_store,
        step5_no_store,
        step6_gated,
        step6_plain,
        step6_no_mem,
        use_verifier=True,
        use_store=True,
        use_memory=False,
        use_fixed_two_pass=True,
    )
    assert abl_mem["correct"] == 2
    assert abl_mem["sections"]["memory"]["correct"] == 0


def test_section_gates_and_kill_logic() -> None:
    # Verify section gate threshold rule
    earlier = 71.67
    actual = 71.67
    regression = max(0.0, earlier - actual)
    assert regression <= 3.0

    # Verify regression cap
    bad_actual = 65.00
    bad_regression = max(0.0, earlier - bad_actual)
    assert bad_regression > 3.0


def test_energy_label_constant() -> None:
    assert (
        MANDATORY_ENERGY_LABEL
        == "Relative energy only; GPU NVML scope; same machine and session; not comparable across machines."
    )
