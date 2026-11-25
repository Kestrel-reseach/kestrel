"""Step 7 tests: integration suite, prompt-bug regression, gates and kill checks."""

from __future__ import annotations

from kestrel.features import GROUP_CAPS, format_item_prompt
from kestrel.halting import Attempt
from scripts.run_step7_integration import (
    MANDATORY_ENERGY_LABEL,
    compute_quality_gate,
    evaluate_configuration,
    paired_bootstrap_delta_ci,
    wilson_score_interval,
)


def test_wilson_score_interval() -> None:
    low, high = wilson_score_interval(50, 100)
    assert 40.0 < low < 50.0
    assert 50.0 < high < 60.0

    zero_low, zero_high = wilson_score_interval(0, 100)
    assert zero_low == 0.0
    assert zero_high > 0.0

    one_low, one_high = wilson_score_interval(100, 100)
    assert one_low < 100.0
    assert one_high == 100.0


def test_paired_bootstrap_delta_ci() -> None:
    a = [True] * 80 + [False] * 20
    b = [True] * 50 + [False] * 50
    lo, hi = paired_bootstrap_delta_ci(a, b, n_bootstrap=500, seed=42)
    assert lo > 0.0
    assert hi >= lo


def test_format_item_prompt_uses_group_caps() -> None:
    assert GROUP_CAPS["math"] == 768
    assert GROUP_CAPS["code"] == 1024
    assert GROUP_CAPS["science_mcq"] == 32
    assert GROUP_CAPS["knowledge_mcq"] == 32

    prompt, cap = format_item_prompt({"group": "code", "prompt": "Write a function."})
    assert cap == 1024
    assert len(prompt) > 0


def test_evaluate_configuration_and_quality_gate() -> None:
    suite = {
        "math": [{"item_id": "math_1", "prompt": "2+2"}],
        "science_mcq": [],
        "knowledge_mcq": [],
        "code": [],
        "knowledge_qa": [{"item_id": "kqa_1", "question": "capital of France"}],
        "memory_probe": [{"item_id": "mem_1", "question": "my name"}],
    }

    checkable_records = {
        "math_1": {
            "item_id": "math_1",
            "group": "math",
            "attempts": [
                {
                    "answer": "4",
                    "score": 0.9,
                    "correct": True,
                    "wall_time_ms": 500.0,
                    "net_energy_joules": 100.0,
                },
                {
                    "answer": "5",
                    "score": 0.2,
                    "correct": False,
                    "wall_time_ms": 500.0,
                    "net_energy_joules": 100.0,
                },
            ],
        }
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
            "stale_fact_error": False,
            "net_energy_joules": 10.0,
            "latency_ms": 100.0,
        }
    }
    step6_plain = {
        "mem_1": {
            "item_id": "mem_1",
            "correct": False,
            "stale_fact_error": True,
            "net_energy_joules": 10.0,
            "latency_ms": 100.0,
        }
    }
    step6_no_mem = {
        "mem_1": {
            "item_id": "mem_1",
            "correct": False,
            "stale_fact_error": False,
            "net_energy_joules": 20.0,
            "latency_ms": 200.0,
        }
    }

    p1 = evaluate_configuration(
        suite,
        checkable_records,
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
    assert p1["sections"]["checkable"]["net_energy_joules"] == 200.0
    assert p1["sections"]["store"]["correct"] == 1
    assert p1["sections"]["memory"]["correct"] == 1

    # Single-pass ablation gets 100 J on checkable
    abl_single = evaluate_configuration(
        suite,
        checkable_records,
        step5_with_store,
        step5_no_store,
        step6_gated,
        step6_plain,
        step6_no_mem,
        use_verifier=True,
        use_store=True,
        use_memory=True,
        use_fixed_two_pass=False,
    )
    assert abl_single["sections"]["checkable"]["net_energy_joules"] == 100.0

    # Quality gate check with failing G-Q (energy 200 > 171.55 cap)
    qg = compute_quality_gate(p1)
    assert qg["G_Q"]["status"] == "FAIL"
    assert qg["G_M"]["status"] == "PASS"
    assert qg["G_R"]["status"] == "PASS"
    assert "gap_closed_pct" in qg["G_F"]


def test_attempt_type_used() -> None:
    att = Attempt(answer="42", score=0.8, correct=True)
    assert att.correct is True


def test_energy_label_constant() -> None:
    assert (
        MANDATORY_ENERGY_LABEL
        == "Relative energy only; GPU NVML scope; same machine and session; not comparable across machines."
    )
