"""Step 6: Real hardware evaluation of write-gated memory on RTX 3050.

Evaluates:
- No memory (prompt with question only, no personal history)
- Plain external memory (prompt with full un-gated history log)
- Write-gated memory (prompt with active, verified, non-stale memory from WriteGatedMemory)

Across all 150 items in suite/validation/memory_probe.jsonl (100 stale, 50 static).
Applies PLAN.md Step 6 pass/kill rules and writes reports/memory_step6.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from evaluation.generators.stale_fact_scorer import score_memory_probe_response
from kestrel.memory_gated import (
    MemoryEntry,
    PlainMemory,
    WriteGatedMemory,
    paired_bootstrap_accuracy_difference,
    step6_gate_decision,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
KESTREL_EVAL_DIR = Path(
    REPO_ROOT.parent / "kestrel-eval"
    if (REPO_ROOT.parent / "kestrel-eval").is_dir()
    else REPO_ROOT
)
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 20260104


class NvmlPowerMonitor:
    """Reads GPU power in milliwatts via NVML."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._handle = None
        self._available = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._available = True
        except Exception:
            self._available = False

    def read_power_watts(self) -> float:
        if not self._available or self._handle is None:
            return 0.0
        try:
            import pynvml

            mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            return float(mw) / 1000.0
        except Exception:
            return 0.0


def extract_question(prompt_text: str) -> str:
    """Extract clean Question: ... portion from the prompt."""
    if "Question:" in prompt_text:
        q_part = prompt_text.split("Question:")[-1].strip()
        return f"Question: {q_part}"
    return prompt_text.strip()


def run_system_evaluation(
    system_name: str,
    items: list[dict[str, Any]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    cache_path: Path,
    device: str = "cuda:0",
    power_monitor: NvmlPowerMonitor | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Run evaluation for one of the three systems on 150 items."""
    if cache_path.is_file():
        cached_results = []
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cached_results.append(json.loads(line))
        if len(cached_results) == len(items):
            print(
                f"Loaded {len(cached_results)} cached predictions for {system_name} from {cache_path}",
                flush=True,
            )
            total_j = sum(r.get("net_energy_joules", 0.0) for r in cached_results)
            return cached_results, total_j

    print(f"Evaluating {system_name} on {len(items)} validation items...", flush=True)

    system_prompt = (
        "You are a helpful personal assistant. "
        "Answer the question with only the direct factual value (e.g. name, date, phone number, or short phrase). "
        "Do not write full sentences or explanations."
    )

    results = []
    total_net_energy_joules = 0.0
    idle_power_w = 6.1444  # Step 0 measured formal idle baseline

    for idx, item in enumerate(items, 1):
        q_part = extract_question(item["prompt"])

        # Construct prompt according to system
        if system_name == "no_memory":
            prompt_content = q_part
        elif system_name == "plain_memory":
            plain = PlainMemory()
            events = item.get("metadata", {}).get("events", [])
            for e_idx, ev in enumerate(events):
                plain.write(
                    MemoryEntry(
                        entry_id=f"{item['item_id']}_{e_idx}",
                        text=ev["content"],
                        source=ev["source"],
                        timestamp=ev["timestamp"],
                        confidence=1.0,
                        attribute=item.get("metadata", {}).get("attribute"),
                    )
                )
            recalled = plain.recall(q_part)
            prompt_content = (
                "Review the personal history log below and answer the question.\n\n"
                f"--- PERSONAL HISTORY LOG ---\n{recalled}\n--- END LOG ---\n\n"
                f"{q_part}"
            )
        elif system_name == "gated_memory":
            wgm = WriteGatedMemory(write_threshold=0.5)
            events = item.get("metadata", {}).get("events", [])
            for e_idx, ev in enumerate(events):
                wgm.write(
                    MemoryEntry(
                        entry_id=f"{item['item_id']}_{e_idx}",
                        text=ev["content"],
                        source=ev["source"],
                        timestamp=ev["timestamp"],
                        confidence=1.0,
                        attribute=item.get("metadata", {}).get("attribute"),
                    )
                )
            recalled = wgm.recall(q_part)
            prompt_content = (
                "Review the personal history log below and answer the question.\n\n"
                f"--- PERSONAL HISTORY LOG ---\n{recalled}\n--- END LOG ---\n\n"
                f"{q_part}"
            )
        else:
            raise ValueError(f"Unknown system {system_name}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_content},
        ]
        full_prompt = (
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            + "Answer:"
        )

        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

        p_before = power_monitor.read_power_watts() if power_monitor else idle_power_w
        t0 = time.perf_counter()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                eos_token_id=[tokenizer.eos_token_id, 198, 271],
            )

        dur_s = time.perf_counter() - t0
        p_after = power_monitor.read_power_watts() if power_monitor else idle_power_w
        p_mean = max(idle_power_w, (p_before + p_after) / 2.0)
        net_energy_j = max(0.01, (p_mean - idle_power_w) * dur_s)
        total_net_energy_joules += net_energy_j

        gen_tokens = out[0][inputs["input_ids"].shape[1] :]
        raw_answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        stale_vals = item.get("metadata", {}).get("stale_values", [])
        is_stale = item.get("metadata", {}).get("is_stale_fact", False)
        target = str(item.get("target", ""))

        score = score_memory_probe_response(
            prediction=raw_answer,
            target=target,
            stale_values=stale_vals,
            is_stale_fact_item=is_stale,
        )

        record = {
            "item_id": item["item_id"],
            "is_stale_fact": is_stale,
            "target": target,
            "stale_values": stale_vals,
            "raw_answer": raw_answer,
            "correct": bool(score["correct"]),
            "stale_fact_error": bool(score["stale_fact_error"]),
            "latency_ms": round(dur_s * 1000, 2),
            "net_energy_joules": round(net_energy_j, 4),
        }
        results.append(record)

        if idx % 50 == 0 or idx == len(items):
            acc_so_far = sum(r["correct"] for r in results) / len(results) * 100
            print(
                f"  [{system_name}] {idx}/{len(items)} processed (Acc: {acc_so_far:.2f}%)",
                flush=True,
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    return results, total_net_energy_joules


def main() -> None:
    print("=== Step 6: Write-Gated Memory Evaluation on RTX 3050 ===")
    val_path = KESTREL_EVAL_DIR / "suite" / "validation" / "memory_probe.jsonl"
    out_path = REPO_ROOT / "reports" / "memory_step6.json"
    cache_no_mem = REPO_ROOT / "reports" / ".cache_step6_no_memory.jsonl"
    cache_plain_mem = REPO_ROOT / "reports" / ".cache_step6_plain_memory.jsonl"
    cache_gated_mem = REPO_ROOT / "reports" / ".cache_step6_gated_memory.jsonl"

    items = []
    with val_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    print(
        f"Loaded {len(items)} validation memory probe items "
        f"({sum(1 for x in items if x.get('metadata', {}).get('is_stale_fact'))} stale, "
        f"{sum(1 for x in items if not x.get('metadata', {}).get('is_stale_fact'))} static)."
    )

    power_monitor = NvmlPowerMonitor(0)

    # Load Model once for all three passes
    print(f"Loading {DEFAULT_BASE_MODEL} in 4-bit NF4...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL, local_files_only=True)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_BASE_MODEL,
        quantization_config=bnb_cfg,
        device_map="cuda:0",
        local_files_only=True,
    )
    model.eval()

    # System 1: No Memory
    res_no, energy_no = run_system_evaluation(
        "no_memory", items, model, tokenizer, cache_no_mem, power_monitor=power_monitor
    )

    # System 2: Plain External Memory
    res_plain, energy_plain = run_system_evaluation(
        "plain_memory",
        items,
        model,
        tokenizer,
        cache_plain_mem,
        power_monitor=power_monitor,
    )

    # System 3: Write-Gated Memory
    res_gated, energy_gated = run_system_evaluation(
        "gated_memory",
        items,
        model,
        tokenizer,
        cache_gated_mem,
        power_monitor=power_monitor,
    )

    # Compute Detailed Metrics
    stale_flags = [
        bool(item.get("metadata", {}).get("is_stale_fact")) for item in items
    ]

    def compute_sys_metrics(
        results: list[dict[str, Any]], energy_total: float
    ) -> dict[str, Any]:
        n = len(results)
        corrects = [r["correct"] for r in results]
        stale_errors = sum(1 for r in results if r["stale_fact_error"])

        stale_corrects = [
            r["correct"] for r, is_s in zip(results, stale_flags, strict=True) if is_s
        ]
        static_corrects = [
            r["correct"]
            for r, is_s in zip(results, stale_flags, strict=True)
            if not is_s
        ]

        acc = sum(corrects) / n
        stale_acc = sum(stale_corrects) / len(stale_corrects) if stale_corrects else 0.0
        static_acc = (
            sum(static_corrects) / len(static_corrects) if static_corrects else 0.0
        )

        n_correct = sum(corrects)
        energy_per_correct = (
            round(energy_total / n_correct, 2) if n_correct > 0 else 99999.0
        )

        return {
            "accuracy": round(acc, 4),
            "accuracy_pct": round(acc * 100, 2),
            "correct": n_correct,
            "total": n,
            "stale_accuracy_pct": round(stale_acc * 100, 2),
            "static_accuracy_pct": round(static_acc * 100, 2),
            "stale_fact_errors": stale_errors,
            "stale_fact_error_rate_pct": round(
                stale_errors / len(stale_corrects) * 100, 2
            )
            if stale_corrects
            else 0.0,
            "total_net_energy_joules": round(energy_total, 2),
            "energy_per_correct_joules": energy_per_correct,
        }

    m_no = compute_sys_metrics(res_no, energy_no)
    m_plain = compute_sys_metrics(res_plain, energy_plain)
    m_gated = compute_sys_metrics(res_gated, energy_gated)

    # Paired Bootstrap Difference between Gated and Plain
    mean_diff, ci_l, ci_u = paired_bootstrap_accuracy_difference(
        [r["correct"] for r in res_gated],
        [r["correct"] for r in res_plain],
        n_resamples=10_000,
        seed=SEED,
        ci_level=0.95,
    )

    gate = step6_gate_decision(
        no_memory_acc=m_no["accuracy"],
        plain_acc=m_plain["accuracy"],
        gated_acc=m_gated["accuracy"],
        plain_stale_errors=m_plain["stale_fact_errors"],
        gated_stale_errors=m_gated["stale_fact_errors"],
        plain_static_acc=m_plain["static_accuracy_pct"] / 100.0,
        gated_static_acc=m_gated["static_accuracy_pct"] / 100.0,
        paired_ci=(ci_l, ci_u),
    )

    report = {
        "step": 6,
        "seed": SEED,
        "n_validation_items": len(items),
        "n_stale_fact_items": sum(stale_flags),
        "n_static_fact_items": len(stale_flags) - sum(stale_flags),
        "base_model": DEFAULT_BASE_MODEL,
        "quantization": "bitsandbytes 4-bit NF4",
        "context_length": 2048,
        "energy_label": "Relative energy only; GPU NVML scope; same machine and session; not comparable across machines.",
        "frontier_reference": {
            "status": "F = gemini-2.5-pro free tier, not run yet (D20)",
            "f_no_history_expected": 0.0,
            "f_full_history_target_bound_pct": 90.0,
        },
        "systems": {
            "no_memory": m_no,
            "plain_memory": m_plain,
            "gated_memory": m_gated,
        },
        "comparison": {
            "accuracy_gain_over_plain_pp": gate.accuracy_gain_pp,
            "paired_bootstrap_ci_95": [gate.paired_ci_lower, gate.paired_ci_upper],
            "static_forgetting_pp": gate.static_forgetting_pp,
            "stale_fact_error_reduction_ratio": gate.stale_error_ratio,
            "plain_stale_errors": gate.plain_stale_errors,
            "gated_stale_errors": gate.gated_stale_errors,
            "stale_error_target_max": gate.stale_error_target_max,
        },
        "gate": {
            "condition_1_accuracy_gain": {
                "gain_pp": gate.accuracy_gain_pp,
                "target_min_gain_pp": 5.0,
                "passes": gate.passes_accuracy_gain,
            },
            "condition_2_ci_excludes_zero": {
                "ci_lower_pp": gate.paired_ci_lower,
                "ci_upper_pp": gate.paired_ci_upper,
                "excludes_zero": gate.passes_ci_excludes_zero,
            },
            "condition_3_forgetting": {
                "static_forgetting_pp": gate.static_forgetting_pp,
                "forgetting_cap_pp": gate.forgetting_cap_pp,
                "passes": gate.passes_forgetting_cap,
            },
            "condition_4_stale_errors": {
                "plain_stale_errors": gate.plain_stale_errors,
                "gated_stale_errors": gate.gated_stale_errors,
                "stale_error_target_max": gate.stale_error_target_max,
                "passes": gate.passes_stale_error_reduction,
            },
            "kill_check": {
                "reduces_stale_errors": gate.passes_stale_error_reduction,
                "rule": "If gating does not reduce stale errors, keep plain memory with timestamps and say so.",
                "passes": gate.passes_stale_error_reduction,
            },
            "gate_decision": gate.gate_decision,
            "recommended_component": gate.recommended_component,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Step 6 Evaluation Summary ===")
    print(
        f"No-Memory Acc    : {m_no['accuracy_pct']:.2f}% (Errors: {m_no['stale_fact_errors']} stale)"
    )
    print(
        f"Plain-Memory Acc : {m_plain['accuracy_pct']:.2f}% (Errors: {m_plain['stale_fact_errors']} stale)"
    )
    print(
        f"Gated-Memory Acc : {m_gated['accuracy_pct']:.2f}% (Errors: {m_gated['stale_fact_errors']} stale)"
    )
    print(
        f"Gain over Plain  : +{gate.accuracy_gain_pp:.2f} pp (95% CI: [{gate.paired_ci_lower}, {gate.paired_ci_upper}] pp)"
    )
    print(f"Static Forgetting: {gate.static_forgetting_pp:.2f} pp (cap <= 3.0 pp)")
    print(
        f"Stale Errors     : {gate.gated_stale_errors} vs plain {gate.plain_stale_errors} (cap <= {gate.stale_error_target_max})"
    )
    print(f"Gate Decision    : {gate.gate_decision}")
    print(f"Recommended      : {gate.recommended_component}")
    print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
