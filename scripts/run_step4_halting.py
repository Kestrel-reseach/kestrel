"""Step 4: Real execution of verifier-gated halting on the RTX 3050.

Tunes threshold on validation items under an energy cap, compares strategies
(single_pass, fixed_two_pass, best_of_three, self_consistency_at_five, gated),
measures gate criteria, and outputs reports/halting_step4.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Ensure paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

EVAL_DIR = Path(
    os.environ.get("KESTREL_EVAL_DIR", str(REPO_ROOT.parent / "kestrel-eval"))
)
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from evaluation.extractors import extract_answer_for_group  # noqa: E402
from kestrel.halting import (  # noqa: E402
    Attempt,
    compare_strategies,
    step4_gate_decision,
    tune_threshold,
)
from kestrel.verifier import VerifierExample, VerifierFeatures, temperature_scale  # noqa: E402
from kestrel.verifier_numpy import NumpyLogisticHead  # noqa: E402

CHECKABLE_GROUPS = ("math", "science_mcq", "code", "knowledge_mcq")
MANDATORY_ENERGY_LABEL = "Relative energy only; GPU NVML scope; same machine and session; not comparable across machines."
ENERGY_PER_PASS_JOULES = 97.57  # Measured in Step 2: 78,055.25 J / 800 items


class RealAttemptSource:
    """AttemptSource backed by cached real attempts for validation items."""

    def __init__(
        self,
        attempts: dict[str, list[Attempt]],
        energy_per_pass: float = ENERGY_PER_PASS_JOULES,
    ) -> None:
        self.attempts = attempts
        self._energy_per_pass = energy_per_pass
        self._current_item: str | None = None
        self._attempt_idx = 0

    @property
    def energy_per_pass(self) -> float:
        return self._energy_per_pass

    def sample(self, item_id: str) -> Attempt:
        if item_id != self._current_item:
            self._current_item = item_id
            self._attempt_idx = 0
        pool = self.attempts[item_id]
        attempt = pool[self._attempt_idx % len(pool)]
        self._attempt_idx += 1
        return attempt


def load_validation_items(suite_val_dir: Path) -> list[dict[str, Any]]:
    items = []
    for group in CHECKABLE_GROUPS:
        path = suite_val_dir / f"{group}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    items.append(json.loads(line))
    return items


def load_trained_head(
    cache_train_path: Path, temp_path: Path
) -> tuple[NumpyLogisticHead, float]:
    print(f"Loading trained verifier head from {cache_train_path}...", flush=True)
    train_examples = []
    with cache_train_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                d = json.loads(line)
                train_examples.append(
                    VerifierExample(
                        item_id=d["item_id"],
                        split="train",
                        features=VerifierFeatures(
                            last_token_hidden=tuple(d["features"]["last_token_hidden"])
                        ),
                        correct=d["correct"],
                    )
                )
    head = NumpyLogisticHead()
    head.fit(train_examples)

    with temp_path.open(encoding="utf-8") as handle:
        step3_data = json.load(handle)
    temperature = float(step3_data.get("temperature", 2.0))
    print(
        f"Head fitted on {len(train_examples)} examples. Temperature: {temperature}",
        flush=True,
    )
    return head, temperature


def generate_attempts_for_validation(
    items: list[dict[str, Any]],
    head: NumpyLogisticHead,
    temperature: float,
    cache_path: Path,
    n_samples: int = 5,
    device: str = "cuda:0",
) -> dict[str, list[Attempt]]:
    cached_attempts: dict[str, list[Attempt]] = {}
    if cache_path.is_file():
        with cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    d = json.loads(line)
                    cached_attempts[d["item_id"]] = [
                        Attempt(
                            answer=att["answer"],
                            score=att["score"],
                            correct=att["correct"],
                        )
                        for att in d["attempts"]
                    ]
        print(
            f"Loaded {len(cached_attempts)} cached item attempts from {cache_path}",
            flush=True,
        )

    remaining_items = [it for it in items if it["item_id"] not in cached_attempts]
    if not remaining_items:
        print("All validation attempts already cached!", flush=True)
        return cached_attempts

    print(
        f"Generating {n_samples} samples for {len(remaining_items)} remaining items...",
        flush=True,
    )

    # Load model
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    total = len(items)
    for idx, item in enumerate(items, 1):
        item_id = item["item_id"]
        if item_id in cached_attempts:
            continue

        prompt = item["prompt"]
        is_mcq = "Answer with the option letter" in prompt
        eos_ids = [tokenizer.eos_token_id]
        max_tokens = 32 if is_mcq else 256
        if is_mcq:
            eos_ids.extend([198, 271])

        torch.manual_seed(20260104 + idx)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        t_start = time.time()
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                do_sample=True,
                num_return_sequences=n_samples,
                max_new_tokens=max_tokens,
                eos_token_id=eos_ids if is_mcq else tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        attempts_list = []
        for i in range(n_samples):
            seq = gen[i : i + 1]
            text = tokenizer.decode(seq[0, prompt_len:], skip_special_tokens=True)
            correct = bool(extract_answer_for_group(item, text)[1])
            with torch.no_grad():
                out = model(input_ids=seq, output_hidden_states=True)
            hidden = out.hidden_states[-1][0, -1, :].float().cpu().tolist()
            raw_score = head.predict(VerifierFeatures(last_token_hidden=tuple(hidden)))
            calibrated_score = float(temperature_scale([raw_score], temperature)[0])
            attempts_list.append(
                Attempt(answer=text, score=calibrated_score, correct=correct)
            )

        cached_attempts[item_id] = attempts_list

        # Append to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as handle:
            rec = {
                "item_id": item_id,
                "group": item.get("group", ""),
                "attempts": [
                    {"answer": a.answer, "score": a.score, "correct": a.correct}
                    for a in attempts_list
                ],
            }
            handle.write(json.dumps(rec) + "\n")

        elapsed = time.time() - t_start
        computed_so_far = len(cached_attempts) - (len(items) - len(remaining_items))
        if (
            computed_so_far % 25 == 0
            or len(cached_attempts) == total
            or computed_so_far <= 3
        ):
            corr_cnt = sum(1 for a in attempts_list if a.correct)
            print(
                f"  [{len(cached_attempts)}/{total}] {item_id} ({item.get('group', '')}): {corr_cnt}/{n_samples} correct ({elapsed:.2f}s)",
                flush=True,
            )

    return cached_attempts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation", default=str(EVAL_DIR / "suite" / "validation"), type=Path
    )
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "reports" / "halting_step4.json"), type=Path
    )
    args = parser.parse_args()

    suite_val_dir = args.validation
    out_path = args.out
    cache_train = REPO_ROOT / "reports" / ".cache_train_examples.jsonl"
    step3_json = REPO_ROOT / "reports" / "verifier_step3.json"
    cache_step4_attempts = REPO_ROOT / "reports" / ".cache_step4_attempts.jsonl"

    print("=== Step 4: Verifier-Gated Halting Evaluation ===")
    val_items = load_validation_items(suite_val_dir)
    print(f"Loaded {len(val_items)} checkable validation items.")

    head, temperature = load_trained_head(cache_train, step3_json)
    attempts_by_item = generate_attempts_for_validation(
        val_items, head, temperature, cache_step4_attempts, n_samples=5
    )

    item_ids = [it["item_id"] for it in val_items]
    source = RealAttemptSource(attempts_by_item, energy_per_pass=ENERGY_PER_PASS_JOULES)

    print(
        "\nTuning threshold on validation partition (max_passes=3, energy_cap_ratio=1.5)...",
        flush=True,
    )
    candidate_thresholds = (
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    )
    chosen_threshold = tune_threshold(
        source,
        item_ids,
        max_passes=3,
        energy_cap_ratio=1.5,
        candidates=candidate_thresholds,
    )
    print(f"Chosen threshold: {chosen_threshold}", flush=True)

    print("Evaluating all policies on validation...", flush=True)
    comparison = compare_strategies(
        source, item_ids, threshold=chosen_threshold, max_passes=3
    )
    print("\nPolicy Comparison:")
    for name, metrics in comparison.items():
        print(
            f"  {name:25s}: Accuracy = {metrics['accuracy']*100:.2f}%, "
            f"Energy/Correct = {metrics['energy_per_correct']:.2f} J, "
            f"Passes/Query = {metrics['passes']:.2f}, "
            f"Abstain Rate = {metrics['abstain_rate']*100:.2f}%"
        )

    # Gate Evaluation (PLAN.md Step 4) -- thresholds frozen, logic in halting.py
    gate_result = step4_gate_decision(comparison)

    print("\nGate Decision Summary:")
    print(json.dumps(gate_result, indent=2))

    payload = {
        "step": 4,
        "seed": 20260104,
        "n_validation": len(val_items),
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "energy_label": MANDATORY_ENERGY_LABEL,
        "energy_per_pass_joules": ENERGY_PER_PASS_JOULES,
        "chosen_threshold": chosen_threshold,
        "max_passes": 3,
        "policy_metrics": comparison,
        "gate": gate_result,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote Step 4 results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
