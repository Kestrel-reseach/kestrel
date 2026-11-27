#!/usr/bin/env python3
"""Phase 1 training: QLoRA GRPO on the checkable train partition.

Hardware-free here. Run this on a free Colab/Kaggle T4 (A owns the run). All heavy
imports are inside main() so the reward/row helpers can be unit-tested without
torch. Outputs go outside the git repo. Validation and the locked partition are
never touched by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
for path in (REPO_ROOT, WORKSPACE_ROOT / "kestrel-eval"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kestrel.phase1_rewards import CHECKABLE_GROUPS, load_train_items  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
CONTEXT_LENGTH = 2048
DEFAULT_STEPS = 500
DEFAULT_GENERATIONS = 8
DEFAULT_COMPLETION_TOKENS = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_rows(
    items: list[dict[str, Any]], tokenizer: Any | None = None
) -> list[dict[str, Any]]:
    """Build TRL dataset rows with canonical prompts and reward metadata."""
    from evaluation.prompts import format_prompt_for_model

    rows: list[dict[str, Any]] = []
    for item in items:
        prompt, prompt_hash, _ = format_prompt_for_model(item, tokenizer=tokenizer)
        rows.append(
            {
                "prompt": prompt,
                "prompt_sha256": prompt_hash,
                "item_id": str(item["item_id"]),
                "group": str(item.get("group", "")),
                "target": str(item.get("target", "")),
                "metadata": json.dumps(item.get("metadata", {}), sort_keys=True),
            }
        )
    return rows


def make_reward_fn(extractor: Callable) -> Callable[..., list[float]]:
    """Create a TRL-compatible reward function using the frozen extractor."""
    from kestrel.phase1_rewards import reward_batch

    def reward_fn(completions: list[str], **columns: Any) -> list[float]:
        n = len(completions)
        items = []
        for idx in range(n):
            metadata = columns.get("metadata", [{}] * n)[idx]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            items.append(
                {
                    "item_id": str(columns.get("item_id", [""] * n)[idx]),
                    "group": str(columns.get("group", [""] * n)[idx]),
                    "target": str(columns.get("target", [""] * n)[idx]),
                    "metadata": metadata or {},
                }
            )
        return reward_batch(items, completions, extractor=extractor)

    return reward_fn


def _private_dir() -> Path:
    configured = os.environ.get("KESTREL_PRIVATE_DIR")
    return (
        Path(configured).resolve()
        if configured
        else (WORKSPACE_ROOT / "kestrel_private").resolve()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-dir",
        type=Path,
        default=WORKSPACE_ROOT / "kestrel-eval" / "suite" / "train",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--num-generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument(
        "--max-completion-tokens", type=int, default=DEFAULT_COMPLETION_TOKENS
    )
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=20260102)
    parser.add_argument("--limit-per-group", type=int, default=None)
    parser.add_argument("--checkpoint-steps", type=int, default=25)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    from kestrel.phase1_rewards import default_extractor

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    paths = [args.suite_dir / f"{group}.jsonl" for group in CHECKABLE_GROUPS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing train files: {missing}")

    items: list[dict[str, Any]] = []
    for path in paths:
        group_items = load_train_items(path, groups=(path.stem,))
        if args.limit_per_group:
            group_items = group_items[: args.limit_per_group]
        items.extend(group_items)
    if not items:
        raise RuntimeError("No checkable train items found")

    out_dir = _private_dir() / "phase1_grpo"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_id": MODEL_ID,
        "context_length": CONTEXT_LENGTH,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "max_completion_tokens": args.max_completion_tokens,
        "learning_rate": args.learning_rate,
        "train_items": len(items),
        "train_files": {path.name: sha256_file(path) for path in paths},
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_ID,
        max_seq_length=CONTEXT_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=32,
        target_modules="all-linear",
        use_gradient_checkpointing="unlock_layers",
        random_state=args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    rows = build_rows(items, tokenizer=tokenizer)
    dataset = Dataset.from_list(rows)

    training_args = GRPOConfig(
        output_dir=str(out_dir),
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_prompt_length=CONTEXT_LENGTH,
        max_completion_length=args.max_completion_tokens,
        beta=0.0,
        save_steps=args.checkpoint_steps,
        logging_steps=10,
        seed=args.seed,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_reward_fn(default_extractor()),
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(str(out_dir / "adapter"))
    print(f"Training complete. Adapter and config in {out_dir}")
    print(f"Train items: {len(items)}; steps: {args.max_steps}; seed: {args.seed}")


if __name__ == "__main__":
    main()
