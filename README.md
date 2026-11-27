# Kestrel

Prototype 1 of a small, local assistant core that tries to get more answers per
joule and per GB than the best same-size quantized LLM, and to close part of the
quality gap to a frontier model on a narrow, locked test slice.

The shared plan, pass/kill gates and step status are kept in the owner's private
Notion workspace, not in this public repository. The locked suite, baselines and
energy harness live in the companion repository
**[Kestrel-reseach/kestrel-eval](https://github.com/Kestrel-reseach/kestrel-eval)**.

## What this is — and what it is not

The core is a pretrained small language model that still emits tokens. It is
**not token-free** and **not** a claim of frontier parity on general benchmarks.
The latent, token-free core is a later research bet and is out of scope here.

The general idea is to separate what the model knows from how hard it thinks.
This repository implements:

1. **Verifier head + verifier-gated halting** — a small trained head predicts
   whether an answer is correct. A query retries, refines, retrieves or
   escalates until the verifier is confident or a pass budget runs out;
   otherwise it abstains.
2. **Knowledge store on SSD** — embeddings and text chunks read by lookup, only
   when confidence is low. This is plain retrieval until it is trained jointly
   with the model; it is not a new architecture.
3. **Write-gated memory** — entries carry source, timestamp, confidence and
   decay. Writes happen only when verified; conflicts keep the old entry
   flagged instead of deleting it.
4. **Optional branches** (timeboxed, not on the product path): a ~100M looped
   core toy (`kestrel/looped_toy/`, Step 8) and a Titans-style learned memory
   module (Step 9).

Left out on purpose: latent sentence decoder, JEPA world model, affect core,
proactive event sentinel, neuromorphic hardware, from-scratch pretraining at
scale, full RL training of the verifier, and cloud use by default.

## Claims to be measured

The outcome is three numbers on the locked suite, reported whatever they are:

1. Quality gap to a frontier reference model **F**.
2. Energy (joules) per correct task versus the best same-hardware quantized
   small LLM with plain retrieval (**B-LLM**).
3. Resident memory (GB of VRAM and RAM) versus **B-LLM**.

Equal-or-better than frontier is plausible only on (a) the owner's-history
memory slice, where a frontier model has no memory unless handed the whole
history, and (b) checkable tasks where retry-until-verified closes part of the
gap. It is not expected on broad knowledge or hard open-ended reasoning at this
size.

The proposed gates are **G-Q** (beats B-LLM on checkable/store groups at no more
energy per correct task, or matches at ≤50% of its energy), **G-F** (reports the
fraction of the B-LLM-to-F gap closed), **G-M** (memory probe at least as good as
F with no history and within 5 points of F with full history; stale-fact errors
at most half of plain external memory) and **G-R** (≤1.5 GB resident VRAM on the
GPU tier, ≤4 GB resident RAM proposed on CPU tier, with joules per correct task
and p50/p95 latency reported). Exact definitions and the acceptance status are
in the owner's private plan; gates are frozen before Step 0 completes and never
changed after seeing results.

## Hardware targets

- **CPU tier:** Ryzen 7 5800H, no NPU.
- **GPU tier:** RTX 3060 Laptop 6 GB. The assistant gets **1–1.5 GB VRAM**; a
  3B model at 4-bit does not fit and is CPU-tier only.
- **Training:** free Colab T4. Sessions are not guaranteed — checkpoint every
  10–20 minutes and test resume.

## Repository layout

```
.
├── kestrel/                    prototype package
│   ├── verifier.py             Step 3 — correctness head + calibration
│   ├── halting.py              Step 4 — verifier-gated halting policy
│   ├── knowledge_store.py      Step 5 — SSD-backed retrieval
│   ├── memory_gated.py         Step 6 — write-gated memory
│   ├── neural_memory.py        Step 9 — Titans-style module (optional)
│   └── looped_toy/             Step 8 — ~100M looped core (optional)
└── tests/                      step-specific tests
```

The module files are currently empty placeholders. Implementation lands with
Steps 3–9, one step at a time, in order.

## Working rules

- Develop from the validation partition. Run the locked test partition **once**,
  at Step 10, with the frozen config.
- Never change a metric or threshold after seeing results. Failed components
  are reported, not hidden.
- Every run writes JSONL records with at least: `run_id`, `git_commit`,
  `config_hash`, `seed`, `split`, `item_id`, `system`, `tokens_generated`,
  `passes`, `verifier_score`, `retrieval_hits`, `bytes_read_from_ssd`,
  `energy_joules`, `wall_time_ms`, `peak_vram_mb`, `correct`. Save config,
  versions, seeds and checkpoint hashes with the run.
- Never commit secrets, real personal data, hostnames, IPs or device
  identifiers. Stage files by name and check the staged list and diff before
  committing. No co-author or AI-attribution trailers in commits or PRs.
- Model versions and sizes dated after early 2026 are unverified; confirm
  before downloading.

## Development

CI (`.github/workflows/ci.yml`) runs on pushes and pull requests to `main` and
performs lightweight checks only: `ruff check .`, `ruff format --check .` and
`pytest -q`. Model inference, energy measurement (NVML / CPU counters) and the
locked suite need the owner's local hardware, so they are **not** run in CI.

Python 3.12 is the CI target. To reproduce the checks locally:

```sh
python -m pip install ruff pytest
ruff check .
ruff format --check .
pytest -q
```

`pytest` currently exits 5 (nothing collected) because tests arrive with later
steps; CI tolerates that specific exit code and fails on real test failures.
