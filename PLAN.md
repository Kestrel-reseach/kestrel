# Kestrel — Prototype 1 Plan

Shared plan for any agent or person working on this project. Read this first, do one step at a time, and update the status list at the bottom when a step passes, fails or is killed.

Full background lives in the owner's Notion workspace (charter, protocol, research pages). This file is the working copy for the code.

## What this is

A small, local assistant core that tries to get more answers per joule and per GB than the best same-size quantized LLM, and to close part of the quality gap to a frontier model on a narrow, locked test slice.

It is **not** token-free and **not** a claim of frontier parity on general benchmarks. The core is a pretrained small language model that still emits tokens. The latent, token-free core is a later research bet and is out of scope here.

Parts built (from the "Orbit-style" idea of separating what the model knows from how hard it thinks):

1. **Verifier head + verifier-gated halting**: a small trained head predicts whether an answer is correct; a query retries, refines, retrieves or escalates until the verifier is confident or a pass budget runs out; otherwise it abstains.
2. **Knowledge store on SSD**: embeddings and text chunks read by lookup, read only when confidence is low. This is plain retrieval until it is trained jointly with the model. Do not call it a new architecture.
3. **Write-gated memory**: entries carry source, timestamp, confidence and decay; writes happen only when verified; conflicts keep the old entry flagged instead of deleting it.
4. **Optional branches** (timeboxed, not on the product path): a ~100M looped-core toy on the Colab T4 (Step 8), and a Titans-style learned memory module (Step 9).

Left out on purpose: latent sentence decoder, JEPA world model, affect core, proactive event sentinel, neuromorphic hardware, from-scratch pretraining at scale, full RL training of the verifier, cloud use by default.

## Layout

```
kestrel/                      (this folder, not a git repo)
├── PLAN.md  AGENTS.md
├── kestrel/                  repo: Kestrel-reseach/kestrel      (the prototype code)
│   ├── kestrel/              verifier.py halting.py knowledge_store.py
│   │                         memory_gated.py neural_memory.py looped_toy/
│   └── tests/
└── kestrel-eval/             repo: Kestrel-reseach/kestrel-eval (suite, baselines, energy)
    ├── evaluation/           manifest.yaml run_matrix.py metrics.py
    ├── energy/               harness.py
    ├── suite/                public/ synthetic/ personal_memory/
    └── baselines/ reports/ tests/
```

Both repos are public under the `Kestrel-reseach` org (`https://github.com/Kestrel-reseach/kestrel` and `https://github.com/Kestrel-reseach/kestrel-eval`). The module files are empty placeholders.

## Hardware

- CPU tier: Ryzen 7 5800H, no NPU. **Owner-accepted test tier (2026-11-18):** Ryzen 7 7435HS, because the 5800H machine is unavailable.
- GPU tier: RTX 3060 Laptop 6 GB. The assistant gets **1–1.5 GB VRAM** (a 3B model at 4-bit does not fit; it is CPU-tier only). **Owner-accepted test tier (2026-11-18):** RTX 3050 Laptop 4 GB (driver 615.78.08). The 1–1.5 GB assistant budget and all frozen gates are unchanged; every result must state which tier produced it, and a re-run on the 3060 is deferred until that hardware is available.
- Training: free Colab T4. Sessions are not guaranteed; checkpoint every 10–20 minutes and test resume.

## Goal as measured claims (not as promises)

Report three numbers on the locked suite, whatever they are:

1. Quality gap to a frontier reference model **F**.
2. Energy (joules) per correct task versus the best same-hardware quantized small LLM with plain retrieval (**B-LLM**).
3. Resident memory (GB of VRAM and RAM) versus B-LLM.

Equal-or-better than frontier is plausible only on (a) the owner's-history memory slice, where a frontier model has no memory unless handed the whole history, and (b) checkable tasks where retry-until-verified closes part of the gap. It is not expected on broad knowledge or hard open-ended reasoning at this size.

## Locked evaluation suite (Step 1)

| Group | Size | Answer check |
| --- | --- | --- |
| Math word problems | 200 | exact numeric match |
| Science multiple choice (ARC-Challenge style) | 200 | answer key |
| Knowledge multiple choice (MMLU style, 10 subjects) | 300 | answer key |
| Code tasks with unit tests | 100 | tests pass |
| Knowledge QA over a fixed corpus snapshot | 300 | key match |
| Personal-memory probe (30 stale-fact items) | 100 | key from source history |

Fixed train / validation / locked-test splits, hashed in the manifest before any tuning. Add privately written or synthetic items, because public items may be in every model's training data; report both. The personal-memory probe uses **synthetic histories only** unless the owner approves real items one by one.

Reference systems: **F** (frontier model, run once on public and synthetic items only; for the memory probe run with no history and with the full history in the prompt), **B-LLM** (best quantized small model from the shortlist, plain retrieval), **P1** (the prototype).

## Pass gates (proposed — owner must accept before Step 0 freezes them)

- **G-Q**: on the checkable and store groups, P1 beats B-LLM by at least 5 percentage points (paired bootstrap 95% interval excludes zero) at no more energy per correct task; or matches B-LLM accuracy at no more than 50% of its energy per correct task.
- **G-F**: report the gap to F and the fraction of the B-LLM-to-F gap that P1 closes. 25% or more on the checkable groups is a strong result.
- **G-M**: on the memory probe, P1 is at least as good as F with no history and within 5 points of F with the full history; stale-fact errors at most half those of plain external memory.
- **G-R**: at most 1.5 GB resident VRAM on the GPU tier; CPU tier proposed 4 GB resident RAM (SSD index excluded, reported separately); report joules per correct task, p50 and p95 latency.
- **No regression**: adding a component must not drop earlier-step results by more than 3 points.

## Rules for every step

- Work from the validation partition. Run the locked test partition **once**, at Step 10, with the frozen config.
- Never change a metric or threshold after seeing results. Failed components are reported, not hidden.
- Every run writes JSONL records with at least: run_id, git_commit, config_hash, seed, split, item_id, system, tokens_generated, passes, verifier_score, retrieval_hits, bytes_read_from_ssd, energy_joules, wall_time_ms, peak_vram_mb, correct. Save config, versions, seeds and checkpoint hashes.
- Energy is joules per correct task, not tokens per second per watt. Record an idle baseline (about 10 minutes), subtract it and also report the total. Calibrate software counters (CPU energy counters, NVML) against a wall meter once; if that is not possible, report relative energy only and say so on every result.
- A pilot is "promising", not "passed", until it holds on locked data.
- Model versions and sizes dated after early 2026 are unverified; confirm before downloading.
- Never commit secrets, real personal data, hostnames, IPs, or device identifiers. Stage files by name, check the staged list and diff before committing. No co-author or AI-attribution trailers in commits or PRs.

## Steps

Times are rough estimates for one person part time. Steps 0–7 and 10 are the product path (about 9–10 weeks). Steps 8–9 are optional timeboxed branches.

### Step 0 — Freeze and instrument (about 1 week) — repo: kestrel-eval
- Manifest (seeds, splits, budgets, thresholds), JSONL schema, power harness (check CPU energy counters exist on this Ryzen, read GPU power through NVML, idle baseline, wall-meter calibration), tests: no future leakage, checkpoint resume, deterministic evaluation, memory write timing.
- **Pass**: same config and seed reproduce results; harness reports joules per task with a stated error against the wall meter.
- **Kill**: if counters cannot be calibrated, report relative energy only.

### Step 1 — Build the locked suite (about 1 week) — repo: kestrel-eval
- Assemble the groups above; write private and synthetic items; build stale-fact items (a fact changes over the history, the later one is the key); split and hash.
- **Pass**: no item in two partitions; hash recorded in the manifest. **Kill**: none, this must finish.

### Step 2 — Baselines and frontier reference (about 1 week) — repo: kestrel-eval
- Take 3–5 candidates from the small-model shortlist (for example the 0.8B and 2B Qwen3.5 small models, Granite 4.0 Nano 1.5B, LFM2), quantize to 4-bit, confirm each fits 1–1.5 GB VRAM with its context. Run each on validation on both tiers with plain retrieval; record accuracy, joules per correct task, resident GB, latency. Choose B-LLM by measurement. Run F once within the owner's cost cap.
- **Pass**: every cell filled, with intervals, from the same harness.
- **Kill**: if no candidate fits with usable accuracy, lower model size or move the base to the CPU tier and rewrite G-R before continuing.

### Step 3 — Verifier head (about 1–1.5 weeks) — repo: kestrel
- Generate base-model answers on the training partition of the checkable groups; label correct/wrong automatically. Features: last-token hidden state, token probabilities, agreement across sampled answers. Train a small head; calibrate on validation (temperature or isotonic). Compare with raw token probability, self-consistency agreement and stated confidence. Gaming tests: held-out task type, paraphrased and adversarial items.
- Metrics: AUROC, expected calibration error, selective accuracy.
- **Pass (proposed)**: calibration error at most 0.10 on held-out data and AUROC at least 0.03 above the best simple baseline (interval excludes zero) on the test partition.
- **Kill**: if it does not beat token probability or self-consistency, use the simpler signal and drop the word "learned".

### Step 4 — Verifier-gated halting (about 1 week) — repo: kestrel
- Policy: generate, score, return if score is at least a threshold; otherwise resample, refine, retrieve or escalate to the larger CPU-tier model, up to a pass budget; then abstain. Cloud escalation is off unless the owner turns it on. Tune the threshold on validation only. Compare with single pass, fixed two passes, fixed best-of-three, self-consistency@5.
- **Pass (proposed)**: at least 5 points over single pass at no more than 1.5x its energy, or the same accuracy as fixed best-of-three at no more than 60% of its energy.
- **Kill**: if it does not beat a fixed two-pass rule at equal compute, drop the gate and keep the fixed rule.

### Step 5 — Knowledge store on SSD (about 1–1.5 weeks) — repo: kestrel
- Index the fixed corpus: small embedding model on CPU, vectors memory-mapped from SSD, approximate or flat index. Measure knowledge-QA accuracy with and without the store, and against a larger quantized no-store model of the same resident RAM. Report accuracy per resident GB, disk size separately. Compare verifier-gated reads with always-read. Count SSD-read energy and latency; report one-time index build cost separately.
- **Pass (proposed)**: at least 10 points over the base model without the store on the store group at no more than 1.3x its energy, and better accuracy per resident GB than the larger no-store model.
- **Kill**: under 3 points of gain, or latency breaks the budget: drop it and record why.

### Step 6 — Write-gated memory (about 1 week) — repo: kestrel
- Entries: text, source, timestamp, confidence, decay or recheck rule; writes only when verified (by the verifier or the owner); entries editable and deletable; conflicts prefer newer and higher-confidence, old entry flagged not deleted. Evaluate on the memory probe including stale-fact items against: no memory, plain external memory, F with no history, F with full history.
- **Pass (proposed)**: 5 points or equivalent over plain external memory (interval excludes zero, at most 3 points forgetting) and stale-fact errors at most half of plain memory.
- **Kill**: if gating does not reduce stale errors, keep plain memory with timestamps and say so.

### Step 7 — Integrate (about 1 week) — repo: kestrel
- Combine the Step 4, 5, 6 components that passed. Router uses the verifier score to choose GPU path, CPU path or abstention. Run the full suite on both tiers; remove one component at a time and rerun; measure joules per correct task, resident GB and latency for each configuration.
- **Pass**: section gates measured on validation with at most 3 points regression versus earlier steps.
- **Kill**: any component whose removal does not hurt accuracy or energy is removed.

### Step 8 — Looped-core toy (optional, timebox 3–5 T4 sessions) — repo: kestrel (kestrel/looped_toy)
- About 100M parameters: prelude, looped block, coda. Train on synthetic checkable tasks (arithmetic, small puzzles, grid or maze) with no pretraining at scale. Compare no loops, fixed loop count, verifier-gated halting at equal parameters and equal compute. Watch training stability (loss spikes), behaviour past the trained loop depth (gains degrade there in published looped models), and halting calibration.
- **Pass (proposed)**: gated halting beats fixed depth at equal compute on held-out tasks (interval excludes zero).
- **Kill**: unstable or no gain after five sessions: stop, publish the negative result, do not scale. Does not feed the product path unless it passes.

### Step 9 — Titans-style memory module (optional, timebox 1–2 weeks) — repo: kestrel
- Only if Step 6 passed. Tiny associative memory with a surprise-and-salience write signal; compare at matched bytes against a FIFO buffer and the Step 6 store. Keep an immutable log so the module can be rebuilt (online updates can be poisoned or drift).
- **Pass**: 5 points on delayed recall at two or more distances, non-inferior task success (within 2 points), at most 3 points forgetting. **Kill**: it only wins with a larger memory budget.

### Step 10 — Final evaluation and report (about 1 week) — repo: kestrel-eval
- Freeze the configuration; run the locked test partition once for F, B-LLM and P1; paired bootstrap intervals; report the three numbers, every ablation and every failed component.
- **Decision**: if G-Q and G-R pass, plan prototype 2 (joint training of store and model, proactive event sentinel, then the latent core). If not, publish the negative result and stop or change path.

## Risks and where they are tested

| Risk | Test |
| --- | --- |
| Verifier gaming (looks confident without being right) | Step 3 gaming tests, held-out task types, calibration error |
| Loop instability | Step 8 stability monitoring and kill gate |
| Retrieval weaker than dense weights on subtle tasks | Step 5, 3-point kill line |
| Variable depth batches poorly on GPUs | Real joules per correct task at batch size one |
| Benchmark contamination | Private and synthetic items reported separately |
| Poisoned or stale memory | Step 6 provenance, decay, stale-fact probe; immutable log in Step 9 |
| Overclaiming against a frontier model | Gates measured on the slice; the report states the slice |

## Owner decisions — resolved 2026-11-17

Do not reopen these without a new, explicit owner decision.

1. **F and cost cap** — deferred to Step 2. `F.model` and `F.cost_cap_usd` stay `UNDEFINED` until then.
2. **Wall power meter** — no meter. Energy is **relative only**; every result must say so.
3. **Gates** — G-Q, G-F, G-M, G-R and no-regression **accepted as written** and frozen in `kestrel-eval/evaluation/manifest.yaml`. Never change a threshold after seeing results.
4. **Cloud escalation** — off.
5. **Real personal data** — synthetic only. Real items require item-by-item owner approval.
6. **Suite sizes** — `sizes_are: per_partition` (1200 items per partition, 3600 total). This keeps locked-test groups at n=100–300 with ±5.7–9.8 pp CIs, so the 5 pp G-Q gate is testable. A `total` reading would give n=20–60 and ±12.7–21.9 pp CIs and was rejected.
7. **Knowledge-QA corpus** — small fixed public Wikipedia snapshot; exact dump URL and SHA-256 to be pinned at Step 1.
8. **Test hardware (2026-11-18)** — owner accepts Ryzen 7 7435HS + RTX 3050 Laptop 4 GB as the available test tier because the RTX 3060 is currently unavailable. G-R thresholds are unchanged (1–1.5 GB assistant VRAM). Every result must name the tier used; a re-run on the RTX 3060 is deferred until it is available.

## Status (update as you go)

- [ ] Step 0 — freeze and instrument
- [ ] Step 1 — locked suite
- [ ] Step 2 — baselines and frontier reference
- [ ] Step 3 — verifier head
- [ ] Step 4 — verifier-gated halting
- [ ] Step 5 — knowledge store
- [ ] Step 6 — write-gated memory
- [ ] Step 7 — integrate
- [ ] Step 8 — looped-core toy (optional)
- [ ] Step 9 — learned memory module (optional)
- [ ] Step 10 — final evaluation and report

Log of results and decisions: add dated lines here, one per finished or killed step, with the numbers and the commit that produced them.

- 2026-11-17 — Owner decisions resolved and recorded above; evaluation manifest frozen (kestrel-eval commit `52f4ed0`). Step 0 implementation merged in kestrel-eval PR #2 (117 tests passing, CI green); repo-local agent instruction files removed in PR #3. Step 0 is **not** marked passed yet: the energy harness still has to be run on the target Ryzen/RTX hardware, and energy will be relative-only by owner decision.
- 2026-11-18 — Owner accepted the alternate test tier (Ryzen 7 7435HS + RTX 3050 Laptop 4 GB) because the RTX 3060 target is unavailable. G-R thresholds unchanged; every result must name the tier used, and the 3060 re-run is deferred until that hardware is available.
