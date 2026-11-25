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

1. **F and cost cap (2026-11-18)** — Gemini 2.5 Pro via the Google AI Studio free tier, cost cap **$0**. Run once on public + synthetic items only, paced across days because the free quota is unpublished; record the exact model version and date. Report G-F as the gap to a **free-tier reference**, not to the current frontier. Free-tier content may be used for training by the provider, so no real personal data is sent.
2. **Wall power meter** — no meter. Energy is **relative only**; every result must say so.
3. **Gates** — G-Q, G-F, G-M, G-R and no-regression **accepted as written** and frozen in `kestrel-eval/evaluation/manifest.yaml`. Never change a threshold after seeing results.
4. **Cloud escalation** — off.
5. **Real personal data** — synthetic only. Real items require item-by-item owner approval.
6. **Suite sizes** — `sizes_are: per_partition` (1200 items per partition, 3600 total). This keeps locked-test groups at n=100–300 with ±5.7–9.8 pp CIs, so the 5 pp G-Q gate is testable. A `total` reading would give n=20–60 and ±12.7–21.9 pp CIs and was rejected.
7. **Knowledge-QA corpus** — small fixed public Wikipedia snapshot; exact dump URL and SHA-256 to be pinned at Step 1.
8. **Test hardware (2026-11-18)** — owner accepts Ryzen 7 7435HS + RTX 3050 Laptop 4 GB as the available test tier because the RTX 3060 is currently unavailable. G-R thresholds are unchanged (1–1.5 GB assistant VRAM). Every result must name the tier used; a re-run on the RTX 3060 is deferred until it is available.
9. **Context length (2026-11-18)** — 2048 tokens, fixed for all evaluations and VRAM checks.
10. **Energy counters (2026-11-18)** — CPU energy `UNMEASURED`; GPU-only relative energy via NVML, labelled on every result.
11. **Non-inferiority margin (2026-11-18)** — 3 pp, the same as no-regression. Proving equality within 3 pp on 300 items may be inconclusive; report the interval.
12. **Step 10 F run on free tier (2026-11-18)** — owner accepts sending locked-test items to Gemini 2.5 Pro free tier, with safeguards: the locked partition is strictly one-time use and cannot be reused for new claims after exposure; send F only what is strictly needed to be scored; build a reserve test set after Step 10 that is never sent to any external API; keep exploring zero-cost / no-retention credits but do not block the run on it. Record the exposure decision in the report.
13. **D13 fictional/post-cutoff diagnostic arm (2026-11-19)** — **approved as diagnostic-only**: a separate ~100-item split for the Step 5 retrieval gate (G-Q), not mixed into the 300 scored knowledge-QA items. It does not alter the frozen suite or its hashes.
14. **D14 Suite v2.0 rebuild (2026-11-19)** — **HELD**. Do not rebuild the suite, bump to manifest v2.0, or replace locked items until the R3 artifact audit, R4 PART 4B and R7 evidence have been provided and reviewed. When released, rebuild must keep public and synthetic arms reported separately, replace the 50 exposed locked items with new seeds, keep verifier hold-out by template family or whole group, and require re-sign-off of Step 1 plus recomputation of Step 2. Gates and thresholds stay frozen (D4).

## Status (update as you go)

- [x] Step 0 — freeze and instrument
- [x] Step 1 — locked suite
- [x] Step 2 — baselines and frontier reference
- [x] Step 3 — verifier head
- [x] Step 4 — verifier-gated halting (evaluated; gate failed, keep fixed rule per kill gate)
- [x] Step 5 — knowledge store (evaluated on RTX 3050; gate PASS)
- [x] Step 6 — write-gated memory (evaluated on RTX 3050; gate PASS)
- [x] Step 7 — integrate (evaluated; section gates PASS no-regression, G-Q FAIL, keep all components)
- [ ] Step 8 — looped-core toy (optional)
- [ ] Step 9 — learned memory module (optional)
- [ ] Step 10 — final evaluation and report

Log of results and decisions: add dated lines here, one per finished or killed step, with the numbers and the commit that produced them.

- 2026-11-17 — Owner decisions resolved and recorded above; evaluation manifest frozen (kestrel-eval commit `52f4ed0`). Step 0 implementation merged in kestrel-eval PR #2 (117 tests passing, CI green); repo-local agent instruction files removed in PR #3. Step 0 is **not** marked passed yet: the energy harness still has to be run on the target Ryzen/RTX hardware, and energy will be relative-only by owner decision.
- 2026-11-18 — Owner accepted the alternate test tier (Ryzen 7 7435HS + RTX 3050 Laptop 4 GB) because the RTX 3060 target is unavailable. G-R thresholds unchanged; every result must name the tier used, and the 3060 re-run is deferred until that hardware is available.
- 2026-11-18 — F chosen as Gemini 2.5 Pro on the Google AI Studio free tier with a $0 cost cap; context length fixed at 2048 tokens; CPU energy recorded as UNMEASURED with GPU-only relative energy. Manifest updated in kestrel-eval.
- 2026-11-18 — D12 resolved: Step 10 locked-test F run proceeds on the free tier with one-time-use, minimal-send and reserve-test-set safeguards.
- 2026-11-18 — **Step 0 passed** (owner sign-off). Manifest frozen at v1.1; energy relative-only with CPU UNMEASURED and GPU NVML. Hardware pilot on accepted RTX 3050 / Ryzen 7435HS tier: formal idle baseline 10 min × 3 (1,800 s, 17,943 samples, pooled mean 6.1444 W ± 0.9472 W; counter-derived 6.1686 W, 0.39% difference); CUDA GEMM pilot mean net 610.93 J over 11.362 s, passing the 3σ noise rule. Kill gate applied per owner decision: no wall meter. Steps 0 implementation and follow-ups in kestrel-eval PRs #2 and #10 (commit `4d0946e`); closure proposal `reports/step0/CLOSURE_PROPOSAL.md`.
- 2026-11-18 — **Step 1 passed** (owner sign-off). Locked suite complete at 3,600 items (1,200 per partition: 200/200/300/100/300/100). Train and validation JSONL in repo; locked-test items held outside git with SHA-256 per group and root hash `22fff5e0…` recorded in the manifest. Leakage audit: zero item-ID overlaps and zero answer-in-prompt leaks across all 3,600 items; near-duplicate detector implemented. Kestrel-eval PR #11 (commit `01185df`), 177 tests passing, 3 skipped, CI green.
- 2026-11-19 — D13 approved as a diagnostic-only fictional/post-cutoff split for Step 5. D14 suite v2.0 rebuild held pending R3/R4/R7 evidence. Phase B output review: the agent will pre-review a sample and summarise it, but the frozen acceptance criterion still requires owner sign-off; the agent summary is advisory only.
- 2026-11-21 — D14 hold released after R3/R4/R7 evidence was supplied. **Step 1 re-signed** on Suite v2: manifest v2.2, 1,250 items per partition (3,750 total per D16), public + synthetic arms reported separately, 50 exposed locked items replaced and blacklisted with a zero-reuse test, locked root hash `15be5be…` (supersedes `22fff5e0…`). Gates unchanged. Kestrel-eval PR #12, 367 tests passing, 4 skipped, CI green. Step 2 remains open: B-LLM is `PENDING_OWNER_CONFIRMATION` per D18 and F has not been run (D20).
- 2026-11-21 — **Step 3 in progress** (Member B, not passed). Real training pipeline merged in kestrel PR #12: QwenBackend feature extraction (1536-d hidden states, token logprobs, self-consistency), numpy logistic head, temperature calibration, and a training CLI that labels against A's frozen extractors. 14 tests pass with fake backends; CI green. Not yet run on real hardware, and B-LLM is still pending per D18 — the gate (ECE ≤ 0.10, AUROC ≥ best baseline + 0.03) has not been measured.
- 2026-11-21 — Evaluation helpers merged in kestrel PR #14: paired-bootstrap AUROC intervals, baseline comparison, gate decision, calibration table and gaming checks (held-out/paraphrase/adversarial). 21 tests pass total. These do not depend on the final B-LLM choice.
- 2026-11-21 — **D18 closed** (owner confirmation, kestrel-eval PR #14, manifest v2.3). B-LLM = Qwen2.5-1.5B-Instruct (4-bit NF4); SmolLM2 and Granite disqualified for exceeding 1.5 GB VRAM, Llama-3.2-1B gated without a token, Qwen3.5-0.8B dominated. Gate baselines set: G-Q 56.88% (CI [53.42, 60.27]) at 171.55 net J/correct, target 61.88%; G-M 22% stale errors, target ≤11%; G-R 1,416 MB VRAM, 120 MB headroom. This supersedes the stale 58.38% / 120.40 J/correct figures in A's handoff docs; the manifest is authoritative. Step 3 real run handed to A via kestrel issue #15 (this environment has no GPU/torch); Step 3 remains in progress until the result is measured. F still not run (D20) and handoff docs still need correcting by A.
- 2026-11-22 — **Step 3 verifier head evaluated on RTX 3050** (real inference on B-LLM Qwen2.5-1.5B 4-bit NF4 across 800 checkable train items and 800 checkable validation items). Results recorded in reports/verifier_step3.json: AUROC = 0.7943, ECE = 0.0862 (≤ 0.10 limit, calibration_ok = true). Learned head beats token probability baseline (AUROC 0.5330) by delta = +0.2613 (95% paired bootstrap CI [0.2043, 0.3163], strictly excludes 0, beats delta_min = 0.03) and beats self-consistency baseline (AUROC 0.5965) by delta = +0.1978 (95% CI [0.1532, 0.2416]). Proposed gate PASSES. Status marked [x] Step 3.
- 2026-11-22 — **Step 4 in progress** (Member B, not passed). Gated-halting policy and baselines merged in kestrel PR #21: `gated(threshold, max_passes)` with abstention, single-pass / fixed-two-pass / best-of-three / self-consistency@5 baselines, validation-only threshold tuning under an energy cap, and a comparison harness. 28 tests pass; CI green. Real threshold tuning and gate measurement handed to A via kestrel issue #22 (no GPU/torch here). Step 4 stays open until the pass/kill rule is measured on validation.
- 2026-11-22 — **Step 4 verifier-gated halting evaluated on RTX 3050** (real inference on B-LLM Qwen2.5-1.5B 4-bit NF4 across 800 checkable validation items). Results recorded in reports/halting_step4.json: threshold tuned on validation to 0.10 under the 1.5x energy cap (471.17 J vs cap 522.70 J; single pass = 28.00% at 348.46 J/correct). Gated halting achieves 29.25% accuracy (1.41 passes/query, 7.5% abstention). Fixed two-pass achieves 31.13% at 626.96 J/correct (2.00 passes); best-of-three achieves 32.75% at 893.77 J/correct (3.00 passes); self-consistency@5 achieves 29.88% at 1632.97 J/correct (5.00 passes). Gate evaluation: Condition 1 (superiority, target ≥33.00%) FAILS at 29.25%; Condition 2 (efficiency, match best-of-three 32.75% at ≤60% energy) FAILS at 29.25%; Kill check: does not beat fixed two-pass on accuracy (29.25% vs 31.13%). Proposed gate FAILS; per PLAN.md Step 4 kill rule, drop the verifier-stopping gate and keep the fixed rule in the product path. Status marked [x] Step 4.
- 2026-11-22 — **Step 4 gate logic corrected** (kestrel PR #28): the kill check now compares accuracy only, per the plan's "at equal compute" rule; lower energy alone no longer counts as beating fixed two-pass. Extracted `step4_gate_decision` into `halting.py` with two regression tests, updated the runner, and regenerated the report's gate field from the existing measurements (thresholds unchanged, decision still FAIL, `beats_or_ties` now `false`). Real rerun with the corrected logic requested from A via kestrel issue #29.
- 2026-11-23 — **Step 4 rerun confirmed on RTX 3050** (kestrel issue #29): executed `scripts/run_step4_halting.py` with corrected `step4_gate_decision` on the 800 checkable validation items; real hardware measurements verified identical (single-pass 28.00% / 348.46 J, fixed two-pass 31.13% / 626.96 J, best-of-three 32.75% / 893.77 J, self-consistency@5 29.88% / 1632.97 J, gated 29.25% / 471.17 J at frozen threshold 0.10); `beats_or_ties` confirmed `false` and gate decision confirmed `FAIL`. 35 tests pass, repo rules clean.
- 2026-11-22 — **Step 5 in progress** (Member B, not passed). SSD knowledge store merged in kestrel PR #24: memory-mapped flat vector index on SSD, `Vectorizer` seam (dependency-free hashing default, small CPU embedding model plugs in), disk size reported separately from resident bytes, and gated-read vs always-read comparison. 5 new tests, 33 total pass; CI green. Real corpus indexing, embedding, accuracy/energy measurement and the pass/kill rule handed to A via kestrel issue #25. Step 5 stays open until measured.
- 2026-11-23 — **Step 5 knowledge store on SSD evaluated on RTX 3050** (real inference on B-LLM Qwen2.5-1.5B 4-bit NF4 across 300 validation Knowledge-QA items with all-MiniLM-L6-v2 CPU vectorizer over 6,065 corpus chunks). Results recorded in reports/knowledge_store_step5.json: store built in 78.50s (15.25 MB disk, 5.95 MB resident RAM). With-store accuracy reaches 71.67% (215/300) vs no-store 5.00% (15/300), achieving +66.67 pp gain (passes >=10 pp gate; real Wikipedia 66.00% vs 7.50%, fictional entities 83.00% vs 0.00%). Energy per correct drops from 1,951.40 J to 137.02 J (passes <=1.3x energy cap of 2,536.82 J). Accuracy per resident GB is 51.74% / GB vs 2.61% / GB for larger no-store model SmolLM2-1.7B 4-bit (passes Condition 2). SSD lookup latency mean = 174.56 ms (p50 = 173.56 ms), well within baseline generation p50 (622.9 ms); verifier-gated reads (threshold 0.10) save 15.67% bytes and 20.15% latency over always-read. Proposed gate PASSES. Status marked [x] Step 5.
- 2026-11-23 — **Step 7 in progress** (Member B, not passed). Integration router merged in kestrel PR #34: routes on verifier score (abstain / retrieve / answer / memory), wires the Step 3 verifier and Step 5 SSD store, keeps fixed two-pass per the Step 4 kill rule, and leaves an optional memory slot for Step 6. Now that Step 6 passed (gated_memory recommended), the memory slot can be filled with WriteGatedMemory. 41 tests pass; CI green. Remaining blocker: F run to close Step 2 (kestrel-eval issue #16, D20). Full-suite Step 7 run, ablations and section gates cannot be measured until F is run. Step 7 stays open until then.
- 2026-11-23 — **Step 6 write-gated memory evaluated on RTX 3050** (real inference on B-LLM Qwen2.5-1.5B 4-bit NF4 across 150 validation personal-memory probe items, including 100 stale-fact items, synthetic histories only per D3). Results recorded in reports/memory_step6.json: No-memory accuracy is 2.67% (4/150). Plain external memory accuracy is 80.00% (120/150; stale 72.00%, static 96.00%) with 28 stale-fact errors (28.00% error rate). Write-gated memory reaches 96.00% accuracy (144/150; stale 96.00%, static 96.00%) with 0 stale-fact errors, achieving a +16.00 pp gain over plain memory (95% paired bootstrap CI [10.0, 22.0] pp, strictly excludes zero). Static forgetting is 0.00 pp (passes <=3.0 pp cap). Stale-fact errors cut from 28 to 0 (ratio 0.0, well within the <=14 error cap). Energy per correct drops from 21.16 J to 17.31 J. Proposed gate PASSES; router wires in gated_memory for Step 7. Status marked [x] Step 6.
- 2026-11-24 — **D21 ratified by owner**: F is `gemini-flash-lite-latest` (gemini-3.5-flash-lite) on the Google AI Studio free tier, superseding D6 because `gemini-2.5-pro` was deprecated upstream (HTTP 404). Manifest `systems.F` updated (kestrel-eval PR #18). **Step 2 closed** with the real F run (kestrel-eval PR #17): Acc_F = 72.64% (908/1250, 95% CI [70.16, 75.12]); public 74.77%, synthetic 75.33%; memory probe full history 90.00% and no history 6.67% (+83.33 pp). 1,400 records with D11 provenance, $0 cost. F is a free-tier reference, not the current frontier; G-F must be labelled as such. Caveats for Step 7/10: F code group scored 0/100 (likely extraction failure, needs checking), and F's 72.64% is over 1,250 items while B-LLM's G-Q baseline is 56.88% on 800 checkable items — G-F needs matched scope. Step 7 full-suite run and ablations now need hardware; only the optional branches and Step 10 remain after that.
- 2026-11-25 — **F result caveats resolved** (kestrel-eval PR #20, issue #19): root cause of F code 0/100 was markdown fence stripping before `ast.parse`; fixed in shared extractor/sandbox and F code rescored live to 72/100 (72.00%, CI [63, 80]). B-LLM's 800 checkable records independently re-scored under the fixed extractor and confirmed unchanged at 455/800 = 56.88% (code 48/100). Matched scope aligned strictly with the G-Q item set (math, science_mcq, knowledge_mcq, code): F = 649/800 = 81.12% (CI [78.38, 83.75]), knowledge_qa (196/300, 65.33%) reported separately as a store group. G-F gap denominator is now 81.12 - 56.88 = **24.24 pp** (previously mis-stated as 18.37 pp using knowledge_qa). Full-suite F accuracy 78.40% (980/1250). These are the authoritative F/G-F numbers for Step 10.
- 2026-11-25 — **Step 7 integration evaluated on RTX 3050** (kestrel PR #41, issue #39). Root-cause fix: Step 3/4 runners passed raw prompts instead of canonical ChatML templates, omitted code entry points and truncated math/code at 256 tokens; a 40-item discriminating test confirmed 10% raw vs 55% canonical, and the extractor was ruled out (0 differences across 800 items). After re-running all 800 checkable items with canonical prompts: P1 integrated 65.60% full suite (820/1250), checkable 57.63% (461/800). Section gates: 0.00 pp regression vs earlier steps (PASS, no-regression sanity only — not product success). Ablations: removing verifier -2.4 pp, store -16.00 pp, memory -11.20 pp, fixed two-pass -2.00 pp; all four components kept. **Product gates (explicit in reports/integration_step7.json): G-Q FAIL** — P1 +0.75 pp vs B-LLM 56.88% with CI [-1.38, 2.88] including zero and energy 297.02 > 171.55 J/cap; **G-F** gap closed 3.09% (below the 25% strong threshold); **G-M PASS** (96.00% memory, 0 stale errors, +6.00 pp vs F full history); **G-R PASS** (1.385 GB resident, 1,246 MB NVML, 1,416 MB at 2048 context ≤ 1,536 MB cap). Overall: G-Q FAIL, G-M/G-R PASS. Store 71.67% vs F 65.33% (+6.34 pp). Prior raw-prompt results in reports/verifier_step3.json and reports/halting_step4.json are superseded on checkable accuracy by this canonical re-run and must be labelled as such in Step 10.
