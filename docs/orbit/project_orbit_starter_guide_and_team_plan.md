# Project Orbit: Starter guide and team work plan

**How to start, what to build first, how to test and track it, and how three people split the work**

> Compiled 16 November 2026 from the project discussions and web research. Week estimates are planning guesses, not measurements. Tool descriptions come from public documentation and third-party summaries; confirm details on each project's own page before relying on them.

## Contents

- [1. Overview and the five modules](#1-overview-and-the-five-modules)
- [2. Starting principles](#2-starting-principles)
- [3. Choosing the code base and setting up](#3-choosing-the-code-base-and-setting-up)
- [4. Building the five modules: order, parallel work and the freeze rule](#4-building-the-five-modules-order-parallel-work-and-the-freeze-rule)
- [5. How to test it](#5-how-to-test-it)
- [6. Git and collaboration for three people](#6-git-and-collaboration-for-three-people)
- [7. Seeing inside the model: the "brain" view](#7-seeing-inside-the-model-the-brain-view)
- [8. Analysing open-source LLMs](#8-analysing-open-source-llms)
- [9. Splitting the work into three](#9-splitting-the-work-into-three)
- [10. The first 10 days and the working rhythm](#10-the-first-10-days-and-the-working-rhythm)
- [11. Gates and kill criteria](#11-gates-and-kill-criteria)
- [12. Day-1 checklist](#12-day-1-checklist)
- [13. Caveats](#13-caveats)
- [14. References](#14-references)

---

## 1. Overview and the five modules

Project Orbit is a proposed efficient language model that separates what a model knows from how hard it thinks. This document answers the practical question of how to start: what to set up, what the code base looks like, how to test it, how three people share one repository, how to look inside the model, how to study existing open models, in what order to build the five modules, and who does what.

| Question | Where it is answered |
| :--- | :--- |
| Where do we start, and which code base? | Sections 2 and 3 |
| In what order are the five modules built, and what runs in parallel? | Section 4 |
| How do we test it? | Section 5 |
| How do three people maintain one project (Git)? | Section 6 |
| Can the model be visualised like a brain? | Section 7 |
| How do we study other open-source LLMs? | Section 8 |
| Who does what? The three-way split | Sections 9 to 11 |
| What is the first-day checklist? | Section 12 |

### 1.1 The five modules

The team defined five modules. The table keeps the team's wording and adds the technical meaning and when each is built. Note that the build order differs from the numbering: Adaptive Halting comes before the Verifier.

| Module | Team definition | Technical meaning | Built in |
| :--- | :--- | :--- | :--- |
| 1. Knowledge Store | External database for retrieving facts. | Two options: external retrieval (RAG), which is simple and independent of the core, or Engram-style lookup memory inside the model, trained with it. | Phase 1 (external, in parallel); Engram version later |
| 2. Reasoning Core | Small neural network with shared weights, executed repeatedly. | Untied prelude, shared looped hybrid block with input injection, untied coda. | Phase 1 |
| 3. Verifier | Predicts whether the current answer is correct. | Small head trained on verifiable tasks (math, code, logic). Optional for version 1. | Phase 3 |
| 4. Adaptive Halting | Decides whether to stop or continue reasoning. | Small head trained on whether one more loop lowers the loss (TaH2-style labels), not on a verifier. | Phase 2 |
| 5. Memory Module | Stores useful information from previous interactions. | Two meanings: conversation memory in the app layer (easy, any time) and test-time neural memory (Titans-style, research, last). | App layer any time; neural memory Phase 4 |

---

## 2. Starting principles

- **Start from a small open code base.** Do not write an LLM from scratch. Get a known pipeline running, then change it.
- **Build a walking skeleton first.** All five modules exist as stubs from day one, so the full pipeline runs end to end. Stubs are replaced one at a time.
- **Always keep a control.** A plain baseline model, trained with the same script, data and compute, is the reference for every Orbit claim.
- **Change one thing at a time.** If everything changes at once, a worse result cannot be explained.
- **Evidence gates.** The next module starts only when the current one passes its written test (Section 11).
- **Everything reproducible.** Every result records the commit, config, seed, data version and FLOPs.
- **Local for debugging, cloud for real runs.** A local GPU is fine for tiny models; real runs need rented or subsidised GPUs.

---

## 3. Choosing the code base and setting up

| Option | What it is | Best use | Caveat |
| :--- | :--- | :--- | :--- |
| nanochat (Karpathy) | Full small-LLM pipeline: tokenizer, pretraining, mid-training, fine-tuning, optional RL, evaluation, chat UI. About 560M parameters for roughly $100 on one 8xH100 node. MIT licence; model depth can be swept for short experiments. | Learning and first prototypes | Cost and size figures come from third-party summaries. |
| OpenLanguageModel (OLM) | PyTorch library where model code reads like the architecture, with Block, Residual, Repeat and Parallel building blocks. Single-node. | Prototyping loops (the Repeat block maps onto a looped core) | Not tried by the team; test before committing. |
| litgpt | Production-oriented library with 20+ architectures, pretraining, fine-tuning, deployment. | Stable training later | Less hackable than nanochat. |
| torchtitan | PyTorch-native pretraining with parallelism and checkpointing; described as the scale-out path in 2026. | Larger multi-GPU runs | Only needed once experiments outgrow a single node. |
| OLMo-core | PyTorch building blocks for the OLMo ecosystem from Allen AI. | Reference for fully open training code | Check current model and MoE support. |

> **Recommendation.** Prototype in `nanochat` or `OLM`. Move to `torchtitan` only if a larger run is justified by results. Reuse an existing open tokenizer at first; training a new one is a later decision.

### 3.1 Environment setup

- Python 3.11 or newer, PyTorch with the right CUDA version, and one pinned dependency lock file shared by all three people.
- Accounts: GitHub or GitLab (private repo), Hugging Face, an experiment tracker (for example Weights and Biases or TensorBoard), and cloud or subsidised compute.
- One shared cloud storage bucket for datasets and checkpoints (never in Git).
- A README that lets a new person reproduce the first tiny run in under an hour.

### 3.2 Repository layout

```text
orbit/
  README.md                   how to set up and reproduce EXP-001
  pyproject.toml              pinned dependencies
  configs/                    baseline_100m.yaml, orbit_100m.yaml
  orbit/
    model/                    baseline.py, orbit_core.py, engram.py,
                              halting.py, verifier.py, memory.py
    data/                     download, tokenize, manifest
    train/                    train.py, optim.py, checkpoint.py
    eval/                     flops_counter.py, benchmarks, code_exec_eval.py
    viz/                      loop_heatmap.py, logit_lens_loops.py
  experiments/                EXP-001.md, EXP-002.md ...
  tests/                      unit and sanity tests
  scripts/                    launch and utility scripts
```

**Key rule:** the baseline and Orbit live in the same repository and use the same training script. Only the model file differs. That keeps the comparison fair.

---

## 4. Building the five modules: order, parallel work and the freeze rule

Short answer: one after another, with some work running in parallel. Dependencies decide the order. The Reasoning Core depends on nothing and everything else needs it. Halting needs the Core. The Verifier needs the Core and verifiable data. The Knowledge Store (external version) is independent. The Memory Module is the riskiest and goes last.

```text
Main chain:
  Phase 1 - Gate 1            Phase 2 - Gate 2            Phase 3 - Gate 3         Phase 4 - later
  [Reasoning Core] ---------> [Adaptive Halting] -------> [Verifier] ------------> [Memory writes]
  (shared block, looped)      (stop or continue)          (is the answer right?)   (test-time memory)

Parallel tracks feeding the main chain:
  - Knowledge Store (external retrieval first; Engram-style lookup inside the model later)
  - Conversation memory (app layer, any time)

Rule:
  Finish and freeze the Core first. Then train Halting and Verifier as small heads on top of it,
  one at a time. Do not train all five together from day one.
```

*Figure 1. Build order.*

| Phase | Build | In parallel | Done when | Estimate |
| :--- | :--- | :--- | :--- | :--- |
| 0 (days 1 to 3) | Stubs for all five modules, baseline model, eval harness | Everyone | End-to-end run works with fake modules | 3 days |
| 1 | Reasoning Core | Knowledge Store (external), eval harness | Matches or beats the plain baseline at equal FLOPs on at least one axis, stable over 3 seeds | 3 to 6 weeks |
| 2 | Adaptive Halting | Knowledge Store (Engram version) | Better accuracy per FLOP than a fixed loop count; does not collapse to one loop | 2 to 4 weeks |
| 3 | Verifier | Conversation memory (app layer) | Well calibrated on held-out verifiable tasks | 3 to 5 weeks |
| 4 | Memory writes (neural) | None | Improves multi-session tasks without hurting the baseline | Only after phases 1 to 3 |

### 4.1 "Simultaneously" has two meanings

- **Parallel work by different people is fine.** One builds the Core, another the Knowledge Store and evaluation, a third the training pipeline.
- **Parallel training of everything together is not.** Train the Core first. Then freeze it and train Halting and Verifier as small heads on top. That is far cheaper and shows exactly what each head contributes.

### 4.2 Interface stubs (day one)

Give every module an agreed interface, a test and a place in the pipeline from the beginning:

```python
class KnowledgeStore:
    lookup = lambda self, ctx: None  # stub


class Halting:
    should_continue = lambda self, h, step: step < 2  # fixed 2 loops


class Verifier:
    score = lambda self, h: 1.0  # stub


class Memory:
    read = lambda self, ctx: None
    write = lambda self, x: None
```

### 4.3 Points where the plan and the original list differ

- **Knowledge Store.** "External database" means retrieval (RAG): quick for a product, independent of the core. Engram-style memory is a lookup table inside the model, trained with it: better for the research paper. Keep one interface so the two can be swapped.
- **Halting versus Verifier.** In the refined plan, halting is trained on whether an extra loop lowers the loss, not on a verifier. The Verifier is therefore optional for version 1.
- **Memory.** Remembering past conversations is an app-layer feature and can be built any time. Test-time neural memory is research and should wait.

> **Rule for moving on.** Do not start the next module until the current one passes its written test. If the Core does not beat the baseline, adding more modules will not fix it. Stop and rethink first.

---

## 5. How to test it

```text
Slower, costlier  ^  Matched-FLOPs comparison (hours to days; per experiment; 3 seeds; baseline vs Orbit)
                  |  Sanity runs (minutes; before merge; overfit one batch; 10M model loss falls)
Faster, cheaper   |  Unit tests (seconds; every commit; shapes, loop count, gradients, seeds)
```

*Figure 2. Three testing layers, from fast and cheap to slow and costly.*

| Layer | What to check | When |
| :--- | :--- | :--- |
| Unit tests | Output shapes are right. With one loop, the core behaves like an ordinary block. Gradients reach every parameter. Same seed gives the same result. | Every commit |
| Sanity runs | The model can overfit one small batch to near-zero loss (if not, something is broken). A 10M-parameter model trained for a few minutes shows falling loss. | Before every merge |
| Matched-FLOPs comparison | Baseline versus Orbit, same data and same training compute, at least three seeds, confidence intervals. Also measure real tokens per second on the target GPU. | Each experiment |

### 5.1 Testing code ability

Do not judge code ability by reading output. Use execution-based tests such as HumanEval or LiveCodeBench, where generated code runs against unit tests and is scored pass or fail. A model of about 100M parameters should be expected to score low; use these tests to compare Orbit with the baseline, not to chase absolute numbers.

### 5.2 Acceptance test per module

| Module | Acceptance test |
| :--- | :--- |
| Reasoning Core | Equal or better than the plain baseline at matched FLOPs, parameters or memory on at least one axis, over three seeds, with no instability. |
| Adaptive Halting | More accuracy per FLOP than a fixed loop count at equal average compute; halting does not collapse to a single loop; works across task difficulty. |
| Verifier | Predicted correctness matches actual correctness on held-out verifiable tasks (calibration curve); cannot be gamed by confident-looking wrong answers. |
| Knowledge Store | Improves factual accuracy per FLOP or per active parameter; lookup latency acceptable; fails gracefully on missing facts. |
| Memory | Improves multi-session tasks; does not hurt the baseline; stable over long sessions. |

---

## 6. Git and collaboration for three people

Use Git in a private GitHub or GitLab repository. It gives history, review and reproducibility, which matter more with three people than with one.

```text
Create branch -> Commit small -> Open pull request -> Teammate reviews -> Tests pass -> Merge to main
(feat/engram-read)  (clear msg)    (what, why, result)   (one approval)     (unit+sanity) (main stays green)

Experiments: Tag the commit (exp-001). Save config, seed and result in experiments/EXP-001.md. Log the run in the tracker.
Never commit: Data files, model checkpoints, API keys or passwords. Put them in .gitignore; store checkpoints in cloud storage or Hugging Face Hub.
```

*Figure 3. Pull-request workflow and the two things never to commit.*

- `main` is always working. Nobody pushes to it directly; protect the branch in repository settings.
- **Short branches and pull requests.** One person opens a PR; another reviews and approves before merging. Branch names like `feat/engram-readonly` or `fix/loop-gradient`.
- **Ownership by folder.** Each person owns a folder (Section 9) and the others review changes there. A `CODEOWNERS` file can automate this.
- **No data or checkpoints in Git.** Use `.gitignore`; store large files in cloud storage or Hugging Face Hub (Git LFS or DVC if versioning is needed).
- **One experiment, one file.** Each `experiments/EXP-00X.md` records the commit hash, config, seed, FLOPs and result. Tag the commit so any result can be reproduced.
- **Track runs and tasks.** Use an experiment tracker for curves and a shared task board (GitHub Issues or Projects) for work items.
- **Keep notebooks out of the main code.** Notebooks are for exploration; anything reused moves into the package.

### 6.1 Everyday commands

```bash
git switch -c feat/orbit-core-v0         # new branch
git add -A && git commit -m "Add sandwich loop core v0 + unit tests"
git push -u origin feat/orbit-core-v0    # then open a pull request
git switch main && git pull              # update after merge
git tag exp-001 && git push --tags       # mark a reproducible experiment
```

### 6.2 Experiment record template

```markdown
# EXP-001: baseline vs Orbit core v0 (30M params)
Commit: <hash>     Date: <date>     Owner: <name>
Hypothesis: sandwich loop (R=3) beats baseline at equal FLOPs
Config: configs/orbit_30m.yaml     Seeds: 1, 2, 3     Data: <manifest id>
Result: <table of accuracy, FLOPs, tokens/s, memory>
Decision: continue / change / stop     Why: <one paragraph>
```

---

## 7. Seeing inside the model: the "brain" view

Yes, it is possible and useful, with realistic expectations. Pretty 3D neuron views exist (for example a public demo that renders neuron activations in 3D), but most are illustrations. The views that actually help are simpler, and the most valuable one for Orbit is specific to loops.

| Level | What it shows | Tools | Effort |
| :--- | :--- | :--- | :--- |
| 1. Training dashboard | Loss, tokens per second, FLOPs, loop statistics over time | Weights and Biases or TensorBoard | Day 1 |
| 2. Per-token loop heatmap | How many loops each token used; text coloured by loop count | Custom (log the halting output) | 1 to 2 days; highest value for Orbit |
| 3. Logit lens per loop | Project the hidden state after each loop onto the vocabulary to watch the answer being refined | Custom; Tuned Lens for a learned version; Ecco | 2 to 4 days |
| 4. Attention and activation views | Attention patterns per head and layer | CircuitsVis, BertViz, TransformerLens, nnsight | 1 to 3 days |
| 5. Engram hit map | Which tokens triggered memory lookups | Custom | 1 day |
| 6. Circuit tracing | Which features cause an output | circuit-tracer, SAELens | Research-level; later |
| Learning aid | How a Transformer works, in the browser | Transformer Explainer | Hours |

- **Custom architecture caveat.** TransformerLens supports only models implemented in it; `nnsight` supports more. For Orbit, plain PyTorch hooks plus custom plots will do most of the work.
- **Pictures are not proof.** A heatmap suggests an explanation; only controlled experiments confirm it. Use visuals for debugging and for the paper's figures.

---

## 8. Analysing open-source LLMs

Study existing models to borrow what works and avoid repeating mistakes. For each model, fill in a one-page architecture card.

| Card field | Where to find it |
| :--- | :--- |
| Total and active parameters, layers, hidden size, heads, experts | `config.json` on the Hugging Face page |
| Attention type, normalisation, position encoding, MoE routing | Model code in the `transformers` library or the official repo |
| Context length, optimizer, precision, tokens trained, data mix | Technical report |
| Licence and usage limits | Model card |

Count parameters without downloading weights (use `trust_remote_code` only for repositories that have been inspected):

```python
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

cfg = AutoConfig.from_pretrained("ORG/MODEL", trust_remote_code=True)
with init_empty_weights():
    m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
print(sum(p.numel() for p in m.parameters()) / 1e9, "B params")
```

| Model to study | What to learn | Suggested owner |
| :--- | :--- | :--- |
| DeepSeek-V4 | Compressed and sparse attention, MoE sizes, low-precision training, efficiency numbers | Person A |
| Qwen3-Next or Kimi Linear | Hybrid linear attention (Gated DeltaNet, about 3:1 layout) | Person A |
| nanochat, OLMo-core, torchtitan | Readable training code, checkpointing, parallelism | Person B |
| OLMo and open data releases | Fully open data, tokenizer and evaluation practice | Person C |

Budget one to two days per model. Each owner presents a 15-minute summary and hands in the architecture card; cards live in a `docs/models` folder.

---

## 9. Splitting the work into three

The split follows skills and dependencies. Each person owns a set of folders and a set of modules, and each has a reviewer.

```text
Person A: Research lead                Person B: Training + infrastructure     Person C: Data, evaluation + product
- Reasoning Core                       - Baseline + training pipeline          - Data pipeline + manifest
- Adaptive Halting                     - FLOPs counter, scaling runs           - Eval harness + benchmarks
- Verifier                             - Batching + serving                    - Knowledge Store, memory
- Experiment design + paper            - Compute budget                        - Pilots, grants, credits
Owns: orbit/model/, configs/           Owns: orbit/train/, scripts/            Owns: orbit/data/, orbit/eval/, orbit/viz/

Shared:
  Weekly Monday plan and Friday review. One experiment log.
  Reviews: A reviews B, B reviews C, C reviews A.
```

*Figure 4. Roles, ownership and shared routines.*

| Field | Person A | Person B | Person C |
| :--- | :--- | :--- | :--- |
| Role | Research lead | Training and infrastructure engineer | Data, evaluation and product lead |
| Skills needed | Architecture, experiment design, scaling-law fits, paper writing | PyTorch, distributed training, GPU efficiency, serving | Data pipelines, benchmarking, statistics, customer and grant work |
| Owns (folders) | `orbit/model/`, `configs/` | `orbit/train/`, `scripts/` | `orbit/data/`, `orbit/eval/`, `orbit/viz/` |
| Owns (modules) | Reasoning Core, Adaptive Halting, Verifier | Training pipeline, scaling, batching and serving | Knowledge Store, conversation memory, verifiable data |
| Reviews PRs from | Person C | Person A | Person B |

```text
Timeline (W0 to W16; weeks are estimates):
  Person A (Research lead)  : Setup + papers [W0–2] -> Reasoning Core: design, v0, ablations [W2–7] -> Adaptive Halting [W7–11] -> Verifier [W11–16]
  Person B (Training/infra) : Setup + nanochat [W0–2] -> Baseline, pipeline, FLOPs [W2–5] -> Scaling runs, Core support [W5–9] -> Halting batching + serving [W9–13] -> Verifier runs [W13–16]
  Person C (Data/eval/prod) : Eval harness, data manifest [W0–2] -> Data pipeline, benchmarks [W2–5] -> Knowledge Store [W5–10] -> Verifiable data, calibration [W10–14] -> Conv. memory [W14–16]

  Gates: Gate 1 Core (~W7) | Gate 2 Halting (~W11) | Gate 3 Verifier (~W16)
```

*Figure 5. Sixteen-week swimlane. The first twelve weeks match the 90-day plan; the Verifier and neural memory run beyond it. Week counts are estimates.*

### 9.1 Person A: research lead

Mission: make the Reasoning Core work, then add Halting and the Verifier, and turn the results into a paper.

| Weeks | Tasks | Deliverable | Done when |
| :--- | :--- | :--- | :--- |
| 0 to 2 | Read the core papers (Huginn, looped-models ablation, TaH2, Engram). Define the Orbit v0 spec and module interfaces. | Model spec (2 pages) and interface stubs | Team agrees on the spec |
| 2 to 7 | Implement the sandwich-loop core (prelude, shared hybrid block, input injection, coda). Ablate fixed versus randomised loop depth, full versus truncated backprop, residual scaling. | `orbit_core.py`, EXP logs, ablation table | Gate 1 passed |
| 7 to 11 | Add the halting head with loss-gain labels on the frozen Core. Measure accuracy per FLOP against fixed loops. | `halting.py`, accuracy-versus-compute curve | Gate 2 passed |
| 11 to 16 | Add the Verifier on verifiable tasks. Calibration tests. Draft the preprint throughout. | `verifier.py`, calibration report, preprint draft | Gate 3 passed |

### 9.2 Person B: training and infrastructure

Mission: make experiments cheap, repeatable and fast, and make Orbit actually run efficiently on real GPUs.

| Weeks | Tasks | Deliverable | Done when |
| :--- | :--- | :--- | :--- |
| 0 to 2 | Set up environment, pinned dependencies and compute accounts. Reproduce a tiny nanochat run end to end. | Working environment and a reproducible tiny run | Everyone can reproduce EXP-000 |
| 2 to 5 | Build the baseline model and shared training script. Add checkpoint and resume, logging, and a FLOPs counter. | `train.py`, `flops_counter.py`, `baseline_100m.yaml` | Baseline trains reproducibly |
| 5 to 9 | Run size-ladder experiments (10M, 30M, 100M, 300M). Support Person A with debugging, stability and scaling-law fits. Manage the compute budget. | Scaling results, budget report | Core gate evidence ready |
| 9 to 13 | Make variable-depth loops efficient: batching, kernels, throughput tests versus the baseline. Build a simple serving endpoint. | Throughput report, serving script | Real tokens per second measured |
| 13 to 16 | Training runs for Verifier experiments; prepare a larger-run plan if gates pass. | Run reports, scale-up plan | Gate 3 evidence ready |

### 9.3 Person C: data, evaluation and product

Mission: supply clean data and trustworthy measurements, build the Knowledge Store, and connect the project to users, credits and funding.

| Weeks | Tasks | Deliverable | Done when |
| :--- | :--- | :--- | :--- |
| 0 to 2 | Evaluation harness skeleton (`lm-evaluation-harness` or `lighteval`). Data manifest listing source, licence and filters. | `eval/` skeleton, manifest v0 | Harness scores the baseline |
| 2 to 5 | Data pipeline: streaming subsets of FineWeb-Edu and others, tokenisation, deduplication, held-out sets. Build 200 to 500 own checkable questions. | `orbit/data/`, hold-out and custom test sets | Data version IDs usable in every run |
| 5 to 10 | External Knowledge Store (retrieval) behind the shared interface; then the Engram-style version with Person A. Loop heatmap and logit-lens visuals. | `engram.py` or retrieval module, viz scripts | Improves factual accuracy per FLOP |
| 10 to 14 | Verifiable data (math, code, logic) with automatic checkers; calibration test set; compute credit and grant applications; first pilot conversations. | Verifiable dataset, applications submitted | Verifier can be trained and scored |
| 14 to 16 | Conversation memory in the app layer; demo and API wrapper. | Demo, API wrapper | Memory improves a multi-session demo |

---

## 10. The first 10 days and the working rhythm

| Days | Task | Owner | Output |
| :--- | :--- | :--- | :--- |
| 1 | Create private repo, branch protection, README, issue board, shared storage | B (with all) | Repo ready |
| 1 to 2 | Everyone reproduces a tiny nanochat run on the same pinned environment | All three | Three working setups |
| 2 to 3 | Write interface stubs for the five modules and one test each | A | Walking skeleton runs end to end |
| 3 to 4 | Evaluation harness skeleton and data manifest v0 | C | Baseline can be scored |
| 3 to 5 | Baseline model in the repo, logging, FLOPs counter | B | EXP-000 baseline logged |
| 5 to 8 | Orbit v0 core: prelude, shared block with 2 to 4 fixed loops, input injection, coda, unit tests | A | `orbit_core.py` merged |
| 6 to 8 | First architecture cards (DeepSeek-V4, nanochat) presented | A, B | Two cards in `docs/models` |
| 9 to 10 | First matched-FLOPs comparison at 10 to 30M parameters, three seeds; write EXP-001 | All three | First honest comparison |

### 10.1 Weekly rhythm and decision rules

- **Monday: 30-minute plan.** Pick experiments and write the success criteria before running them.
- **Friday: 45-minute review.** Compare results with the criteria; decide continue, change or stop; update the experiment log.
- **Daily:** a short written update in the team channel (done, next, blocked).
- **Decision rights:** each owner decides inside their area, using data. Cross-area changes (interfaces, data format) need a written proposal and one approval from each affected owner.
- **Compute budget:** a monthly limit; Person B approves large runs.

### 10.2 Handoffs between people

| From | To | What | Needed by |
| :--- | :--- | :--- | :--- |
| A | B, C | Module interfaces and the model spec | Day 3 |
| B | A | Training script, FLOPs counter, compute access | Week 3 |
| C | A, B | Eval harness, held-out sets, data format | Week 3 |
| B | C | Compute for Knowledge Store training | Week 5 |
| A | C | Engram interface details and loop logging hooks (for visuals) | Week 5 |
| C | A | Verifiable dataset with checkers | Week 10 |

### 10.3 If someone is overloaded or missing

- Pause the lowest-priority parallel track first (conversation memory, circuit tracing, serving). The main chain (Core, Halting, Verifier) takes priority.
- Every critical piece is understood by at least two people: A and B both know the training script; A and C both know the eval harness; B and C both know the data pipeline.
- If a gate slips by more than two weeks, hold a rescope meeting rather than silently extending.

---

## 11. Gates and kill criteria

| Gate | Evidence required | If it fails |
| :--- | :--- | :--- |
| Gate 1: Core (about week 7) | Beats the plain baseline at equal FLOPs, parameters or memory on at least one axis; three seeds; stable; real tokens per second measured. | Run ablations for two more weeks; then publish a clean negative result and pivot to data, tokeniser or fine-tuning. |
| Gate 2: Halting (about week 11) | More accuracy per FLOP than fixed loops at equal average compute; no collapse to one loop. | Fall back to fixed loops; drop adaptive depth from the first release. |
| Gate 3: Verifier (about week 16) | Calibrated predictions on held-out verifiable tasks; no obvious gaming. | Ship without it; keep as a research item. |
| Overall | Preprint, code and baselines released; at least one pilot conversation held. | Re-plan scope, team and funding before scaling up. |

---

## 12. Day-1 checklist

| Done | Item | Owner |
| :--- | :--- | :--- |
| [ ] | Private repo created; main branch protected; all three invited | B |
| [ ] | README with setup steps; pinned dependency file | B |
| [ ] | Hugging Face and experiment-tracker accounts created | All |
| [ ] | Shared cloud storage bucket created | B |
| [ ] | Compute programs applied for (IndiaAI portal, TPU Research Cloud, NVIDIA Inception, startup credits) | C |
| [ ] | nanochat cloned and tiny run reproduced by each person | All |
| [ ] | Interface stubs and one test per module committed | A |
| [ ] | Equity, vesting and IP assignment discussed and written down (take legal advice) | All |
| [ ] | Monday and Friday meeting slots fixed | All |

---

## 13. Caveats

- Week estimates and phase lengths are planning guesses. Adjust after the first real runs.
- `nanochat` cost and size details come from third-party summaries of the project; check the repository for current numbers.
- `OLM` was found in a research abstract and has not been tested by the team.
- Interpretability and visualisation tools change quickly; confirm that a tool supports the model and library versions in use.
- The Orbit combination (Engram memory, looped MoE, loss-gain halting) is a hypothesis. No paper testing it at matched FLOPs was found, but the search was not exhaustive.
- Legal, licensing and equity matters need professional advice.

---

## 14. References

Sources used for this guide (URLs as found). Check each source directly before relying on it.

### Code bases and training

- `nanochat`: `https://github.com/karpathy/nanochat` (announcement thread: `https://github.com/karpathy/nanochat/discussions/1`); summaries at `everydev.ai/tools/nanochat` and `scriptbyai.com/nanochat-open-chatgpt-alternative`
- *OpenLanguageModel: Readable and Composable Small-Language-Model Pretraining*, `https://arxiv.org/abs/2607.16669`
- *Self-Training a Small LLM From Scratch (2026 guide)*, `https://codersera.com/blog/self-training-small-llm-complete-guide-2026/`
- `torchtitan`: `https://github.com/pytorch/torchtitan`; `OLMo-core`: `https://github.com/allenai/OLMo-core`

### Visualisation and interpretability

- *A Primer on the Inner Workings of Transformer-based Language Models*, `https://arxiv.org/pdf/2405.00208`
- *Mechanistic Interpretability guide (TransformerLens, SAELens, nnsight)*, `https://aisecurityandsafety.org/en/guides/mechanistic-interpretability/`
- *circuit-tracer: A New Library for Finding Feature Circuits*, `https://aclanthology.org/2025.blackboxnlp-1.14.pdf`
- *Transformer Explainer*, `https://dl.acm.org/doi/10.1145/3772318.3791725`
- *LLM Circuits Visualizer (3D demo)*, `https://github.com/MaxwellCalkin/llm-circuit-visualizer`
- *Advanced Interpretability Techniques for Tracing LLM Activations (logit lens)*, `https://dejan.ai/blog/advanced-interpretability-techniques-for-tracing-llm-activations/`
- `nnterp`: `https://pypi.org/project/nnterp/`

### Orbit research background (from the earlier article)

- Huginn, `arXiv:2502.05171`; *Towards Looped Models Done Right*, `https://arxiv.org/pdf/2610.23036`; TaH2, `https://arxiv.org/pdf/2609.35748`; RecurTrace, `https://arxiv.org/pdf/2609.03379`
- Engram, `https://arxiv.org/pdf/2601.07372`; Titans, `https://arxiv.org/pdf/2501.00663`; DeepSeek-V4, `https://arxiv.org/abs/2606.19348`
