# Project Orbit: Designing a more efficient large language model

**Research, architecture, efficiency estimates, data, compute, funding, team and a 90-day plan**

> Compiled 14 November 2026 from web research. Most technical evidence comes from recent preprints at small scale. Estimates are labelled as estimates. Nothing here is legal, financial or investment advice.

## Contents

- [Introduction: What is Orbit?](#introduction-what-is-orbit)
- [1. Executive summary](#1-executive-summary)
- [2. Background: why India lags and where the opening is](#2-background-why-india-lags-and-where-the-opening-is)
- [3. The efficiency toolbox used by current open models](#3-the-efficiency-toolbox-used-by-current-open-models)
- [4. Architecture: old versus new](#4-architecture-old-versus-new)
- [5. How answer efficiency has changed over time](#5-how-answer-efficiency-has-changed-over-time)
- [6. The Orbit concept](#6-the-orbit-concept)
- [7. Refined training method: what is supported and what is not](#7-refined-training-method-what-is-supported-and-what-is-not)
- [8. Risk register](#8-risk-register)
- [9. Efficiency: existing models today and Orbit in one to two years](#9-efficiency-existing-models-today-and-orbit-in-one-to-two-years)
- [10. How to compare a model and prove it is better](#10-how-to-compare-a-model-and-prove-it-is-better)
- [11. Data: where to get it, how much, and the legal basics](#11-data-where-to-get-it-how-much-and-the-legal-basics)
- [12. Compute, funding and publishing](#12-compute-funding-and-publishing)
- [13. Operating a team of three](#13-operating-a-team-of-three)
- [14. The 90-day plan and go / no-go gates](#14-the-90-day-plan-and-go--no-go-gates)
- [15. Caveats, unverified items and glossary](#15-caveats-unverified-items-and-glossary)
- [16. References](#16-references)

---

## Introduction: What is Orbit?

Orbit is a proposed design for a more efficient large language model (LLM). Rather than simply making a model bigger, it aims to extract more useful answers from the same compute, memory and money by changing how the model is built and trained. The name is a working title for a research programme, not a released product.

The central principle is to **separate what a model knows from how hard it thinks**. In most current models, facts and reasoning share one very large stack of weights, and every token pays for all of it. Orbit splits the work into parts:

- **Knowledge.** Facts live in a large, sparse memory that is read by lookup (the approach DeepSeek published as Engram), held in cheap DRAM or SSD rather than in GPU-resident weights.
- **Reasoning.** A small, shared block of layers, the core, is reused in a loop. Easy tokens take one pass; hard tokens take more.
- **Decision.** A small head decides, token by token, whether another loop is worth its cost. It is trained on whether an extra loop actually lowers the prediction loss.
- **Optional later stage.** Test-time memory that the model can update during use (Titans-style).

**Why it might matter.** Reasoning models today usually think by writing out thousands of visible tokens, which is slow and expensive. Thinking in the model's internal (latent) space, with depth that scales with difficulty, could cut that cost. Research on looped models reports parameter-efficiency gains of roughly two to three times in some settings, but many looped models lose to ordinary models when compute is matched. This article sets out what is known, what is unproven, how Orbit could be tested, and what a small team needs to try it.

---

## 1. Executive summary

This article consolidates the research behind Project Orbit, a proposal for a world-class language model with a genuinely new architecture: a check of prior work, an efficiency comparison, and a practical plan to start.

- **Not the first, and that is fine.** India already has Sarvam (30B and 105B models released in 2026) and other IndiaAI Mission models. The credible pitch is not "first" but "more efficient at something specific".
- **Every ingredient of Orbit is already published:** looped (recurrent-depth) models, looped mixture-of-experts, adaptive depth, DeepSeek's Engram conditional memory, and Titans test-time memory. The nearest prior work, RecurTrace, already pairs loop memory with learned halting.
- **The open, ownable question:** can Engram-style memory take over fact recall so that loops are spent only on reasoning? No paper testing that at matched FLOPs turned up in the research behind this article. The search was not exhaustive, so verification on Google Scholar and arXiv is advisable.
- **Honest warning:** looped models often lose to ordinary models at matched compute. The scientific task is to find out whether Orbit is an exception.
- **Efficiency:** the efficiency of closed frontier models is not publicly documented. For Orbit, the planning scenarios are no gain, about 30% cheaper per correct answer (base), or about 2x cheaper (optimistic), always measured against a same-size baseline. These are assumptions, not forecasts.
- **Moving target:** the price of a fixed capability has been falling about 10x per year (published estimates vary from about 5x to 50x), so any edge is measured against tomorrow's baseline.
- **Data is not the bottleneck.** Open corpora hold trillions of tokens (FineWeb 15T, Dolma 3T, Nemotron-CC 6.3T). Compute and filtering quality are the constraints.
- **Local training is not efficient for real runs.** Cloud credits and subsidised compute are the practical route; the local GPU is for debugging tiny models.
- A team of three can work for research-scale models (about 0.1B to 3B parameters) over 6 to 12 months, not for frontier-scale pretraining.
- **First 90 days:** audit literature, secure compute, reproduce baselines, run the Engram x loops x decider ablation, publish a preprint, then decide go or no-go.

---

## 2. Background: why India lags and where the opening is

The starting question for the project was why India has produced so few frontier models. The short version, with the evidence behind it:

| Factor | What the research found | Implication |
| :--- | :--- | :--- |
| Compute and capital | One analysis attributes 57% to 63% of the India-US AI gap to capital and compute, and at most 5% to talent. The government gave Sarvam 4,096 GPUs, about 50 times fewer than frontier training uses. | Out-design rather than out-spend. |
| Talent retention | A tracker of top-tier papers found India keeps only about 20% of its top researchers. Entry-level AI engineers earn roughly 7 to 8 times less than in the US. | Hire globally or remote; offer equity. |
| Research spend | R&D is about 0.84% of GDP; the IT industry grew on services, not core technology. | Expect thin local research ecosystem; build ties abroad. |
| Hardware | GPUs in India are NVIDIA H100, H200 and Blackwell class; none designed or made in India. | Rent cloud capacity; plan around export and supply swings. |
| Data | Indian-language data is thin compared with English, and much of it is machine-translated. | Quality filtering is an advantage. |
| Tokenizer tax | English-centric tokenizers split Indic text into many more tokens: Tamil gets about 48% and Hindi about 58% of the effective context in GPT-4o. An Indic tokenizer reported 39.5% better fertility and 44% higher inference throughput than LLaMA4. | A concrete efficiency lever if Indic languages are ever served. |

> **Takeaway.** The gap is mostly money and compute, not brains. That is why an efficiency-first architecture is a sensible bet for a small team: it attacks the constraint a small team actually faces.

---

## 3. The efficiency toolbox used by current open models

Before inventing anything, know what the best open labs already do. Each row is a lever that cut cost or memory.

| Technique | What it saves | Evidence |
| :--- | :--- | :--- |
| Mixture-of-Experts (MoE) | Compute per token: only some experts run. | DeepSeek-V4-Pro: 1.6T parameters, 49B active. V4-Flash: 284B, 13B active. |
| Low-precision training | Memory and time. | DeepSeek-V3 trained 14.8T tokens in 2.788M GPU hours (about $5.6M at $2/GPU-hour, final run only). |
| Muon optimizer | Faster, stabler convergence. | Adopted by Kimi, GLM and DeepSeek-V4. |
| Compressed + sparse attention | KV cache and attention FLOPs at long context. | V4-Pro at 1M tokens: 27% of V3.2 FLOPs and 10% of KV cache; about 2% of a standard GQA8 cache. |
| Hybrid linear attention | Memory and speed at long context. | Gated DeltaNet in roughly 3:1 hybrids in Qwen3-Next and Kimi Linear. |
| Data curriculum | Tokens needed for a given quality. | "Web early, curated late" over 10.6T tokens in one 2026 report. |
| On-policy distillation | Merges specialist skills cheaply. | DeepSeek-V4 distils domain specialists instead of one unified RL stage. |

---

## 4. Architecture: old versus new

Figure 1 shows the four eras. Figure 2 compares an old dense stack, a current MoE-plus-hybrid-attention model, and Orbit side by side.

```text
1. Dense Transformer (2017–2022)
   - Every layer unique; every parameter used for every token
   - Full attention; KV cache grows with context
   - Efficiency came mainly from bigger models + more data

2. GQA / MLA + Mixture-of-Experts (2023–2024)
   - Smaller KV cache (grouped / latent attention)
   - MoE: pay only for active parameters
   - DeepSeek-V3: FP8 training, 14.8T tokens in 2.788M GPU hours

3. Sparse + compressed attention, Muon, FP4/FP8 (2025–2026)
   - DeepSeek-V4: hybrid CSA + HCA attention, 1M context
   - At 1M tokens: 27% of FLOPs, 10% of KV cache vs V3.2
   - Muon optimizer, mHC residuals, on-policy distillation

4. Research frontier (2026+): loops, memory, linear hybrids
   - Hybrid linear/full attention (about 3:1) in Qwen3-Next, Kimi Linear
   - Looped / recurrent-depth models: reuse blocks, adaptive depth
   - Conditional memory (Engram) and test-time memory (Titans)
```

*Figure 1. How efficiency levers accumulated, 2017 to the research frontier.*

```text
OLD: Dense Transformer                TODAY: MoE + hybrid attention           PROPOSED: Orbit (hypothesis)

Output                                Output                                 Output
  ^                                     ^                                      ^
Layer 6: attention + dense FFN        Sparse/compressed attn + Router->few   Coda (untied): writes the answer
Layer 5: attention + dense FFN        Linear-style attn      + Router->few     ^
Layer 4: attention + dense FFN        Sparse/compressed attn + Router->few   Decider: another loop worth it?
Layer 3: attention + dense FFN        Linear-style attn      + Router->few     ^
Layer 2: attention + dense FFN        Sparse/compressed attn + Router->few   Shared core x R (hybrid attn + MoE, reused each loop)
Layer 1: attention + dense FFN        Linear-style attn      + Router->few     ^
  ^                                     ^                                    Prelude (untied): reads the input once
Tokens in                             Tokens in                                ^
                                                                             Tokens in + Engram memory (N-gram lookup, cheap DRAM/SSD)

Notes:
- OLD: All parameters used for every token; KV cache grows with context; depth = number of distinct layers.
- TODAY: Only a slice of experts active (49B of 1.6T in V4-Pro); KV cache ~10% of V3.2 at 1M tokens; FP8/FP4 + Muon for cheaper training.
- PROPOSED: Thinks longer only where it helps; facts live in memory, not weights; unproven combination.
```

*Figure 2. Old dense Transformer vs today's MoE hybrid vs the proposed Orbit design. Orbit is a hypothesis, not a proven design.*

### 4.1 Closed frontier models

Closed frontier models are not open source. Their developers do not publish architecture, parameter counts or serving cost, so these cannot be analysed directly. The best public reference for a frontier-class open design is DeepSeek-V4. Beating a closed model on a task does not require knowing its internals: it requires testing the model as a black box (Section 10).

```text
Single-token inference FLOPs at 1M context       KV cache size at 1M context
- DeepSeek V3.2    : 100%                        - DeepSeek V3.2    : 100%
- DeepSeek V4-Pro  :  27%                        - DeepSeek V4-Pro  :  10%
- DeepSeek V4-Flash:  10%                        - DeepSeek V4-Flash:   7%
```

*Figure 3. DeepSeek-V4 versus V3.2 at a 1M-token context (V3.2 = 100%). Source: DeepSeek-V4 technical report.*

---

## 5. How answer efficiency has changed over time

| Measure | Finding | Source / caveat |
| :--- | :--- | :--- |
| Price to reach a fixed benchmark level | Fell between 9x and 900x per year, median about 50x (Epoch AI, March 2025). Restricted to data after Jan 2024, the median was higher. | Epoch AI; depends on benchmark chosen. |
| Average price decline for a fixed performance level | About 47% per quarter since 2023, or about 13x per year. Prices fall fastest right after a level is first reached; two years later the pace roughly halves (about 4.7x per year). | Epoch AI, "The plunging price of thought". |
| Industry rule of thumb | About 10x cheaper per year for equal capability (a16z "LLMflation"). | Commentary; consistent with Epoch's range. |
| Reasoning models | Cheaper tokens, but reasoning models can use 10 to 100 times more tokens per task. | Secondary source; the effect depends on the workload. |
| Open-weight share of routed tokens | Open models passed proprietary ones in OpenRouter token share in June 2026 (about 60/40), but a token share converts to a much smaller dollar share. | OpenRouter is one marketplace, not the whole market. |

> **Why this matters for Orbit.** Cheaper tokens do not always mean cheaper answers: long visible reasoning chains cost tokens. Thinking in latent space (loops) is a plausible way to cut that, which is exactly the opening Orbit targets. But the baseline keeps getting cheaper, so Orbit must beat the baseline of the day it launches.

---

## 6. The Orbit concept

**Core principle.** Separate what the model knows from how hard it thinks. Facts live in a cheap, large memory read by lookup. Reasoning runs in a small shared core that loops a variable number of times, decided token by token.

```text
Tokens (text in) -> Embed (token vectors)
                       |---> Engram conditional memory (n-gram key -> table lookup, read-only first; stored off-GPU)
                       v
                    Prelude (untied layers, run once) ---[input injection (prelude output re-fed)]---+
                       |                                                                             |
                       v                                                                             v
             Shared recurrent core: hybrid attention (3 linear : 1 full) + MoE FFN (run R times) <--+
                       |                                                                             |
                       v                                                                             |
                    Decider (small head, trained on loss gain: does one more loop lower the loss?) --+ (loop again, up to R)
                       | (exit)
                       v
                    Coda (untied layers) -> logits

Stages:
- Stage 1: prelude + core + coda + Engram read-only
- Stage 2: add the decider (post-training)
- Stage 3+: memory writes, RL
- Later, only if stages 1-3 work: Titans-style test-time memory writes
```

*Figure 4. Orbit data flow. Stage 1 builds the stable core and read-only memory; the decider and memory writes come later.*

### 6.1 What is already published (prior-art map)

| Orbit component | Published work | Status |
| :--- | :--- | :--- |
| Prelude / loop / coda sandwich + input injection | Huginn (arXiv 2502.05171); "Towards Looped Models Done Right" (2610.23036) | Established (small scale) |
| Looped MoE and loop scaling laws | Scaling Laws for Looped MoE (2609.40316); Parcae (2604.12946); iso-depth laws (2604.21106); Mixture-of-Recursions; MoEUT | Established (one trillion-token claim) |
| Loss-gain adaptive depth | TaH2 (2609.35748): post-training, 1.7B scale | Established (small scale) |
| Halting + loop memory | RecurTrace (2609.03379) | Partly explored |
| Conditional memory (knowledge store) | DeepSeek Engram (2601.07372); Lngram (2605.24869); EngramEdit (2610.10533); tokenizer-agnostic Engram (2607.29065) | Established (27B scale) |
| Test-time memory writes | Titans (2501.00663); TNT (2511.07343) | Medium-scale evidence only |
| Engram memory + looped MoE + loss-gain decider, matched FLOPs | Not found in the research for this article | **OPEN QUESTION** |

*Note: the knowledge-store component builds directly on DeepSeek's published Engram work, which should be cited as its foundation.*

### 6.2 The testable hypothesis

Engram's authors report that conditional memory relieves a model's early layers from reconstructing static patterns, effectively deepening it for reasoning. In a looped model, that might mean fewer loops are needed on knowledge-heavy tokens, so the decider can spend loops on reasoning. That is a hypothesis to test with ablations, not an established fact.

---

## 7. Refined training method: what is supported and what is not

A refined plan was proposed for Orbit. Each claim was checked against the papers. The plan is largely grounded in real work, but it overstates the evidence in places and mixes recipes that were never tested together.

| Add-on | Verdict | Why |
| :--- | :--- | :--- |
| Sandwich layout (prelude, loop, coda) | **KEEP** | Controlled ablation at 730M parameters and 336B tokens: MATH500 rose from 20.80 to 32.80 vs full-stack looping at matched logical depth; won on 8 of 10 benchmarks; advantage transfers to MoE. |
| Input injection | **ADD** | Improves knowledge retrieval and context use. Random initial state and separate recurrent states gave mixed or negative returns. |
| DeepLoop residual scaling | **TEST** | Real, but evaluated at GPT-2 small/medium scale with Post-LN DeepNorm; a hybrid MoE like Orbit's is outside that regime. Treat as a hyperparameter. |
| Weighted LM loss at every depth | **UNVERIFIED** | This is Ouro's deep supervision. Huginn used none (truncated backprop). No evidence was found that it reduces token budgets. Ablate it; also test full vs truncated BPTT. |
| Entropy/KL on exit gate in stage 1 | **CUT** | Contradicts "no gate yet". That regulariser belongs to a learned exit gate. Use randomised depth sampling instead. |
| Fixed 2 to 4 loops | **CHANGE** | The mean recurrence used in training caps the test-time frontier; compute-optimal training raises loops and data together. |
| TaH2 lookahead depth labels | **KEEP\*** | Labels come from measured loss gain, online; adds under 3% parameters. \*But TaH2 uses loss-based labels, not answer verification. |
| Verifier head + RLVR on depth decisions | **EXPERIMENT** | Not in TaH2. A reasonable own idea, but unproven; test separately. |
| Read-only knowledge store, then Titans writes | **DEFER** | Engram read-only is supported; writable memory lacks frontier-scale evidence. |
| On-policy distillation across loop depths | **CUT for now** | No support found for loop-depth distillation. |

### 7.1 Key warnings from the literature

- **Looped models often lose at matched compute.** TaH2's authors found that post-trained looped transformers have steeper test-time scaling slopes but underperform non-looped baselines at matched test-time compute. TaH2's own gain was about 3.4 points on AIME at 1.7B.
- **Loops can forget.** One 1.7B looped model peaked at two loops on MathQA (54.7%) and fell to 47.5% at sixteen. Penalty- or confidence-based halting rules collapsed to a single loop or stopped too late.
- **Looped MoE results need careful reading.** One paper reports a looped MoE matching a roughly 2x larger non-looped MoE at trillion-token scale; a reviewer notes its "compute-matched" comparison used wall-clock time, not FLOPs, and the looped model used about 1.42x the compute.
- **Scale of evidence.** Nearly all results are at 3.5B parameters or below and mostly from preprints.
- **Hardware.** Variable depth batches poorly. One serving system reports 1.32x to 1.53x throughput over full-depth continuous batching for Ouro on one H100, showing that systems work matters.

### 7.2 Recommended stage order

| Stage | What | Gate |
| :--- | :--- | :--- |
| A | Plain hybrid MoE (control) vs sandwich-looped hybrid MoE with input injection, randomised depth, matched FLOPs, parameters and memory. | Orbit wins on at least one fair axis. |
| B | Add Engram read-only; ablate Engram on/off x loops x decider. | Memory reduces needed loops or raises accuracy per FLOP. |
| C | TaH2-style loss-gain decider as post-training; measure accuracy per doubling of test-time FLOPs. | Beats non-looped baseline at matched test-time compute. |
| D | Standard post-training: SFT then RLVR on final answers. | Task accuracy and calibration hold up. |
| E (defer) | Titans-style memory writes, verifier-gated depth with RL, loop-depth distillation. | Only after A to D succeed. |

---

## 8. Risk register

| Risk | Why it matters | Mitigation |
| :--- | :--- | :--- |
| Loops do not beat baseline at matched FLOPs | The central claim fails. | Run the control first; publish a clean negative result if so. |
| Loop instability or gate collapse | Training diverges or halting degenerates to one loop. | Randomised depth, residual scaling ablation, loss-gain labels. |
| Verifier or metric gaming | Model looks confident without being right. | Calibration tests on held-out verifiable tasks; separate experiment. |
| Memory retrieval quality | Lookup memory can lag dense weights on subtle tasks. | Mix MoE and Engram per the U-shaped allocation law; evaluate by task type. |
| Systems inefficiency | FLOP savings vanish in real throughput. | Measure tokens per second on real GPUs from stage A. |
| Moving baseline | Competitors get cheaper every year. | Pick a niche and re-benchmark every quarter. |
| Data licence issues | Common Crawl pages keep their original copyright. | Data manifest; legal review before commercial release. |
| Team and funding | Three people, limited credits. | Stage gates, kill criteria, credits first. |

---

## 9. Efficiency: existing models today and Orbit in one to two years

### 9.1 Closed frontier models today

The efficiency of closed frontier models cannot be stated as a defensible number. Developers do not publish parameter counts, active parameters, hardware or compute per answer, and a model has no instrument inside itself that measures it. What can be measured from outside is real and enough for comparison: latency, tokens generated per answer, price per million tokens (from the provider's current pricing page), and accuracy on the target tasks. Section 10 shows how to turn those into cost per correct answer.

For context, an open reference point is DeepSeek-V4-Flash, listed at about $0.14 input and $0.28 output per million tokens on its first-party API. That is the kind of price Orbit would compete against on general tasks, and it shows why a small team should choose a narrow niche.

### 9.2 Orbit after 1 to 2 years: three scenarios (planning estimates)

The metric is cost per correct answer on the target task, versus a same-size baseline trained on the same data at the same compute. The numbers below are planning assumptions anchored on the best published results, not predictions.

| Scenario | Assumption | What it means |
| :--- | :--- | :--- |
| No gain (100) | Loops fail to pay for themselves at matched FLOPs, or the gate collapses. | Publish the negative result; pivot to data, tokenizer or fine-tuning product using what was learned. |
| Base (about 70) | Modest gains: sandwich loops and Engram give roughly 30% lower cost per correct answer on reasoning tasks at about 1 to 3B active parameters. | A credible niche model and a solid paper. The planning case. |
| Optimistic (about 50) | Parameter efficiency near the best published anchor (a looped MoE matching a model about 2x larger), and the decider adds real test-time savings. | A strong open-weight efficiency story. Gains do not simply multiply; this needs proof. |

```text
A. Orbit cost per correct answer                    B. Price of a fixed capability over time
   (same-size baseline = 100; planning assumptions)    (today = 100; arithmetic from published rates)
   - No gain    : 100                                  - 4.7x / yr: 100 (Today), 21 (1 yr), 4.5 (2 yrs)
   - Base case  :  70                                  - 10x / yr : 100 (Today), 10 (1 yr), 1.0 (2 yrs)
   - Optimistic :  50                                  - 13x / yr : 100 (Today), 7.7 (1 yr), 0.6 (2 yrs)
   (Lower is better. 50 = twice as cheap per correct answer.)
```

*Figure 5. Left: Orbit scenarios (assumptions). Right: how a fixed capability gets cheaper over time at 4.7x, 10x and 13x per year (arithmetic from published rates).*

### 9.3 Realistic timeline

- **Year 1:** research validation at 0.1B to 3B parameters, preprint, released code, first pilot conversations. Cost is modest if credits are used.
- **Year 2:** if the gates pass, a 7B to 30B-total MoE (a few billion active parameters) for one vertical, hosted behind an OpenAI-compatible API. This needs real funding.
- **Expected position:** competitive with same-size open models on a narrow reasoning task at lower serving cost. Not a match for frontier models on general capability.
- Because the baseline gets cheaper each year, an Orbit advantage of 30% to 50% against today's baseline may shrink or vanish by launch unless the team keeps re-measuring and improving.

```text
Conceptual target (NOT measured results):
  Accuracy (y-axis) vs Compute per answer in FLOPs or dollars, log scale (x-axis)
  - Orbit if the idea works: curve sits above and left of the same-size baseline at matched FLOPs
  - Orbit if loops fail to pay for themselves: sits below/right of the baseline
```

*Figure 6. The picture to be produced with real data. Curves are conceptual.*

---

## 10. How to compare a model and prove it is better

Treat every competitor as a black box: same questions in, answers scored, numbers compared.

### 10.1 Two separate comparisons

| Comparison | Against | Purpose |
| :--- | :--- | :--- |
| 1. Architecture proof | A plain baseline of its own: same data, parameters and training compute; only the idea differs. | Shows the architecture works. |
| 2. Product proof | Same-size open models, and frontier models only on the chosen task. | Shows where the model stands and why someone should switch. |

### 10.2 Rules for an honest comparison

- Define "better" and the threshold before running anything (for example, 10% higher accuracy at equal cost).
- Identical conditions: same prompts, temperature, maximum length and attempts.
- Hidden test set; never tune on what is reported; prefer new or private problems to avoid contamination.
- Several seeds; report spread, not the best run.
- Compare compute, not just accuracy. Give "thinking" models the same token budget.
- Report confidence intervals (bootstrap over questions and seeds). If the interval overlaps zero, do not claim an improvement.
- For open-ended answers, use blind pairwise win rate with randomised order; never use the model under test as judge.

### 10.3 Metrics

| Metric | Why it matters |
| :--- | :--- |
| Accuracy (pass@1, exact match) | Basic quality. |
| Accuracy vs compute curve | The headline chart; shows whether extra loops or tokens help. |
| Cost per correct answer | The real efficiency number. |
| Tokens generated per answer | Shows whether the model thinks in latent space instead of long text. |
| Latency, tokens per second | User experience. |
| Memory use | Hardware cost. |
| Failure cases | Where it breaks and why. |

**Suggested tests:** GSM8K, MATH, AIME-style (reasoning); MMLU-Pro, GPQA (knowledge); HumanEval, LiveCodeBench (code); IFEval (instructions); long-context retrieval; plus 200 to 500 checkable questions of its own from the target use case. Tools: `lm-evaluation-harness` or `lighteval`; save every API response with model version, date and settings. Check each provider's terms of service before using any model outputs beyond evaluation.

### 10.4 Why people actually switch

- **Price drives routed volume.** One analysis of OpenRouter reports 84% of tokens go through non-frontier models, which deliver about 77% of frontier performance at about 2.5% of the price (secondary source; treat loosely).
- **Price alone is not enough.** OpenRouter's own study finds users choose along capability, latency, price and trust.
- **Deployment friction decides.** A survey summarised by one source found 79% of developers use open models but only about 53% of those teams reach production, versus 63% for closed models, mainly due to tooling and infrastructure.
- **A small team cannot win a general price war.** Frontier-adjacent open models are listed at around $0.14 per million input tokens.

**Switch checklist:** a measurable win on one task that the customer cares about; no worse than about 5% on everything else; an OpenAI-compatible API and open weights; a pilot on the customer's own data; and a trust reason such as privacy or self-hosting.

**Proof package:** one headline accuracy-versus-compute chart; reproducible evaluation scripts and released checkpoints; a blind side-by-side demo; a cost-per-correct-answer table; a model card listing data, limits and known failures.

---

## 11. Data: where to get it, how much, and the legal basics

Open datasets already contain trillions of tokens. Subsets can be streamed from Hugging Face rather than downloading everything.

```text
Common Crawl (raw web snapshots)
  |---> FineWeb (15T tokens, filtered, ODC-By)
  |       '---> FineWeb-Edu (1.3T, classifier-picked educational text) --\
Code + math (The Stack v2, math-focused sets) --------------------------+--> Training mix (web + code + math + reasoning + custom filters)
Dolma (3T tokens, mixed sources, ODC-By) ------------------------------/
Nemotron-CC / Zyda-2 (6.3T / 5T tokens, dedup + quality filters) -----/

Keep a manifest of every source, licence and filter applied.
```

*Figure 7. A practical open-data pipeline. Filtering and mixing are where a small team can add value.*

| Dataset | Size | Licence / note |
| :--- | :--- | :--- |
| FineWeb | 15T tokens from 96 Common Crawl snapshots | ODC-By. |
| FineWeb-Edu | 1.3T tokens; classifier-selected educational text | ODC-By; small models trained on it did notably better on knowledge tasks. |
| DCLM | Model-filtered web subset; size varies by version | Alongside FineWeb-Edu, among the best open sets for small models in one survey. |
| Dolma | 3T tokens: web, papers, code, books, encyclopedic text | ODC-By; documented provenance. |
| Nemotron-CC | 6.3T tokens, about 30% synthetic rephrasings | Check the licence on the dataset card. |
| Zyda-2 | About 5T tokens, cross-deduplicated, model-filtered | Check the card. |
| The Stack v2 (code) | Large code corpus | Usable only with per-document licences and opt-outs respected. |
| FineWeb2 (multilingual) | Over a thousand languages | Check the card. |

- **How much is needed.** The sandwich-loop ablation used 336B tokens at 730M parameters. Stage-1 experiments need tens to a few hundred billion tokens, not trillions.
- **Cost precedent.** A paper titled *"JetMoE: Reaching Llama2 Performance with 0.1M Dollars"* shows small budgets can reach competitive small models.
- **Legal care (not legal advice).** ODC-By covers the compilation; Common Crawl pages keep their original copyright. Keep a manifest of every source, licence and filter, and take legal advice before commercial release.
- **Where value can be added:** better quality classifiers, more math, code and verifiable reasoning data, deduplication and careful mixing. Scraping the web is not a good place to start.

### 11.1 Local versus cloud

Local training is not efficient for real runs because of memory limits, throughput and the lack of fast multi-GPU communication. This is an engineering judgement rather than a cited finding. The local machine is best used for debugging and models of about 10M to 100M parameters. Cloud GPUs are for real runs, with checkpoints saved often so runs can be stopped and resumed cheaply.

---

## 12. Compute, funding and publishing

### 12.1 Compute sources (verify each on its official page)

| Source | What it offers | Notes |
| :--- | :--- | :--- |
| IndiaAI Compute Portal | Subsidised GPUs from empanelled providers, reported at about Rs 65 per GPU-hour with monthly application windows (another listing says up to 40% off from about Rs 45). | Rates conflict across sources. Startups: DPIIT recognition plus either Rs 50 lakh revenue or Rs 1 crore funding; researchers and students at recognised institutions can also register. Not a cash grant. |
| Google TPU Research Cloud | Free Cloud TPU capacity for accepted researchers. | Results must be shared openly: fits a publish-first plan. |
| NVIDIA Inception | Free membership, no equity; gateway to partner credits. | One source notes the widely repeated "$100,000" credit figure is not on NVIDIA's own page. |
| Google for Startups Cloud | Up to $350,000 for AI-first startups (Google's own listing). | Tiers depend on stage. |
| AWS Activate, Microsoft Founders Hub | Credits reported up to $200,000 and $150,000. | Secondary sources; confirm. |
| Lambda Research Grant, Kaggle | Up to $5,000 credits; about 30 GPU hours + 20 TPU hours per week free. | Good for early experiments. |

### 12.2 How to save compute

- Train a size ladder (100M, 300M, 1B) and fit scaling laws instead of training big.
- Start from public code: TaH2, DeepLoop and Engram all have GitHub repositories.
- Always compare against a plain hybrid MoE at matched FLOPs and memory, with several seeds.
- Use checkpoint-and-resume to take advantage of cheaper interruptible instances.

### 12.3 Funding path

Order of operations: credits first; then a small-scale result and a preprint; then incorporate and approach investors. In this field, investors fund evidence. Startups aiming at the IndiaAI route should register for DPIIT recognition early.

### 12.4 Publishing path

- Post a preprint on arXiv first. ICLR allows papers that are on arXiv. arXiv's current endorsement rules for first-time authors were not verified; check early.
- ICLR 2027 main track is closed (full-paper deadline was in September 2026).
- ICLR 2027 workshop contributions: suggested deadline 1 February 2027 (official).
- ICML 2027: one aggregator lists about 22 January 2027 (abstract 16 January); that is tight for new experiments. NeurIPS 2027 is expected around May 2027. Confirm on official pages.
- A clean negative result with strong baselines is still publishable and protects credibility.

---

## 13. Operating a team of three

Short answer: yes, for research and a first product at 0.1B to 3B parameters; no, for frontier-scale pretraining. This is a judgement from the scale involved, not a cited study.

```text
Research lead                        Training + infra engineer              Data, eval + product lead
- Architecture + hypotheses          - Training code, distributed runs      - Data pipeline + filters
- Ablations, scaling-law fits        - GPU efficiency, serving              - Benchmarks, pilots
- The paper                          - Variable-depth batching              - Credits, grants, customers

Shared weekly loop:
  Mon: choose experiments. Fri: review vs pre-written success criteria.
  One experiment log. Monthly compute budget with one approver.
```

*Figure 8. Suggested roles and the shared operating loop.*

### 13.1 Operating rules

- **One owner per area.** The owner decides, using data.
- **Weekly rhythm.** Monday choose experiments; Friday review against success criteria written in advance.
- **Experiment log.** Every run records config, data version, seed, FLOPs and result in one shared place.
- **Kill criteria.** For example: if the model does not beat the baseline at matched FLOPs by week 10, change direction.
- **Compute budget.** A monthly spend limit and one person who approves large runs.
- **No single points of failure.** Data pipeline, training code and eval scripts are each understood by at least two people.
- **Write the paper early** so missing experiments are obvious.

### 13.2 Legal basics (take legal advice)

Agree equity split and vesting, IP assignment so code and models belong to the company, and written ownership of each component. Disputes here sink many small teams.

### 13.3 Where three people struggle

Large-scale pretraining, safety and red-team work, and customer support. Plan to partner, hire later, or narrow scope.

---

## 14. The 90-day plan and go / no-go gates

```text
Timeline: W0 ---- W2 ---- W4 ---- W6 ---- W8 ---- W10 ---- W12
  [W0–W2 ] Literature audit, accounts, compute
  [W2–W6 ] Reproduce baselines (100–300M)
  [W6–W10] Ablations: Engram x loops x decider
  [W10–12] Preprint + code release
  [W12   ] Go / no-go on scale-up
```

*Figure 9. Twelve-week plan.*

| Weeks | Do | Gate to continue |
| :--- | :--- | :--- |
| 0 to 2 | Full Scholar/arXiv audit; register for compute programs; DPIIT recognition; reading list of about 10 core papers. | Reading list done and compute secured. |
| 2 to 6 | Reproduce hybrid MoE vs sandwich loop at 100M to 300M, using public code. | Baselines match published trends. |
| 6 to 10 | Engram on/off x loops x decider, matched FLOPs, multiple seeds. | A win on tokens per FLOP, accuracy or memory. |
| 10 to 12 | Preprint and code release; decide on scale-up to 1B to 3B. | Go / no-go review against pre-written criteria. |

> If the memory-substitutes-loops hypothesis fails cleanly, publish that anyway. A clear negative result with solid baselines is still a paper, and it keeps credibility intact for the next idea.

---

## 15. Caveats, unverified items and glossary

### 15.1 What could not be verified

- Whether anyone has published the exact Engram + looped MoE + loss-gain decider combination. Google Scholar could not be queried directly.
- Architecture, size and serving cost of closed frontier models: not public.
- arXiv's current endorsement policy; exact funding and credit amounts (these change; some sources conflict); IndiaAI portal rates.
- The three efficiency scenarios are planning estimates, anchored on published results but not measured on Orbit.
- Several statistics come from secondary sources (OpenRouter analyses, survey summaries, aggregator sites). Treat them as indicative.
- Most technical papers are recent preprints and have not been independently replicated.

### 15.2 Glossary

| Term | Meaning |
| :--- | :--- |
| LLM | Large language model: a neural network trained to predict the next token. |
| MoE | Mixture-of-Experts: only some sub-networks run per token. |
| KV cache | Memory of past tokens kept during generation; grows with context length. |
| Looped / recurrent-depth model | Reuses the same block several times, trading extra compute for depth without extra parameters. |
| Prelude / coda | Untied layers before and after the looped block. |
| Engram | DeepSeek's conditional memory: a table lookup keyed by recent tokens. |
| Titans | A design with memory that is updated at test time. |
| FLOPs | Floating-point operations; a measure of compute. |
| RLVR | Reinforcement learning with verifiable rewards (answers checked automatically). |
| BPTT | Backpropagation through time; truncated BPTT limits how far gradients flow. |
| Matched FLOPs | Comparing models that use the same amount of compute, so differences come from design, not budget. |
| Contamination | Test questions leaking into training data, inflating scores. |

---

## 16. References

Papers and pages consulted in this research (titles paraphrased; URLs as found). Check each source directly before citing it.

### Looped / recurrent-depth models

- Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn)*, `arXiv:2502.05171`
- *Towards Looped Models Done Right, Part I* (topology, input injection, recurrent state), `https://arxiv.org/pdf/2610.23036`
- *DeepLoop: Depth Scaling for Looped Transformers*, `https://arxiv.org/pdf/2607.13491`
- *Improving Test-Time Scaling with Adaptive Looped Transformers (TaH2)*, `https://arxiv.org/pdf/2609.35748`; code: `https://github.com/Abtaha/TaH`
- *RecurTrace: Adaptive Latent Reasoning with Loop-Time Memory*, `https://arxiv.org/pdf/2609.03379`
- *Scaling Laws for Looped Mixture of Experts*, `https://arxiv.org/pdf/2609.40316`
- *Parcae: Scaling Laws For Stable Looped Language Models*, `https://arxiv.org/pdf/2604.12946`
- *How Much Is One Recurrence Worth? Iso-Depth Scaling Laws*, `https://arxiv.org/pdf/2604.21106`
- *Loop the Loopies!*, `https://arxiv.org/pdf/2607.16051`
- *Looped World Models* (background on the looped lineage), `https://arxiv.org/pdf/2606.18208`

### Memory

- *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs (Engram)*, `https://arxiv.org/pdf/2601.07372`; code: `https://github.com/deepseek-ai/Engram`
- *Lngram: N-gram Conditional Memory in Latent Space*, `https://arxiv.org/pdf/2605.24869`
- *EngramEdit: Decoupled Knowledge Updates in LLMs through Conditional Memory*, `https://arxiv.org/pdf/2610.10533`
- *Tokenizer-Agnostic Engram Module*, `https://arxiv.org/pdf/2607.29065`
- *Titans: Learning to Memorize at Test Time*, `https://arxiv.org/pdf/2501.00663`
- *TNT: Improving Chunkwise Training for Test-Time Memorization*, `https://arxiv.org/pdf/2511.07343`

### Open model architectures and efficiency

- *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*, `https://arxiv.org/abs/2606.19348`
- *DeepSeek-V3 Technical Report*, `https://arxiv.org/pdf/2412.19437`
- Epoch AI, *LLM inference prices have fallen rapidly but unequally across tasks*, `https://epoch.ai/data-insights/llm-inference-price-trends`
- Epoch AI, *The plunging price of thought*, `https://epoch.ai/publications/the-plunging-price-of-thought`
- OpenRouter, *State of AI: 100T Token LLM Usage Study*, `https://openrouter.ai/state-of-ai`
- OpenRouter, *The Open Weight Models that Matter: June 2026*, `https://openrouter.ai/blog/insights/the-open-weight-models-that-matter-june-2026/`

### Data

- *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*, NeurIPS 2024 Datasets and Benchmarks
- *GneissWeb* (covers Nemotron-CC and Zyda-2 sizes), `https://arxiv.org/pdf/2502.14907`
- *Small Language Models: Survey, Measurements, and Insights*, `https://arxiv.org/pdf/2409.15790`
- *JetMoE: Reaching Llama2 Performance with 0.1M Dollars*, `https://arxiv.org/pdf/2404.07413`

### India context

- *Sarvam AI (overview)*, `https://en.wikipedia.org/wiki/Sarvam_AI`
- *Talent-Rich, Compute-Poor: A Layer Decomposition of India's Position in the Global AI Race*, `https://www.researchsquare.com/article/rs-11292693/v1`
- *Measuring the Tokenization Premium*, `https://arxiv.org/pdf/2608.09046`; *IndicSuperTokenizer*, `https://arxiv.org/pdf/2511.03237`
- *IndiaAI compute portal explainer*, `https://productgrowth.in/insights/ai-ml/indiaai-compute-portal-startup-gpu-access/`

### Funding and publishing

- Free GPU cloud credits (2026) roundups: `spheron.network`, `thegrantindex.com`, `aquanode.io`
- ICLR 2027 author guidelines and workshop guidelines, `https://iclr.cc/Conferences/2027/`
