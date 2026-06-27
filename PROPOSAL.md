# outten/lmm — Large Math Model Proposal

A proposal for building a specialized Ollama-native model (outten/lmm) that
excels at mathematics — from elementary arithmetic through research-level
problems. Unlike general-purpose LLMs that treat math as one capability among
many, outten/lmm treats mathematics as its primary domain.

This is an early-stage proposal. Several decisions need your input (see
§11 Open Decisions) before training begins.

---

## 1. Goals

Primary
  - Solve arithmetic, algebra, calculus, linear algebra, statistics,
    number theory, combinatorics, and competition problems reliably
  - Show step-by-step reasoning that is verifiable
  - Use computational tools (sympy, numerical Python) when helpful

Secondary
  - Generate proofs and explain math concepts clearly
  - Self-correct when intermediate steps are wrong
  - Produce executable, reproducible code for every math response
    (see §13 — Code-First Doctrine)

Non-goals (initially)
  - Multi-modal math (handwritten equations, images)
  - Broad natural-language capability
  - Multi-language support (English first)

---

## 2. Scope of Mathematical Capability

  Tier  Subject                          Target accuracy
  ----  -------------------------------  ---------------
   1    Arithmetic (+, -, ×, ÷)             99%+
   2    Pre-algebra (word problems)         95%+
   3    Algebra (linear, quadratic, sys)     90%+
   4    Geometry (plane, coordinate)        85%+
   5    Pre-calculus (trig, logs, seq)      85%+
   6    Calculus (limits, deriv, integ)     80%+
   7    Linear algebra (matrices, eigen)    80%+
   8    Statistics & probability            80%+
   9    Number theory                      75%+
  10    Combinatorics & graph theory        75%+
  11    Competition (AMC/AIME/Olympiad)     50%+
  12    Research-level (open problems)      best-effort

The accuracy targets are approximate and will be calibrated against
public benchmarks (MATH, GSM8K, AIME).

---

## 3. Base Model Selection

Three strategies, each with tradeoffs.

Option A — Start math-specialized
  Base: Qwen2.5-Math-7B-Instruct or DeepSeek-Math-7B
  Pro:  Already math-trained. Less work, higher baseline.
  Con:  Smaller (7B); may need 14B+ for hardest problems.

Option B — Start strong general
  Base: Llama 3.3 70B or Qwen2.5-72B
  Pro:  Strong reasoning foundation, higher capability ceiling.
  Con:  Larger; needs more compute and data to specialize.

Option C — Distill from frontier
  Base: DeepSeek-R1-Distill-Qwen-7B (or similar reasoning-distilled)
  Pro:  Inherits reasoning patterns from a frontier model.
  Con:  Reasoning style may be opinionated and hard to override.

Recommended path:
  - v0.1 uses Option A for fast iteration
  - v1.0 targets Option B for the capability ceiling
  - Option C is explored in parallel as a comparison

The current Modelfile uses llama3.2 (general, 3B) — too small to be a
viable LMM. The first decision is which base to switch to.

---

## 4. Training Data Strategy

Mix of curated and synthetic datasets.

Public datasets to leverage
  - NuminaMath-CoT         — 860K competition-style w/ step solutions
  - MetaMathQA             — 395K augmented GSM8K/MATH problems
  - OpenMathInstruct-2     — 14M problems w/ verified solutions
  - MATH (Hendrycks)       — 12.5K competition (gold eval set)
  - GSM8K                  — 8.5K grade school (standard benchmark)
  - PRM800K                — process supervision, step-level rewards
  - Orca-Math              — 200K word problems

Synthetic data generation
  - Programmatic variation of seed problems
  - Verifier-based filtering (sympy equivalence, numerical checks)
  - Rejection sampling: solve with model, keep only correct traces

Critical insight
  The biggest gains in math reasoning (per recent research) come from
  PROCESS-LEVEL supervision, not just final-answer accuracy. Datasets
  like PRM800K and tool-integrated reasoning traces should be weighted
  heavily.

Anti-contamination
  Held-out test set must never appear in training. MATH, GSM8K,
  and AIME test splits are explicitly excluded from any SFT mix.

---

## 5. Fine-tuning Methodology

Three training phases, in order. Phase 3 (tool/code execution) is no
longer optional — it is the defining capability per §13 (Code-First
Doctrine). All three phases incorporate code-output training data.

Phase 1 — Supervised Fine-Tuning (SFT)
  Format (Code-First per §13):
    ### Question
    <problem text>
    ### Approach
    <brief method, 1-3 sentences>
    ### Code
    <python block — numpy/pandas/sympy/matplotlib as appropriate>
    <optional mermaid block for diagrams>
    ### Output
    <result of running the code>
  Method:   LoRA or QLoRA for efficiency; full FT if compute allows
  Tool:     Unsloth (2-5× faster, lower memory) or LLaMA-Factory
  Dataset bias: code-with-solution pairs, not text-only answers

Phase 2 — Preference Optimization
  Generate K candidate responses per problem
  Score with:
    - sympy-based equivalence check on the code's output
    - execution success (does the code run without error?)
    - process reward model (if PRM800K-style data available)
  Train with DPO (Direct Preference Optimization) on (correct, incorrect)
  pairs, or GRPO (Group Relative Policy Optimization) à la DeepSeek-R1.
  A correct response is one whose code runs AND produces the expected
  output — not merely one whose prose "looks right."

Phase 3 — Code Execution (core, not optional)
  Train model to emit ```python``` and ```mermaid``` blocks
  A companion runtime (Ollama + sandbox service) executes Python via
  subprocess or Pyodide, and renders Mermaid via mermaid-cli (mmdc)
  Output is injected back into the model's context so it can verify
  and iterate
  Effectively gives the model a calculator, a chart engine, and a
  symbolic algebra system it can trust — and the user gets the same
  code to run anytime.

Initial hyperparameters
  SFT lr:        2e-5
  DPO lr:        5e-7
  Epochs:        2-3 (SFT), 1-2 (DPO)
  Sequence len:  4096 (truncate longer proofs)
  Batch size:    as large as fits w/ grad accumulation

---

## 6. Evaluation & Benchmarking

Standard benchmarks
  - MATH (5000 problems, 7 subjects)
  - GSM8K (grade school)
  - AIME (recent years)
  - AMC 10/12 (recent years)
  - CollegeMath (domain-specific)
  - OlympiadBench

Custom internal eval
  - Held-out set: 50-200 problems spanning all 12 tiers
  - Adversarial cases: numerical traps, ambiguous wording, common mistakes
  - Step-level manual annotation on a subset

Quality metrics
  - Code-correctness rate     (does the code execute without errors?)
  - Output-equivalence rate   (does the code produce the expected result?)
  - Final-answer accuracy     (verified by running the code, not by string match)
  - Step-level correctness    (on annotated subset)
  - Hallucination rate        (undefined symbols, made-up theorems, magic numbers)

Eval cadence
  - After every checkpoint
  - Side-by-side comparisons across runs
  - Watch for regression on simpler tiers when optimizing for harder ones

---

## 7. Build & Deployment Pipeline

  [Base HF model]
      │ fine-tune (Unsloth / LLaMA-Factory)
      ▼
  [HF adapter or merged model]
      │ convert to GGUF (llama.cpp)
      ▼
  [GGUF file]
      │ Modelfile + ollama create
      ▼
  [outten/lmm:tag]
      │ ollama push
      ▼
  [ollama run outten/lmm]
      │ companion runtime extracts code blocks
      ▼
  [sandboxed Python (subprocess/Pyodide)]
      │ mermaid-cli (mmdc)
      ▼
  [executed output → rendered charts → user]

Make targets (proposed)
  make train        # fine-tuning
  make convert      # HF → GGUF
  make build        # Ollama Modelfile build
  make eval         # benchmark suite (code-verified)
  make run          # ollama run + companion runtime
  make sandbox-test # smoke-test the Python/Mermaid sandbox
  make push         # publish to registry

---

## 8. Infrastructure Requirements

Compute
  Training
    7B  QLoRA :  1× A100/H100 (80GB) suffices
    70B QLoRA :  4× A100/H100 minimum
  Storage
    ~200 GB for datasets
    ~50 GB per model checkpoint
  Inference testing
    Consumer GPU or CPU via Ollama

Software stack
  Python 3.11+
  unsloth, transformers, peft, trl, accelerate
  llama.cpp (GGUF conversion)
  ollama (local deployment & registry)
  sympy, numpy, pandas, scipy (verification + the code the model writes)
  matplotlib, plotly (chart rendering inside generated code)
  mermaid-cli (mmdc) (diagram rendering)
  Pyodide or Docker-based sandbox (code execution runtime)
  pytest (eval harness)

---

## 9. Milestones

v0.1 — Foundation (2-4 weeks)
  - Pick base model
  - SFT on code-with-solution pairs (MetaMathQA, OpenMathInstruct,
    plus new (question → code → output) conversions)
  - Train in the §13 response format from day one
  - Convert to GGUF, build Ollama Modelfile
  - Eval on GSM8K + MATH (code-verified)
  Deliverable: outten/lmm:v0.1 — code-first math through algebra

v0.2 — Code execution at scale (4-8 weeks)
  - Add competition data (NuminaMath) with code-first format
  - DPO scored on code execution + output equivalence
  - Companion runtime: sandboxed Python + mermaid-cli renderer
  - Eval suite extended: "produce code that does X" tasks
  Deliverable: outten/lmm:v0.2 — code-first calculus + competition-viable

v0.3 — Process supervision + scale (8-12 weeks)
  - Integrate PRM800K-style process rewards (verifiable via code)
  - Scale to 14B or 72B base
  - Self-improvement loop (STaR) with verifier as judge
  Deliverable: outten/lmm:v0.3 — Olympiad-competitive code-first math

v1.0 — Research-grade (12+ weeks)
  - Formal verification hooks (Lean/Coq?) for proof outputs
  - Public eval suite release (code-verified, reproducible)
  - Reproducible training pipeline
  Deliverable: outten/lmm:v1.0

---

## 10. Risks & Mitigations

  Risk                                       Mitigation
  ----------------------------------------   --------------------------------
  Catastrophic forgetting of general         Mix ~10-20% general instruction
  capability                                  data into training mix
  Hallucination of fake theorems             Process supervision penalizes
                                             unfounded claims; code-first
                                             format forces verifiable output
  Compute limits block scaling               QLoRA + smaller models first;
                                             scale up only after methodology
                                             proven
  Distribution cost if public release        Quantized GGUF (Q4_K_M); doc
                                             the build process for users
  Evaluation gaming on benchmarks            Held-out test set never in
                                             training mix; answers verified
                                             by code execution, not regex
  Base-model license forbids redistribution  Audit before training; prefer
                                             Apache 2.0 / MIT bases
  Sandbox escape / malicious generated      Subprocess sandbox with no
  code damaging host                          network, restricted FS, timeout,
                                             resource caps; consider Docker
                                             or gVisor for stronger isolation
  User trusts model output without running   Default to "code must be re-run"
  the code themselves                         messaging; show output of the
                                             executed code in the response

---

## 11. Open Decisions

These need your call before training starts.

  1. Base model
     Math-specialized (Qwen2.5-Math-7B), general (Llama 3.3 70B),
     or reasoning-distilled (DeepSeek-R1-Distill)?
     Recommendation: Qwen2.5-Math-7B for v0.1.

  2. Compute access
     Local GPU(s), cloud (which provider: Lambda Labs, RunPod, AWS),
     or rented H100s?
     Determines realistic model size and iteration speed.

  3. Code-first from day 1 — SETTLED
     outten/lmm follows the §13 Code-First Doctrine: every response
     contains code (Python and/or Mermaid) that the user can re-run.
     Training, evaluation, and deployment all assume this format.

  4. Sandbox technology
     Pyodide (browser-style, in-process), Docker subprocess
     (heavier isolation), or another approach?
     Affects security model and which Python packages are available.

  5. Distribution
     Push to public Ollama registry (ollama.com/outten/lmm) or
     keep private?

  6. Repository layout
     Where do datasets, training scripts, eval harness live?
     Monorepo here, or separate repos for each phase?

  7. Versioning convention
     Tag scheme: v0.1 / v0.2 / v0.3 / v1.0, or date-based
     (2026.06.27), or content-hash?

  8. Evaluation philosophy
     Optimize for benchmark scores, or for "useful on real problems
     a human would ask"?
     Recommendation: both — benchmarks for tracking, human-curated
     set for ground truth.

---

## 12. Immediate Next Steps (Once Decisions Are Made)

  1. Initialize the repo: training/, eval/, data/, scripts/   ← DONE
  2. Pull and quantize the chosen base model                 ← DEFERRED
  3. Download and curate training data mix                  ← DEFERRED
  4. Write the SFT training script (Unsloth + LoRA)         ← DONE (scaffolding)
  5. Build a held-out eval set (MATH test split + custom)   ← DONE (smoke set)
  6. Establish the eval harness (pytest, sympy verifier)    ← DONE
  7. First training run → first Ollama build → first bench  ← DEFERRED

Estimated time from decisions → v0.1: 2-4 weeks with adequate compute.

See STATUS.md for an item-by-item breakdown of what the repo
currently contains versus what's still proposal-only, and README.md
for how to run the runtime, eval harness, and training pipeline.

---

## 13. Code-First Doctrine

The defining design principle of outten/lmm.

Core principle
  Computers compute. LLMs hallucinate. For any non-trivial math response,
  the LMM writes code that performs the math — and the math comes from
  deterministic code execution, not from the model itself.

What this means in practice
  - Every numeric answer is produced by running code, not by sampling
    tokens
  - Every chart, plot, or visualization is generated by code
    (Mermaid for diagrams, matplotlib/plotly for data charts)
  - The user receives:
      1. The code itself (so it can be re-run anytime, with no tokens
         and no model)
      2. The output of running that code
      3. A short narrative explaining what's happening
  - Comparisons (data source A vs data source B) are handled by writing
    code that loads both, computes the correlation/statistics, and
    produces side-by-side visualizations

Response shape (target format)
    ### Question
    <user's question>

    ### Approach
    <brief explanation of method, ~1-3 sentences>

    ### Code
    ```python
    # Python: numerical / symbolic computation, charts, file I/O
    ```

    ```mermaid
    flowchart / sequence / class / state diagram
    ```

    ### Output
    <result of executing the code — text, table, chart path>

    ### Re-run anytime
    The code above is self-contained. Save it to a file and run it
    directly — no LLM, no tokens, deterministic output.

Why this is the right architecture
  - Correctness   Code is verifiable. A sympy expression that returns
                  42 either evaluates to 42 or it doesn't.
  - Reproducibility The same code with the same inputs produces the
                  same output forever.
  - Cheap to re-run Once the LMM has produced the code, you don't
                  need the LMM to run it again. No tokens, no latency,
                  no API cost.
  - Auditability  Users can inspect exactly what math was done.
  - Combines well with tool use (Phase 3) — the model literally writes
                  the code that gets executed.

Implications for training
  - The SFT corpus must be heavy on (problem → solution-with-code)
    pairs, not just (problem → answer) pairs
  - Code blocks must be syntactically valid Python / valid Mermaid
  - Style: idiomatic pandas/numpy/matplotlib for data; sympy for
    symbolic; mermaid for diagrams
  - The model should default to code even when a one-line answer would
    suffice, so the answer is always verifiable

Example pair the model is trained to produce
  Question:  "Correlate daily close prices of SPY and QQQ for 2025."
  Approach:  Load both series, align dates, compute Pearson + Spearman
             correlation, plot both price series and a scatter.
  Code:     yfinance (or csv) load → pandas align → scipy.stats.pearsonr
             → matplotlib.pyplot.subplots(2,1) → savefig
  Mermaid:  flow showing data sources → alignment → stats → chart
  Output:   correlation matrix, p-values, chart.png

Implications for evaluation (§6)
  - "Final-answer accuracy" must be code-verified: execute the code,
    check the result, not just string-match the model's answer
  - New metric: code-correctness rate (does it run without errors?)
  - New metric: output-equivalence rate (does the code produce the
    expected result?)
  - The held-out test set must include "produce code that does X" tasks,
    not only "what is X" tasks

Implications for deployment (§7)
  - Ollama serves the model; a small companion service (or user-side
    script) executes the code blocks in a sandbox and feeds results
    back into the chat
  - Sandbox: subprocess in a restricted env (no network, file write
    limited to a working dir, timeout). Docker or Pyodide-based
    execution are both options
  - Output rendering: matplotlib → PNG/SVG, Mermaid → render via
    mermaid-cli (mmdc) to SVG, plotly → HTML

Anti-patterns to avoid
  - The model saying "the answer is 3.14159..." with no code shown
  - Hand-computed arithmetic in the model's prose when it could be
    `compute(x)`
  - Mermaid diagrams that don't actually visualize the answer
  - "Magic numbers" in code without provenance comments

This is what makes outten/lmm a Large Math Model rather than a
math-flavored chatbot: every numerical claim is backed by code that
can be inspected and re-executed.

---
