# outten/lmm — implementation plan & status

This file tracks what the repo contains today, what's still
proposal-only, and the order in which new work should land.

See PROPOSAL.md for the *design*. This file is the *delivery*.

## Status legend

  [x]  shipped, tested
  [~]  in progress
  [ ]  proposed, not yet built

## v0.1 — Foundation

Runtime (sandbox/)
  [x] extract_code_blocks — fence parsing
  [x] python_subprocess runtime — resource limits, env sandboxing
  [x] mermaid runtime — mmdc wrapper, error extraction
  [x] run_sandboxed — orchestrates extraction + execution
  [x] render_followup — formats execution results for the model
  [x] ollama_cli — one-shot CLI: model → sandbox → optional summary
  [x] 12 unit tests covering extraction, python, mermaid

Training (training/)
  [x] reformat_dataset.py — MetaMathQA + OMI -> our schema
  [x] sft_train.py — Unsloth + QLoRA SFT (NVIDIA GPU required)
  [x] convert_to_gguf.py — HF -> GGUF for Ollama
  [x] configs/v0.1.yaml — hyperparameters
  [x] data/example.jsonl — 3 hand-crafted smoke examples
  [ ] First real fine-tuned checkpoint (depends on GPU access)

Eval (eval/)
  [x] harness/runner.py — code-verified eval loop
  [x] harness/download_benchmarks.py — GSM8K + MATH downloader
  [x] benchmarks/smoke.jsonl — 13 tier-1..7 problems
  [ ] benchmarks/gsm8k_test.jsonl — full GSM8K (post-download)
  [ ] benchmarks/math_test.jsonl — full MATH (post-download)

Distribution
  [x] Modelfile (placeholder + code-first system prompt)
  [x] Makefile (install, build, run, eval, train, convert, push)
  [ ] First outten/lmm:v0.1 published to ollama.com

Community
  [x] README.md
  [x] CONTRIBUTING.md
  [x] LICENSE (Apache 2.0)
  [x] .github/workflows/ci.yml (sandbox tests + import smoke)
  [x] .gitignore
  [x] examples/ — tier 1, 5, 8 worked examples
  [x] docs/ — code-first format spec
  [ ] GH repo created (handled outside this commit)

## v0.2 — Code execution at scale

  [ ] DPO training script (Phase 2 of methodology)
  [ ] Sympy-based equivalence check for preference scoring
  [ ] Code-verified eval on full GSM8K
  [ ] NuminaMath-CoT integration in reformat_dataset
  [ ] PRM800K-style process reward signals

## v0.3 — Process supervision + scale

  [ ] Process reward model training
  [ ] Larger base (14B / 72B)
  [ ] STaR self-improvement loop
  [ ] Docker-backed sandbox runtime for stronger isolation

## v1.0 — Research-grade

  [ ] Lean/Coq formal verification hooks
  [ ] Public reproducible eval suite
  [ ] Model card + responsible-AI disclosure

## Open decisions (from PROPOSAL.md §11)

These still need calls:
  - [ ] Base model for v0.1 (Qwen2.5-Math-7B recommended)
  - [ ] Compute provider (Lambda Labs / RunPod / AWS / on-prem)
  - [ ] Distribution: public ollama registry or private
  - [ ] Sandbox: Pyodide vs Docker subprocess
  - [ ] Repo: monorepo (current) or split per phase