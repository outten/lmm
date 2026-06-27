"""Eval harness for outten/lmm.

Code-verified evaluation (the key innovation vs string-matching):

  1. Pose a problem to the model.
  2. The model emits a response containing ```python``` blocks.
  3. We execute the blocks in the sandbox.
  4. We extract a numeric/structured "answer" from the code's output
     (using a problem-specific extractor).
  5. We compare against the ground truth with a tolerance/equivalence
     check — NEVER a regex on the model's prose.

This module also supports the legacy fallback (string-match) for
models that don't yet emit code, but reports it separately.

Usage:
    python -m eval.harness.run_benchmark \\
        --benchmark eval/benchmarks/gsm8k.jsonl \\
        --model outten/lmm:v0.1 \\
        --output eval/results/gsm8k.json
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Allow imports from project root when run as `python -m eval.harness.runner`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sandbox import run_sandboxed  # noqa: E402
from sandbox.ollama_cli import ollama_generate  # noqa: E402


@dataclass
class ProblemResult:
    problem_id: str
    tier: int
    question: str
    expected: object
    model_response: str = ""
    code_blocks: int = 0
    code_executed_ok: bool = False
    extracted_answer: object = None
    correct: bool = False
    error: Optional[str] = None
    execution_time_s: float = 0.0


@dataclass
class BenchmarkSummary:
    benchmark: str
    model: str
    n_problems: int = 0
    n_correct: int = 0
    n_code_executed: int = 0
    accuracy: float = 0.0
    code_correctness_rate: float = 0.0
    per_tier: dict = field(default_factory=dict)
    results: list[ProblemResult] = field(default_factory=list)

    def finalize(self):
        self.accuracy = self.n_correct / self.n_problems if self.n_problems else 0.0
        self.code_correctness_rate = (
            self.n_code_executed / self.n_problems if self.n_problems else 0.0
        )
        for r in self.results:
            key = f"tier_{r.tier}"
            t = self.per_tier.setdefault(key, {"n": 0, "correct": 0, "code_ok": 0})
            t["n"] += 1
            if r.correct:
                t["correct"] += 1
            if r.code_executed_ok:
                t["code_ok"] += 1

    def to_dict(self):
        return {
            "benchmark": self.benchmark,
            "model": self.model,
            "n_problems": self.n_problems,
            "n_correct": self.n_correct,
            "n_code_executed": self.n_code_executed,
            "accuracy": self.accuracy,
            "code_correctness_rate": self.code_correctness_rate,
            "per_tier": self.per_tier,
            "results": [asdict(r) for r in self.results],
        }


def extract_numeric_answer(text: str) -> Optional[float]:
    """Last number printed in stdout — a deliberately simple extractor.

    Replace per-benchmark with smarter extractors as needed (e.g.,
    GSM8K extracts the value after '####', MATH extracts from \boxed{}).
    The point is the *executor* produced it, not the model's prose.
    """
    # Prefer the last line that contains "= <number>"
    for line in reversed(text.splitlines()):
        m = re.search(r"=\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    # Fallback: last number in the text
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return None


def numeric_close(a: float, b: float, rel: float = 1e-3, abs_: float = 1e-6) -> bool:
    """Compare two numbers with relative + absolute tolerance."""
    if a is None or b is None:
        return False
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), abs_)


def evaluate_problem(record: dict, model: str, host: str = "http://127.0.0.1:11434",
                     timeout_s: float = 30.0) -> ProblemResult:
    question = record["question"]
    expected = record.get("answer")
    tier = record.get("tier", 0)

    pr = ProblemResult(
        problem_id=record.get("id", str(hash(question))),
        tier=tier,
        question=question,
        expected=expected,
    )

    # 1. Ask the model
    prompt = f"### Question\n{question}\n"
    try:
        response = ollama_generate(model, prompt, host=host)
    except Exception as e:
        pr.error = f"ollama error: {e}"
        return pr
    pr.model_response = response

    # 2. Execute code blocks
    with tempfile.TemporaryDirectory(prefix="lmm-eval-") as tmp:
        run = run_sandboxed(response, workdir=Path(tmp), timeout_s=timeout_s)
        pr.code_blocks = len(run.results)
        pr.code_executed_ok = run.ok and any(r.ok for r in run.results)
        pr.execution_time_s = run.total_time_s
        if run.results and run.results[0].stdout:
            pr.extracted_answer = extract_numeric_answer(run.results[0].stdout)

    # 3. Compare
    if isinstance(expected, (int, float)) and isinstance(pr.extracted_answer, (int, float)):
        pr.correct = numeric_close(float(expected), float(pr.extracted_answer))
    elif expected is not None and pr.extracted_answer is not None:
        pr.correct = str(expected).strip() == str(pr.extracted_answer).strip()

    return pr


def run_benchmark(benchmark_path: Path, model: str, host: str = "http://127.0.0.1:11434",
                  output: Optional[Path] = None, limit: Optional[int] = None,
                  timeout_s: float = 30.0) -> BenchmarkSummary:
    summary = BenchmarkSummary(benchmark=str(benchmark_path), model=model)
    with benchmark_path.open() as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            record = json.loads(line)
            pr = evaluate_problem(record, model=model, host=host, timeout_s=timeout_s)
            summary.results.append(pr)
            summary.n_problems += 1
            if pr.correct:
                summary.n_correct += 1
            if pr.code_executed_ok:
                summary.n_code_executed += 1
            status = "OK" if pr.correct else "  "
            print(f"[{i+1:4d}] {status}  tier={pr.tier}  extracted={pr.extracted_answer!r:>20}  "
                  f"expected={pr.expected!r:>20}  code_ok={pr.code_executed_ok}")
    summary.finalize()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            json.dump(summary.to_dict(), f, indent=2)
    return summary


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--benchmark", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args()

    summary = run_benchmark(
        benchmark_path=args.benchmark,
        model=args.model,
        host=args.host,
        output=args.output,
        limit=args.limit,
        timeout_s=args.timeout,
    )
    print()
    print(f"=== {summary.benchmark} on {summary.model} ===")
    print(f"Problems:   {summary.n_problems}")
    print(f"Correct:    {summary.n_correct}  ({summary.accuracy:.1%})")
    print(f"Code ran:   {summary.n_code_executed}  ({summary.code_correctness_rate:.1%})")
    print()
    print("Per tier:")
    for tier, stats in sorted(summary.per_tier.items()):
        acc = stats["correct"] / stats["n"] if stats["n"] else 0.0
        print(f"  {tier}: {stats['correct']}/{stats['n']}  ({acc:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())