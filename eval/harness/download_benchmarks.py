"""Download public math benchmarks into eval/benchmarks/.

Datasets are converted to the jsonl schema the eval harness expects:
    {"id": "...", "tier": N, "question": "...", "answer": ...}

We deliberately drop the test-split fields we DON'T want in any
training data (PROPOSAL.md §4 anti-contamination). The downloaded
files live in eval/benchmarks/ — never copy them into training/data/.

Usage:
    python eval/harness/download_benchmarks.py gsm8k
    python eval/harness/download_benchmarks.py math
    python eval/harness/download_benchmarks.py --all
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval" / "benchmarks"


def write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records):>6} records to {path}")


def gsm8k_test() -> list[dict]:
    """GSM8K test split. 1319 problems, grade school arithmetic."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    out = []
    for i, row in enumerate(ds):
        # GSM8K answers end with "#### N". We strip that for the prompt
        # and use N as the expected answer.
        full = row["answer"]
        m = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", full)
        if not m:
            continue
        answer_str = m.group(1).replace(",", "")
        out.append({
            "id": f"gsm8k-test-{i:04d}",
            "tier": 2,
            "question": row["question"],
            "answer": float(answer_str) if "." in answer_str else int(answer_str),
        })
    return out


def math_test() -> list[dict]:
    """Hendrycks MATH test split. 5000 problems, 7 subjects."""
    from datasets import load_dataset
    ds = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
    out = []
    for i, row in enumerate(ds):
        # MATH answers are wrapped in \boxed{...}
        m = re.search(r"\\boxed\{([^}]+)\}", row["solution"])
        if not m:
            continue
        out.append({
            "id": f"math-test-{i:04d}",
            "tier": _math_tier(row.get("type", "")),
            "question": row["problem"],
            "answer": m.group(1).strip(),
            "subject": row.get("type", ""),
        })
    return out


def _math_tier(subject: str) -> int:
    """Map MATH subjects to our 12-tier scale (rough)."""
    return {
        "Algebra": 3,
        "Counting & Probability": 6,
        "Geometry": 4,
        "Intermediate Algebra": 5,
        "Number Theory": 9,
        "Prealgebra": 2,
        "Precalculus": 5,
    }.get(subject, 6)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("which", nargs="*", choices=["gsm8k", "math"],
                   help="Which benchmarks to download")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    targets = ["gsm8k", "math"] if args.all else args.which
    if not targets:
        print("specify a benchmark (gsm8k, math) or --all", file=sys.stderr)
        return 2

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("Install the datasets library:", file=sys.stderr)
        print("  pip install datasets", file=sys.stderr)
        return 3

    if "gsm8k" in targets:
        write_jsonl(gsm8k_test(), OUT / "gsm8k_test.jsonl")
    if "math" in targets:
        write_jsonl(math_test(), OUT / "math_test.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())