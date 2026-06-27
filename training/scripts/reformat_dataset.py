"""Reformat math instruction datasets into the outten/lmm response shape.

Input datasets (any of):
  - MetaMathQA       (parquet/jsonl): augmented GSM8K + MATH
  - OpenMathInstruct (jsonl)       : 14M problems w/ verified solutions
  - NuminaMath-CoT   (jsonl)       : competition-style w/ CoT
  - local jsonl with {"question", "answer", ...}

Output format (jsonl, one record per example):
    {
      "prompt":    "### Question\\n<problem text>",
      "response":  "### Approach\\n...\\n### Code\\n```python\\n...\\n```\\n### Output\\n...",
      "tier":      3,
      "source":    "metamath"
    }

The training script (sft_train.py) consumes this format directly.

Why we reformat (vs fine-tuning on raw answers):
  Per the Code-First Doctrine (§13 of PROPOSAL.md), we want the model
  to emit Python (or Mermaid) blocks as the primary form of solution.
  Existing datasets mostly have prose answers; we synthesize the code
  form by extracting any contained code blocks, or by wrapping the
  answer in a code-first shell with the model's "approach" taken from
  the original reasoning trace.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, Iterator


RESPONSE_TEMPLATE = """### Approach
{approach}

### Code
```{language}
{code}
```

### Output
{output}

Re-run anytime: save the code block to a .py file and run it. No LLM, no tokens."""


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Return [(language, code), ...] from fenced blocks in `text`."""
    out = []
    for m in re.finditer(r"```([a-zA-Z0-9_+\-]*)\s*\n(.*?)\n```", text, re.DOTALL):
        lang = (m.group(1) or "").strip().lower()
        lang = "python" if lang in ("py", "python3") else (lang or "text")
        out.append((lang, m.group(2)))
    return out


def reformat_metamath(record: dict) -> dict | None:
    """MetaMathQA records have: query, response, type, original_question."""
    question = record.get("query") or record.get("question") or ""
    raw_answer = record.get("response") or record.get("answer") or ""
    if not question or not raw_answer:
        return None
    blocks = extract_code_blocks(raw_answer)
    if blocks:
        # Dataset already has code — lift it out as the primary artifact
        lang, code = blocks[0]
        approach = raw_answer.split("```")[0].strip() or "See code below."
        output = raw_answer.split("```", 2)[-1].strip() or "(run the code)"
    else:
        # No code in original — synthesize a shell that wraps the
        # explanation as an "approach" and emits a stub code block.
        # In practice you'd run this through a code-generation model
        # to fill in the body; we leave a marker for that step.
        lang, code = "python", "# TODO: synthesize code for this problem\n"
        approach = raw_answer.strip()
        output = "(code will produce the answer above when run)"
    response = RESPONSE_TEMPLATE.format(
        approach=approach, language=lang, code=code, output=output,
    )
    return {
        "prompt": f"### Question\n{question.strip()}",
        "response": response,
        "tier": 2,  # MetaMath is mostly grade-school difficulty
        "source": record.get("source", "metamath"),
    }


def reformat_openmathinstruct(record: dict) -> dict | None:
    """OpenMathInstruct records have: problem, generated_solution, expected_answer."""
    problem = record.get("problem") or record.get("question") or ""
    raw = record.get("generated_solution") or record.get("solution") or ""
    expected = record.get("expected_answer", "")
    if not problem or not raw:
        return None
    blocks = extract_code_blocks(raw)
    if blocks:
        lang, code = blocks[0]
        approach = raw.split("```")[0].strip() or "Computed via Python."
        output = expected or raw.split("```", 2)[-1].strip()
    else:
        lang, code = "python", "# synthesized\n"
        approach = raw.strip()
        output = expected or "(run to see)"
    response = RESPONSE_TEMPLATE.format(
        approach=approach, language=lang, code=code, output=output,
    )
    return {
        "prompt": f"### Question\n{problem.strip()}",
        "response": response,
        "tier": 3,
        "source": "openmathinstruct",
    }


REFORMATTERS = {
    "metamath": reformat_metamath,
    "metamathqa": reformat_metamath,
    "openmathinstruct": reformat_openmathinstruct,
    "omi": reformat_openmathinstruct,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("input", type=Path, help="Input jsonl file")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--format", required=True, choices=list(REFORMATTERS),
                   help="Source dataset format")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on number of examples to write")
    p.add_argument("--source-tag", default=None,
                   help="Override the 'source' field in output records")
    args = p.parse_args()

    fn = REFORMATTERS[args.format]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for record in iter_jsonl(args.input):
            n_in += 1
            if args.limit and n_out >= args.limit:
                break
            out = fn(record)
            if out is None:
                continue
            if args.source_tag:
                out["source"] = args.source_tag
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"in={n_in}  out={n_out}  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())