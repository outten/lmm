"""CLI: chat with an Ollama model and execute its code blocks.

Usage:
    python -m sandbox.ollama_cli "What is the integral of x^2 from 0 to 1?"
    python -m sandbox.ollama_cli --model outten/lmm --workdir /tmp/lmm --show-code

Wired so that a single turn produces:
    1. Model emits a response (potentially containing ```python and
       ```mermaid blocks)
    2. Sandbox extracts and executes the code
    3. Results are printed (and, if --followup is set, fed back to the
       model for a summary turn)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow `python -m sandbox.ollama_cli` to find the package when run
# from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sandbox import run_sandboxed, render_followup  # noqa: E402

import urllib.request  # noqa: E402
import urllib.error  # noqa: E402


def ollama_generate(model: str, prompt: str, host: str = "http://127.0.0.1:11434",
                    options: dict | None = None) -> str:
    """Call Ollama's /api/generate endpoint. Non-streaming.

    Uses stdlib only so the CLI works without an `ollama` Python lib.
    """
    payload = {"model": model, "prompt": prompt, "stream": False}
    if options:
        payload["options"] = options
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate", data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    return obj.get("response", "")


def main() -> int:
    p = argparse.ArgumentParser(description="Chat with Ollama + run code blocks.")
    p.add_argument("prompt", nargs="*", help="Prompt text (or read from stdin).")
    p.add_argument("--model", default=os.environ.get("LMM_MODEL", "outten/lmm"))
    p.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    p.add_argument("--workdir", default=None,
                   help="Where to run code. Default: a fresh tempdir.")
    p.add_argument("--keep-workdir", action="store_true",
                   help="Don't delete the tempdir after running.")
    p.add_argument("--show-code", action="store_true",
                   help="Print each code block before executing.")
    p.add_argument("--followup", action="store_true",
                   help="After execution, send the results back to the model "
                        "and print its summary turn.")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--memory-mb", type=int, default=512)
    p.add_argument("--json", action="store_true",
                   help="Emit a machine-readable JSON result.")
    args = p.parse_args()

    if args.prompt:
        prompt = " ".join(args.prompt)
    else:
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("error: empty prompt", file=sys.stderr)
        return 2

    workdir = Path(args.workdir) if args.workdir else None
    if workdir is not None:
        workdir.mkdir(parents=True, exist_ok=True)

    response = ollama_generate(args.model, prompt, host=args.host)

    if args.show_code:
        from sandbox import extract_code_blocks
        for blk in extract_code_blocks(response):
            print(f"--- [{blk.language}] (block {blk.index}) ---")
            print(blk.code)
            print()

    run = run_sandboxed(
        response, workdir=workdir,
        timeout_s=args.timeout, memory_mb=args.memory_mb,
    )

    if args.json:
        out = {
            "model_response": response,
            "blocks": [
                {"language": b.language, "code": b.code, "index": b.index, "line": b.line}
                for b in run.blocks
            ],
            "results": [
                {
                    "language": r.language,
                    "ok": r.ok,
                    "stdout": r.stdout,
                    "stderr": r.stderr,
                    "returncode": r.returncode,
                    "error": r.error,
                    "artifacts": [str(a) for a in r.artifacts],
                    "execution_time_s": r.execution_time_s,
                }
                for r in run.results
            ],
            "workdir": str(run.workdir) if run.workdir else None,
            "ok": run.ok,
            "total_time_s": run.total_time_s,
        }
        print(json.dumps(out, indent=2))
    else:
        print("=== Model response ===")
        print(response)
        print()
        print("=== Execution ===")
        print(run.summary())

    if args.followup and run.blocks:
        follow = render_followup(response, run)
        summary = ollama_generate(args.model, follow, host=args.host)
        print()
        print("=== Model summary ===")
        print(summary)

    if args.keep_workdir and run.workdir is not None:
        print(f"\n(workdir kept at: {run.workdir})", file=sys.stderr)

    return 0 if run.ok else 1


if __name__ == "__main__":
    sys.exit(main())