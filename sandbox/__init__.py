"""outten/lmm sandbox — extract code blocks and execute them.

Main entry point: `run_sandboxed(model_output, workdir)` returns a
`SandboxRun` containing every code block that was extracted and the
result of running it.

Design contract:
  - The model emits fenced code blocks:
        ```python
        # ...
        ```

        ```mermaid
        graph TD; A-->B
        ```
  - We identify each block's language from the fence info string.
  - We dispatch to the matching runtime.
  - Results are returned in execution order.
  - Output is truncated to keep chat context bounded.

This module is intentionally framework-free — no Ollama, no HTTP.
Callers wire it up to whatever chat surface they want.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .runtimes import ALL_RUNTIMES, RuntimeResult


# Match a fenced code block with an optional info string.
#   ```python
#   ...
#   ```
#   ```mermaid
#   ...
#   ```
# Captures group 1 = language (lowercased, may be empty), group 2 = body.
_FENCE_RE = re.compile(
    r"```([a-zA-Z0-9_+\-]*)\s*\n(.*?)\n```",
    re.DOTALL,
)


@dataclass
class CodeBlock:
    """A code block extracted from the model's output."""
    language: str
    code: str
    index: int
    line: int  # line in the model output where the block starts


@dataclass
class SandboxRun:
    """Outcome of running all code blocks in a single model response."""
    blocks: List[CodeBlock] = field(default_factory=list)
    results: List[RuntimeResult] = field(default_factory=list)
    workdir: Optional[Path] = None
    total_time_s: float = 0.0

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results) if self.results else True

    def summary(self, max_chars: int = 2000) -> str:
        if not self.blocks:
            return "(no code blocks found in model output)"
        lines = []
        for blk, res in zip(self.blocks, self.results):
            lines.append(f"### Block {blk.index} [{blk.language}]")
            lines.append(res.summary(max_chars=max_chars))
            lines.append("")
        return "\n".join(lines).rstrip()


def extract_code_blocks(model_output: str) -> List[CodeBlock]:
    """Pull all fenced code blocks out of `model_output`.

    Language normalization:
      - "py"       -> "python"
      - anything   -> lowercased as-is; runtime registry decides if known
    """
    blocks: List[CodeBlock] = []
    for i, m in enumerate(_FENCE_RE.finditer(model_output)):
        lang_raw = (m.group(1) or "").strip().lower()
        lang = "python" if lang_raw in ("py", "python3") else lang_raw
        line = model_output[:m.start()].count("\n") + 1
        blocks.append(CodeBlock(
            language=lang,
            code=m.group(2),
            index=i,
            line=line,
        ))
    return blocks


def run_sandboxed(
    model_output: str,
    workdir: Path | None = None,
    timeout_s: float = 30.0,
    memory_mb: int = 512,
    only_languages: Optional[set[str]] = None,
    custom_runtimes: Optional[dict] = None,
) -> SandboxRun:
    """Extract and execute all code blocks in `model_output`.

    Args:
        model_output: The raw text the model emitted.
        workdir:      Where to execute. Created if missing. If None, a
                      fresh tempdir is made and removed on completion.
        timeout_s:    Per-block wall-clock timeout.
        memory_mb:    Memory cap for Python subprocesses.
        only_languages: If given, skip blocks whose language isn't in this set.
        custom_runtimes: Override or extend the default runtime registry.

    Returns:
        SandboxRun with blocks, results, and the workdir (caller's
        responsibility to clean up if they supplied their own).
    """
    runtimes = {**ALL_RUNTIMES, **(custom_runtimes or {})}
    all_blocks = extract_code_blocks(model_output)
    # Only dispatch blocks whose language is in the runtime registry.
    # Empty info strings and unknown languages are left as plain
    # markdown in the model's output (e.g. ```text``` showing sample
    # output).
    blocks = [b for b in all_blocks if b.language in runtimes]
    if only_languages is not None:
        blocks = [b for b in blocks if b.language in only_languages]

    if workdir is None:
        import tempfile
        workdir = Path(tempfile.mkdtemp(prefix="lmm-sandbox-"))
        cleanup = True
    else:
        cleanup = False
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    results: List[RuntimeResult] = []
    start = time.time()
    for blk in blocks:
        runtime_cls = runtimes.get(blk.language)
        if runtime_cls is None:
            results.append(RuntimeResult(
                ok=False,
                language=blk.language,
                error=f"no runtime registered for language: {blk.language!r}",
            ))
            continue
        runtime = runtime_cls()
        results.append(runtime.execute(
            code=blk.code, workdir=workdir,
            timeout_s=timeout_s, memory_mb=memory_mb,
        ))

    sandbox_run = SandboxRun(
        blocks=blocks, results=results, workdir=workdir,
        total_time_s=time.time() - start,
    )

    if cleanup:
        # Best-effort cleanup; leave it if anything went wrong so a
        # human can inspect.
        import shutil
        if sandbox_run.ok:
            shutil.rmtree(workdir, ignore_errors=True)

    return sandbox_run


def render_followup(model_output: str, sandbox_run: SandboxRun) -> str:
    """Compose a follow-up prompt that gives the model the execution
    results so it can summarize or self-correct.

    The exact format is what the model is fine-tuned to expect during
    training; this is the format we use as a contract.
    """
    parts = ["### Execution Results", ""]
    if not sandbox_run.blocks:
        parts.append("(no code blocks were found in your previous response)")
        parts.append("")
        parts.append("Reminder: please respond using fenced code blocks.")
    else:
        for blk, res in zip(sandbox_run.blocks, sandbox_run.results):
            parts.append(f"#### Block {blk.index} `{blk.language}`")
            parts.append("")
            parts.append(res.summary(max_chars=1500))
            parts.append("")
    parts.append("### Now respond")
    parts.append("Summarize the result for the user. If something failed,")
    parts.append("fix the code and try again.")
    return "\n".join(parts)