"""Base class for code runtimes."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RuntimeResult:
    """Outcome of executing a single code block.

    Attributes:
        ok:               True if execution finished without raising.
        stdout:           Captured standard output (truncated to 50 KB).
        stderr:           Captured standard error (truncated to 50 KB).
        returncode:       Process exit code, or 0 if not applicable.
        error:            Exception/error message if ok is False.
        artifacts:        List of files produced by the execution
                          (relative to the working dir). Charts, CSVs,
                          rendered SVGs, etc.
        execution_time_s: Wall-clock duration.
        language:         The language tag this runtime handled.
    """
    ok: bool
    language: str
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    error: Optional[str] = None
    artifacts: list[Path] = field(default_factory=list)
    execution_time_s: float = 0.0

    def summary(self, max_chars: int = 2000) -> str:
        """Human-readable summary for display."""
        head = f"[{self.language}] ok={self.ok} returncode={self.returncode}"
        head += f" ({self.execution_time_s:.2f}s)"
        if self.error:
            head += f"  error: {self.error[:max_chars]}"
        if self.stdout:
            head += f"\n--- stdout ---\n{self.stdout[:max_chars]}"
        if self.stderr:
            head += f"\n--- stderr ---\n{self.stderr[:max_chars]}"
        if self.artifacts:
            head += f"\n--- artifacts ---\n"
            head += "\n".join(f"  {a}" for a in self.artifacts)
        return head


class Runtime(abc.ABC):
    """Abstract base for a sandboxed code executor."""

    language: str = "abstract"

    @abc.abstractmethod
    def execute(self, code: str, workdir: Path, timeout_s: float = 30.0,
                memory_mb: int = 512) -> RuntimeResult:
        """Execute the given code in workdir.

        Implementations must:
          - Write `code` to a temporary file inside workdir
          - Run it with the configured isolation (subprocess, sandbox)
          - Enforce timeout_s and memory_mb
          - Capture stdout/stderr, returncode
          - Detect produced artifacts in workdir (files newer than the
            start of execution)
          - Never block longer than timeout_s + a small grace period
        """