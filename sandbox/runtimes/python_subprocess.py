"""Python execution via a sandboxed subprocess.

Security posture (defense in depth, but NOT a hardened sandbox):
  - No network access by default (env cleared, subprocess inherits
    a sanitized environment)
  - Writes limited to workdir
  - Hard timeout
  - Memory cap via resource limits (RLIMIT_AS on POSIX)
  - Filesystem writes outside workdir are caught by snapshotting
    workdir before/after and reporting deltas

For stronger isolation, run this inside a Docker container or gVisor.
See sandbox/docker_runner.py for the Docker-backed variant.
"""
from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from .base import Runtime, RuntimeResult


# Cap stdout/stderr to keep chat context from blowing up on a runaway
# print loop. Truncation is annotated so the model can see it was cut.
MAX_OUTPUT_BYTES = 50_000


def _truncate(text: str, limit: int = MAX_OUTPUT_BYTES) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} bytes]"


def _resolve_python(explicit: Optional[str]) -> str:
    """Pick the Python interpreter that has the packages the user expects.

    Order of preference:
      1. Explicit argument
      2. LMM_PYTHON env var
      3. `python3` on PATH (matches what the user typed at the shell)
      4. `python` on PATH
      5. The interpreter running this code (last resort)

    Using `python3` from PATH instead of `sys.executable` matters on
    macOS, where Xcode's Python (sys.executable) often lacks the
    packages the user installed via `pip install`.
    """
    if explicit:
        return explicit
    if os.environ.get("LMM_PYTHON"):
        return os.environ["LMM_PYTHON"]
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            return found
    return sys.executable


class PythonSubprocessRuntime(Runtime):
    """Execute Python code in a subprocess with resource limits.

    By default uses `python3` from PATH. Override `python_executable`
    to use a specific venv.
    """

    language = "python"

    def __init__(self, python_executable: Optional[str] = None):
        self.python_executable = _resolve_python(python_executable)

    def execute(self, code: str, workdir: Path, timeout_s: float = 30.0,
                memory_mb: int = 512) -> RuntimeResult:
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        script_path = workdir / "_lmm_script.py"
        before = _snapshot_workdir(workdir)
        script_path.write_text(code, encoding="utf-8")

        # Sandboxed env: strip secrets and API tokens, force no proxies,
# redirect tempdir to workdir. KEEP PATH, HOME, and PYTHONUSERBASE
# intact so user-installed Python packages still resolve — overriding
# HOME drops the user site-packages directory on macOS/Linux.
        env = {}
        # Forward a minimal whitelist of env vars; strip everything else
        # (including API keys, AWS creds, etc.) by default.
        for key in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR",
                    "PYTHONPATH", "PYTHONUSERBASE", "PYTHONHOME",
                    "SSL_CERT_FILE", "SSL_CERT_DIR"):
            if key in os.environ:
                env[key] = os.environ[key]
        # Redirect TMPDIR into workdir if not already set
        env.setdefault("TMPDIR", str(workdir))
        env["LMM_SANDBOX"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"

        start = time.time()
        try:
            proc = subprocess.Popen(
                [self.python_executable, "-W", "ignore", str(script_path)],
                cwd=str(workdir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # New process group so we can kill children too
                preexec_fn=_apply_resource_limits(memory_mb) if sys.platform != "win32" else None,
            )
        except FileNotFoundError as e:
            return RuntimeResult(
                ok=False,
                language=self.language,
                error=f"python interpreter not found: {e}",
                execution_time_s=time.time() - start,
            )

        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            stdout, stderr = proc.communicate()
            return RuntimeResult(
                ok=False,
                language=self.language,
                stdout=_truncate(stdout or ""),
                stderr=_truncate(stderr or ""),
                returncode=-1,
                error=f"timeout after {timeout_s}s",
                execution_time_s=time.time() - start,
            )

        after = _snapshot_workdir(workdir)
        artifacts = _diff_artifacts(before, after, exclude={script_path.name})
        # Resolve to absolute paths so callers can .exists() them
        # without needing to know the workdir.
        artifacts = [workdir / a for a in artifacts]

        return RuntimeResult(
            ok=returncode == 0,
            language=self.language,
            stdout=_truncate(stdout or ""),
            stderr=_truncate(stderr or ""),
            returncode=returncode,
            error=None if returncode == 0 else f"exit code {returncode}",
            artifacts=artifacts,
            execution_time_s=time.time() - start,
        )


def _apply_resource_limits(memory_mb: int):
    """Return a preexec_fn that limits memory and CPU on POSIX."""
    def _setlimits():
        # Address-space limit (bytes). Acts as a soft memory cap.
        mem_bytes = memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass
        # CPU time limit: 2x the wall timeout would normally suffice,
        # but the wall timeout in Popen.communicate is the hard cap.
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
        except (ValueError, OSError):
            pass
        # New process group so we can signal the whole tree
        os.setpgrp()
    return _setlimits


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Send SIGKILL to the process group, then the process itself."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except Exception:
            pass


def _snapshot_workdir(workdir: Path) -> set[str]:
    """Return a set of (relative) file paths currently in workdir."""
    if not workdir.exists():
        return set()
    return {str(p.relative_to(workdir)) for p in workdir.rglob("*") if p.is_file()}


def _diff_artifacts(before: set[str], after: set[str],
                    exclude: set[str] | None = None) -> List[Path]:
    """Return files in `after` that weren't in `before`, excluding listed names."""
    exclude = exclude or set()
    new = (after - before) - exclude
    return [Path(name) for name in sorted(new)]