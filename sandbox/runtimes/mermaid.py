"""Mermaid diagram rendering via mermaid-cli (mmdc).

The model emits a ```mermaid fenced block; we write it to a .mmd file
and invoke mmdc to produce an SVG. The SVG is the "executed output."
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from .base import Runtime, RuntimeResult


class MermaidRuntime(Runtime):
    """Render Mermaid diagrams to SVG via mermaid-cli.

    Requires `mmdc` on PATH. Install with:
        npm install -g @mermaid-js/mermaid-cli

    Or use the convenience script: scripts/install_mmdc.sh
    """

    language = "mermaid"

    def __init__(self, mmdc_executable: str | None = None,
                 puppeteer_config: Path | None = None):
        self.mmdc = mmdc_executable or shutil.which("mmdc") or "mmdc"
        self.puppeteer_config = puppeteer_config

    def execute(self, code: str, workdir: Path, timeout_s: float = 30.0,
                memory_mb: int = 512) -> RuntimeResult:
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        if shutil.which(self.mmdc) is None and not Path(self.mmdc).exists():
            return RuntimeResult(
                ok=False,
                language=self.language,
                error=(
                    "mermaid-cli (mmdc) not found on PATH. "
                    "Install with: npm install -g @mermaid-js/mermaid-cli"
                ),
            )

        mmd_path = workdir / "_lmm_diagram.mmd"
        svg_path = workdir / "_lmm_diagram.svg"
        mmd_path.write_text(code, encoding="utf-8")

        cmd = [self.mmdc, "-i", str(mmd_path), "-o", str(svg_path), "-q"]
        if self.puppeteer_config:
            cmd += ["-p", str(self.puppeteer_config)]

        start = time.time()
        try:
            proc = subprocess.run(
                cmd, cwd=str(workdir),
                capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return RuntimeResult(
                ok=False,
                language=self.language,
                error=f"mmdc timeout after {timeout_s}s",
                execution_time_s=time.time() - start,
            )

        ok = proc.returncode == 0 and svg_path.exists()
        # Try to extract a parse error from mmdc's JSON output if it failed
        error = None if ok else _extract_mmdc_error(proc.stderr or proc.stdout)
        artifacts = [svg_path.resolve()] if ok else []
        return RuntimeResult(
            ok=ok,
            language=self.language,
            stdout="",
            stderr=_truncate(proc.stderr or ""),
            returncode=proc.returncode,
            error=error,
            artifacts=artifacts,
            execution_time_s=time.time() - start,
        )


def _truncate(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} bytes]"


def _extract_mmdc_error(text: str) -> str | None:
    """mmdc sometimes prints JSON-encoded error lines. Pull the message out."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and '"message"' in line:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "message" in obj:
                    return str(obj["message"])
            except json.JSONDecodeError:
                continue
    # Fall back to first non-empty line
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return "mmdc failed (no error output)"