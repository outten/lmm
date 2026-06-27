"""Tests for the sandbox runtime.

Run with:
    python -m pytest tests/
or:
    python -m unittest discover tests
"""
from __future__ import annotations

import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path

from sandbox import extract_code_blocks, run_sandboxed
from sandbox.runtimes.mermaid import MermaidRuntime
from sandbox.runtimes.python_subprocess import PythonSubprocessRuntime


class ExtractCodeBlocksTests(unittest.TestCase):
    def test_single_python_block(self):
        text = "hello\n```python\nprint(1)\n```\nbye"
        blocks = extract_code_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].language, "python")
        self.assertEqual(blocks[0].code.strip(), "print(1)")
        self.assertEqual(blocks[0].line, 2)

    def test_multiple_blocks(self):
        text = textwrap.dedent("""\
            ```python
            a = 1
            ```
            prose
            ```mermaid
            graph TD; A-->B
            ```
        """)
        blocks = extract_code_blocks(text)
        self.assertEqual(len(blocks), 2)
        self.assertEqual([b.language for b in blocks], ["python", "mermaid"])

    def test_empty_fence_not_a_code_block(self):
        """Plain markdown fences with no info string should still parse,
        but our dispatcher ignores them (no registered language)."""
        text = "```\nplain text\n```"
        blocks = extract_code_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].language, "")

    def test_unknown_language(self):
        text = "```ruby\nputs 'hi'\n```"
        blocks = extract_code_blocks(text)
        self.assertEqual(blocks[0].language, "ruby")


class PythonRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp(prefix="lmm-test-"))

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_simple_print(self):
        run = run_sandboxed(
            "```python\nprint('hello world')\n```",
            workdir=self.workdir, timeout_s=10,
        )
        self.assertTrue(run.ok)
        self.assertEqual(len(run.results), 1)
        self.assertIn("hello world", run.results[0].stdout)

    def test_sympy(self):
        run = run_sandboxed(
            "```python\nimport sympy as sp\n"
            "x = sp.symbols('x')\n"
            "print(sp.integrate(x**2, (x, 0, 1)))\n"
            "```",
            workdir=self.workdir, timeout_s=15,
        )
        self.assertTrue(run.ok, msg=run.results[0].stderr)
        self.assertIn("1/3", run.results[0].stdout)

    def test_matplotlib_artifact(self):
        run = run_sandboxed(
            "```python\n"
            "import matplotlib\nmatplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "plt.plot([1, 2, 3], [1, 4, 9])\n"
            "plt.savefig('chart.png')\n"
            "print('saved')\n"
            "```",
            workdir=self.workdir, timeout_s=30,
        )
        self.assertTrue(run.ok, msg=run.results[0].stderr)
        self.assertEqual(len(run.results[0].artifacts), 1)
        self.assertEqual(run.results[0].artifacts[0].name, "chart.png")
        self.assertTrue(run.results[0].artifacts[0].exists())

    def test_timeout(self):
        run = run_sandboxed(
            "```python\nimport time; time.sleep(60)\n```",
            workdir=self.workdir, timeout_s=1,
        )
        self.assertFalse(run.ok)
        self.assertIn("timeout", (run.results[0].error or "").lower())

    def test_exit_code_captured(self):
        run = run_sandboxed(
            "```python\nimport sys; sys.exit(7)\n```",
            workdir=self.workdir, timeout_s=5,
        )
        self.assertFalse(run.ok)
        self.assertEqual(run.results[0].returncode, 7)

    def test_unknown_language_skipped(self):
        run = run_sandboxed(
            "```text\nnot code\n```\n```python\nprint(1)\n```",
            workdir=self.workdir, timeout_s=5,
        )
        # Only the python block should be executed; text is left alone.
        self.assertTrue(run.ok)
        self.assertEqual(len(run.results), 1)
        self.assertEqual(run.results[0].language, "python")


class MermaidRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp(prefix="lmm-test-"))

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    @unittest.skipUnless(shutil.which("mmdc"), "mermaid-cli not installed")
    def test_simple_diagram(self):
        run = run_sandboxed(
            "```mermaid\ngraph TD; A-->B; B-->C;\n```",
            workdir=self.workdir, timeout_s=20,
        )
        self.assertTrue(run.ok, msg=run.results[0].error or run.results[0].stderr)
        self.assertEqual(len(run.results[0].artifacts), 1)
        svg = run.results[0].artifacts[0]
        self.assertEqual(svg.suffix, ".svg")
        self.assertTrue(svg.exists())

    @unittest.skipUnless(shutil.which("mmdc"), "mermaid-cli not installed")
    def test_invalid_diagram(self):
        run = run_sandboxed(
            "```mermaid\nnot a valid diagram @#$%\n```",
            workdir=self.workdir, timeout_s=20,
        )
        self.assertFalse(run.ok)


if __name__ == "__main__":
    unittest.main()