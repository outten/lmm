"""Code execution runtimes for the outten/lmm sandbox.

Each runtime knows how to execute one language/dialect and return:
    - stdout / captured output
    - any files produced (charts, data files)
    - exit status and any errors

Runtimes enforce resource limits (timeout, memory, network isolation)
so a model that emits adversarial code cannot damage the host.
"""
from .base import Runtime, RuntimeResult  # noqa: F401
from .python_subprocess import PythonSubprocessRuntime  # noqa: F401
from .mermaid import MermaidRuntime  # noqa: F401

ALL_RUNTIMES = {
    "python": PythonSubprocessRuntime,
    "mermaid": MermaidRuntime,
}