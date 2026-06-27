# Contributing to outten/lmm

Thanks for helping build a Large Math Model. The design philosophy
("code-first") is the most important thing to understand before
contributing — see [PROPOSAL.md §13](PROPOSAL.md#13-code-first-doctrine).

## Where to start

| You want to...                                  | Look here                                       |
| ----------------------------------------------- | ----------------------------------------------- |
| Improve the runtime / add a language            | `sandbox/runtimes/`                             |
| Add test cases                                  | `tests/test_sandbox.py`                         |
| Add benchmark problems                          | `eval/benchmarks/` (jsonl, see schema in harness)|
| Add a new public dataset adapter                | `training/scripts/reformat_dataset.py`          |
| Improve training hyperparameters                | `training/configs/`                             |
| Add a doc or tutorial                           | `docs/`                                         |
| Improve the model itself                        | `training/scripts/sft_train.py` + `Modelfile`   |
| File a bug                                      | GitHub Issues (label: `bug`)                    |
| Propose a design change                         | GitHub Discussions (category: `ideas`)          |

## Development setup

```bash
git clone https://github.com/outten/lmm
cd ollama-lmm

# Runtime deps (used by the sandbox)
make install

# Dev deps (training, eval)
make install-dev

# Run the test suite
make sandbox-test
```

## Code style

- Python: PEP 8, type hints where reasonable, docstrings on public
  functions. Run `ruff check .` if installed.
- Modelfile: keep system prompts short and behavior-focused.
- Markdown: terminal-friendly plain text where possible; one blank
  line between sections; avoid nested lists more than two deep.

## Tests

The sandbox tests in `tests/` are the gate. They run on every PR via
GitHub Actions. Adding code without tests is fine for prototypes, but
anything merged to `main` should have corresponding test coverage.

```bash
# Quick: just the sandbox
make sandbox-test

# Full: include eval-harness smoke test
python3 -m eval.harness.runner \
    --benchmark eval/benchmarks/smoke.jsonl \
    --model outten/lmm:latest --limit 13
```

## Adding a new language runtime

The runtime registry in `sandbox/runtimes/__init__.py` is the contract.
Implement the `Runtime` ABC (see `base.py`), then register it:

```python
# sandbox/runtimes/my_lang.py
from .base import Runtime, RuntimeResult

class MyLangRuntime(Runtime):
    language = "mylang"
    def execute(self, code, workdir, timeout_s=30.0, memory_mb=512):
        ...

# sandbox/runtimes/__init__.py
from .my_lang import MyLangRuntime
ALL_RUNTIMES["mylang"] = MyLangRuntime
```

Add a test in `tests/test_sandbox.py` and you're done — any model that
emits a ```mylang fence will now execute via your runtime.

## Anti-contamination (training data hygiene)

The test splits of GSM8K, MATH, and AIME must NEVER appear in any
training set. If you add a dataset adapter or downloader, explicitly
exclude these splits. The eval harness does NOT mix train and test
data — it always evaluates against held-out benchmarks.

If you suspect contamination, open an issue with the label
`contamination`.

## Pull request checklist

- [ ] Code passes `make sandbox-test`
- [ ] New behavior has a corresponding test
- [ ] New public function/CLI flag has a docstring
- [ ] PROPOSAL.md updated if the change is design-level
- [ ] README updated if user-facing
- [ ] No training-set leakage from any benchmark's test split

## Communication

- **Bugs / small features**: GitHub Issues
- **Design questions / ideas**: GitHub Discussions
- **Security issues**: see `SECURITY.md` (or open a private issue)

## Code of conduct

Be kind, assume good faith, focus on the math. This project exists to
make math more accessible and more correct — let's keep the tone
matching the goal.