# outten/lmm — Documentation

Long-form documentation lives here.

## Index

- [PROPOSAL.md](../PROPOSAL.md) — design proposal, methodology, milestones
- [SPECIFICATION.md](../SPECIFICATION.md) — original 3-line spec
- [CLOUD_GPU.md](CLOUD_GPU.md) — provider comparison, pricing, recommendations
- [Architecture decision records](#adrs) — significant design choices

## Architecture Decision Records

Each ADR captures a meaningful design choice, the context that
motivated it, and the consequences. Living list:

  - ADR-001: Code-first response format (see PROPOSAL.md §13)
  - ADR-002: Sandbox as a separate process (vs in-process execution)
  - ADR-003: Code-verified evaluation (vs string-match on prose)
  - ADR-004: Mermaid for diagrams (vs ASCII art, SVG hand-written)

(Proper ADR files to be added as decisions get reviewed.)

## Code-first format specification

This is the canonical response shape the model is trained to emit.
Used by training/data/example.jsonl and the reformat_dataset.py
converter.

```
### Question
<user's question>

### Approach
<1-3 sentence explanation of method>

### Code
```python
# python code that performs the math
```

```mermaid
# optional mermaid diagram
```

### Output
<text/table/file-path showing the result of running the code>

Re-run anytime: save the code block to a .py file and run it. No LLM, no tokens.
```

The runtime (sandbox/__init__.py) extracts blocks by language tag
and dispatches them to the matching runtime. Unknown or empty
languages are left as plain markdown.