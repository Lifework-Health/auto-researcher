# Auto Researcher v2.1

Auto Researcher v2.1 introduces a typed, resumable LangGraph control plane around
scientific components. PR 1 is intentionally offline: deterministic mock agents,
direct search, evaluation, verification, SQLite checkpoints, and append-only
scientific provenance demonstrate one complete research cycle without external
services.

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e . --no-deps
.venv/bin/auto-researcher run --mock --run-id demo --thread-id demo-thread --max-cycles 1
.venv/bin/auto-researcher provenance --run-id demo
```

See [the architecture](docs/architecture/V2_1_ARCHITECTURE.md) for component
boundaries and explicit non-goals.
