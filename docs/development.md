# Development

Use `uv`, never `pip`. The active project has no mandatory runtime dependencies;
development dependencies are locked in `uv.lock`.

## Verification

```console
uv sync --frozen
uv run --frozen pytest
uv run --frozen ruff check src tests
uv run --frozen mypy
git diff --check
```

Ordinary development is offline and CPU-only. Tests use temporary directories
for outputs. They must not inspect or modify the user-owned `external/prime-rl`
checkout, retained historical evidence, provider state, models, or GPUs.

## Research workflow

1. Express the question in a small analysis or test.
2. Reuse `redco.contracts`, `redco.algo`, and `redco.env` rather than creating a
   versioned protocol stack.
3. Keep exploratory output under ignored `runs/` or `.artifacts/` paths.
4. Promote only the compact result needed to support the current research claim.
5. Delete abandoned analyses from the active tree; Git retains them.

A new experiment should not add readiness builders, authorization schemas,
terminal evidence codecs, or campaign-specific test frameworks unless the user
explicitly asks for production-grade operational assurance.
