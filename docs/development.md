# Development

Use Python 3.12 or newer and `uv`.

```console
uv sync --frozen
uv run --frozen pytest
uv run --frozen ruff check src tests
uv run --frozen mypy
git diff --check
```

The package has no mandatory runtime dependencies. Ordinary tests are offline and
CPU-only, use temporary output directories, and must not access provider state,
models, GPUs, or the user-owned `external/prime-rl` checkout.

For research changes:

1. State the question and failure criterion first.
2. Add the smallest analysis or test that can answer it.
3. Reuse `redco.contracts`, `redco.algo`, and `redco.env`.
4. Keep full outputs ignored and promote only compact results.
5. Remove abandoned experiment code; recover it from Git if needed later.
