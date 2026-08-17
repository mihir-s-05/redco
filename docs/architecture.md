# Architecture

The code has three inward-pointing layers:

```text
analysis
   |
   v
algorithms and replay environment
   |
   v
contracts and integrity helpers
```

| Package | Responsibility |
|---|---|
| `redco.contracts` | Canonical immutable research values |
| `redco.integrity` | SHA-256 and digest validation |
| `redco.algo` | Branching and training calculations |
| `redco.env` | Artifacts, tracing, caching, commands, and replay |
| `redco.analysis` | Small experiments and diagnostics |

Core packages do not import analyses. Imports must not perform filesystem,
network, provider, model, or GPU actions.

Keep behavior in one owner, prefer typed values and direct functions over
registries, and split files by scientific responsibility before they become hard
to review. Experimental outputs belong under ignored paths until a compact result
is deliberately promoted.
