# Development and verification

Use `uv`, never `pip`. Keep `uv.lock` stable unless a dependency change is
separately authorized and reviewed. Windows and WSL virtual environments are
not binary-compatible; use a platform-specific environment. Put large temporary
artifacts on `D:` or in the operating-system temporary directory.

Ordinary development is offline and CPU-only. It does not authorize network,
Prime, provider/model, Parquet-source, GPU, wallet, provisioning, training, or
scientific execution. Do not touch user-owned `external/prime-rl` or a paused
automation merely because a test can discover it.

## Canonical verifier

The repository verifier owns root/`src`/`scripts` and reviewed local-environment
import paths plus exact test membership. From an already provisioned environment
whose uv cache contains the explicit extra below, run:

```console
uv run --offline --frozen --with jsonschema==4.26.0 python scripts/verify_repository.py pytest
```

The default `required-green` profile fails on regressions. Other pytest profiles
are selected with `pytest --profile NAME`:

| Class | Meaning |
|---|---|
| `required-green` | Portable checks that must execute and pass; includes safe Prime/security nodes |
| `platform-linux`, `platform-windows` | Real OS process/filesystem contracts, reported separately |
| `optional-stack` | Imports and reports every separately supplied optional module before collection; missing imports or any non-pass from an incompatible stack exit nonzero |
| `retained-provenance` | Checks requiring authenticated retained-local evidence |
| `inherited-known-failure` | Explicit baseline sentinels; never counted as required green |
| refused profiles | Protected monitor, user-owned external checkout, authenticated source/data-stack/Windows material |

The protected capacity-monitor test is never collected; its contract and audit
digest are authenticated directly. Every executed row reports selected, passed,
failed, errored, skipped, deselected, protected, and unavailable/preignored
counts. Preflight-unavailable and refused rows instead name every missing or
prohibited prerequisite and exit nonzero before collection. The verifier clears
and reports ambient `PYTEST_ADDOPTS` and `PYTEST_PLUGINS`, disables third-party
pytest plugin autoload before importing pytest, and fails if final required
collection differs from its reviewed selected-node count or identity digest.
Use `pytest --collect-only` to authenticate that membership without executing
tests; arbitrary pytest argument passthrough is intentionally unsupported.

Static verification uses the same owner:

```console
uv run --offline --frozen python scripts/verify_repository.py ruff
uv run --offline --frozen python scripts/verify_repository.py mypy
uv run --offline --frozen python scripts/verify_repository.py compile
git diff --check
```

Ruff is repository-wide with exact-digest-gated exceptions for three frozen
files. Mypy is strict for the affected production boundary; broad legacy mypy
remains classified inherited debt and is not falsely advertised as green.

## Verification by risk

| Change | Minimum evidence |
|---|---|
| Parser, path, or schema | Positive plus malformed, traversal, alias, and unexpected-field cases |
| Hash/canonical bytes | Golden bytes, mutation tests, historical digest and newline checks |
| Ledger/receipt | Transition, prior-hash, torn-write, lock, seal, and evidence tampering tests |
| Publication/builder | Two fresh roots, byte equality, check-only, no-mtime, alias and rollback tests |
| Deletion | Four-class reference scan, import/build/test scan, retained successor proof |
| Process boundary | Intended Windows/WSL/Linux process, signal, containment, and path checks |

Writers target an OS-temporary directory, never the repository evidence tree.
Check-only validates existing bytes without writes, missing-output creation, or
mtime changes. Never regenerate frozen evidence merely to satisfy changed code.

A clean clone and retained-local evidence are distinct matrices. Missing local
evidence must be reported, never downloaded or reconstructed implicitly. The
standalone `environments/redco_credit_v1` suite remains unavailable until it has
a separately reviewed lock. See [provenance](provenance.md) for retention and
known frozen exceptions.
