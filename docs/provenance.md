# Provenance and recovery

This is a research repository. Git history is the authoritative archive for old
experiments; the current checkout is reserved for active, understandable work.

## Normalized history catalog

`provenance/history-v1.jsonl` describes every regular tracked file at pre-cleanup
commit `53a7c67c9cb6df39e44454f364aaf3c9ca352966`.

The first row records the source commit, source tree, file count, and aggregate
row digest. Every later row records:

- path and role;
- Git blob;
- raw SHA-256 and byte count;
- file format and executable bit;
- safe primitive top-level schema metadata when available.

The catalog is an index, not a rewritten evidence format. It intentionally omits
nested authority values, credentials, provider identifiers, scientific payloads,
and inferred equivalence.

## Recover a retired file

Inspect or restore exact historical bytes with Git:

```console
git show 53a7c67c9cb6df39e44454f364aaf3c9ca352966:PATH
git restore --source=53a7c67c9cb6df39e44454f364aaf3c9ca352966 -- PATH
```

Restoration makes the file active again and therefore requires a normal review.
Do not rewrite old signed payloads and claim they are the original evidence.

## Local artifacts

Ignored `runs/`, `.redco/`, and `.artifacts/` trees may contain valuable local
research evidence. They are not part of the lean source checkout. Large retained
units should be copied to archival storage with a SHA-256 manifest before their
local working copies are removed.
