# Provenance

Git history is the archive for retired experiments. The active checkout contains
only maintained code and compact results.

`provenance/history-v1.jsonl` indexes every regular file from the large historical
snapshot at commit `53a7c67c9cb6df39e44454f364aaf3c9ca352966`. Each record includes
the path, Git blob, raw SHA-256, byte count, file format, and executable bit.

Inspect or restore exact historical bytes with:

```console
git show 53a7c67c9cb6df39e44454f364aaf3c9ca352966:PATH
git restore --source=53a7c67c9cb6df39e44454f364aaf3c9ca352966 -- PATH
```

Restored files become active code and require normal review. Ignored local
`runs/`, `.redco/`, and `.artifacts/` directories are not part of the Git archive;
preserve important local outputs separately with a SHA-256 manifest.
