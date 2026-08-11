# Provenance catalog

Redco preserves historical artifact bytes exactly. It does not rewrite old
configs, reports, datasets, environments, patches, or retained run records into
a new schema: signatures, hashes, Git blobs, newline conventions, and historical
claims all refer to the original bytes.

`history-v1.jsonl` is a normalized **retirement catalog** for the complete
tracked repository at its recorded source commit and tree. Each file row has one
stable shape:

- repository path and role (`source`, `test`, `command`, `intent`, `dataset`,
  `environment`, `patch`, `evidence`, `retained-run`, or project metadata);
- exact Git blob and raw SHA-256;
- byte count, format, and executable bit;
- only safe primitive top-level metadata such as `schema_version`, `domain`,
  `state`, `status`, `disposition`, and `purpose` when present.

The catalog deliberately omits credentials, provider identifiers, nested
authority objects, scientific contents, and inferred equivalence. A catalog row
does not replace, authorize, supersede, or reinterpret its payload.

The historical payload can be recovered exactly with Git, for example:

```console
git show SOURCE_COMMIT:PATH
```

The one-time generator read committed Git blobs directly, so checkout newline
conversion could not change catalog identities. The first JSONL row binds the
source commit, source tree, file count, and SHA-256 of all subsequent canonical
rows. The generator was intentionally not retained: the catalog is a recovery
index, not a second provenance framework. Local ignored evidence remains
separate from this reproducible tracked snapshot.
