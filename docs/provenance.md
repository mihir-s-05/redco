# Provenance and retention

Configs, reports, datasets, runs, patches, tests, and environments form Redco's
audit trail. Age, failure, a newer version, or ignored Git status does not make
bytes disposable. Preserve scientific meaning, authority boundaries, and exact
historical claims.

## Evidence and bindings

Preregistrations freeze intent; amendments change it explicitly; protocols and
datasets are executable inputs; manifests bind members; audits and terminal
reports record decisions; receipts and ledgers bind authority/lifecycle; patches
and environments bind runtime deltas. Retain negative and superseded evidence,
not only successful endpoints. A digest is not itself a recoverable copy.

Before changing or deleting a path, classify four distinct reference classes:

1. tracked live/runtime/build/CI/import/subprocess/package references;
2. tracked immutable provenance, path, Git-blob, or exact-byte bindings;
3. ignored/untracked narrative, run, archive, cache, or evidence references;
4. Git-history-only recovery, proving the local object exists, the commit/path is
   reachable from retained history, and the recovered bytes match the digest.

Search path, basename, node ID, raw SHA-256, builders, check-only owners,
manifests, archives, fixture imports, and ignored evidence. “Historical” and
“unreferenced import” are not deletion proofs.

Bindings have different address modes:

- A historical digest or Git blob freezes recorded bytes while a reviewed later
  source path may evolve.
- A live-worktree binding freezes the current path until a successor retires the
  promise.
- A versioned evidence path is enduring; change semantics under a new path and
  domain with an explicit predecessor relationship.

Never rewrite old evidence to match current code. A successor preserves the old
object, binds its predecessor, states whether it repairs/supersedes/extends it,
and authenticates both sides. When a repair receipt must bind a reviewed commit,
create it in a direct child; do not make a self-referential same-commit claim.
Mass-formatting historical JSON is forbidden because canonical no-newline,
newline-terminated, and pretty-printed families have distinct identities.

Exact-bound tests require the same path/node/digest/import scan before edits.
An independent hash, signature, canonicalization, or safe-path oracle must not
call the production helper it is intended to verify.

## Authenticated exceptions

- `production.log` and `production.stderr` in the Stage-D production-replay CPU
  evidence are equal but separately listed output channels; retain both.
- The two 2026-07-29 Stage-C4 warmstart hardware/preregistration audit files are
  equal but represent separate review events; retain both.
- `reports/rlm-recorded-raf-projection-result-2026-07-26.json` records
  `c95bc5c5dc64261edea8470542a890dbaa9ae20d7a1ab0e00ad2b6b4ac1d357d`
  for the ignored raw RAF projection payload, not the condensed report itself.
  Preserve that digest indirection rather than reinterpreting the old field.

The frozen Stage-C4 V2–V4 verifier also preserves known legacy weaknesses:
manifest paths/duplicates/link escape and completeness are not fully rejected;
SFT JSON accepts non-finite values and duplicate steps overwrite; empty renderer
`checks` can pass through `all([])`. Repair only in a versioned successor.

## Known Phase-A trust defect

The V13 Phase-A v1 approval anchor correctly binds Foundation-F registry and
behavior bytes. Later reviewed selection work changed the live registry,
decoder, tests, collection, and publication owners. The later lineage does not
supersede the v1 anchor, so current live Phase-A build/check-only paths fail
closed against the historical anchor.

Do not update the old anchor or Foundation hashes. A safe repair must either
verify v1 against exact Foundation Git blobs, introduce an independently
reviewed current-anchor v2, or retire the live builder while retaining a
historical verifier.

## Local evidence and generated data

A clean clone omits most `runs/`, `.redco/`, `.artifacts/`, nested narrative
reports, vendor snapshots, and lifecycle state. Tests must report absent
retained evidence and must not fetch or regenerate it. Synthetic fixtures may
test mechanics but cannot claim to authenticate a historical run.

`.pytest_cache`, `.mypy_cache`, `.ruff_cache`, coverage files, bytecode, and an
explicitly obsolete environment are generated residue. Ignored runs, archives,
ledgers, manifests, and vendor trees are not caches. Retain or archive them as
complete authenticated units unless redundancy or obsolescence is proven.
