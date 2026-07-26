# Local run artifacts

Paid-run outputs are synchronized back under this directory before compute is
terminated. The contents are intentionally ignored by Git because logs, rollout
records, and checkpoints can be large; this README remains tracked.

Each run directory must contain the resolved Prime-RL configs, command transcript,
component logs, metrics, pin manifest, patch hash, resource metadata, and terminal
status. Smoke runs omit model checkpoints unless checkpoint behavior is itself
under test.
