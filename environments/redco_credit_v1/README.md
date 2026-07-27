# redco-credit-v1

This is the restricted Stage-C environment. Each episode first samples one
untargeted root routing decision, then precommits one eligible depth-one sub-call
target and runs four same-policy seats from the identical action prompt: one
original and three alternatives. Every output, including an invalid one, is
executed by the finite probe and receives its true deterministic reward; nothing
is rejected or resampled. The root route adds a deterministic background reward,
so the suite measures whether local branch credit recovers target signal while
trajectory broadcast must also resolve upstream variation.

Both full-suffix and graph-sliced replay are evaluated for every action. They must
produce the exact same reward before the configured arm's result is accepted.
The environment records branch identity, action source, pre-action selection
features, replay mode, equivalence, and cost meters in each native verifiers trace.
The restricted graph's declared work model contains four target-independent
post-target audit events: full-suffix replay would revisit them, while the sliced
arm restores them and executes only the two target descendants. Its 1/3 logical
work fraction is logged separately from actual evaluation cost. Because this gate
executes both reward paths on every action to enforce in-loop equivalence, it is
not a wall-time or GPU-savings claim; Stage B supplies the empirical RAF evidence.

This establishes estimator behavior on a deliberately restricted depth-one
substrate. It does not claim general Python or multi-turn RLM transfer.
