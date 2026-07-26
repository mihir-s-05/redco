# ReDCO RLM trace audit

A minimal native `verifiers.v1` taskset for checking whether real RLM rollouts
carry the prompt, action, graph, usage, and timing fields required by ReDCO.
It uses the built-in `RLMHarness`; no custom harness is provided.

## Develop

From the pinned verifiers checkout, run with uv and an editable taskset:

```bash
uv run --with-editable /path/to/redco/environments/redco_rlm_trace_v1 \
  validate redco-rlm-trace-v1 -n 4 --runtime.type subprocess --rich false
```

For a running prime-rl vLLM endpoint, the packaged runner selects verifiers'
built-in RLM harness, pins its source revision, sets `max_depth = 1`, and uses
the train client so the trace contains exact token IDs:

```bash
cd external/prime-rl/deps/verifiers
UV_PROJECT_ENVIRONMENT=/tmp/redco-verifiers-env uv run --frozen --no-dev --python 3.12 \
  --with-editable /path/to/redco/environments/redco_rlm_trace_v1 \
  python -m redco_rlm_trace_v1.run_audit \
  --output-dir /path/to/redco/runs/stage-b/rlm-trace-audit/live
```

The saved `traces.jsonl` is the input to
`redco.analysis.verifiers_trace_audit`.

This taskset is an instrumentation probe, not a learning benchmark. Its marker
reward only confirms that the harness completed the requested data lookup.
