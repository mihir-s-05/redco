import verifiers.v1 as vf


class RedcoRlmTraceData(vf.TaskData):
    """One deterministic trace-audit prompt."""

    answer: str


class RedcoRlmTraceTask(vf.Task[RedcoRlmTraceData]):
    """Score the marker and persist native trace-coverage counters."""

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del runtime
        linked_calls = sum(call.node is not None for call in trace.calls)
        tokenized_nodes = sum(bool(node.token_ids) for node in trace.nodes)
        sampled_nodes = sum(node.sampled for node in trace.nodes)
        trace.info["redco_trace_audit"] = {
            "model_calls": len(trace.calls),
            "linked_model_calls": linked_calls,
            "message_nodes": len(trace.nodes),
            "tokenized_nodes": tokenized_nodes,
            "sampled_nodes": sampled_nodes,
            "branches": len(trace.branches),
        }

    @vf.reward(weight=1.0)
    async def reward(self, trace: vf.Trace) -> float:
        return float(self.data.answer in trace.last_reply)

    @vf.metric
    async def linked_call_fraction(self, trace: vf.Trace) -> float:
        if not trace.calls:
            return 0.0
        return sum(call.node is not None for call in trace.calls) / len(trace.calls)

    @vf.metric
    async def tokenized_node_fraction(self, trace: vf.Trace) -> float:
        if not trace.nodes:
            return 0.0
        return sum(bool(node.token_ids) for node in trace.nodes) / len(trace.nodes)


class RedcoRlmTraceConfig(vf.TasksetConfig):
    num_tasks: int = 4
    """How many tasks to build."""


class RedcoRlmTraceTaskset(vf.Taskset[RedcoRlmTraceTask, RedcoRlmTraceConfig]):
    def load(self) -> list[RedcoRlmTraceTask]:
        tasks: list[RedcoRlmTraceTask] = []
        for index in range(self.config.num_tasks):
            answer = f"REDCO-{index:03d}"
            prompt = (
                "This is an RLM trace-instrumentation task. Use the available "
                "IPython workflow and make at least one programmatic recursive "
                "model call before answering. Inspect the following records and "
                f"return the marker attached to the largest value: "
                f"[('ignore', {index}), ('target', {100 + index}, '{answer}'), "
                f"('other', {50 + index})]. End with the marker {answer}."
            )
            data = RedcoRlmTraceData(
                idx=index,
                prompt=prompt,
                answer=answer,
            )
            tasks.append(RedcoRlmTraceTask(data, self.config.task))
        return tasks
