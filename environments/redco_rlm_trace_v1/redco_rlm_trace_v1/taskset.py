from typing import Literal

import verifiers.v1 as vf

PROMPT_PROTOCOL = "forced-recursion-v2"
MULTI_CHILD_PROMPT_PROTOCOL = "forced-four-child-recursion-v2"


class RedcoRlmTraceData(vf.TaskData):
    """One deterministic trace-audit prompt."""

    answer: str
    prompt_protocol: str


class RedcoRlmTraceTask(vf.Task[RedcoRlmTraceData]):
    """Score the marker and persist native trace-coverage counters."""

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del runtime
        linked_calls = sum(call.node is not None for call in trace.calls)
        structured_calls = [
            call for call in trace.calls if getattr(call, "rlm", None) is not None
        ]
        exact_recursive_calls = [
            call
            for call in structured_calls
            if call.rlm is not None
            and call.rlm.parent_session_id is not None
            and call.rlm.parent_turn is not None
        ]
        tokenized_nodes = sum(bool(node.token_ids) for node in trace.nodes)
        sampled_nodes = sum(node.sampled for node in trace.nodes)
        trace.info["redco_trace_audit"] = {
            "prompt_protocol": self.data.prompt_protocol,
            "model_calls": len(trace.calls),
            "linked_model_calls": linked_calls,
            "structured_model_calls": len(structured_calls),
            "exact_recursive_model_calls": len(exact_recursive_calls),
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
    profile: Literal["single", "multi_child"] = "single"
    """Trace shape requested by the instrumentation prompt."""


class RedcoRlmTraceTaskset(vf.Taskset[RedcoRlmTraceTask, RedcoRlmTraceConfig]):
    def load(self) -> list[RedcoRlmTraceTask]:
        tasks: list[RedcoRlmTraceTask] = []
        for index in range(self.config.num_tasks):
            if self.config.profile == "multi_child":
                markers = [
                    f"REDCO-{index:03d}-{suffix}"
                    for suffix in ("A", "B", "C", "D")
                ]
                answer = "|".join(markers)
                prompts = [
                    f"Return only the marker {marker}" for marker in markers
                ]
                one_line_call = (
                    "import asyncio; child_results = await asyncio.gather("
                    f"*(rlm(p) for p in {prompts!r})); "
                    'print("|".join(item.answer for item in child_results))'
                )
                prompt = (
                    "This is a recursive map-reduce instrumentation protocol. "
                    "Your first action MUST be one IPython tool call. In that call, "
                    "execute this exact single-line Python statement before doing "
                    f"anything else: {one_line_call}\n"
                    "Do not answer until all four recursive calls return. Then "
                    f"return exactly this combined marker: {answer}"
                )
                protocol = MULTI_CHILD_PROMPT_PROTOCOL
            else:
                answer = f"REDCO-{index:03d}"
                prompt = (
                    "This is an instrumentation protocol, not a reasoning test. "
                    "Your first action MUST be an IPython tool call. In that call, "
                    "execute exactly this Python statement before doing anything else: "
                    f'result = await rlm("Return only the marker {answer}")\n'
                    "Then print result.answer in IPython. Do not answer the user until "
                    "that recursive call has returned. Afterward, inspect these records "
                    "and return the marker attached to the largest value: "
                    f"[('ignore', {index}), "
                    f"('target', {100 + index}, '{answer}'), "
                    f"('other', {50 + index})]. End with the marker {answer}."
                )
                protocol = PROMPT_PROTOCOL
            data = RedcoRlmTraceData(
                idx=index,
                prompt=prompt,
                answer=answer,
                prompt_protocol=protocol,
            )
            tasks.append(RedcoRlmTraceTask(data, self.config.task))
        return tasks
