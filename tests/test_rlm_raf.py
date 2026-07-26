from __future__ import annotations

from redco.analysis.rlm_raf import run_rlm_raf_campaign
from redco.env.dynamic_replay import EventRole


def test_rlm_shaped_dynamic_topology_campaign_passes() -> None:
    report = run_rlm_raf_campaign(
        seed=23,
        programs=100,
        alternatives_per_program=3,
    )

    assert report.passed_rlm_shaped_cpu_proxy
    assert report.paired_branches == 300
    assert report.deterministic_failures == 0
    assert report.topology_divergences > 0
    assert report.added_events > 0
    assert report.removed_events > 0
    assert report.sliced_suffix_events < report.full_suffix_events
    assert report.modeled_sliced_policy_token_raf < report.modeled_full_policy_token_raf
    assert report.empirical_real_trace_status == "pending_recorded_or_live_rlm_traces"
    assert set(report.full_work_by_role) == {role.value for role in EventRole}
