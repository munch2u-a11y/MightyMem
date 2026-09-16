from __future__ import annotations

from context_loom.demo import run_demo
from context_loom.evaluation import PROJECTS, run_benchmark


def test_demo_proves_dynamic_context_and_capability_change() -> None:
    result = run_demo()
    change = result["dynamic_change"]
    assert result["integrity"] == "ok"
    assert len(set(change["context_ids"])) == 4
    assert "read-file" in change["capabilities_by_stage"]["inspect"]
    assert "write-file" in change["capabilities_by_stage"]["act"]
    assert "read-file" in change["capabilities_by_stage"]["verify"]
    assert "report-result" in change["capabilities_by_stage"]["report"]
    assert change["source_read_in_action"] is True
    assert change["write_receipt_in_verify"] is True
    assert change["reread_receipt_in_report"] is True
    assert all(receipt["verified"] for receipt in result["receipts"].values())
    assert all(
        packet["token_count"] <= packet["max_input_tokens"]
        for packet in result["packets"].values()
    )
    assert result["packets"]["verify"]["metadata"]["historical_query"] is False
    assert result["final_task"]["stage"] == "report"
    assert result["final_task"]["status"] == "ready_to_report"


def test_full_deterministic_benchmark() -> None:
    result = run_benchmark()
    assert result["projects"] == len(PROJECTS) == 24
    assert result["cases_per_arm"] == 240
    assert result["total_compilations"] == 720
    assert result["integrity"] == "ok"

    recent = result["summary"]["recent_only"]
    semantic = result["summary"]["semantic_recent"]
    loom = result["summary"]["context_loom"]
    # The recency baseline is no longer capped at a top-k, so it is a genuinely
    # stronger control than it used to be -- it recalls real evidence now.  The
    # claim under test is that Loom still beats it, not that it scores zero.
    assert loom["evidence_micro_recall"] > recent["evidence_micro_recall"]
    assert loom["evidence_micro_recall"] == 1
    assert loom["full_evidence_hit_rate"] == 1
    assert loom["evidence_micro_recall"] >= semantic["evidence_micro_recall"]
    assert loom["mean_evidence_lane_precision"] >= semantic["mean_evidence_lane_precision"]
    for metric in (
        "tool_recall", "fragment_or_focus_fidelity", "prohibited_exclusion_rate",
        "scope_isolation_rate", "budget_compliance_rate", "current_input_last_rate",
    ):
        assert loom[metric] == 1


def test_benchmark_rows_have_auditable_invariants() -> None:
    result = run_benchmark()
    loom_rows = [row for row in result["rows"] if row["arm"] == "context_loom"]
    assert all(row["budget_ok"] and row["current_input_last"] for row in loom_rows)
    assert all(row["scope_ok"] and row["prohibited_ok"] for row in loom_rows)
    assert {row["category"] for row in loom_rows} == set(result["by_category"]["context_loom"])
