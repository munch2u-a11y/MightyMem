"""Historical standalone compiler diagnostic; this is not a product benchmark."""
from __future__ import annotations

import json
import math
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .capabilities import default_capabilities
from .compiler import CompilerConfig, ContextCompiler
from .engine import ContextLoom
from .models import EvidenceStatus, MemoryRecord, RecordKind
from .retrieval import HashingSemanticEmbedder, RecencyLane, SemanticLane
from .store import LoomStore


PROJECTS = [
    "Maple", "Pebble", "Cedar", "Orchid", "Harbor", "Juniper", "Nimbus", "Copper",
    "Willow", "Quartz", "Saffron", "Falcon", "Kestrel", "Ember", "Mosaic", "Atlas",
    "Birch", "Lagoon", "Pioneer", "Solstice", "Linden", "Aurora", "Comet", "Meadow",
]


@dataclass(slots=True)
class BenchmarkCase:
    id: str
    category: str
    scope: str
    query: str
    now: float
    gold_record_ids: tuple[str, ...] = ()
    prohibited_record_ids: tuple[str, ...] = ()
    task_id: str | None = None
    expected_capability_ids: tuple[str, ...] = ()
    expected_fragments: tuple[str, ...] = ()


def _seed_benchmark(store: LoomStore) -> list[BenchmarkCase]:
    embedder = HashingSemanticEmbedder()
    loom = ContextLoom(
        ":memory:",
        install_default_capabilities=False,
    )
    # Use the caller's store while retaining high-level seed helpers.
    loom.store.close()
    loom.store = store
    loom.compiler = ContextCompiler(store, embedding_provider=embedder)
    for capability in default_capabilities():
        store.register_capability(capability)
    cases: list[BenchmarkCase] = []
    for index, name in enumerate(PROJECTS):
        scope = name.lower()
        base = 10_000.0 + index * 1_000
        old = loom.remember_user_statement(
            f"Project {name} originally used PostgreSQL {10 + index % 5}.",
            scope=scope,
            record_id=f"{scope}-database-old",
            created_at=base,
            effective_at=base,
            importance=0.65,
        )
        new_version = 16 + index % 4
        correction = loom.correct(
            old.id,
            f"Project {name} now uses PostgreSQL {new_version}; migration completed.",
            scope=scope,
            record_id=f"{scope}-database-current",
            created_at=base + 10,
            effective_at=base + 10,
            importance=0.82,
        )
        negative = loom.remember_user_statement(
            f"The {name} archive has not been deleted and must remain available.",
            scope=scope,
            record_id=f"{scope}-archive-status",
            created_at=base + 20,
            importance=0.75,
        )
        deadline = loom.remember_user_statement(
            f"Project {name}'s review port is {5200 + index} and its deadline is 2026-10-{(index % 20) + 1:02d}.",
            scope=scope,
            record_id=f"{scope}-exact-values",
            created_at=base + 30,
            importance=0.7,
        )
        location = loom.remember_user_statement(
            f"The launch venue for Project {name} is the north workshop.",
            scope=scope,
            record_id=f"{scope}-location",
            created_at=base + 40,
            importance=0.6,
        )
        preference = loom.remember_preference(
            f"For {name}, I prefer compact Markdown with citations and an uncertainty line.",
            scope=scope,
            record_id=f"{scope}-preference",
            created_at=base + 50,
            importance=0.98,
        )
        read = loom.record_tool_result(
            f"Read {scope}/notes.md with sha256={index:064x}.",
            scope=scope,
            verified=True,
            receipt_id=f"receipt-{scope}-read",
            verifier="content_and_sha256",
            now=base + 60,
        )
        procedure = loom.remember_procedure(
            f"For Project {name} summaries, inspect the source, write a concise Markdown file, "
            "reread it and compare hashes before reporting completion.",
            trajectory_ids=(read.id,),
            reliability=0.9,
            verified=True,
            scope=scope,
            record_id=f"{scope}-summary-procedure",
            created_at=base + 70,
            importance=0.8,
        )
        # Same-topic material creates a real ranking problem: a useful compiler
        # must find the current, exact, or procedural record without packing the
        # entire topical neighborhood into the prompt.
        for distractor_index in range(16):
            loom.remember_user_statement(
                f"Archived planning note {distractor_index} for Project {name}: database migration, "
                f"PostgreSQL compatibility, launch venue, review service port {6100 + distractor_index}, "
                "summary formatting, and file verification were discussed but not adopted.",
                scope=scope,
                record_id=f"{scope}-topical-distractor-{distractor_index}",
                created_at=base + 71 + distractor_index * 0.1,
                importance=0.08,
                metadata={"applicability": "rejected"},
            )
        # A tempting but unsupported model-created claim must stay out of ordinary evidence.
        loom.record(
            RecordKind.FACT,
            f"Project {name} owns an interplanetary spaceport.",
            scope=scope,
            record_id=f"{scope}-unsupported",
            created_at=base + 80,
            source_ids=(location.id,),
            evidence_status=EvidenceStatus.MODEL_INFERRED,
            metadata={"support_assessment": "unchecked"},
            importance=1.0,
        )
        # Newer conversational noise makes plain recent windows weak.
        for noise_index in range(10):
            loom.remember_user_statement(
                f"Routine aside {noise_index} for {name} about coffee and weather.",
                scope=scope,
                record_id=f"{scope}-aside-{noise_index}",
                created_at=base + 100 + noise_index,
                importance=0.05,
            )
        inspect_task = loom.begin_task(
            f"{scope}-inspect-task",
            f"Inspect Project {name}'s notes and establish its current technical state.",
            scope=scope,
            stage="inspect",
            focus_stack=["read source notes", "resolve versions", "retain the archive exception"],
            open_questions=[f"Which database version is current for {name}?"],
            completion_conditions=["source read receipt exists", "current and historical state separated"],
            now=base + 150,
        )
        act_task = loom.begin_task(
            f"{scope}-act-task",
            f"Create a verified summary file for Project {name}.",
            scope=scope,
            stage="act",
            focus_stack=["write summary", "reread", "compare hash"],
            completion_conditions=["write receipt", "matching read-back"],
            mandatory_capability_ids=["write-file"],
            now=base + 151,
        )
        now = base + 200
        cases.extend(
            [
                BenchmarkCase(
                    f"{scope}-current", "current_correction", scope,
                    f"Which PostgreSQL version does Project {name} use now?", now,
                    (correction.id,), (old.id,), expected_fragments=(str(new_version),),
                ),
                BenchmarkCase(
                    f"{scope}-history", "historical_state", scope,
                    f"What database did Project {name} use originally, and what does it use now?", now,
                    (old.id, correction.id), expected_fragments=(str(10 + index % 5), str(new_version)),
                ),
                BenchmarkCase(
                    f"{scope}-negation", "negation", scope,
                    f"Has the {name} archive been deleted?", now,
                    (negative.id,), expected_fragments=("not",),
                ),
                BenchmarkCase(
                    f"{scope}-exact", "exact_values", scope,
                    f"What are the review port and deadline for Project {name}?", now,
                    (deadline.id,), expected_fragments=(str(5200 + index), f"2026-10-{(index % 20) + 1:02d}"),
                ),
                BenchmarkCase(
                    f"{scope}-semantic", "semantic_paraphrase", scope,
                    f"Where will Project {name} launch?", now,
                    (location.id,), expected_fragments=("north workshop",),
                ),
                BenchmarkCase(
                    f"{scope}-personal", "subjective_preference", scope,
                    f"How should you format the {name} deliverable for me?", now,
                    (preference.id,), expected_fragments=("Markdown", "uncertainty"),
                ),
                BenchmarkCase(
                    f"{scope}-procedure", "verified_procedure", scope,
                    f"How should you create and verify the Project {name} summary?", now,
                    (procedure.id,), expected_fragments=("reread", "hash"),
                ),
                BenchmarkCase(
                    f"{scope}-focus", "focus_continuity", scope,
                    f"Continue inspecting {name} after our unrelated conversation.", now,
                    task_id=inspect_task.id,
                    expected_capability_ids=("read-file",),
                    expected_fragments=("read source notes", "resolve versions"),
                ),
                BenchmarkCase(
                    f"{scope}-action", "phase_tool_admission", scope,
                    f"Create and save the Project {name} summary file now.", now,
                    task_id=act_task.id,
                    expected_capability_ids=("write-file",),
                ),
                BenchmarkCase(
                    f"{scope}-unsupported", "unsupported_claim_gate", scope,
                    f"Does Project {name} own an interplanetary spaceport?", now,
                    prohibited_record_ids=(f"{scope}-unsupported",),
                ),
            ]
        )
    return cases


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def run_benchmark(database: str | Path | None = None) -> dict:
    temporary = None
    if database is None:
        temporary = tempfile.TemporaryDirectory(prefix="context-loom-eval-")
        database = Path(temporary.name) / "benchmark.sqlite"
    store = LoomStore(database)
    try:
        cases = _seed_benchmark(store)
        provider = HashingSemanticEmbedder()
        config = CompilerConfig(model_context_tokens=720, reserved_output_tokens=180, max_visible_capabilities=3)
        arms = {
            "recent_only": ContextCompiler(
                store, config=config, embedding_provider=provider, lanes=[RecencyLane()]
            ),
            "semantic_recent": ContextCompiler(
                store,
                config=config,
                embedding_provider=provider,
                lanes=[SemanticLane(provider), RecencyLane()],
            ),
            "context_loom": ContextCompiler(store, config=config, embedding_provider=provider),
        }
        rows = []
        for arm_name, compiler in arms.items():
            for case in cases:
                packet = compiler.compile(
                    case.query,
                    scope=case.scope,
                    task_id=case.task_id,
                    now=case.now,
                )
                admitted = {
                    source_id
                    for entry in packet.manifest
                    if entry.admitted
                    for source_id in entry.source_record_ids
                }
                visible = {capability.id for capability in packet.capabilities}
                gold = set(case.gold_record_ids)
                recalled = len(gold & admitted)
                admitted_records = {record_id for record_id in admitted if store.get_record(record_id)}
                evidence_lane_records = {
                    source_id
                    for entry in packet.manifest
                    if entry.admitted and set(entry.lanes) - {"recent"}
                    for source_id in entry.source_record_ids
                    if store.get_record(source_id)
                }
                relevant_admitted = len(gold & admitted_records)
                focus_ok = all(fragment in packet.rendered for fragment in case.expected_fragments)
                tool_ok = set(case.expected_capability_ids) <= visible
                prohibited_ok = not (set(case.prohibited_record_ids) & admitted)
                scope_ok = all(store.get_record(record_id).scope == case.scope for record_id in admitted_records)
                rows.append(
                    {
                        "arm": arm_name,
                        "case_id": case.id,
                        "category": case.category,
                        "token_count": packet.token_count,
                        "budget": packet.max_input_tokens,
                        "budget_ok": packet.token_count <= packet.max_input_tokens,
                        "current_input_last": packet.rendered.endswith(case.query),
                        "gold_count": len(gold),
                        "gold_recalled": recalled,
                        "full_evidence_hit": not gold or recalled == len(gold),
                        "record_precision": (
                            relevant_admitted / len(admitted_records) if admitted_records and gold else None
                        ),
                        "evidence_lane_precision": (
                            len(gold & evidence_lane_records) / len(evidence_lane_records)
                            if evidence_lane_records and gold else None
                        ),
                        "tool_ok": tool_ok,
                        "focus_or_fragment_ok": focus_ok,
                        "prohibited_ok": prohibited_ok,
                        "scope_ok": scope_ok,
                        "visible_capabilities": sorted(visible),
                        "admitted_record_ids": sorted(admitted_records),
                    }
                )
        summary = {}
        for arm_name in arms:
            arm_rows = [row for row in rows if row["arm"] == arm_name]
            evidence_rows = [row for row in arm_rows if row["gold_count"]]
            tool_rows = [row for row in arm_rows if any(
                case.id == row["case_id"] and case.expected_capability_ids for case in cases
            )]
            precision_values = [row["record_precision"] for row in evidence_rows if row["record_precision"] is not None]
            evidence_precision_values = [
                row["evidence_lane_precision"]
                for row in evidence_rows
                if row["evidence_lane_precision"] is not None
            ]
            tokens = [float(row["token_count"]) for row in arm_rows]
            summary[arm_name] = {
                "cases": len(arm_rows),
                "evidence_micro_recall": sum(row["gold_recalled"] for row in evidence_rows)
                / max(1, sum(row["gold_count"] for row in evidence_rows)),
                "full_evidence_hit_rate": sum(row["full_evidence_hit"] for row in evidence_rows)
                / max(1, len(evidence_rows)),
                "mean_record_precision": statistics.mean(precision_values) if precision_values else 0.0,
                "mean_evidence_lane_precision": (
                    statistics.mean(evidence_precision_values) if evidence_precision_values else 0.0
                ),
                "tool_recall": sum(row["tool_ok"] for row in tool_rows) / max(1, len(tool_rows)),
                "fragment_or_focus_fidelity": sum(row["focus_or_fragment_ok"] for row in arm_rows)
                / len(arm_rows),
                "prohibited_exclusion_rate": sum(row["prohibited_ok"] for row in arm_rows) / len(arm_rows),
                "scope_isolation_rate": sum(row["scope_ok"] for row in arm_rows) / len(arm_rows),
                "budget_compliance_rate": sum(row["budget_ok"] for row in arm_rows) / len(arm_rows),
                "current_input_last_rate": sum(row["current_input_last"] for row in arm_rows) / len(arm_rows),
                "mean_tokens": statistics.mean(tokens),
                "p95_tokens": _percentile(tokens, 0.95),
            }
        by_category = {}
        for arm_name in arms:
            by_category[arm_name] = {}
            for category in sorted({case.category for case in cases}):
                group = [row for row in rows if row["arm"] == arm_name and row["category"] == category]
                by_category[arm_name][category] = {
                    "cases": len(group),
                    "full_evidence_hit_rate": sum(row["full_evidence_hit"] for row in group) / len(group),
                    "fragment_or_focus_fidelity": sum(row["focus_or_fragment_ok"] for row in group) / len(group),
                    "tool_recall": sum(row["tool_ok"] for row in group) / len(group),
                    "mean_tokens": statistics.mean(row["token_count"] for row in group),
                }
        return {
            "benchmark": "context-loom-synthetic-v1",
            "projects": len(PROJECTS),
            "cases_per_arm": len(cases),
            "total_compilations": len(rows),
            "configuration": asdict(config),
            "embedding_provider": provider.identity,
            "token_counter": "regex-v1",
            "summary": summary,
            "by_category": by_category,
            "rows": rows,
            "integrity": store.integrity_check(),
            "limitations": [
                "Synthetic evidence-selection benchmark; it does not measure language-model answer quality.",
                "The deterministic hashing provider includes an auditable concept lexicon; it is not a learned sentence embedder.",
                "All arms use the same phase-aware capability selector, so the benchmark tests tool admission but not a learned router.",
            ],
        }
    finally:
        store.close()
        if temporary is not None:
            temporary.cleanup()


def write_benchmark(result: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
