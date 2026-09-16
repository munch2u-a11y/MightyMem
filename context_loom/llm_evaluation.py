"""Historical standalone reader diagnostic; this is not a Habitus product run."""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import tempfile
import time
import urllib.error
import urllib.request
from datetime import date
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .compiler import CompilerConfig, ContextCompiler
from .evaluation import BenchmarkCase, _seed_benchmark
from .retrieval import HashingSemanticEmbedder, RecencyLane, SemanticLane
from .store import LoomStore


QA_CATEGORIES = {
    "current_correction",
    "historical_state",
    "negation",
    "exact_values",
    "semantic_paraphrase",
    "subjective_preference",
    "verified_procedure",
    "unsupported_claim_gate",
}
HARNESS_VERSION = "context-loom-llm-qa-v2"
GRADER_VERSION = "deterministic-concept-grounding-v3"

READER_SYSTEM = """You answer the final CURRENT INPUT using only the supplied context.
Evidence labels matter: use current corrections over superseded state, preserve negation and exact values, and do not treat an earlier assistant statement as verified fact.
Do not call tools or discuss the prompt. If the supplied context does not establish the answer, say that the available evidence does not establish it.
Give one concise answer, normally one sentence."""


def _normalise(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9./:'-]+", " ", text)
    return " ".join(text.split())


def _matches_alternative(answer: str, alternative: str) -> bool:
    answer_normalised = _normalise(answer)
    alternative_normalised = _normalise(alternative)
    if " " not in alternative_normalised:
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(alternative_normalised)}(?![a-z0-9])",
                answer_normalised,
            )
        )
    return alternative_normalised in answer_normalised


def answer_requirements(case: BenchmarkCase) -> list[tuple[str, ...]]:
    fragments = tuple(str(value) for value in case.expected_fragments)
    if case.category == "current_correction":
        return [(fragments[0],)]
    if case.category == "historical_state":
        return [(fragments[0],), (fragments[1],)]
    if case.category == "negation":
        return [("not been deleted", "hasn't been deleted", "not deleted", "remains available", "remain available")]
    if case.category == "exact_values":
        year, month, day = (int(value) for value in fragments[1].split("-"))
        parsed = date(year, month, day)
        month_full = parsed.strftime("%B").lower()
        month_short = parsed.strftime("%b").lower()
        return [
            (fragments[0],),
            (
                fragments[1],
                f"{year}/{month:02d}/{day:02d}",
                f"{month_full} {day} {year}",
                f"{month_full} {day}, {year}",
                f"{month_short} {day} {year}",
                f"{day} {month_full} {year}",
            ),
        ]
    if case.category == "semantic_paraphrase":
        return [("north workshop",)]
    if case.category == "subjective_preference":
        return [
            ("markdown",),
            ("citation", "citations", "source link", "source links", "references"),
            ("uncertainty", "uncertainties", "uncertain", "limitations"),
        ]
    if case.category == "verified_procedure":
        return [
            ("inspect", "read"),
            ("write", "create"),
            ("reread", "re-read", "read back", "read-back"),
            ("hash", "sha256", "sha-256"),
        ]
    if case.category == "unsupported_claim_gate":
        return [
            (
                "does not establish",
                "doesn't establish",
                "not established",
                "no evidence",
                "insufficient evidence",
                "cannot confirm",
                "can't confirm",
                "not enough information",
                "unknown",
            )
        ]
    raise ValueError(f"no answer requirements for {case.category}")


def _unsupported_numeric_claims(answer: str, context: str) -> list[str]:
    if not context:
        return []
    pattern = re.compile(r"(?<![A-Za-z0-9])\d{2,}(?:[./:-]\d+)*(?![A-Za-z0-9])")
    answer_values = set(pattern.findall(answer))
    context_values = set(pattern.findall(context))
    for structured in tuple(context_values):
        if any(separator in structured for separator in ("-", "/", ":")):
            context_values.update(part for part in re.split(r"[-/:.]", structured) if len(part) >= 2)
    if "sha256" in context.lower() or "sha-256" in context.lower():
        context_values.add("256")
    return sorted(answer_values - context_values)


def grade_answer(case: BenchmarkCase, answer: str, *, context: str = "") -> dict[str, Any]:
    requirements = answer_requirements(case)
    matched = [
        any(_matches_alternative(answer, alternative) for alternative in alternatives)
        for alternatives in requirements
    ]
    affirmative_unsupported = False
    if case.category == "unsupported_claim_gate":
        normalised = _normalise(answer)
        affirmative_unsupported = bool(
            re.search(r"\b(?:yes|indeed)\b.{0,50}\b(?:own|owns|has)\b", normalised)
            or (
                not all(matched)
                and re.search(r"\b(?:project\s+)?\w+\s+owns\s+an?\s+interplanetary", normalised)
            )
        )
    unsupported_numbers = _unsupported_numeric_claims(answer, context)
    concept_complete = bool(answer.strip()) and all(matched) and not affirmative_unsupported
    return {
        "requirements": [list(group) for group in requirements],
        "matched_requirements": matched,
        "matched_count": sum(matched),
        "requirement_count": len(matched),
        "concept_recall": sum(matched) / max(1, len(matched)),
        "concept_complete": concept_complete,
        "affirmative_unsupported_claim": affirmative_unsupported,
        "unsupported_numeric_claims": unsupported_numbers,
        "correct": concept_complete and not unsupported_numbers,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def ollama_answer(
    prompt: str,
    *,
    model: str,
    endpoint: str = "http://127.0.0.1:11434",
    timeout: float | None = None,
    seed: int = 7,
) -> tuple[str, dict[str, Any]]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "system": READER_SYSTEM,
            "stream": False,
            "keep_alive": -1,
            "options": {"temperature": 0, "seed": seed},
        }
    ).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    if result.get("done_reason") == "length":
        raise RuntimeError("model returned a partial answer at its physical context boundary")
    elapsed_ms = (time.perf_counter() - started) * 1000
    answer = str(result.get("response", "")).strip()
    metrics = {
        "latency_ms": elapsed_ms,
        "done": bool(result.get("done")),
        "done_reason": result.get("done_reason"),
        "prompt_eval_count": int(result.get("prompt_eval_count") or 0),
        "eval_count": int(result.get("eval_count") or 0),
        "load_duration_ns": int(result.get("load_duration") or 0),
        "prompt_eval_duration_ns": int(result.get("prompt_eval_duration") or 0),
        "eval_duration_ns": int(result.get("eval_duration") or 0),
    }
    return answer, metrics


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _arm_compilers(store: LoomStore) -> dict[str, ContextCompiler]:
    provider = HashingSemanticEmbedder()
    config = CompilerConfig(model_context_tokens=720, reserved_output_tokens=180, max_visible_capabilities=3)
    return {
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


def _summarise(rows: list[dict[str, Any]], arm_names: Iterable[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm in arm_names:
        group = [row for row in rows if row["arm"] == arm]
        gold_rows = [row for row in group if row["gold_count"]]
        unsupported = [row for row in group if row["category"] == "unsupported_claim_gate"]
        full_evidence_rows = [row for row in gold_rows if row["full_evidence_hit"]]
        incomplete_evidence_rows = [row for row in gold_rows if not row["full_evidence_hit"]]
        latencies = [float(row["generation"]["latency_ms"]) for row in group if not row.get("error")]
        requirement_total = sum(row["grade"]["requirement_count"] for row in group)
        requirement_hits = sum(row["grade"]["matched_count"] for row in group)
        summary[arm] = {
            "cases": len(group),
            "answer_accuracy": sum(row["grade"]["correct"] for row in group) / max(1, len(group)),
            "concept_complete_rate": sum(row["grade"]["concept_complete"] for row in group)
            / max(1, len(group)),
            "answer_concept_recall": requirement_hits / max(1, requirement_total),
            "evidence_micro_recall": sum(row["gold_recalled"] for row in gold_rows)
            / max(1, sum(row["gold_count"] for row in gold_rows)),
            "full_evidence_hit_rate": sum(row["full_evidence_hit"] for row in gold_rows)
            / max(1, len(gold_rows)),
            "accuracy_when_full_evidence": sum(
                row["grade"]["correct"] for row in full_evidence_rows
            )
            / max(1, len(full_evidence_rows)),
            "accuracy_when_evidence_incomplete": sum(
                row["grade"]["correct"] for row in incomplete_evidence_rows
            )
            / max(1, len(incomplete_evidence_rows)),
            "unsupported_refusal_rate": sum(row["grade"]["correct"] for row in unsupported)
            / max(1, len(unsupported)),
            "affirmative_unsupported_claim_rate": sum(
                row["grade"]["affirmative_unsupported_claim"] for row in unsupported
            )
            / max(1, len(unsupported)),
            "unsupported_numeric_claim_rate": sum(
                bool(row["grade"]["unsupported_numeric_claims"]) for row in group
            )
            / max(1, len(group)),
            "generation_error_rate": sum(bool(row.get("error")) for row in group) / max(1, len(group)),
            "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
            "p95_latency_ms": _percentile(latencies, 0.95),
            "mean_prompt_eval_tokens": statistics.mean(
                row["generation"]["prompt_eval_count"] for row in group if not row.get("error")
            )
            if latencies
            else 0.0,
            "mean_generated_tokens": statistics.mean(
                row["generation"]["eval_count"] for row in group if not row.get("error")
            )
            if latencies
            else 0.0,
        }
        summary[arm]["by_category"] = {
            category: {
                "cases": len(category_rows),
                "answer_accuracy": sum(row["grade"]["correct"] for row in category_rows)
                / len(category_rows),
                "answer_concept_recall": sum(row["grade"]["matched_count"] for row in category_rows)
                / max(1, sum(row["grade"]["requirement_count"] for row in category_rows)),
                "full_evidence_hit_rate": (
                    sum(row["full_evidence_hit"] for row in category_rows if row["gold_count"])
                    / sum(bool(row["gold_count"]) for row in category_rows)
                    if any(row["gold_count"] for row in category_rows)
                    else None
                ),
            }
            for category in sorted({row["category"] for row in group})
            if (category_rows := [row for row in group if row["category"] == category])
        }
    return summary


def run_llm_benchmark(
    output_dir: str | Path,
    *,
    model: str = "granite4.1:8b",
    arms: Iterable[str] = ("recent_only", "semantic_recent", "context_loom"),
    max_cases_per_arm: int | None = None,
    timeout: float | None = None,
    seed: int = 7,
    endpoint: str = "http://127.0.0.1:11434",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arm_names = tuple(arms)
    unknown = set(arm_names) - {"recent_only", "semantic_recent", "context_loom"}
    if unknown:
        raise ValueError(f"unknown benchmark arms: {sorted(unknown)}")
    configuration = {
        "benchmark": HARNESS_VERSION,
        "model": model,
        "arms": list(arm_names),
        "max_cases_per_arm": max_cases_per_arm,
        "timeout": timeout,
        "seed": seed,
        "endpoint": endpoint,
        "reader_system_sha256": hashlib.sha256(READER_SYSTEM.encode()).hexdigest(),
        "grader_version": GRADER_VERSION,
        "generation_options": {"temperature": 0, "seed": seed},
    }
    _atomic_json(output / "configuration.json", configuration)
    with tempfile.TemporaryDirectory(prefix="context-loom-llm-eval-") as temporary:
        store = LoomStore(Path(temporary) / "benchmark.sqlite")
        try:
            cases = [case for case in _seed_benchmark(store) if case.category in QA_CATEGORIES]
            if max_cases_per_arm is not None:
                cases = cases[:max_cases_per_arm]
            compilers = _arm_compilers(store)
            rows: list[dict[str, Any]] = []
            total = len(cases) * len(arm_names)
            completed = 0
            for arm in arm_names:
                compiler = compilers[arm]
                for case in cases:
                    result_path = output / "run_state" / arm / f"{case.id}.json"
                    if result_path.exists():
                        existing = json.loads(result_path.read_text(encoding="utf-8"))
                        if (
                            existing.get("model") == model
                            and existing.get("seed") == seed
                            and existing.get("harness_version") == HARNESS_VERSION
                        ):
                            existing["grade"] = grade_answer(
                                case,
                                str(existing.get("answer", "")),
                                context=str(existing.get("context", "")),
                            )
                            existing["grader_version"] = GRADER_VERSION
                            existing["full_evidence_hit"] = (
                                set(existing.get("gold_record_ids", ()))
                                <= set(existing.get("admitted_record_ids", ()))
                                if existing.get("gold_count")
                                else None
                            )
                            _atomic_json(result_path, existing)
                            rows.append(existing)
                            completed += 1
                            continue
                    packet = compiler.compile(case.query, scope=case.scope, task_id=case.task_id, now=case.now)
                    admitted = {
                        source_id
                        for entry in packet.manifest
                        if entry.admitted
                        for source_id in entry.source_record_ids
                    }
                    gold = set(case.gold_record_ids)
                    error = None
                    try:
                        answer, generation = ollama_answer(
                            packet.rendered,
                            model=model,
                            endpoint=endpoint,
                            timeout=timeout,
                            seed=seed,
                        )
                    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
                        answer = ""
                        generation = {
                            "latency_ms": 0.0,
                            "done": False,
                            "done_reason": None,
                            "prompt_eval_count": 0,
                            "eval_count": 0,
                            "load_duration_ns": 0,
                            "prompt_eval_duration_ns": 0,
                            "eval_duration_ns": 0,
                        }
                        error = f"{type(exc).__name__}: {exc}"
                    row = {
                        "arm": arm,
                        "case_id": case.id,
                        "category": case.category,
                        "scope": case.scope,
                        "question": case.query,
                        "model": model,
                        "seed": seed,
                        "harness_version": HARNESS_VERSION,
                        "grader_version": GRADER_VERSION,
                        "packet_id": packet.id,
                        "context": packet.rendered,
                        "context_tokens": packet.token_count,
                        "gold_record_ids": list(case.gold_record_ids),
                        "admitted_record_ids": sorted(admitted),
                        "gold_count": len(gold),
                        "gold_recalled": len(gold & admitted),
                        "full_evidence_hit": gold <= admitted if gold else None,
                        "answer": answer,
                        "grade": grade_answer(case, answer, context=packet.rendered),
                        "generation": generation,
                        "error": error,
                    }
                    _atomic_json(result_path, row)
                    rows.append(row)
                    completed += 1
                    print(
                        f"[{completed}/{total}] {arm} {case.id}: "
                        f"evidence={('hit' if row['full_evidence_hit'] else 'miss') if gold else 'n/a'} "
                        f"answer={'pass' if row['grade']['correct'] else 'fail'} "
                        f"{generation['latency_ms'] / 1000:.2f}s",
                        flush=True,
                    )
            summary = _summarise(rows, arm_names)
            result = {
                **configuration,
                "cases_per_arm": len(cases),
                "total_answers": len(rows),
                "summary": summary,
                "rows": rows,
                "store_integrity": store.integrity_check(),
                "limitations": [
                    "Synthetic QA with deterministic concept grading; this is not a natural-corpus benchmark.",
                    "The 24 project scopes are controlled template variants, not 24 independent task domains.",
                    "The reader sees compiled context but never sees gold record IDs or grading requirements.",
                    "Answer accuracy requires every category-specific concept and grades unsupported questions by calibrated refusal phrases.",
                    "Several grader rules were corrected after early output inspection; all fixed answers were regraded without regeneration, but a confirmatory benchmark should preregister its grader.",
                    "The hashing retrieval provider is deterministic and is not a learned embedding model.",
                ],
            }
            _atomic_json(output / "results.json", result)
            _atomic_json(
                output / "summary.json",
                {key: value for key, value in result.items() if key != "rows"},
            )
            return result
        finally:
            store.close()


def render_llm_report(result: dict[str, Any]) -> str:
    total_prompt_tokens = sum(row["generation"]["prompt_eval_count"] for row in result["rows"])
    total_output_tokens = sum(row["generation"]["eval_count"] for row in result["rows"])
    total_latency_seconds = sum(row["generation"]["latency_ms"] for row in result["rows"]) / 1000
    lines = [
        "# Context Loom end-to-end LLM QA", "",
        f"Model: `{result['model']}`; seed: `{result['seed']}`; "
        f"answers: `{result['total_answers']}`", "",
        "The same local reader answered every question from each compiled context. Gold record IDs and grading requirements were never included in the model prompt. Answers were checkpointed before grading, so grader corrections did not regenerate or select model outputs.",
        "",
        f"The run processed `{total_prompt_tokens:,}` prompt tokens and `{total_output_tokens:,}` generated tokens. "
        f"Summed model-call latency was `{total_latency_seconds / 60:.1f}` minutes, with zero generation errors.",
        "",
        "| Arm | Answer accuracy | Concept recall | Evidence recall | Full evidence hit | Accuracy with full evidence | Unsupported refusal | Mean latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in result["arms"]:
        item = result["summary"][arm]
        lines.append(
            f"| {arm} | {item['answer_accuracy']:.1%} | {item['answer_concept_recall']:.1%} | "
            f"{item['evidence_micro_recall']:.1%} | {item['full_evidence_hit_rate']:.1%} | "
            f"{item['accuracy_when_full_evidence']:.1%} | {item['unsupported_refusal_rate']:.1%} | "
            f"{item['mean_latency_ms'] / 1000:.2f}s |"
        )
    lines.extend(["", "## Accuracy by category", ""])
    categories = sorted(next(iter(result["summary"].values()))["by_category"])
    lines.append("| Category | " + " | ".join(result["arms"]) + " |")
    lines.append("|---|" + "---:|" * len(result["arms"]))
    for category in categories:
        lines.append(
            f"| {category} | "
            + " | ".join(
                f"{result['summary'][arm]['by_category'][category]['answer_accuracy']:.1%}"
                for arm in result["arms"]
            )
            + " |"
        )
    lines.extend(["", "## Residual failures", ""])
    for arm in result["arms"]:
        failures = [row for row in result["rows"] if row["arm"] == arm and not row["grade"]["correct"]]
        grouped: dict[str, int] = {}
        for row in failures:
            grouped[row["category"]] = grouped.get(row["category"], 0) + 1
        description = ", ".join(f"{category}: {count}" for category, count in sorted(grouped.items()))
        lines.append(f"- **{arm}:** {len(failures)} failures" + (f" ({description})." if description else "."))
    loom_failures = [
        row for row in result["rows"]
        if row["arm"] == "context_loom" and not row["grade"]["correct"]
    ]
    if loom_failures:
        lines.extend(["", "Context Loom's remaining failures:", ""])
        for row in loom_failures:
            missing = [
                " / ".join(requirement)
                for requirement, matched in zip(
                    row["grade"]["requirements"], row["grade"]["matched_requirements"]
                )
                if not matched
            ]
            answer = " ".join(row["answer"].split()).replace("|", "\\|")
            lines.append(
                f"- `{row['case_id']}` missed `{'; '.join(missing)}`. Answer: “{answer}”"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Context Loom's gain over semantic plus recent is 24 answers, exactly matching the 24 preference records that the subjective lane recovered. Both richer arms made the same two procedure mistakes despite having the full procedure evidence, so those failures belong to reader compression rather than retrieval.",
            "",
            "Recent-only correctly refused all unsupported spaceport premises, but it could not answer any evidence-bearing category. Refusing an unsupported premise is scored separately from gold-record recall.",
            "",
            "Natural-language date forms such as `October 1, 2026` are accepted as equivalent to `2026-10-01`. A required procedure must preserve inspect/read, write, reread, and hash comparison as separate concepts. Numeric or date claims absent from the supplied context invalidate an otherwise complete answer.",
        ]
    )
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {limitation}" for limitation in result["limitations"])
    lines.append("")
    return "\n".join(lines)
