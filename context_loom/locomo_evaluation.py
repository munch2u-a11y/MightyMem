"""LoCoMo retrieval diagnostics and a retracted standalone reader harness.

Only ``run_locomo_retrieval`` represents the current component diagnostic. The
legacy ``run_locomo`` path bypasses Habitus and is retained solely so historical
artifacts remain reproducible; it must not be reported as a product benchmark.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .compiler import CompilerConfig
from .engine import ContextLoom
from .entity_graph import EntityExtractor, EntityGraphStore
from .models import EvidenceStatus, RecordKind
from .multi_hop_lane import MultiHopGraphLane, YCipherDepthSearchLane
from .deployment import verify_ollama_server_runtime
from .retrieval import (
    EmbeddingProvider,
    OllamaEmbeddingProvider,
    clean_embedding_payload,
    scope_blind_text,
)


HARNESS_VERSION = "context-loom-locomo-v1"
GRADER_VERSION = "category-aware-strict-llm-v1"
CATEGORY_NAMES = {
    1: "Multi-Hop",
    2: "Temporal",
    3: "Open-Domain",
    4: "Single-Hop",
    5: "Adversarial",
}
ANSWER_SYSTEM = """Answer the memory question using only the supplied Context Loom packet.
Preserve who said or experienced each fact. Resolve relative dates against the dated conversation records when possible.
The question may contain a false premise or attribute another person's experience to the wrong person. If so, reject or correct the premise instead of repeating it.
If the evidence does not establish an answer, say so. Give one concise answer."""
VERDICT_RE = re.compile(r"\b(PASS|FAIL)\b", re.IGNORECASE)
PREFERENCE_RE = re.compile(
    r"\b(?:prefer|prefers|preferred|favorite|favourite|fave|top pick|ideal)\b",
    re.IGNORECASE,
)


def parse_session_time(value: str) -> float:
    return datetime.strptime(value.strip(), "%I:%M %p on %d %B, %Y").timestamp()


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode()).hexdigest()[:24]}"


def session_keys(conversation: dict[str, Any]) -> list[str]:
    return sorted(
        (
            key
            for key in conversation
            if key.startswith("session_") and not key.endswith("_date_time")
        ),
        key=lambda key: int(key.split("_")[1]),
    )


def seed_conversation(
    conversation: dict[str, Any],
    *,
    database: str | Path,
    embedding_provider: EmbeddingProvider,
    compiler_context_tokens: int = 1500,
    compiler_reserved_output_tokens: int = 400,
) -> tuple[ContextLoom, dict[str, str], float]:
    loom = ContextLoom(
        database,
        embedding_provider=embedding_provider,
        install_default_capabilities=False,
        config=CompilerConfig(
            model_context_tokens=compiler_context_tokens,
            reserved_output_tokens=compiler_reserved_output_tokens,
            section_limits={
                "observations": 0,
                "procedures": 0,
                "private": 0,
            },
            minimum_utility=None,
        ),
    )
    graph_store = EntityGraphStore(loom.store)
    extractor = EntityExtractor()
    multi_hop_lane = MultiHopGraphLane(loom.store, graph_store, extractor)
    y_cipher_lane = YCipherDepthSearchLane(loom.store, graph_store)
    existing_names = {getattr(l, "name", "") for l in loom.compiler.lanes}
    if "multi_hop_graph" not in existing_names:
        loom.compiler.lanes.append(multi_hop_lane)
    if "y_cipher_depth" not in existing_names:
        loom.compiler.lanes.append(y_cipher_lane)

    record_to_dialog: dict[str, str] = {}
    latest = 0.0
    record_ids: list[str] = []
    for key in session_keys(conversation):
        number = key.split("_")[1]
        date_text = str(conversation[f"session_{number}_date_time"])
        session_time = parse_session_time(date_text)
        latest = max(latest, session_time)
        turns = conversation[key]
        for position, turn in enumerate(turns):
            dialog_id = str(turn["dia_id"])
            speaker = str(turn["speaker"])
            text = str(turn["text"])
            if turn.get("blip_caption"):
                text += f" Shared image: {turn['blip_caption']}."
            record_id = stable_id("locomo_turn", dialog_id)
            preference = bool(PREFERENCE_RE.search(text))
            loom.record(
                RecordKind.PREFERENCE if preference else RecordKind.USER,
                f"[{dialog_id} | {date_text}] {speaker}: {text}",
                scope="locomo",
                record_id=record_id,
                created_at=session_time + position,
                importance=0.85 if preference else 0.5,
                confidence=1.0,
                evidence_status=EvidenceStatus.USER_REPORTED,
                metadata={
                    "dia_id": dialog_id,
                    "speaker": speaker,
                    "session": key,
                    "session_date_time": date_text,
                },
            )
            latest = max(latest, session_time + position)
            graph_store.index_record(
                record_id,
                text,
                speaker=speaker,
                extractor=extractor,
            )

            record_to_dialog[record_id] = dialog_id
            record_ids.append(record_id)
        marker_id = stable_id("locomo_session", key)
        loom.record(
            RecordKind.EPISODE,
            f"{key} occurred at {date_text}.",
            scope="locomo",
            record_id=marker_id,
            created_at=session_time,
            importance=0.75,
            confidence=1.0,
            evidence_status=EvidenceStatus.SYSTEM_DERIVED,
            metadata={"session": key, "support_assessment": "extractive"},
        )
        record_ids.append(marker_id)

    if isinstance(embedding_provider, OllamaEmbeddingProvider):
        provider_key = embedding_provider.identity + ":clean-payload-v2"
        missing: list[tuple[str, str]] = []
        for record_id in record_ids:
            if loom.store.get_embedding(record_id, provider_key) is None:
                record = loom.store.get_record(record_id)
                if record is not None:
                    missing.append((record_id, clean_embedding_payload(record.content, "locomo")))
        for offset in range(0, len(missing), 64):
            batch = missing[offset : offset + 64]
            vectors = embedding_provider.embed_many([text for _, text in batch])
            for (record_id, _), vector in zip(batch, vectors):
                loom.store.put_embedding(record_id, provider_key, vector)
    return loom, record_to_dialog, latest + 1.0


def retrieval_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise exact LoCoMo evidence-ID retrieval without reader judgments."""

    categories: dict[str, dict[str, Any]] = {}
    for category_id, name in CATEGORY_NAMES.items():
        group = [row for row in rows if row["category"] == category_id]
        if group:
            gold_count = sum(row["evidence_count"] for row in group)
            hits = sum(row["evidence_hits"] for row in group)
            categories[name] = {
                "category_id": category_id,
                "questions": len(group),
                "gold_evidence_ids": gold_count,
                "evidence_hits": hits,
                "evidence_micro_recall": hits / max(1, gold_count),
                "mean_question_evidence_recall": statistics.mean(
                    row["evidence_recall"] for row in group
                ),
                "any_evidence_hit_rate": sum(row["any_evidence_hit"] for row in group)
                / len(group),
                "full_evidence_hit_rate": sum(row["full_evidence_hit"] for row in group)
                / len(group),
            }
    gold_count = sum(row["evidence_count"] for row in rows)
    hits = sum(row["evidence_hits"] for row in rows)
    micro_recall = hits / max(1, gold_count)
    average_tokens = statistics.mean(row["injected_tokens"] for row in rows)
    return {
        "questions": len(rows),
        "gold_evidence_ids": gold_count,
        "evidence_hits": hits,
        "evidence_micro_recall": micro_recall,
        "mean_question_evidence_recall": statistics.mean(
            row["evidence_recall"] for row in rows
        ),
        "any_evidence_hit_rate": sum(row["any_evidence_hit"] for row in rows)
        / max(1, len(rows)),
        "full_evidence_hit_rate": sum(row["full_evidence_hit"] for row in rows)
        / max(1, len(rows)),
        "avg_injected_whitespace_tokens": average_tokens,
        "avg_compiler_tokens": statistics.mean(row["compiler_tokens"] for row in rows),
        "recall_token_efficiency_ratio": 100.0 * micro_recall / (average_tokens / 1000.0),
        "categories": categories,
    }


def render_retrieval_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Context Loom retrieval on one complete LoCoMo conversation",
        "",
        f"- Dataset conversation: index {result['conversation_index']} (`{result['sample_id']}`)",
        f"- Questions: {summary['questions']} of {summary['questions']} (no sampling)",
        f"- Exact evidence micro-recall: **{summary['evidence_micro_recall']:.1%} "
        f"({summary['evidence_hits']}/{summary['gold_evidence_ids']} evidence IDs)**",
        f"- Full evidence hit rate: **{summary['full_evidence_hit_rate']:.1%}**",
        f"- Any evidence hit rate: **{summary['any_evidence_hit_rate']:.1%}**",
        f"- Mean injected payload: **{summary['avg_injected_whitespace_tokens']:.1f} whitespace tokens**",
        f"- Recall/token efficiency: **{summary['recall_token_efficiency_ratio']:.2f} recall points per 1k packet tokens**",
        "",
        "| Category | Questions | Evidence IDs | Micro recall | Any evidence | Full evidence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["categories"].items():
        lines.append(
            f"| {name} | {row['questions']} | {row['gold_evidence_ids']} | "
            f"{row['evidence_micro_recall']:.1%} | {row['any_evidence_hit_rate']:.1%} | "
            f"{row['full_evidence_hit_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "Recall is scored by exact overlap between LoCoMo's labeled `dia_id` evidence and the source IDs in Context Loom's admission manifest. It does not use an LLM judge and does not measure whether a reader can infer the final answer from partial or alternate evidence.",
            "",
            "The injected-token figure covers the complete rendered Context Loom packet, including its header, evidence labels, and current question, using whitespace splitting for comparability with FP-AMB. It excludes the outer reader prompt, model template, and generated answer.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_locomo_retrieval(
    output_dir: str | Path,
    *,
    dataset_path: str | Path = os.environ.get("LOCOMO_DATASET", "locomo10.json"),
    conversation_index: int = 1,
    embed_model: str = "qwen3-embedding:0.6b",
    ollama_url: str = "http://127.0.0.1:11434",
    kv_cache_type: str = "f16",
    ollama_server_pid: int | None = None,
    timeout: float | None = None,
    compiler_context_tokens: int = 1500,
    compiler_reserved_output_tokens: int = 400,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    sample = dataset[conversation_index]
    questions = sample["qa"]
    source_files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    if ollama_server_pid is None:
        raise ValueError("local retrieval diagnostic requires --ollama-server-pid")
    embedder = OllamaEmbeddingProvider(
        model=embed_model,
        endpoint=ollama_url,
        timeout=timeout,
        num_gpu=-1,
    )
    server_runtime = verify_ollama_server_runtime(
        embedder,
        server_pid=ollama_server_pid,
        kv_cache_type=kv_cache_type,
    )
    show_request = urllib.request.Request(
        embedder.endpoint + "/api/show",
        data=json.dumps({"model": embed_model}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(show_request, timeout=timeout) as response:
        embedding_model_manifest = response.read()
    intended_lanes = [
        "semantic", "exact", "recent", "subjective", "procedure", "correction",
        "multi_hop_graph", "y_cipher_depth",
    ]
    configuration = {
        "harness_version": HARNESS_VERSION,
        "evidence_class": "retrieval_component_diagnostic",
        "mode": "retrieval_only",
        "entrypoint": "context_loom.locomo_evaluation.run_locomo_retrieval",
        "output_path": str(output.resolve()),
        "source_revision": "no usable git repository; source_sha256 is authoritative",
        "source_sha256": source_hashes,
        "python": sys.version,
        "platform": os.uname().sysname + " " + os.uname().release,
        "cpu_count": os.cpu_count(),
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
        "conversation_index": conversation_index,
        "sample_id": sample.get("sample_id"),
        "question_count": len(questions),
        "question_ids": list(range(len(questions))),
        "embed_model": embed_model,
        "ollama_url": embedder.endpoint,
        "ollama_server_runtime": server_runtime,
        "kv_cache_type": kv_cache_type,
        "embedding_model_manifest_sha256": hashlib.sha256(embedding_model_manifest).hexdigest(),
        "embedding_num_gpu": -1,
        "memory_store": str((output / "loom.sqlite").resolve()),
        "retrieval_lanes": intended_lanes,
        "context_compiler": "context_loom.compiler.ContextCompiler",
        "token_counter": "regex-v1",
        "compiler_context_tokens": compiler_context_tokens,
        "compiler_reserved_output_tokens": compiler_reserved_output_tokens,
        "compiler_max_input_tokens": compiler_context_tokens - compiler_reserved_output_tokens,
        "minimum_candidate_utility_cutoff": None,
        "retrieval_candidate_limit": None,
        "entity_graph_hop_limit": None,
        "answer_model": None,
        "enabled_tools": [],
        "grader": "exact overlap of official LoCoMo evidence dialogue IDs with admitted source IDs",
        "parity_verified": False,
    }
    atomic_json(output / "composition-manifest.json", configuration)
    loom, record_to_dialog, now = seed_conversation(
        sample["conversation"],
        database=output / "loom.sqlite",
        embedding_provider=embedder,
        compiler_context_tokens=compiler_context_tokens,
        compiler_reserved_output_tokens=compiler_reserved_output_tokens,
    )
    actual_lanes = [lane.name for lane in loom.compiler.lanes]
    if actual_lanes != intended_lanes:
        loom.close()
        raise RuntimeError(f"retrieval composition mismatch: {actual_lanes!r}")
    configuration["parity_verified"] = True
    configuration["retrieval_lanes"] = actual_lanes
    atomic_json(output / "configuration.json", configuration)
    atomic_json(output / "composition-manifest.json", configuration)
    rows: list[dict[str, Any]] = []
    try:
        for index, qa in enumerate(questions):
            question = str(qa["question"])
            packet = loom.compiler.compile(question, scope="locomo", mode="reply", now=now)
            admitted_record_ids = {
                source_id
                for entry in packet.manifest
                if entry.admitted
                for source_id in entry.source_record_ids
            }
            admitted_dialog_ids = sorted(
                record_to_dialog[record_id]
                for record_id in admitted_record_ids
                if record_id in record_to_dialog
            )
            gold = [str(value) for value in qa.get("evidence", [])]
            hits = sorted(set(gold) & set(admitted_dialog_ids))
            row = {
                "question_id": index,
                "category": int(qa["category"]),
                "category_name": CATEGORY_NAMES[int(qa["category"])],
                "question": question,
                "expected_answer": qa.get("answer"),
                "adversarial_distractor": qa.get("adversarial_answer"),
                "gold_evidence_ids": gold,
                "admitted_dialog_ids": admitted_dialog_ids,
                "evidence_hits": len(hits),
                "evidence_count": len(gold),
                "evidence_recall": len(hits) / max(1, len(gold)),
                "any_evidence_hit": bool(hits),
                "full_evidence_hit": set(gold) <= set(admitted_dialog_ids),
                "compiler_tokens": packet.token_count,
                "injected_tokens": len(packet.rendered.split()),
                "context": packet.rendered,
                "packet_id": packet.id,
            }
            rows.append(row)
            if index == 0 or (index + 1) % 5 == 0 or index + 1 == len(questions):
                print(
                    f"[{index + 1:03d}/{len(questions)}] {row['category_name']:<12} "
                    f"evidence={len(hits)}/{len(gold)}",
                    flush=True,
                )
            if (index + 1) % 5 == 0:
                atomic_json(output / "partial_results.json", {"configuration": configuration, "rows": rows})
    finally:
        loom.close()
    result = {
        **configuration,
        "benchmark": "LoCoMo one complete conversation retrieval",
        "rows": rows,
        "summary": retrieval_summary(rows),
    }
    atomic_json(output / "results.json", result)
    atomic_json(output / "summary.json", result["summary"])
    (output / "REPORT.md").write_text(render_retrieval_report(result), encoding="utf-8")
    return result


def build_grade_prompt(question: str, expected: str, answer: str, *, adversarial: bool, distractor: str) -> str:
    if adversarial:
        return f"""Grade this adversarial memory question strictly.
The question contains a false premise, usually assigning one person's experience to another person. The distractor is what an answer would say if it accepted that false premise.

Question: {question}
Distractor: {distractor}
Student answer: {answer}

Output PASS only if the answer clearly rejects or corrects the false premise, says the claimed information is unavailable for the named person, or correctly attributes it to someone else.
Output FAIL if it accepts the distractor as true of the named person, is evasive without challenging the premise, or is off-topic.
Output only PASS or FAIL."""
    return f"""Grade this memory answer strictly.
Question: {question}
Expected answer: {expected}
Student answer: {answer}

Output PASS only if the student explicitly states the key fact in the expected answer. Direct paraphrases, synonyms, and equivalent numeric or date forms count. Implication alone, a list of alternatives, or claiming the information is missing does not count.
Output only PASS or FAIL."""


def parse_verdict(value: str) -> bool | None:
    match = VERDICT_RE.search(value.strip())
    return None if match is None else match.group(1).upper() == "PASS"


def ollama_generate(
    prompt: str,
    *,
    model: str,
    system: str = "",
    num_ctx: int,
    num_predict: int | None = None,
    timeout: float | None = None,
) -> tuple[str, dict[str, Any]]:
    options = {
        "temperature": 0,
        "seed": 7,
        "num_gpu": -1,
        "num_ctx": num_ctx,
    }
    if num_predict is not None:
        options["num_predict"] = num_predict
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "keep_alive": -1,
            "options": options,
        }
    ).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    text = str(result.get("response", "")).strip()
    if result.get("done_reason") == "length":
        raise RuntimeError("model reached its declared generation boundary before completing")
    return text, {
        "latency_ms": (time.perf_counter() - started) * 1000,
        "prompt_eval_count": int(result.get("prompt_eval_count") or 0),
        "eval_count": int(result.get("eval_count") or 0),
        "done": bool(result.get("done")),
        "done_reason": result.get("done_reason"),
    }


def gemini_generate(
    prompt: str,
    *,
    model: str = "gemini-2.5-flash",
    system: str = "",
    api_key: str | None = None,
    timeout: float = 30.0,
) -> tuple[str, dict[str, Any]]:
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key and os.path.exists(".env"):
            with open(".env") as f:
                for line in f:
                    if line.startswith("GOOGLE_API_KEY="):
                        api_key = line.strip().split("=", 1)[1].strip("\"'")
    if not api_key:
        raise RuntimeError("No GOOGLE_API_KEY found in environment or .env file")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload_data: dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0},
    }
    if system:
        payload_data["system_instruction"] = {"parts": [{"text": system}]}

    payload = json.dumps(payload_data).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.loads(response.read())

    text = ""
    candidates = result.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts:
            text = parts[0].get("text", "").strip()

    latency_ms = (time.perf_counter() - started) * 1000
    usage = result.get("usageMetadata", {})
    return text, {
        "latency_ms": latency_ms,
        "prompt_eval_count": usage.get("promptTokenCount", 0),
        "eval_count": usage.get("candidatesTokenCount", 0),
        "done": True,
        "done_reason": candidates[0].get("finishReason") if candidates else None,
    }




def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _checkpoint_matches(value: dict[str, Any], *, model: str, question: str, context_hash: str) -> bool:
    return (
        value.get("harness_version") == HARNESS_VERSION
        and value.get("grader_version") == GRADER_VERSION
        and value.get("model") == model
        and value.get("question") == question
        and value.get("context_hash") == context_hash
        and value.get("answer")
        and value.get("judge_verdict") in {"PASS", "FAIL"}
    )


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories: dict[str, dict[str, Any]] = {}
    for category_id, name in CATEGORY_NAMES.items():
        group = [row for row in rows if row["category"] == category_id]
        if group:
            categories[name] = {
                "category_id": category_id,
                "questions": len(group),
                "correct": sum(row["correct"] for row in group),
                "accuracy": sum(row["correct"] for row in group) / len(group),
                "evidence_micro_recall": sum(row["evidence_hits"] for row in group)
                / max(1, sum(row["evidence_count"] for row in group)),
                "full_evidence_hit_rate": sum(row["full_evidence_hit"] for row in group) / len(group),
            }
    full = [row for row in rows if row["full_evidence_hit"]]
    incomplete = [row for row in rows if not row["full_evidence_hit"]]
    return {
        "questions": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "answer_accuracy": sum(row["correct"] for row in rows) / max(1, len(rows)),
        "evidence_micro_recall": sum(row["evidence_hits"] for row in rows)
        / max(1, sum(row["evidence_count"] for row in rows)),
        "full_evidence_hit_rate": sum(row["full_evidence_hit"] for row in rows) / max(1, len(rows)),
        "accuracy_with_full_evidence": sum(row["correct"] for row in full) / max(1, len(full)),
        "accuracy_with_incomplete_evidence": sum(row["correct"] for row in incomplete)
        / max(1, len(incomplete)),
        "misses_with_full_evidence": sum(not row["correct"] for row in full),
        "misses_with_incomplete_evidence": sum(not row["correct"] for row in incomplete),
        "avg_injected_whitespace_tokens": statistics.mean(row["injected_tokens"] for row in rows),
        "avg_compiler_tokens": statistics.mean(row["compiler_tokens"] for row in rows),
        "total_answer_prompt_tokens": sum(row["answer_metrics"]["prompt_eval_count"] for row in rows),
        "total_answer_output_tokens": sum(row["answer_metrics"]["eval_count"] for row in rows),
        "total_judge_prompt_tokens": sum(row["judge_metrics"]["prompt_eval_count"] for row in rows),
        "total_judge_output_tokens": sum(row["judge_metrics"]["eval_count"] for row in rows),
        "categories": categories,
    }


def render_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Context Loom on one complete LoCoMo conversation",
        "",
        f"- Dataset conversation: index {result['conversation_index']} (`{result['sample_id']}`)",
        f"- Questions: {summary['questions']} of {summary['questions']} (no sampling)",
        f"- Reader and judge: `{result['model']}`",
        f"- Answer accuracy: **{summary['answer_accuracy']:.1%} ({summary['correct']}/{summary['questions']})**",
        f"- Exact evidence micro-recall: **{summary['evidence_micro_recall']:.1%}**",
        f"- Full evidence hit rate: **{summary['full_evidence_hit_rate']:.1%}**",
        f"- Mean injected payload: **{summary['avg_injected_whitespace_tokens']:.1f} whitespace tokens**",
        "",
        "| Category | Questions | Accuracy | Evidence recall | Full evidence |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in summary["categories"].items():
        lines.append(
            f"| {name} | {row['questions']} | {row['accuracy']:.1%} | "
            f"{row['evidence_micro_recall']:.1%} | {row['full_evidence_hit_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Error boundary",
            "",
            f"- Accuracy when all labeled evidence was admitted: {summary['accuracy_with_full_evidence']:.1%}.",
            f"- Accuracy with incomplete labeled evidence: {summary['accuracy_with_incomplete_evidence']:.1%}.",
            f"- Misses despite full evidence: {summary['misses_with_full_evidence']}.",
            f"- Misses with incomplete evidence: {summary['misses_with_incomplete_evidence']}.",
            "",
            "LoCoMo category 5 is treated as adversarial: its `adversarial_answer` is a distractor, and a response passes only when it rejects or corrects the false premise. Non-adversarial responses use a strict semantic LLM judge. The same local model acts as reader and judge, so this score includes judge variance and should be interpreted alongside exact evidence-ID recall.",
            "",
            "The run uses the official local `locomo10.json`, one entire conversation, a fixed 4,096-token model context, qwen3 embeddings, Context Loom's source and admission manifests, and resumable per-question checkpoints. Gold answers and evidence IDs are never placed in the reader prompt.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_locomo(
    output_dir: str | Path,
    *,
    dataset_path: str | Path = os.environ.get("LOCOMO_DATASET", "locomo10.json"),
    conversation_index: int = 1,
    model: str = "granite4.1:8b",
    embed_model: str = "qwen3-embedding:0.6b",
    timeout: float | None = None,
    compiler_context_tokens: int = 3600,
    compiler_reserved_output_tokens: int = 400,
) -> dict[str, Any]:
    """Reproduce the historical standalone QA diagnostic, not the product."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    sample = dataset[conversation_index]
    questions = sample["qa"]
    embedder = OllamaEmbeddingProvider(model=embed_model, timeout=timeout, num_gpu=-1)
    loom, record_to_dialog, now = seed_conversation(
        sample["conversation"],
        database=output / "loom.sqlite",
        embedding_provider=embedder,
        compiler_context_tokens=compiler_context_tokens,
        compiler_reserved_output_tokens=compiler_reserved_output_tokens,
    )
    configuration = {
        "harness_version": HARNESS_VERSION,
        "grader_version": GRADER_VERSION,
        "dataset_path": str(Path(dataset_path).resolve()),
        "conversation_index": conversation_index,
        "sample_id": sample.get("sample_id"),
        "question_count": len(questions),
        "question_ids": list(range(len(questions))),
        "model": model,
        "embed_model": embed_model,
        "model_context_tokens": 4096,
        "compiler_context_tokens": compiler_context_tokens,
        "compiler_reserved_output_tokens": compiler_reserved_output_tokens,
        "temperature": 0,
        "seed": 7,
    }
    atomic_json(output / "configuration.json", configuration)
    rows: list[dict[str, Any]] = []
    try:
        for index, qa in enumerate(questions):
            question = str(qa["question"])
            packet = loom.compiler.compile(question, scope="locomo", mode="reply", now=now)
            context_hash = hashlib.sha256(packet.rendered.encode()).hexdigest()
            checkpoint_path = output / "run_state" / f"q{index:03d}.json"
            checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
            if not _checkpoint_matches(
                checkpoint, model=model, question=question, context_hash=context_hash
            ):
                answer_prompt = f"[CONTEXT LOOM PACKET]\n{packet.rendered}\n\nAnswer the final current input."
                answer, answer_metrics = ollama_generate(
                    answer_prompt,
                    model=model,
                    system=ANSWER_SYSTEM,
                    num_ctx=4096,
                    timeout=timeout,
                )
                adversarial = int(qa["category"]) == 5
                expected = str(qa.get("answer") or "")
                distractor = str(qa.get("adversarial_answer") or "")
                judge_prompt = build_grade_prompt(
                    question,
                    expected,
                    answer,
                    adversarial=adversarial,
                    distractor=distractor,
                )
                judge, judge_metrics = ollama_generate(
                    judge_prompt,
                    model=model,
                    num_ctx=4096,
                    timeout=timeout,
                )
                verdict = parse_verdict(judge)
                if verdict is None:
                    raise RuntimeError(f"unparseable judge verdict for question {index}: {judge!r}")
                checkpoint = {
                    "harness_version": HARNESS_VERSION,
                    "grader_version": GRADER_VERSION,
                    "model": model,
                    "question": question,
                    "context_hash": context_hash,
                    "answer": answer,
                    "answer_metrics": answer_metrics,
                    "judge": judge,
                    "judge_verdict": "PASS" if verdict else "FAIL",
                    "judge_metrics": judge_metrics,
                }
                atomic_json(checkpoint_path, checkpoint)

            admitted_record_ids = {
                source_id
                for entry in packet.manifest
                if entry.admitted
                for source_id in entry.source_record_ids
            }
            admitted_dialog_ids = sorted(
                record_to_dialog[record_id]
                for record_id in admitted_record_ids
                if record_id in record_to_dialog
            )
            gold = [str(value) for value in qa.get("evidence", [])]
            hits = sorted(set(gold) & set(admitted_dialog_ids))
            row = {
                "question_id": index,
                "category": int(qa["category"]),
                "category_name": CATEGORY_NAMES[int(qa["category"])],
                "question": question,
                "expected_answer": qa.get("answer"),
                "adversarial_distractor": qa.get("adversarial_answer"),
                "gold_evidence_ids": gold,
                "admitted_dialog_ids": admitted_dialog_ids,
                "evidence_hits": len(hits),
                "evidence_count": len(gold),
                "evidence_recall": len(hits) / max(1, len(gold)),
                "full_evidence_hit": set(gold) <= set(admitted_dialog_ids),
                "compiler_tokens": packet.token_count,
                "injected_tokens": len(packet.rendered.split()),
                "context": packet.rendered,
                "packet_id": packet.id,
                "answer": checkpoint["answer"],
                "answer_metrics": checkpoint["answer_metrics"],
                "judge": checkpoint["judge"],
                "judge_metrics": checkpoint["judge_metrics"],
                "correct": checkpoint["judge_verdict"] == "PASS",
            }
            rows.append(row)
            if index == 0 or (index + 1) % 5 == 0 or index + 1 == len(questions):
                print(
                    f"[{index + 1:03d}/{len(questions)}] {row['category_name']:<12} "
                    f"answer={'PASS' if row['correct'] else 'FAIL'} evidence={len(hits)}/{len(gold)}",
                    flush=True,
                )
            if (index + 1) % 5 == 0:
                atomic_json(output / "partial_results.json", {"configuration": configuration, "rows": rows})
    finally:
        loom.close()
    result = {
        **configuration,
        "benchmark": "LoCoMo one complete conversation",
        "rows": rows,
        "summary": summarise(rows),
    }
    atomic_json(output / "results.json", result)
    atomic_json(output / "summary.json", result["summary"])
    (output / "REPORT.md").write_text(render_report(result), encoding="utf-8")
    return result
