from __future__ import annotations

import json
import urllib.request
import pytest
from pathlib import Path

from context_loom import ContextLoom, CompilerConfig, RecordKind, EvidenceStatus, OllamaEmbeddingProvider
from context_loom.topical import TopicalTracker, TopicalFrame
from context_loom.summarizer import RollingFactSummarizer, FactSummaryItem

OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
CHAT_MODEL = "granite4.1:8b"
EMBED_MODEL = "qwen3-embedding:0.6b"


def is_ollama_online() -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_ENDPOINT}/api/tags")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def query_real_ollama(prompt: str, system_prompt: str = "", model: str = CHAT_MODEL) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 300},
    }

    req = urllib.request.Request(
        f"{OLLAMA_ENDPOINT}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=45.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"].strip()


@pytest.mark.skipif(not is_ollama_online(), reason="Local Ollama server is offline")
def test_live_ollama_multi_turn_and_rolling_summary(tmp_path: Path):
    """Live end-to-end integration test using real local Ollama models."""
    db_path = tmp_path / "live_ollama_mind.sqlite"
    embedder = OllamaEmbeddingProvider(model=EMBED_MODEL, endpoint=OLLAMA_ENDPOINT)
    loom = ContextLoom(db_path, embedding_provider=embedder)
    tracker = TopicalTracker(loom.store, decay=0.70)

    class RealOllamaAdapter:
        def complete(self, prompt: str, **kwargs) -> str:
            return query_real_ollama(prompt, system_prompt="You are a strict JSON extractor.", model=CHAT_MODEL)

    summarizer = RollingFactSummarizer(loom, model_adapter=RealOllamaAdapter())

    # --- Turn 1: Preference & Project Initialization ---
    user_msg_1 = "Hi! My name is Dr. Sarah Chen, Chief Architect of Project Chronos."
    loom.log(
        speaker="Sarah",
        content=user_msg_1,
        kind=RecordKind.PREFERENCE,
        scope="live_user",
        importance=0.9,
        evidence_status=EvidenceStatus.USER_REPORTED,
    )
    
    # Simulate Y-path activation
    tracker.observe_turn(
        y_paths=[{
            "target_concept_id": "concept:project_chronos",
            "target_concept_name": "Project Chronos Architecture",
            "endpoint_score": 0.95,
            "path_nodes": ["self_root", "concept:project_chronos"],
        }],
        output_trunk="SPEAK",
    )

    packet1 = loom.compiler.compile(user_msg_1, scope="live_user")
    prompt1 = f"{tracker.frame().rendered}\n\n{packet1.rendered}"
    reply1 = query_real_ollama(user_msg_1, system_prompt=prompt1)
    
    loom.log(
        speaker="Assistant",
        content=reply1,
        kind=RecordKind.ASSISTANT,
        scope="live_user",
        evidence_status=EvidenceStatus.UNVERIFIED,
    )

    assert len(reply1) > 0

    # --- Turn 2: Critical Technical Constraint ---
    user_msg_2 = "Project Chronos must strictly use Rust for the core daemon and SQLite for local metadata."
    loom.log(
        speaker="Sarah",
        content=user_msg_2,
        kind=RecordKind.PREFERENCE,
        scope="live_user",
        importance=0.95,
        evidence_status=EvidenceStatus.USER_REPORTED,
    )

    packet2 = loom.compiler.compile(user_msg_2, scope="live_user")
    prompt2 = f"{tracker.frame().rendered}\n\n{packet2.rendered}"
    reply2 = query_real_ollama(user_msg_2, system_prompt=prompt2)

    loom.log(
        speaker="Assistant",
        content=reply2,
        kind=RecordKind.ASSISTANT,
        scope="live_user",
        evidence_status=EvidenceStatus.UNVERIFIED,
    )

    # --- Real Model Rolling Fact Summary Generation ---
    active_records = loom.store.list_records(scope="live_user")
    summary_item = summarizer.summarize_exchanges(active_records[:4])
    assert isinstance(summary_item, FactSummaryItem)
    
    ingested_ids = summarizer.ingest_summary(summary_item, scope="live_user")
    assert len(ingested_ids) > 0

    # Close first session
    loom.close()

    # --- Session 2: Process Termination & Cold Restart Verification ---
    loom_restarted = ContextLoom(db_path, embedding_provider=embedder)
    query_restarted = "What is my name, my role, and what are the strict language and database constraints for Project Chronos?"
    packet_restarted = loom_restarted.compiler.compile(query_restarted, scope="live_user")

    real_final_reply = query_real_ollama(query_restarted, system_prompt=f"Answer directly from evidence:\n{packet_restarted.rendered}")

    # Verify real LLM factual recall accuracy
    lowered_reply = real_final_reply.casefold()
    assert "sarah" in lowered_reply or "chen" in lowered_reply
    assert "rust" in lowered_reply
    assert "sqlite" in lowered_reply

    loom_restarted.close()
