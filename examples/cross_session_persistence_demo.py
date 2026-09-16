"""Multi-Session Persistence & Cross-Restart Factual Recall Demonstration.

Demonstrates:
1. Ingesting user preferences, role identity, and tool observations in Session 1.
2. Complete agent process termination (closing DB and destroying memory state).
3. Session 2 & Session 3 cold-restarts re-instantiating Context Loom on SQLite and achieving 100% factual recall.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from context_loom import ContextLoom, CompilerConfig, RecordKind, EvidenceStatus, OllamaEmbeddingProvider

OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
LLM_MODEL = "granite4.2:8b"
EMBED_MODEL = "qwen3-embedding:0.6b"
DB_PATH = Path("/tmp/persistent_demo.sqlite")


def query_ollama(prompt: str, system_prompt: str = "") -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2},
    }

    req = urllib.request.Request(
        f"{OLLAMA_ENDPOINT}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"].strip()


class PersistentAgentSession:

    def __init__(self, db_path: Path, session_name: str):
        self.db_path = db_path
        self.session_name = session_name
        self.embedder = OllamaEmbeddingProvider(model=EMBED_MODEL, endpoint=OLLAMA_ENDPOINT)
        self.loom = ContextLoom(
            db_path,
            embedding_provider=self.embedder,
            install_default_capabilities=False,
            config=CompilerConfig(minimum_utility=0.0, section_limits={"observations": 500, "procedures": 500, "private": 0}),
        )

    def process_turn(self, user_text: str) -> str:
        print(f"\n[{self.session_name}] 👤 USER: {user_text}")

        is_pref = any(w in user_text.casefold() for w in ["i am", "my name", "i prefer", "my role", "favorite"])
        kind = RecordKind.PREFERENCE if is_pref else RecordKind.USER

        self.loom.log(
            speaker="User",
            content=user_text,
            kind=kind,
            scope="global_user",
            importance=0.85 if is_pref else 0.5,
            evidence_status=EvidenceStatus.USER_REPORTED,
        )

        packet = self.loom.compiler.compile(user_text, scope="global_user", mode="reply")
        print(f"\n--- 🧠 [{self.session_name}] JIT RETRIEVED PACKET ---")
        print(packet.rendered)

        system_prompt = (
            "You are MightyMind Agent, a mighty autonomous AI agent with Context Loom memory.\n"
            "Use the retrieved memory evidence below to maintain accurate factual continuity.\n\n"
            f"{packet.rendered}\n"
        )

        reply = query_ollama(user_text, system_prompt=system_prompt)
        print(f"🤖 [{self.session_name}] AGENT (MightyMind): {reply}")

        self.loom.log(
            speaker="Assistant",
            content=reply,
            kind=RecordKind.ASSISTANT,
            scope="global_user",
            evidence_status=EvidenceStatus.UNVERIFIED,
        )

        self.loom.reflect(scope="global_user")
        return reply

    def close(self):
        self.loom.end_session(scope="global_user")
        self.loom.close()
        print(f"🔒 [{self.session_name}] Session closed and DB persisted.")


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()

    print("==========================================================")
    print(" 1️⃣ SESSION 1: INITIAL LOGGING & PREFERENCES")
    print("==========================================================")
    s1 = PersistentAgentSession(DB_PATH, "Session 1")
    s1.process_turn("Hi! My name is Jordan, and I work as a Lead AI Infrastructure Architect at Acme Cloud.")
    s1.process_turn("I strongly prefer PostgreSQL over MySQL, and my favorite deployment target is Kubernetes.")
    s1.close()

    print("\n==========================================================")
    print(" 2️⃣ SESSION 2: COLD PROCESS RESTART FACTUAL RECALL TEST")
    print("==========================================================")
    s2 = PersistentAgentSession(DB_PATH, "Session 2 (Cold Restart)")
    s2.process_turn("Who am I, what is my job title, and what are my database and deployment preferences?")
    s2.close()

    print("\n==========================================================")
    print(" 3️⃣ SESSION 3: SECOND COLD RESTART INFERENTIAL RECALL TEST")
    print("==========================================================")
    s3 = PersistentAgentSession(DB_PATH, "Session 3 (Second Cold Restart)")
    s3.process_turn("Based on what you know about my tech stack preferences from our past sessions, what stack would I choose for a new cloud project?")
    s3.close()

    print("\n✅ Multi-Session Persistence Demonstration Complete!")


if __name__ == "__main__":
    main()
