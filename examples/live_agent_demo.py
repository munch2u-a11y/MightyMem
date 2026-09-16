"""Live Agent Demonstration using Context Loom & Local Ollama LLM.

Demonstrates:
1. Multi-turn dialogue with Context Loom JIT Context Compilations.
2. Code creation and tool execution with verified observation logging.
3. Accurate memory recall across turns.
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from pathlib import Path

from context_loom import ContextLoom, CompilerConfig, RecordKind, EvidenceStatus, OllamaEmbeddingProvider

OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
LLM_MODEL = "granite4.2:8b"
EMBED_MODEL = "qwen3-embedding:0.6b"
DB_PATH = Path("/tmp/live_agent_demo.sqlite")


def query_ollama(prompt: str, system_prompt: str = "") -> str:
    """Query local Ollama server."""
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


class LiveAgent:

    def __init__(self, db_path: Path = DB_PATH):
        if db_path.exists():
            db_path.unlink()
        self.embedder = OllamaEmbeddingProvider(model=EMBED_MODEL, endpoint=OLLAMA_ENDPOINT)
        self.loom = ContextLoom(
            db_path,
            embedding_provider=self.embedder,
            install_default_capabilities=False,
            config=CompilerConfig(minimum_utility=0.0, section_limits={"observations": 500, "procedures": 500, "private": 0}),
        )

    def process_turn(self, user_text: str) -> str:
        print(f"\n👤 USER: {user_text}")

        is_pref = any(w in user_text.casefold() for w in ["i am", "my name", "i love", "my role", "i prefer"])
        kind = RecordKind.PREFERENCE if is_pref else RecordKind.USER

        self.loom.log(
            speaker="User",
            content=user_text,
            kind=kind,
            scope="live_session",
            importance=0.85 if is_pref else 0.5,
            evidence_status=EvidenceStatus.USER_REPORTED,
        )

        packet = self.loom.compiler.compile(user_text, scope="live_session", mode="reply")
        print("\n--- 🧠 CONTEXT LOOM JIT RENDERED PACKET ---")
        print(packet.rendered)

        system_prompt = (
            "You are MightyMind Agent, a mighty autonomous AI agent with Context Loom memory.\n"
            "Use the retrieved memory evidence below to maintain precise personal recall.\n\n"
            f"{packet.rendered}\n"
        )

        reply = query_ollama(user_text, system_prompt=system_prompt)
        print(f"\n🤖 AGENT (MightyMind): {reply}")

        self.loom.log(
            speaker="Assistant",
            content=reply,
            kind=RecordKind.ASSISTANT,
            scope="live_session",
            evidence_status=EvidenceStatus.UNVERIFIED,
        )

        self.loom.reflect(scope="live_session")
        return reply


def main():
    print(f"🚀 Starting MightyMind Live Agent Demo using LLM '{LLM_MODEL}' + Context Loom!")
    agent = LiveAgent()

    # Turn 1
    agent.process_turn("Hi! I am Alex. I am a Senior Data Engineer who loves automated health scripts.")

    # Turn 2
    agent.process_turn("Can you create a Python script in 'scratch/disk_check.py' that measures free disk space and writes a summary to 'scratch/report.md', then run it?")

    # Create & execute script
    script_path = Path("scratch/disk_check.py")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text('''import os, shutil
total, used, free = shutil.disk_usage("/")
with open("scratch/report.md", "w") as f:
    f.write(f"# Disk Report\\nFree Space: {free / (1024**3):.2f} GB\\n")
print("[SUCCESS] Disk report written to scratch/report.md")
''')
    proc = subprocess.run(["python3", str(script_path)], capture_output=True, text=True)

    agent.loom.log(
        speaker="Assistant",
        content=f"Executed script '{script_path.name}'. Output: {proc.stdout.strip()}",
        kind=RecordKind.OBSERVATION,
        scope="live_session",
        evidence_status=EvidenceStatus.TOOL_VERIFIED,
    )
    agent.loom.reflect(scope="live_session")

    # Turn 3: Memory Recall
    agent.process_turn("Can you recall my name, my role, and summarize what script we just created and executed?")
    print("\n✅ Live Agent Demonstration Complete!")


if __name__ == "__main__":
    main()
