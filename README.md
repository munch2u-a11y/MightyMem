# MightyMem (MightyMemory)

> **Mighty Local Autonomous Agents Powered by Context Loom JIT Memory & Habitus AI**

**MightyMem** gives small local open-weight LLMs (3B–8B parameter models like Granite 4.2 & Qwen 3.5) persistent memory continuity, cold-restart state retention, and structured context compilation—running 100% locally on your machine.

---

## ⚡ The Core Problem: Why Small Models Fail on Plain RAG

A standard 8B model equipped with a basic RAG setup and raw tool schemas quickly gets **overwhelmed**. When flooded with 15k–30k tokens of uncompressed chat logs and verbose tool schemas, small local models experience context degradation:
- They drop system instructions or forget early conversational context.
- They hallucinate tool outputs (e.g. claiming a script executed when it failed or was never called).
- They struggle with multi-turn identity and user preference retention across cold restarts.

**MightyMem solves this through Cognitive Offloading.** Instead of forcing the LLM to search through noisy chat logs, ContextLoom compiles a concise, structured **~1,000-token JIT Context Header** containing exact evidence, user preferences, and execution receipts.

---

## 💡 Key Benefits & Benchmark Results

- 🚀 **Enhanced Agent Continuity on Local Hardware**: Enables `granite4.2:8b` and `qwen3.5:4b` to maintain focused long-term interaction without prompt token overload.
- 🔒 **Persistent Multi-Session Memory**: Stores facts and preferences in source-preserved SQLite records across process restarts.
- 📊 **79.4% Context Recall on FP-AMB Benchmark**: Evaluated across 60 multi-session conversations and 262 ground-truth questions ([`docs/FPAMB_SCORECARD.md`](docs/FPAMB_SCORECARD.md)).
- 🛡️ **Receipt-Backed Tool Execution**: Offloads tool execution to local runtimes and injects verified execution receipts (`TOOL_VERIFIED`), preventing hallucinated action claims.
- 💬 **Multi-Channel Router Support**: Built-in support for **Telegram**, **Discord**, and **Piper Voice (TTS/STT)** adapters with per-user isolated memory scopes.

---

## 🏗️ How It Works: Cognitive Offloading

MightyMem offloads heavy reasoning, memory search, and state tracking away from the LLM’s parameter budget into deterministic background algorithms:

1. **JIT Context Packet Compilation**:  
   Compiles candidate memories into a 3-section structured header (`[ASSERTED & DERIVED EVIDENCE]`, `[BOUNDED PERSONAL CONTINUITY]`, `[RECENCY & CONVERSATIONAL CONTINUITY]`). The LLM receives exact, ranked evidence rather than noisy raw dialogue.

2. **Non-Language Vector Concept Vaults (Y-Cypher)**:  
   Multi-turn dialogue dynamics are compacted into language-independent vector concept nodes. Memory retrieval queries concepts and n-grams rather than relying solely on exact keyword overlap.

3. **Topical Node Density Reflection Engine**:  
   Identifies concept clusters with high record density ($\ge 3$ records) and automatically consolidates them into factually grounded reflection records (`RecordKind.FACT`, `EvidenceStatus.SYSTEM_DERIVED`).

4. **Tool Use Cortex & Verified Observations**:  
   Offloads heavy computation, script execution, and host actions to local runtimes. Tool outcomes are logged as `RecordKind.OBSERVATION` with `EvidenceStatus.TOOL_VERIFIED`, providing execution receipts for the local LLM.

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/mightymem/mightymem.git
cd mightymem
pip install -e .
```


### 2. Multi-Channel Router CLI Command
```bash
# Telegram Channel Router
context-loom channel --adapter telegram

# Discord Channel Router (polling target channel)
context-loom channel --adapter discord --discord-channel-id 1234567890123456789

# Local Piper Voice (TTS/STT) Channel Router
context-loom channel --adapter piper-voice --piper-model-path en_US-lessac-medium.onnx
```

### 3. Safe Credentials Management CLI
```bash
# Save tokens securely to ~/.mightymem/.env (prompts for value with hidden typing)
context-loom credentials set TELEGRAM_BOT_TOKEN
context-loom credentials set DISCORD_BOT_TOKEN
context-loom credentials set DISCORD_CHANNEL_ID
context-loom credentials list
```


### 4. Python API Example
```python
from context_loom import ContextLoom, CompilerConfig, OllamaEmbeddingProvider, RecordKind, EvidenceStatus

# 1. Initialize MightyMem memory engine with local embeddings
embedder = OllamaEmbeddingProvider(model="qwen3-embedding:0.6b")
loom = ContextLoom("mightymem_memory.sqlite", embedding_provider=embedder)

# 2. Log user preference into user memory scope
loom.log(
    speaker="User",
    content="Hi! I am Alex, a Senior Data Engineer who loves PostgreSQL and Kubernetes.",
    kind=RecordKind.PREFERENCE,
    scope="user_usr_1001",
    evidence_status=EvidenceStatus.USER_REPORTED,
)

# 3. Compile JIT Context Packet for query
packet = loom.compiler.compile("What is user's database preference?", scope="user_usr_1001")

# 4. Inject packet.rendered into LLM system prompt
print(packet.rendered)

# 5. Run reflection pass before closing session
loom.reflect()
loom.close()
```

---

## 📚 Technical Documentation & Subsystems

For deep-dive technical specifications on individual subsystems, see:

- 🤖 **[Habitus AI Agent Integration](docs/HABITUS_INTEGRATION.md)**: Details on `HabitusLoomAgent`, taste schemas, bounded runtime execution, and agentic loop orchestration.
- 🏗️ **[Architecture & Subsystems](docs/ARCHITECTURE.md)**: Memory classification, dual-axis X-Y membrane, and 3-section packet rendering.
- 🧠 **[Topical Reflection Engine](docs/REFLECTION_ENGINE.md)**: Density detection algorithms and reflection ingestion.
- 📊 **[Benchmarks & Evaluation](docs/BENCHMARKS.md)**: LoCoMo reference metrics and empirical overview.
- 📈 **[FP-AMB Scorecard](docs/FPAMB_SCORECARD.md)**: Full evaluation scorecard across 262 benchmark questions.

---

## 💻 Demonstration Scripts

Try the ready-to-run interactive local agent examples in `examples/`:

- **[live_agent_demo.py](examples/live_agent_demo.py)**: Interactive MightyMem agent chat + Python script creation & execution with local Ollama.
- **[cross_session_persistence_demo.py](examples/cross_session_persistence_demo.py)**: Multi-session process restart factual recall test.

---

## 📄 License

MightyMem is distributed under the terms of the **[Apache License 2.0](LICENSE)**.


