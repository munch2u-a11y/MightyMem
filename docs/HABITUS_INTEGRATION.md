# Habitus AI Agent Integration & Architecture

This document details the **Habitus AI** integration within **MightyMem**, explaining how `HabitusLoomAgent` pairs small open-weight LLMs with **ContextLoom**'s JIT memory compiler, why this architecture is necessary, and how it enables small models (3B–8B parameters) to perform reliable multi-turn reasoning without context degradation.

---

## 1. Why Habitus AI Integration is Necessary

Small local models (such as `granite4.2:8b` or `qwen3.5:4b`) have tight parameter budgets and limited context handling compared to massive 70B+ cloud models. When deployed with a standard naive RAG framework and raw tool definitions:

1. **Context Overwhelm & Prompt Bloat**: Naive RAG retrieves raw dialogue chunks that quickly fill up context windows (10k–30k tokens). Small 8B models lose track of initial system instructions, leading to goal drift.
2. **Tool Outcome Hallucination**: Without receipt-backed verification, small models often invent execution outputs (e.g. asserting *"I saved the file"* when the script was never executed).
3. **Multi-Session Continuity Loss**: Standard conversational agents wipe memory across restarts or rely on static vector databases that lack graph relationships and concept-level consolidation.

### The Solution: Cognitive Offloading

`HabitusLoomAgent` solves this by decoupling **Execution & Tool Orchestration** (`habitus_ai`) from **Memory Indexing & Context Compilation** (`ContextLoom`):

```
┌─────────────────────────────────────────────────────────────┐
│                 Multi-Channel Comms Layer                   │
│         (Telegram / Discord / Piper Voice Adapters)         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                ┌──────────────▼──────────────┐
                │        ChannelRouter        │
                │  (User Profile Scope Isolation)│
                └──────────────┬──────────────┘
                               │
                ┌──────────────▼──────────────┐
                │      HabitusLoomAgent       │
                │   (Ollama / Local 8B Model) │
                └──────┬───────────────┬──────┘
                       │               │
        1. Query JIT   │               │ 3. Execute Actions
           Header      │               │    via Bounded Runtime
                       ▼               ▼
        ┌──────────────────┐     ┌──────────────────┐
        │   ContextLoom    │     │ Bounded Runtime  │
        │ Memory Compiler  │     │ (File & Tools)   │
        └──────────────────┘     └──────────────────┘
```

By presenting the 8B LLM with a **compact ~1,000 token JIT Context Header** (containing verified evidence, user preferences, and recent dialogue continuity), the model operates cleanly without prompt noise.

---

## 2. Core Architecture & Subsystems

### `HabitusLoomAgent` (`context_loom/habitus_agent.py`)

The `HabitusLoomAgent` is the central orchestration agent. It handles:

- **Initialization & Pairing**: Connects an Ollama local model (`LocalModels`) to a persistent `ContextLoom` database instance.
- **Taste Schemas**: Configures behavioral tendencies using pre-built taste schemas:
  - `builder`: Pragmatic, execution-oriented, code & tool focused.
  - `deliberate`: Analytical, double-checks facts and reasoning.
  - `curious`: Exploratory, asks clarifying questions.
  - `balanced`: Default balanced behavior.
- **Turn Loop (`handle`)**:
  1. Ingests user input into user-isolated memory scope (`RecordKind.USER`).
  2. Compiles a JIT Context Packet via `loom.compiler.compile()`.
  3. Prepends the compiled header (`packet.rendered`) to the LLM system prompt.
  4. Generates response / tool call recommendations.
  5. Executes tools via `BoundedFileRuntime` and records stdout/receipts as `EvidenceStatus.TOOL_VERIFIED`.
  6. Ingests assistant response into `ContextLoom` (`RecordKind.ASSISTANT`).

---

## 3. Bounded Runtime & Tool Execution Receipts

To prevent tool execution hallucinations:

- **`BoundedFileRuntime`** ([`runtime.py`](../context_loom/runtime.py)): Restricts script execution and file operations within safe workspace paths, enforcing strict timeouts.
- **Verified Receipts**: Tool execution outputs are tagged with `RecordKind.OBSERVATION` and `EvidenceStatus.TOOL_VERIFIED`.
- **Admittance Guarantee**: When the agent compiles context for subsequent turns, `TOOL_VERIFIED` receipts take precedence over unverified assistant claims (`UNVERIFIED`). The 8B LLM reads concrete execution receipts in its system prompt, ensuring accurate responses.

---

## 4. Multi-User Comms Integration

The Habitus integration extends seamlessly to multi-user environments via [`context_loom/channels/`](../context_loom/channels):


- **User Profile Mapping**: `UserProfileStore` maps external channel user IDs (Telegram, Discord, Piper Voice) to a unique `profile_id`.
- **Scope Isolation**: Every user gets a distinct `memory_scope = "user_usr_<id>"`.
- **Channel Routing**: `ChannelRouter` passes messages to `HabitusLoomAgent` scoped to the active user profile, ensuring User A's memory and preferences are never exposed to User B.

---

## 5. Comparative Advantage: MightyMem vs. Naive RAG + Tools

| Dimension | Naive 8B RAG + Tool Schema | MightyMem + Habitus AI Agent |
| :--- | :--- | :--- |
| **Context Window Load** | High (15k–30k raw tokens) | Low (~1,000 compiled tokens) |
| **Tool Execution** | Unverified / Prone to Hallucination | Receipt-Backed (`TOOL_VERIFIED`) |
| **Memory Isolation** | Global / Unsegmented | Per-User `memory_scope` |
| **Cold Restart Continuity** | Lost or unindexed raw log dumps | Source-preserved JIT compilation |
| **Hardware Requirement** | Heavy / Expensive | 100% Local (3B–8B models) |

By combining Habitus AI's agentic loop with ContextLoom's cognitive offloading, MightyMem empowers local 8B models to maintain coherent long-term conversations and tool workflows that would otherwise overwhelm an 8B model.
