from __future__ import annotations

import json
import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from context_loom.models import MemoryRecord, RecordKind, EvidenceStatus


@dataclass(frozen=True)
class FactSummaryItem:
    key_facts: tuple[str, ...] = ()
    user_constraints: tuple[str, ...] = ()
    decisions_made: tuple[str, ...] = ()
    unresolved_tasks: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "key_facts": list(self.key_facts),
            "user_constraints": list(self.user_constraints),
            "decisions_made": list(self.decisions_made),
            "unresolved_tasks": list(self.unresolved_tasks),
        }


class RollingFactSummarizer:
    """Extracts compact factual summaries from dialogue history and ingests them into ContextLoom."""

    def __init__(
        self,
        loom: Any,
        model_adapter: Any = None,
        *,
        max_raw_exchanges: int = 2,
    ) -> None:
        self.loom = loom
        self.model_adapter = model_adapter
        self.max_raw_exchanges = max(1, int(max_raw_exchanges))

    def _build_prompt(self, records: Sequence[MemoryRecord]) -> str:
        transcript = "\n".join(f"{r.metadata.get('speaker', r.kind.value)}: {r.content}" for r in records)
        return (
            "Extract a concise factual summary of the conversation below into strict JSON format with keys:\n"
            '  "key_facts": [string],\n'
            '  "user_constraints": [string],\n'
            '  "decisions_made": [string],\n'
            '  "unresolved_tasks": [string]\n\n'
            "Transcript:\n"
            f"{transcript}\n\n"
            "JSON Summary:"
        )

    def summarize_exchanges(self, records: Sequence[MemoryRecord]) -> FactSummaryItem:
        if not records:
            return FactSummaryItem()

        if self.model_adapter is None:
            # Fallback heuristic extraction if model adapter is not provided
            facts = [r.content.strip() for r in records if r.content.strip()]
            return FactSummaryItem(key_facts=tuple(facts))

        prompt = self._build_prompt(records)
        raw_response = ""
        if hasattr(self.model_adapter, "complete"):
            raw_response = self.model_adapter.complete(prompt)
        elif callable(self.model_adapter):
            raw_response = self.model_adapter(prompt)

        try:
            # Extract JSON block
            text = str(raw_response).strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(text)
            return FactSummaryItem(
                key_facts=tuple(str(x) for x in parsed.get("key_facts", [])),
                user_constraints=tuple(str(x) for x in parsed.get("user_constraints", [])),
                decisions_made=tuple(str(x) for x in parsed.get("decisions_made", [])),
                unresolved_tasks=tuple(str(x) for x in parsed.get("unresolved_tasks", [])),
            )
        except (json.JSONDecodeError, AttributeError, ValueError):
            return FactSummaryItem(
                key_facts=tuple(r.content[:120] for r in records[:3]),
            )

    def ingest_summary(self, item: FactSummaryItem, scope: str = "default") -> list[str]:
        saved_ids: list[str] = []
        now = time.time()

        categories = [
            ("key_facts", item.key_facts, "Key Fact"),
            ("user_constraints", item.user_constraints, "User Constraint"),
            ("decisions_made", item.decisions_made, "Decision"),
            ("unresolved_tasks", item.unresolved_tasks, "Unresolved Task"),
        ]

        for cat_key, items, label in categories:
            for text in items:
                if not text.strip():
                    continue
                record_id = f"fact_summary_{uuid.uuid4().hex[:12]}"
                content = f"[ROLLING SUMMARY - {label}] {text.strip()}"
                record = MemoryRecord(
                    id=record_id,
                    content=content,
                    kind=RecordKind.FACT,
                    scope=scope,
                    evidence_status=EvidenceStatus.SYSTEM_DERIVED,
                    created_at=now,
                    metadata={"summary_category": cat_key, "source": "rolling_fact_summarizer"},
                )
                self.loom.store.append_record(record)
                saved_ids.append(record_id)

        return saved_ids
