"""Topical Node Density & Overlap Reflection Engine for Context Loom."""
from __future__ import annotations

import logging
from typing import Any, Callable

from .models import EvidenceStatus, MemoryRecord, RecordKind

logger = logging.getLogger(__name__)


class TopicalReflectionEngine:
    """Detects dense Y-concept nodes (record density >= min_records) and consolidates
    their connected records into unified, factually grounded reflection records.
    """

    def __init__(self, loom: Any):
        self.loom = loom

    def find_dense_concept_nodes(
        self, scope: str = "default", min_records: int = 3
    ) -> list[dict[str, Any]]:
        """Find concept nodes that have at least `min_records` linked records.

        Returns a list of dicts with concept node details and linked record IDs.
        """
        query = """
            SELECT c.id as concept_id, c.name as concept_name, COUNT(DISTINCT l.record_id) as record_count,
                   GROUP_CONCAT(DISTINCT l.record_id) as record_ids
            FROM concept_nodes c
            JOIN concept_record_links l ON c.id = l.concept_id
            JOIN records r ON l.record_id = r.id
            WHERE c.id != 'self_root' AND r.scope = ?
            GROUP BY c.id, c.name
            HAVING record_count >= ?
            ORDER BY record_count DESC
        """
        try:
            rows = self.loom.graph_store.connection.execute(
                query, (scope, min_records)
            ).fetchall()
        except Exception:
            fallback_query = """
                SELECT c.id as concept_id, c.name as concept_name, COUNT(DISTINCT l.record_id) as record_count,
                       GROUP_CONCAT(DISTINCT l.record_id) as record_ids
                FROM concept_nodes c
                JOIN concept_record_links l ON c.id = l.concept_id
                WHERE c.id != 'self_root'
                GROUP BY c.id, c.name
                HAVING record_count >= ?
                ORDER BY record_count DESC
            """
            rows = self.loom.graph_store.connection.execute(
                fallback_query, (min_records,)
            ).fetchall()

        results = []
        for row in rows:
            rec_ids = row["record_ids"].split(",") if row["record_ids"] else []
            results.append(
                {
                    "concept_id": row["concept_id"],
                    "concept_name": row["concept_name"],
                    "record_count": row["record_count"],
                    "record_ids": rec_ids,
                }
            )
        return results

    def default_synthesizer(
        self, concept_name: str, records_text: list[str]
    ) -> str:
        """Default synthesizer that joins record texts under the concept topic."""
        joined = "; ".join(records_text)
        return f"Consolidated fact regarding {concept_name}: {joined}"

    def consolidate_and_ingest_topical_node(
        self,
        concept_id: str,
        synthesized_text: str,
        scope: str = "default",
        source_ids: list[str] | None = None,
    ) -> MemoryRecord:
        """Ingests a synthesized topical reflection record into Context Loom
        and links it directly to the Y-concept node.
        """
        sources = list(source_ids) if source_ids else []
        ref_record = self.loom.log(
            speaker="System",
            content=synthesized_text,
            kind=RecordKind.FACT,
            scope=scope,
            evidence_status=EvidenceStatus.SYSTEM_DERIVED,
            source_ids=sources,
            metadata={
                "origin": "topical_reflection",
                "y_concept_id": concept_id,
                "support_assessment": "entailed",
                "source_ids": sources,
            },
        )

        self.loom.graph_store.link_concept_record(concept_id, ref_record.id)

        x_terms, _ = self.loom.shadow_x_cypher.parse_input(synthesized_text)
        for term in x_terms:
            if len(term) > 2:
                self.loom.graph_store.link_x_membrane_to_y_concept(
                    term, concept_id
                )

        return ref_record

    def run_reflection_pass(
        self,
        scope: str = "default",
        min_records: int = 3,
        synthesizer: Callable[[str, list[str]], str] | None = None,
    ) -> list[MemoryRecord]:
        """Runs a complete topical reflection pass over all dense concept nodes in the given scope."""
        dense_nodes = self.find_dense_concept_nodes(
            scope=scope, min_records=min_records
        )
        if not dense_nodes:
            return []

        synth_fn = synthesizer or self.default_synthesizer
        created_records: list[MemoryRecord] = []

        for node_info in dense_nodes:
            cid = node_info["concept_id"]
            cname = node_info["concept_name"]
            rec_ids = node_info["record_ids"]

            texts = []
            for rid in rec_ids:
                rec = self.loom.store.get_record(rid)
                if rec and rec.content:
                    texts.append(rec.content)

            if not texts:
                continue

            synthesized = synth_fn(cname, texts)
            ref_rec = self.consolidate_and_ingest_topical_node(
                concept_id=cid,
                synthesized_text=synthesized,
                scope=scope,
                source_ids=rec_ids,
            )
            created_records.append(ref_rec)

        return created_records
