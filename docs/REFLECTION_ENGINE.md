# Topical Node Density & Overlap Reflection Engine

The **Topical Reflection Engine** (`TopicalReflectionEngine`) automates background consolidation of dense memory nodes into concise, factually grounded inferential records.

## Algorithm Overview

1. **Density Detection**:
   Queries SQLite graph storage for Y-concept nodes linked to $\ge 3$ records in the target scope:
   ```sql
   SELECT c.id, c.name, COUNT(DISTINCT l.record_id) as record_count
   FROM concept_nodes c
   JOIN concept_record_links l ON c.id = l.concept_id
   WHERE c.id != 'self_root'
   GROUP BY c.id, c.name
   HAVING record_count >= 3
   ```

2. **Synthesis & Ingestion**:
   Consolidates connected observations into a unified statement and logs it as:
   - `kind = RecordKind.FACT`
   - `evidence_status = EvidenceStatus.SYSTEM_DERIVED`
   - `metadata = {"origin": "topical_reflection", "y_concept_id": concept_id, "support_assessment": "entailed"}`

3. **X-Membrane Cross-Axis Linking**:
   Links terms in the synthesized statement directly back to the concept vault, enabling immediate 100% precision keyword and cosine similarity matches on inferential queries.

## Usage Example

```python
from context_loom import ContextLoom, TopicalReflectionEngine

loom = ContextLoom("memory.sqlite")
reflection_engine = TopicalReflectionEngine(loom)

# Run reflection pass over scope
new_facts = reflection_engine.run_reflection_pass(scope="default", min_records=3)
print(f"Ingested {len(new_facts)} synthesized reflection facts.")
```
