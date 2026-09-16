"""Relational Entity Graph construction and topological storage for Context Loom."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
import re
import sqlite3
import time
from typing import Any, Sequence

from .store import LoomStore


class EntityKind(str, Enum):
    PERSON = "person"
    LOCATION = "location"
    TOPIC = "topic"
    INTEREST = "interest"
    PROJECT = "project"
    EVENT = "event"
    ORGANIZATION = "organization"
    TIME_INTERVAL = "time_interval"
    OTHER = "other"


class RelationKind(str, Enum):
    FAMILY_OF = "family_of"
    WORKS_WITH = "works_with"
    WORKS_AT = "works_at"
    LOCATED_IN = "located_in"
    INTERESTED_IN = "interested_in"
    ASSOCIATED_WITH = "associated_with"
    CO_OCCURRED_WITH = "co_occurred_with"
    OWNS = "owns"


@dataclass(frozen=True)
class Entity:
    name: str
    kind: EntityKind
    canonical_name: str = ""

    def __post_init__(self) -> None:
        if not self.canonical_name:
            object.__setattr__(self, "canonical_name", self.name.casefold().strip(" \t\n\r\"'.,;:!?()[]{}"))


@dataclass(frozen=True)
class Relation:
    source: str
    relation: RelationKind
    target: str
    record_id: str | None = None
    confidence: float = 1.0


ENTITY_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "from",
    "by", "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "of", "in", "on", "off", "over", "under",
    "again", "further", "once", "here", "there", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "don",
    "should", "now", "what", "which", "who", "whom", "whose", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "would", "could", "ought", "i", "you",
    "he", "she", "it", "we", "they", "my", "your", "his", "her", "its", "our", "their",
    "mine", "yours", "hers", "ours", "theirs", "myself", "yourself", "himself", "herself",
    "itself", "ourselves", "themselves", "tell", "find", "show", "explain", "list", "give",
    "said", "say", "says", "told", "ask", "asked", "yeah", "yes", "nope", "sure", "okay",
    "agreed", "hello", "hi", "hey", "bye", "goodbye", "thanks", "thank", "welcome", "please",
    "started", "starting", "opened", "opening", "working", "works", "worked", "teach", "teaches",
}


class EntityExtractor:
    """Extract entities and typed relations from conversational text."""

    def __init__(self) -> None:
        self._kinship_pattern = re.compile(
            r"\b([A-Z][a-z]+)'s\s+(brother|sister|father|mother|son|daughter|husband|wife|parent|child)\s+(?:is\s+)?([A-Z][a-z]+)\b",
            re.IGNORECASE,
        )
        self._my_kinship_pattern = re.compile(
            r"\b([A-Z][a-z]+):\s+My\s+(brother|sister|father|mother|son|daughter|husband|wife)\s+([A-Z][a-z]+)\b",
            re.IGNORECASE,
        )
        self._works_pattern = re.compile(
            r"\b([A-Z][a-z]+)\s+(?:works\s+(?:at|as|with|for)|collaborat(?:e|ing)\s+with)\s+([A-Z][a-zA-Z0-9_\s]+)\b",
            re.IGNORECASE,
        )
        self._location_pattern = re.compile(
            r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        )
        self._owns_pattern = re.compile(
            r"\b([A-Z][a-z]+)\s+(?:drives|has|owns|bought|uses|started|opened|runs|operates)\s+(?:a|an|the)?\s*([A-Za-z0-9_\s-]{3,35})\b",
            re.IGNORECASE,
        )
        self._multiword_proper_pattern = re.compile(
            r"\b([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+)\b"
        )
        self._concept_compound_pattern = re.compile(
            r"\b([a-z]{3,20}\s+(?:studio|store|shop|business|company|firm|agency|venture|market|enterprise|startup|salon|boutique|practice|clinic|bakery|restaurant|cafe))\b",
            re.IGNORECASE,
        )
        self._occupation_pattern = re.compile(
            r"\b(?:working\s+as|works\s+as|work\s+as|worked\s+as|job\s+as|hired\s+as|career\s+as|role\s+as|position\s+as)\s+(?:a|an)?\s*([a-z]{3,25}(?:\s+[a-z]{3,25})?)\b",
            re.IGNORECASE,
        )
        self._common_occupations = {
            "banker", "architect", "engineer", "lawyer", "doctor", "nurse", "teacher",
            "consultant", "accountant", "designer", "artist", "developer", "manager",
            "director", "scientist", "researcher", "analyst", "writer", "journalist",
            "chef", "baker", "pilot", "dentist", "pharmacist", "therapist", "musician",
        }
        self._temporal_date_pattern = re.compile(
            r"(?:\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b|"
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b|"
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b|"
            r"\b(?:yesterday|today|tomorrow|last\s+(?:week|month|year)|next\s+(?:week|month|year))\b|"
            r"\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)\b)",
            re.IGNORECASE,
        )

    def extract(self, text: str) -> tuple[list[Entity], list[Relation]]:
        entities: dict[str, Entity] = {}
        relations: list[Relation] = []

        def add_entity(name: str, kind: EntityKind) -> None:
            clean = name.strip(" \t\n\r\"'.,;:!?()[]{}")
            canon = clean.casefold()
            if not canon or len(canon) < 2:
                return
            if canon in ENTITY_STOPWORDS:
                return
            if canon in entities:
                # Upgrade generic kind to specific kind
                if entities[canon].kind in {EntityKind.OTHER, EntityKind.PERSON} and kind not in {EntityKind.OTHER, EntityKind.PERSON}:
                    entities[canon] = Entity(name=clean, kind=kind, canonical_name=canon)
            else:
                entities[canon] = Entity(name=clean, kind=kind, canonical_name=canon)

        # 1. Temporal entity extraction
        for match in self._temporal_date_pattern.finditer(text):
            temporal_str = match.group(0).strip(" \t\n\r\"'.,;:!?()[]{}")
            if temporal_str:
                add_entity(temporal_str, EntityKind.TIME_INTERVAL)

        # 2. Kinship pattern 1: Alex's brother Sam
        for match in self._kinship_pattern.finditer(text):
            src, rel_str, tgt = match.group(1), match.group(2), match.group(3)
            add_entity(src, EntityKind.PERSON)
            add_entity(tgt, EntityKind.PERSON)
            relations.append(Relation(source=src, relation=RelationKind.FAMILY_OF, target=tgt))
            relations.append(Relation(source=tgt, relation=RelationKind.FAMILY_OF, target=src))

        # 3. Kinship pattern 2: [Speaker] Alex: My brother Sam
        for match in self._my_kinship_pattern.finditer(text):
            src, rel_str, tgt = match.group(1), match.group(2), match.group(3)
            add_entity(src, EntityKind.PERSON)
            add_entity(tgt, EntityKind.PERSON)
            relations.append(Relation(source=src, relation=RelationKind.FAMILY_OF, target=tgt))
            relations.append(Relation(source=tgt, relation=RelationKind.FAMILY_OF, target=src))

        # 4. Works / Collaborates pattern
        for match in self._works_pattern.finditer(text):
            src, tgt = match.group(1), match.group(2).strip(" \t\n\r\"'.,;:!?()[]{}")
            add_entity(src, EntityKind.PERSON)
            tgt_lower = tgt.casefold()
            tgt_kind = (
                EntityKind.ORGANIZATION
                if any(w in tgt_lower for w in ["inc", "corp", "llc", "ltd", "studio", "store", "company", "firm", "agency"])
                else EntityKind.PERSON
            )
            add_entity(tgt, tgt_kind)
            relations.append(Relation(source=src, relation=RelationKind.WORKS_WITH, target=tgt))

        # 5. Owns / Drives / Started / Opened pattern
        for match in self._owns_pattern.finditer(text):
            src = match.group(1).strip()
            tgt = match.group(2).strip(" \t\n\r\"'.,;:!?()[]{}")
            tgt = re.sub(r"^(?:a|an|the)\s+", "", tgt, flags=re.IGNORECASE).strip(" \t\n\r\"'.,;:!?()[]{}")
            if src.casefold() not in ENTITY_STOPWORDS and tgt.casefold() not in ENTITY_STOPWORDS and len(tgt) >= 2:
                add_entity(src, EntityKind.PERSON)
                tgt_lower = tgt.casefold()
                tgt_kind = (
                    EntityKind.ORGANIZATION
                    if any(w in tgt_lower for w in ["store", "studio", "shop", "firm", "company", "business", "bakery", "cafe"])
                    else EntityKind.OTHER
                )
                add_entity(tgt, tgt_kind)
                relations.append(Relation(source=src, relation=RelationKind.OWNS, target=tgt))

        # 6. Location pattern
        for match in self._location_pattern.finditer(text):
            loc = match.group(1).strip(" \t\n\r\"'.,;:!?()[]{}")
            if loc and loc[0].isupper() and loc.casefold() not in {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}:
                add_entity(loc, EntityKind.LOCATION)

        # 7. Salient compound noun phrases (e.g. "dance studio", "clothing store")
        for match in self._concept_compound_pattern.finditer(text):
            compound = match.group(1).strip(" \t\n\r\"'.,;:!?()[]{}")
            if compound and compound.casefold() not in ENTITY_STOPWORDS:
                add_entity(compound, EntityKind.TOPIC)

        # 8. Occupations / Roles
        for match in self._occupation_pattern.finditer(text):
            occ = match.group(1).strip(" \t\n\r\"'.,;:!?()[]{}")
            if occ and occ.casefold() not in ENTITY_STOPWORDS:
                add_entity(occ, EntityKind.TOPIC)

        for word in re.findall(r"\b[a-z]{4,20}\b", text.lower()):
            if word in self._common_occupations:
                add_entity(word, EntityKind.TOPIC)

        # 9. Multi-word proper noun chunks
        for match in self._multiword_proper_pattern.finditer(text):
            chunk = match.group(1).strip(" \t\n\r\"'.,;:!?()[]{}")
            words = chunk.split()
            if words[0].casefold() in {"the", "this", "that", "there", "here"} and len(words) > 1:
                chunk = " ".join(words[1:])
            chunk_lower = chunk.casefold()
            if any(w in chunk_lower for w in ["city", "state", "street", "avenue", "ave", "road", "tower", "center", "park", "beach", "lake", "river"]):
                add_entity(chunk, EntityKind.LOCATION)
            elif any(w in chunk_lower for w in ["inc", "corp", "llc", "studio", "store", "company", "firm", "agency", "hospital", "school", "university", "department"]):
                add_entity(chunk, EntityKind.ORGANIZATION)
            else:
                add_entity(chunk, EntityKind.OTHER)

        # 10. Capitalized entity fallback
        words = re.findall(r"\b[A-Z][a-z]{1,20}\b", text)
        for word in words:
            word_clean = word.strip(" \t\n\r\"'.,;:!?()[]{}")
            if word_clean.casefold() not in ENTITY_STOPWORDS and len(word_clean) >= 2:
                add_entity(word_clean, EntityKind.PERSON)

        return list(entities.values()), relations


class EntityGraphStore:
    """Persistent graph table storage inside SQLite for Context Loom."""

    def __init__(self, store: LoomStore) -> None:
        self.store = store
        self.connection = store.connection
        self._init_tables()

    def _init_tables(self) -> None:
        with self.connection:
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS entity_nodes (
                    canonical_name TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS entity_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    target TEXT NOT NULL,
                    record_id TEXT,
                    confidence REAL DEFAULT 1.0,
                    FOREIGN KEY(source) REFERENCES entity_nodes(canonical_name),
                    FOREIGN KEY(target) REFERENCES entity_nodes(canonical_name)
                );
                CREATE TABLE IF NOT EXISTS record_entities (
                    record_id TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    PRIMARY KEY(record_id, canonical_name)
                );
                CREATE TABLE IF NOT EXISTS concept_nodes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    dense_vector TEXT,
                    created_at REAL NOT NULL,
                    parent_id TEXT,
                    y_depth INTEGER DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS x_y_cross_axis_links (
                    x_canonical_text TEXT NOT NULL,
                    y_concept_id TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    PRIMARY KEY(x_canonical_text, y_concept_id)
                );
                CREATE TABLE IF NOT EXISTS y_node_turn_activations (
                    node_id TEXT NOT NULL,
                    turn_record_id TEXT NOT NULL,
                    activated_at REAL NOT NULL,
                    PRIMARY KEY(node_id, turn_record_id)
                );
                CREATE TABLE IF NOT EXISTS concept_record_links (
                    concept_id TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    PRIMARY KEY(concept_id, record_id)
                );
                CREATE TABLE IF NOT EXISTS relation_cooccurrences (
                    relation_a TEXT NOT NULL,
                    relation_b TEXT NOT NULL,
                    cooccurrence_count INTEGER DEFAULT 1,
                    shared_target TEXT,
                    last_activated REAL NOT NULL,
                    PRIMARY KEY(relation_a, relation_b)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_edges_identity
                    ON entity_edges(source, relation, target, COALESCE(record_id, ''));
                CREATE INDEX IF NOT EXISTS idx_concept_record_links_concept ON concept_record_links(concept_id);
                CREATE INDEX IF NOT EXISTS idx_concept_record_links_record ON concept_record_links(record_id);
                CREATE INDEX IF NOT EXISTS idx_entity_edges_source ON entity_edges(source);
                CREATE INDEX IF NOT EXISTS idx_entity_edges_target ON entity_edges(target);
                CREATE INDEX IF NOT EXISTS idx_entity_edges_relation ON entity_edges(relation);
                CREATE INDEX IF NOT EXISTS idx_record_entities_name ON record_entities(canonical_name);
                CREATE INDEX IF NOT EXISTS idx_record_entities_record_id ON record_entities(record_id);
                CREATE INDEX IF NOT EXISTS idx_relation_cooccurrences_a ON relation_cooccurrences(relation_a);
            """)

    def add_entity(self, name: str, kind: EntityKind = EntityKind.OTHER) -> str:
        clean = name.strip(" \t\n\r\"'.,;:!?()[]{}")
        canon = clean.casefold()
        if not canon:
            return ""
        kind_val = kind.value if isinstance(kind, Enum) else str(kind)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO entity_nodes (canonical_name, name, kind)
                VALUES (?, ?, ?)
                ON CONFLICT(canonical_name) DO UPDATE SET
                    kind = CASE
                        WHEN entity_nodes.kind = 'other' AND excluded.kind != 'other' THEN excluded.kind
                        WHEN entity_nodes.kind = 'person' AND excluded.kind NOT IN ('person', 'other') THEN excluded.kind
                        ELSE entity_nodes.kind
                    END,
                    name = CASE
                        WHEN length(excluded.name) > length(entity_nodes.name) THEN excluded.name
                        ELSE entity_nodes.name
                    END
                """,
                (canon, clean, kind_val),
            )
        return canon

    def add_relation(
        self,
        source: str,
        relation: RelationKind | str,
        target: str,
        record_id: str | None = None,
        confidence: float = 1.0,
        source_kind: EntityKind | None = None,
        target_kind: EntityKind | None = None,
    ) -> None:
        src_kind = source_kind or EntityKind.OTHER
        tgt_kind = target_kind or EntityKind.OTHER
        src_canon = self.add_entity(source, src_kind)
        tgt_canon = self.add_entity(target, tgt_kind)
        rel_str = relation.value if isinstance(relation, Enum) else str(relation)
        if not src_canon or not tgt_canon:
            return
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO entity_edges (source, relation, target, record_id, confidence) VALUES (?, ?, ?, ?, ?)",
                (src_canon, rel_str, tgt_canon, record_id, confidence),
            )

    def link_record_entity(
        self,
        record_id: str,
        entity_name: str,
        entity_kind: EntityKind | None = None,
    ) -> None:
        clean = entity_name.strip(" \t\n\r\"'.,;:!?()[]{}")
        canon = clean.casefold()
        if not canon:
            return
        self.add_entity(clean, entity_kind or EntityKind.OTHER)
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO record_entities (record_id, canonical_name) VALUES (?, ?)",
                (record_id, canon),
            )

    def add_cooccurrence(
        self,
        entities: Sequence[str],
        record_id: str,
        temporal_weight: float = 1.0,
    ) -> None:
        if not entities or not record_id:
            return

        canon_map: dict[str, str] = {}
        for ent in entities:
            clean = ent.strip(" \t\n\r\"'.,;:!?()[]{}")
            canon = clean.casefold()
            if canon and canon not in canon_map:
                canon_map[canon] = clean

        canons = list(canon_map.keys())
        if not canons:
            return

        # Link all entities to record
        for canon in canons:
            self.link_record_entity(record_id, canon_map[canon])

        # Pairwise reciprocal CO_OCCURRED_WITH edges
        rel_str = RelationKind.CO_OCCURRED_WITH.value
        with self.connection:
            for i in range(len(canons)):
                for j in range(i + 1, len(canons)):
                    c1, c2 = canons[i], canons[j]
                    # Direct direction
                    self.connection.execute(
                        """
                        INSERT OR IGNORE INTO entity_edges (source, relation, target, record_id, confidence)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (c1, rel_str, c2, record_id, temporal_weight),
                    )
                    # Reciprocal direction
                    self.connection.execute(
                        """
                        INSERT OR IGNORE INTO entity_edges (source, relation, target, record_id, confidence)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (c2, rel_str, c1, record_id, temporal_weight),
                    )

    def index_record(
        self,
        record_id: str,
        text: str,
        *,
        speaker: str | None = None,
        extractor: EntityExtractor | None = None,
    ) -> tuple[list[Entity], list[Relation]]:
        """Index one record through the shared entity, relation, and speaker path."""
        active_extractor = extractor or EntityExtractor()
        entities, relations = active_extractor.extract(text)
        for entity in entities:
            self.link_record_entity(record_id, entity.name, entity.kind)
        names = [entity.name for entity in entities]
        if speaker and speaker.strip():
            self.link_record_entity(record_id, speaker, EntityKind.PERSON)
            names.insert(0, speaker)
        self.add_cooccurrence(names, record_id=record_id)
        for relation in relations:
            self.add_relation(
                relation.source,
                relation.relation,
                relation.target,
                record_id=record_id,
            )
        return entities, relations

    def neighbors(self, entity_name: str) -> list[dict[str, Any]]:
        canon = entity_name.casefold().strip(" \t\n\r\"'.,;:!?()[]{}")
        rows = self.connection.execute(
            """
            SELECT e.target AS entity, e.relation, e.record_id, e.confidence, n.name, n.kind
            FROM entity_edges e
            JOIN entity_nodes n ON e.target = n.canonical_name
            WHERE e.source = ?
            UNION
            SELECT e.source AS entity, e.relation, e.record_id, e.confidence, n.name, n.kind
            FROM entity_edges e
            JOIN entity_nodes n ON e.source = n.canonical_name
            WHERE e.target = ?
            """,
            (canon, canon),
        ).fetchall()
        return [
            {
                "entity": row["name"],
                "canonical_name": row["entity"],
                "relation": row["relation"],
                "record_id": row["record_id"],
                "confidence": row["confidence"],
                "kind": row["kind"],
            }
            for row in rows
        ]

    def records_for_entity(self, entity_name: str) -> list[str]:
        canon = entity_name.casefold().strip(" \t\n\r\"'.,;:!?()[]{}")
        rows = self.connection.execute(
            "SELECT record_id FROM record_entities WHERE canonical_name = ?", (canon,)
        ).fetchall()
        return [row["record_id"] for row in rows]

    def entity_degree(self, entity_canon: str) -> int:
        """Compute topological degree of entity (connected edges + linked records)."""
        edge_row = self.connection.execute(
            "SELECT COUNT(*) AS c FROM entity_edges WHERE source = ? OR target = ?",
            (entity_canon, entity_canon),
        ).fetchone()
        edge_count = edge_row["c"] if edge_row else 0

        rec_row = self.connection.execute(
            "SELECT COUNT(*) AS c FROM record_entities WHERE canonical_name = ?",
            (entity_canon,),
        ).fetchone()
        rec_count = rec_row["c"] if rec_row else 0

        return edge_count + rec_count

    def multi_hop_records(
        self,
        seed_entity_names: Sequence[str],
        max_hops: int | None = None,
    ) -> list[dict[str, Any]]:
        visited_entities: set[str] = set()
        visited_records: dict[str, dict[str, Any]] = {}

        current_frontier = [name.casefold().strip(" \t\n\r\"'.,;:!?()[]{}") for name in seed_entity_names if name.strip()]

        hop = 1
        while current_frontier and (max_hops is None or hop <= max_hops):
            next_frontier: list[str] = []
            base_hop_score = max(0.40, 0.95 * (0.85 ** (hop - 1)))

            for entity_canon in current_frontier:
                if not entity_canon or entity_canon in visited_entities:
                    continue
                visited_entities.add(entity_canon)

                deg = self.entity_degree(entity_canon)
                # Log2 degree discounting: W(e) = 1.0 / log2(2.0 + deg(e))
                deg_weight = 1.0 / math.log2(2.0 + deg)

                # Direct record links
                recs = self.records_for_entity(entity_canon)
                for rec_id in recs:
                    rec_score = base_hop_score * deg_weight
                    if rec_id in visited_records:
                        existing = visited_records[rec_id]
                        existing["score"] = 1.0 - (1.0 - existing["score"]) * (1.0 - rec_score)
                        existing["seed_count"] = existing.get("seed_count", 1) + 1
                    else:
                        visited_records[rec_id] = {
                            "record_id": rec_id,
                            "hop": hop,
                            "score": rec_score,
                            "via_entity": entity_canon,
                            "seed_count": 1,
                        }

                # Neighbor entity expansion
                neighs = self.neighbors(entity_canon)
                for n in neighs:
                    n_canon = n["canonical_name"]
                    n_deg = self.entity_degree(n_canon)
                    n_deg_weight = 1.0 / math.log2(2.0 + n_deg)
                    path_weight = math.sqrt(deg_weight * n_deg_weight) * float(n.get("confidence", 1.0))
                    is_cooccurrence = n["relation"] == RelationKind.CO_OCCURRED_WITH.value
                    if is_cooccurrence:
                        path_weight *= 0.20
                    edge_score = base_hop_score * path_weight

                    if n["record_id"]:
                        n_rec_id = n["record_id"]
                        if n_rec_id in visited_records:
                            existing = visited_records[n_rec_id]
                            existing["score"] = 1.0 - (1.0 - existing["score"]) * (1.0 - edge_score)
                            existing["seed_count"] = existing.get("seed_count", 1) + 1
                        else:
                            visited_records[n_rec_id] = {
                                "record_id": n_rec_id,
                                "hop": hop,
                                "score": edge_score,
                                "via_entity": f"{entity_canon}->{n['relation']}->{n_canon}",
                                "seed_count": 1,
                            }
                    if (
                        not is_cooccurrence
                        and n_canon not in visited_entities
                        and n_canon not in next_frontier
                    ):
                        next_frontier.append(n_canon)

            current_frontier = next_frontier
            hop += 1

        return list(visited_records.values())

    def add_concept_node(
        self,
        concept_id: str,
        name: str,
        kind: EntityKind = EntityKind.OTHER,
        dense_vector: list[float] | None = None,
        created_at: float | None = None,
        parent_id: str | None = None,
        y_depth: int = 1,
    ) -> VectorConceptNode:
        timestamp = time.time() if created_at is None else created_at
        vec_str = json.dumps(dense_vector) if dense_vector else ""
        kind_val = kind.value if isinstance(kind, Enum) else str(kind)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO concept_nodes (id, name, kind, dense_vector, created_at, parent_id, y_depth)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    kind = excluded.kind,
                    parent_id = COALESCE(excluded.parent_id, concept_nodes.parent_id),
                    y_depth = excluded.y_depth,
                    dense_vector = CASE
                        WHEN length(excluded.dense_vector) > 0 THEN excluded.dense_vector
                        ELSE concept_nodes.dense_vector
                    END
                """,
                (concept_id, name, kind_val, vec_str, timestamp, parent_id, y_depth),
            )
        row = self.connection.execute(
            "SELECT name, kind, dense_vector, created_at, parent_id, y_depth FROM concept_nodes WHERE id = ?",
            (concept_id,),
        ).fetchone()
        stored_vector = list(json.loads(row["dense_vector"])) if row and row["dense_vector"] else []
        stored_kind = EntityKind(row["kind"]) if row and row["kind"] in {item.value for item in EntityKind} else kind
        return VectorConceptNode(
            id=concept_id,
            name=row["name"] if row else name,
            kind=stored_kind,
            dense_vector=stored_vector,
            created_at=float(row["created_at"]) if row else timestamp,
            parent_id=row["parent_id"] if row else parent_id,
            y_depth=int(row["y_depth"]) if row and row["y_depth"] is not None else y_depth,
        )

    def link_concept_record(self, concept_id: str, record_id: str, weight: float = 1.0) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO concept_record_links (concept_id, record_id, weight) VALUES (?, ?, ?)",
                (concept_id, record_id, weight),
            )

    def record_relation_cooccurrence(
        self,
        relation_a: str,
        relation_b: str,
        shared_target: str = "",
        timestamp: float | None = None,
    ) -> None:
        """Record dynamic co-occurrence between relation verbs in dialogue memory."""
        rel_a, rel_b = sorted([relation_a.casefold(), relation_b.casefold()])
        ts = time.time() if timestamp is None else timestamp
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO relation_cooccurrences (relation_a, relation_b, cooccurrence_count, shared_target, last_activated)
                VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(relation_a, relation_b) DO UPDATE SET
                    cooccurrence_count = cooccurrence_count + 1,
                    shared_target = CASE WHEN length(excluded.shared_target) > 0 THEN excluded.shared_target ELSE shared_target END,
                    last_activated = excluded.last_activated
                """,
                (rel_a, rel_b, shared_target, ts),
            )

    def dynamic_relation_similarity(
        self,
        relation_a: str,
        relation_b: str,
        embedder: Any = None,
        now: float | None = None,
    ) -> float:
        """Estimate lexical relation similarity without treating co-occurrence as equivalence."""
        rel_a, rel_b = relation_a.casefold(), relation_b.casefold()
        if rel_a == rel_b:
            return 1.0

        # Signal 1: Cosine Similarity of Relation Embeddings
        cos_sim = 0.0
        if embedder is not None:
            try:
                v1 = embedder.embed(rel_a)
                v2 = embedder.embed(rel_b)
                cos_sim = max(0.0, _cosine_sim(v1, v2))
            except Exception:
                cos_sim = 0.0

        return min(1.0, max(0.0, cos_sim))

    def dynamic_relation_cluster(
        self,
        relation: str,
        embedder: Any = None,
        threshold: float = 0.80,
    ) -> set[str]:
        """Find all relation verbs dynamically equivalent to the query relation."""
        rel_clean = relation.casefold()
        rows = self.connection.execute(
            "SELECT DISTINCT relation FROM entity_edges WHERE relation != 'co_occurred_with'"
        ).fetchall()
        known_relations = {r["relation"].casefold() for r in rows}
        known_relations.add(rel_clean)

        cluster = {rel_clean}
        for r in known_relations:
            if r != rel_clean:
                score = self.dynamic_relation_similarity(rel_clean, r, embedder=embedder)
                if score >= threshold:
                    cluster.add(r)
        return cluster

    def compact_concept_nodes(
        self,
        embedding_provider: Any = None,
        *,
        allowed_record_ids: set[str] | None = None,
    ) -> list[VectorConceptNode]:
        """Build exact relation concepts from the records eligible for this retrieval.

        Distinct predicates are deliberately kept distinct. Mere occurrence in
        the same utterance is evidence of association, not evidence that the
        predicates mean the same thing.
        """
        from collections import defaultdict
        edges = self.connection.execute(
            "SELECT source, relation, target, record_id FROM entity_edges "
            "WHERE relation != 'co_occurred_with' ORDER BY source, relation, target, record_id"
        ).fetchall()
        if allowed_record_ids is not None:
            edges = [edge for edge in edges if edge["record_id"] in allowed_record_ids]
        if not edges:
            return []

        clusters: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            key = (edge["source"].casefold(), edge["relation"].casefold())
            clusters[key].append(edge)

        concept_nodes: list[VectorConceptNode] = []
        for (source, relation_head), cluster_edges in clusters.items():
            targets = sorted(list({e["target"] for e in cluster_edges}))
            concept_name = f"{source.capitalize()} {relation_head.replace('_', ' ')}: {', '.join(targets)}"
            concept_id = f"concept:{source}:{relation_head}"

            existing = self.connection.execute(
                "SELECT name, dense_vector FROM concept_nodes WHERE id = ?",
                (concept_id,),
            ).fetchone()
            vec: list[float] = []
            if existing and existing["name"] == concept_name and existing["dense_vector"]:
                vec = list(json.loads(existing["dense_vector"]))
            elif embedding_provider is not None:
                vec = embedding_provider.embed(concept_name)

            node = self.add_concept_node(
                concept_id=concept_id,
                name=concept_name,
                kind=EntityKind.TOPIC,
                dense_vector=vec,
            )
            concept_nodes.append(node)

            cluster_record_ids = [edge["record_id"] for edge in cluster_edges if edge["record_id"]]
            if cluster_record_ids:
                placeholders = ",".join("?" for _ in cluster_record_ids)
                with self.connection:
                    self.connection.execute(
                        f"DELETE FROM concept_record_links WHERE concept_id = ? AND record_id IN ({placeholders})",
                        (concept_id, *cluster_record_ids),
                    )
            for edge in cluster_edges:
                if edge["record_id"]:
                    self.link_concept_record(concept_id, edge["record_id"], weight=1.0)

        return concept_nodes

    def y_cipher_depth_search(
        self,
        query: str,
        embedding_provider: Any = None,
        top_k: int | None = None,
        allowed_record_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform Y-cipher depth vector search through language-agnostic VectorConceptNodes."""
        if embedding_provider is None:
            return []

        try:
            query_vec = embedding_provider.embed(query)
        except Exception:
            return []

        if not query_vec:
            return []

        rows = self.connection.execute("SELECT id, name, kind, dense_vector FROM concept_nodes").fetchall()
        if not rows:
            return []

        concept_scores: list[tuple[str, str, float]] = []
        for r in rows:
            cid = r["id"]
            cname = r["name"]
            vec_str = r["dense_vector"]
            if vec_str:
                try:
                    c_vec = json.loads(vec_str)
                    sim = _cosine_sim(query_vec, c_vec)
                    if sim > 0.0:
                        # Density scaling: fetch number of linked records
                        link_count = self.connection.execute(
                            "SELECT COUNT(*) as cnt FROM concept_record_links WHERE concept_id = ?", (cid,)
                        ).fetchone()["cnt"]
                        density_factor = 1.0 + 0.15 * math.log(1 + link_count)
                        score = min(1.0, sim * density_factor * 1.35)
                        concept_scores.append((cid, cname, score))
                except Exception:
                    pass

        concept_scores.sort(key=lambda x: x[2], reverse=True)
        results: list[dict[str, Any]] = []
        seen_records: set[str] = set()

        for cid, cname, score in concept_scores:
            links = self.connection.execute(
                "SELECT record_id, weight FROM concept_record_links WHERE concept_id = ?", (cid,)
            ).fetchall()
            for link in links:
                rec_id = link["record_id"]
                if (
                    rec_id not in seen_records
                    and (allowed_record_ids is None or rec_id in allowed_record_ids)
                ):
                    seen_records.add(rec_id)
                    results.append({
                        "record_id": rec_id,
                        "concept_id": cid,
                        "concept_name": cname,
                        "score": score,
                    })

        return results if top_k is None else results[:top_k]

    def get_relation_cooccurrences(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT relation_a, relation_b, cooccurrence_count, shared_target, last_activated FROM relation_cooccurrences"
        ).fetchall()
        return [dict(row) for row in rows]

    def get_or_create_self_node(self) -> VectorConceptNode:
        """Get or create root Self node (origin for all Y-axis concept traversals)."""
        row = self.connection.execute(
            "SELECT id, name, kind, dense_vector, created_at, parent_id, y_depth FROM concept_nodes WHERE id = 'self_root'"
        ).fetchone()
        if row:
            return VectorConceptNode(
                id=row["id"],
                name=row["name"],
                kind=EntityKind.PERSON,
                dense_vector=json.loads(row["dense_vector"]) if row["dense_vector"] else [],
                created_at=float(row["created_at"]),
                parent_id=row["parent_id"],
                y_depth=int(row["y_depth"]),
            )

        node = self.add_concept_node(
            concept_id="self_root",
            name="Self Origin",
            kind=EntityKind.PERSON,
            dense_vector=[],
            parent_id=None,
            y_depth=0,
        )
        return node

    def create_vector_concept_node(
        self,
        name: str,
        embedding: list[float] | None = None,
        parent_id: str | None = None,
        y_depth: int = 1,
        kind: EntityKind = EntityKind.TOPIC,
    ) -> VectorConceptNode:
        import hashlib
        name_slug = re.sub(r'[^a-zA-Z0-9_]', '_', name.casefold())
        name_hash = hashlib.sha256(name.casefold().encode()).hexdigest()[:10]
        concept_id = f"concept_y{y_depth}_{name_slug[:16]}_{name_hash}"
        return self.add_concept_node(
            concept_id=concept_id,
            name=name,
            kind=kind,
            dense_vector=embedding,
            parent_id=parent_id,
            y_depth=y_depth,
        )

    def link_x_membrane_to_y_concept(self, x_canonical_text: str, y_concept_id: str, weight: float = 1.0) -> None:
        x_clean = x_canonical_text.casefold().strip()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO x_y_cross_axis_links (x_canonical_text, y_concept_id, weight)
                VALUES (?, ?, ?)
                ON CONFLICT(x_canonical_text, y_concept_id) DO UPDATE SET weight = excluded.weight
                """,
                (x_clean, y_concept_id, weight),
            )

    def y_cypher_find_paths(
        self,
        query: str,
        x_terms: Sequence[str],
        embedding_provider: Any = None,
        top_k: int = 5,
    ) -> list[YCypherPath]:
        """Find paths of least resistance from Self root -> Y-concept nodes -> X-membrane terms."""
        self_node = self.get_or_create_self_node()
        query_clean = query.casefold().strip()

        # Step 1: Find Y-concept nodes directly linked to query's X-membrane terms
        linked_concept_ids: set[str] = set()
        for x_term in x_terms:
            rows = self.connection.execute(
                "SELECT y_concept_id FROM x_y_cross_axis_links WHERE x_canonical_text = ?",
                (x_term.casefold().strip(),),
            ).fetchall()
            for r in rows:
                linked_concept_ids.add(r["y_concept_id"])

        # Step 2: Fetch candidate Y-concept nodes
        all_concepts = self.connection.execute(
            "SELECT id, name, kind, dense_vector, parent_id, y_depth FROM concept_nodes WHERE id != 'self_root'"
        ).fetchall()

        if not all_concepts:
            return []

        query_vec = embedding_provider.embed(query) if embedding_provider else []

        paths: list[YCypherPath] = []
        for c in all_concepts:
            cid = c["id"]
            cname = c["name"]
            pid = c["parent_id"] or self_node.id
            vec_str = c["dense_vector"]

            score = 0.0
            if vec_str and query_vec:
                try:
                    cvec = json.loads(vec_str)
                    sim = _cosine_sim(query_vec, cvec)
                    score = max(0.0, sim)
                except Exception:
                    pass

            if cid in linked_concept_ids:
                score = max(score, 0.5) * 1.5

            if score < 0.35:
                continue

            # Path sequence: Self -> Parent -> Child
            node_ids = [self_node.id]
            if pid != self_node.id:
                node_ids.append(pid)
            node_ids.append(cid)

            # Records in concept vault
            recs = self.connection.execute(
                "SELECT record_id FROM concept_record_links WHERE concept_id = ?", (cid,)
            ).fetchall()
            record_ids = [r["record_id"] for r in recs]

            paths.append(
                YCypherPath(
                    target_concept_id=cid,
                    target_concept_name=cname,
                    node_ids=node_ids,
                    total_score=min(1.0, score),
                    record_ids=record_ids,
                )
            )

        paths.sort(key=lambda p: p.total_score, reverse=True)
        return paths[:top_k]

    def get_records_in_concept_vaults_for_term(self, term: str) -> list[str]:
        term_clean = term.casefold().strip()
        rows = self.connection.execute(
            """
            SELECT DISTINCT l.record_id
            FROM x_y_cross_axis_links x
            JOIN concept_record_links l ON x.y_concept_id = l.concept_id
            WHERE x.x_canonical_text = ?
            """,
            (term_clean,),
        ).fetchall()
        return [r["record_id"] for r in rows]

    def activate_y_node_turn(self, node_id: str, turn_record_id: str) -> None:
        """Record turn-tied activation metadata on Y-concept node."""
        now = time.time()
        with self.connection:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO y_node_turn_activations (node_id, turn_record_id, activated_at)
                VALUES (?, ?, ?)
                """,
                (node_id, turn_record_id, now),
            )


def _cosine_sim(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (max(1e-9, n1) * max(1e-9, n2))


@dataclass(frozen=True)
class YCypherPath:
    target_concept_id: str
    target_concept_name: str
    node_ids: list[str]
    total_score: float
    record_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VectorConceptNode:
    id: str
    name: str
    kind: EntityKind
    dense_vector: list[float] = field(default_factory=list)
    created_at: float = 0.0
    parent_id: str | None = None
    y_depth: int = 1

