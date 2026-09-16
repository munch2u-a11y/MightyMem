"""Multi-Hop Graph Propagation retrieval lane for Context Loom."""
from __future__ import annotations

import re
from typing import Sequence

from .entity_graph import EntityExtractor, EntityGraphStore
from .models import Candidate
from .retrieval import RetrievalContext, RetrievalLane, candidate_for
from .store import LoomStore

QUESTION_STOPWORDS = {
    "where", "what", "when", "who", "whom", "whose", "why", "how", "which",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "have", "has", "had", "can", "could", "would", "should", "will", "shall",
    "tell", "find", "show", "give", "explain", "both", "all", "each", "in",
    "common", "between", "the", "this", "that", "there", "here", "my", "your",
    "his", "her", "their", "our", "and", "or", "but", "with", "from", "for",
    "about", "to", "at", "by", "after", "before",
}


DECLARATIVE_ATTRIBUTE_TERMS = {
    "identity": {"trans", "transgender", "woman", "man", "gender", "pronouns", "identity"},
    "relationship": {"partner", "husband", "wife", "spouse", "married", "single", "dating", "together", "relationship"},
    "status": {"partner", "husband", "wife", "spouse", "married", "single", "dating", "together"},
    "career": {"counseling", "counselor", "job", "work", "degree", "school", "university", "field", "profession", "occupation", "career"},
    "education": {"school", "degree", "university", "college", "studied", "study", "major", "field", "education"},
    "activities": {"hobby", "painting", "paint", "running", "race", "hiking", "sports", "music", "art", "activities", "activity"},
    "interests": {"hobby", "likes", "enjoys", "interested", "passion", "favorite", "interests", "interest"},
    "where": {"from", "lived", "relocated", "moved", "born", "city", "state", "house", "apartment"},
    "moved": {"from", "lived", "relocated", "moved", "born", "city"},
    "move": {"from", "lived", "relocated", "moved", "born", "city"},
}


class MultiHopGraphLane(RetrievalLane):
    """Nominate candidates reached via multi-hop entity-relational graph propagation."""

    name = "multi_hop_graph"

    def __init__(
        self,
        store: LoomStore,
        graph_store: EntityGraphStore | None = None,
        extractor: EntityExtractor | None = None,
        max_hops: int | None = None,
        top_k: int | None = None,
    ) -> None:
        self.store = store
        self.graph_store = graph_store or EntityGraphStore(store)
        self.extractor = extractor or EntityExtractor()
        self.max_hops = max_hops
        self.top_k = top_k

    def nominate(self, context: RetrievalContext) -> Sequence[Candidate]:
        # Extract query seed entities
        extracted_entities, _ = self.extractor.extract(context.query)
        entity_names = [
            e.name for e in extracted_entities
            if e.canonical_name not in QUESTION_STOPWORDS and len(e.canonical_name) >= 2
        ]

        # Also pull any capitalized tokens from query as potential seed entities (filtering stopwords)
        capitalized = re.findall(r"\b[A-Z][a-z]{1,15}\b", context.query)
        clean_capitalized = [
            cap for cap in capitalized
            if cap.casefold() not in QUESTION_STOPWORDS and len(cap) >= 2
        ]
        all_seeds = list(dict.fromkeys(entity_names + clean_capitalized))

        if not all_seeds:
            return []

        multi_hop_hits = self.graph_store.multi_hop_records(all_seeds, max_hops=self.max_hops)

        # Query terms for relevance conditioning
        query_terms = {
            t.casefold()
            for t in re.findall(r"\b[a-zA-Z0-9_]{3,}\b", context.query)
            if t.casefold() not in QUESTION_STOPWORDS
        }

        candidates: list[Candidate] = []
        seen_record_ids: set[str] = set()
        eligible_records = {record.id: record for record in context.records}

        for hit in multi_hop_hits:
            rec_id = hit["record_id"]
            record = eligible_records.get(rec_id)
            if record is None:
                continue

            seen_record_ids.add(rec_id)
            base_score = float(hit.get("score", 0.5))

            # Query-relevance conditioning
            record_words = re.findall(r"\b[a-zA-Z0-9_]{3,}\b", record.content.lower())
            record_terms = set(record_words)
            overlap = len(query_terms & record_terms)
            relevance_boost = 1.0
            if query_terms:
                relevance_boost += 0.5 * (overlap / len(query_terms))

            # Entity-Attribute Anchoring: boost records linked to seed entities that express matching declarative facts
            for q_term in query_terms:
                if q_term in DECLARATIVE_ATTRIBUTE_TERMS:
                    attr_overlap = DECLARATIVE_ATTRIBUTE_TERMS[q_term] & record_terms
                    if attr_overlap:
                        relevance_boost = max(relevance_boost, 2.2)
                        break

            # Conversational filler dampening (e.g. "Jon: Bye!", "Gina: ;)")
            length_factor = 0.35 if len(record.content.split()) <= 3 else 1.0

            final_score = max(0.05, min(1.0, base_score * relevance_boost * length_factor))
            via_entity = hit.get("via_entity", "entity link")
            candidates.append(
                candidate_for(
                    record,
                    self.name,
                    final_score,
                    f"multi-hop graph propagation ({via_entity})",
                )
            )

        candidates.sort(
            key=lambda c: c.lane_scores.get(self.name, 0.0),
            reverse=True,
        )
        return candidates if self.top_k is None else candidates[: self.top_k]


class YCipherDepthSearchLane(RetrievalLane):
    """Nominate candidates reached via language-agnostic VectorConceptNode depth search."""

    name = "y_cipher_depth"

    def __init__(
        self,
        store: LoomStore,
        graph_store: EntityGraphStore | None = None,
        top_k: int | None = None,
    ) -> None:
        self.store = store
        self.graph_store = graph_store or EntityGraphStore(store)
        self.top_k = top_k

    def nominate(self, context: RetrievalContext) -> Sequence[Candidate]:
        if not context.embedding_provider:
            return []
        eligible_records = {record.id: record for record in context.records}
        eligible_ids = set(eligible_records)
        y_hits = self.graph_store.y_cipher_depth_search(
            context.query,
            context.embedding_provider,
            top_k=self.top_k,
            allowed_record_ids=eligible_ids,
        )
        candidates: list[Candidate] = []
        seen: set[str] = set()
        for y_hit in y_hits:
            rec_id = y_hit["record_id"]
            if rec_id not in seen:
                seen.add(rec_id)
                record = eligible_records.get(rec_id)
                if record is not None:
                    candidates.append(
                        candidate_for(
                            record,
                            self.name,
                            float(y_hit["score"]),
                            f"y-cipher vector concept {y_hit['concept_name']}",
                        )
                    )
        return candidates
