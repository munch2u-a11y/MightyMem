"""X-Axis Membrane and Shadow X-Cypher for surface linguistic & n-gram activation."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import sqlite3
import time
from typing import Sequence

from .store import LoomStore


@dataclass
class XMembraneNode:
    canonical_text: str
    kind: str = "word"  # word, phrase, entity_relation
    activation_count: int = 1
    last_activated_at: float = field(default_factory=time.time)


@dataclass
class XMembraneEdge:
    source: str
    target: str
    co_occurrence_count: int = 1
    weight: float = 1.0


class ShadowXCypher:
    """Shadow X-Cypher for parsing raw inputs into membrane nodes, edges, and n-gram groupings."""

    def __init__(self) -> None:
        self._stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "in", "on", "at", "to", "for", "with", "by", "of", "from", "and", "or",
            "but", "if", "my", "your", "his", "her", "their", "our", "it", "this", "that",
        }

    def parse_input(self, text: str) -> tuple[list[str], list[tuple[str, ...]]]:
        """Extract unigram terms, n-gram phrases, and entity-relation tuple groupings."""
        words = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.casefold())
        meaningful_terms = [w for w in words if w not in self._stopwords]

        # Extract n-grams and tuple groupings (e.g. ("tom", "enjoys", "painting"))
        tuple_groupings: list[tuple[str, ...]] = []
        clean_words = [w for w in words if w not in self._stopwords or w in ("in", "at", "by", "to")]

        for i in range(len(clean_words) - 1):
            tuple_groupings.append(tuple(clean_words[i : i + 2]))
            if i < len(clean_words) - 2:
                tuple_groupings.append(tuple(clean_words[i : i + 3]))

        all_terms = list(dict.fromkeys(meaningful_terms + [w for tup in tuple_groupings for w in tup]))
        return all_terms, tuple_groupings


class XMembrane:
    """X-Axis Membrane tracking rolling BM25/n-gram activation counters and regional softmax weights."""

    def __init__(self, store: LoomStore) -> None:
        self.store = store
        self.connection = store.connection
        self._init_tables()

    def _init_tables(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS x_membrane_nodes (
                    canonical_text TEXT PRIMARY KEY,
                    kind TEXT NOT NULL DEFAULT 'word',
                    activation_count INTEGER NOT NULL DEFAULT 1,
                    last_activated_at REAL NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS x_membrane_edges (
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    co_occurrence_count INTEGER NOT NULL DEFAULT 1,
                    weight REAL NOT NULL DEFAULT 1.0,
                    PRIMARY KEY (source, target)
                )
                """
            )

    def activate(self, items: Sequence[str]) -> None:
        """Increment activation counters (+1) for words/phrases/edges."""
        now = time.time()
        canonical_items = [item.casefold().strip() for item in items if item.strip()]

        with self.connection:
            for item in canonical_items:
                kind = "phrase" if " " in item else "word"
                self.connection.execute(
                    """
                    INSERT INTO x_membrane_nodes (canonical_text, kind, activation_count, last_activated_at)
                    VALUES (?, ?, 1, ?)
                    ON CONFLICT(canonical_text) DO UPDATE SET
                        activation_count = activation_count + 1,
                        last_activated_at = ?
                    """,
                    (item, kind, now, now),
                )

            # Update co-occurrence edges between consecutive items
            for i in range(len(canonical_items) - 1):
                src, tgt = sorted([canonical_items[i], canonical_items[i + 1]])
                if src != tgt:
                    self.connection.execute(
                        """
                        INSERT INTO x_membrane_edges (source, target, co_occurrence_count, weight)
                        VALUES (?, ?, 1, 1.0)
                        ON CONFLICT(source, target) DO UPDATE SET
                            co_occurrence_count = co_occurrence_count + 1
                        """,
                        (src, tgt),
                    )

    def get_node(self, canonical_text: str) -> XMembraneNode | None:
        text = canonical_text.casefold().strip()
        cursor = self.connection.execute(
            "SELECT canonical_text, kind, activation_count, last_activated_at FROM x_membrane_nodes WHERE canonical_text = ?",
            (text,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return XMembraneNode(
            canonical_text=row[0],
            kind=row[1],
            activation_count=row[2],
            last_activated_at=row[3],
        )

    def get_regional_softmax_weights(self, query_terms: Sequence[str]) -> dict[str, float]:
        """Compute regional softmax weights across query terms based on activation frequency."""
        canonical_terms = list(dict.fromkeys(t.casefold().strip() for t in query_terms if t.strip()))
        if not canonical_terms:
            return {}

        raw_scores: dict[str, float] = {}
        for term in canonical_terms:
            node = self.get_node(term)
            count = node.activation_count if node else 1
            raw_scores[term] = float(count)

        max_score = max(raw_scores.values())
        exp_scores = {term: math.exp(score - max_score) for term, score in raw_scores.items()}
        sum_exp = sum(exp_scores.values())
        if sum_exp == 0:
            return {term: 1.0 / len(canonical_terms) for term in canonical_terms}

        return {term: score / sum_exp for term, score in exp_scores.items()}
