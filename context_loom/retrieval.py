from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from .models import Candidate, EvidenceStatus, MemoryRecord, RecordKind, TaskState
from .store import LoomStore


WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")
PUNCT_STRIP = ".:,;!?-_/()[]{}'\""
STOPWORDS = {
    "a", "about", "after", "an", "and", "are", "as", "at", "be", "been", "before",
    "by", "can", "could", "d", "did", "do", "does", "find", "for", "from", "had",
    "has", "have", "how", "i", "in", "is", "it", "its", "know", "ll", "m", "me",
    "my", "now", "of", "on", "or", "our", "project", "re", "s", "should", "show",
    "tell", "that", "the", "their", "them", "then", "this", "to", "use", "used",
    "uses", "ve", "was", "we", "were", "what", "when", "where", "which", "who",
    "whom", "whose", "why", "will", "with", "you", "your",
}
HISTORICAL_RE = re.compile(
    r"\b(previous|previously|original|originally|used to|historical|history|at the time|"
    r"prior state|old version|before (?:the )?(?:change|correction|migration|update))\b",
    re.IGNORECASE,
)
FUTURE_RE = re.compile(
    r"\b(future|planned|plan to|scheduled|upcoming|will|going to)\b",
    re.IGNORECASE,
)


def stem_token(word: str) -> str:
    if len(word) <= 3 or any(char.isdigit() for char in word):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 4:
        return word[:-3]
    if word.endswith("ed") and len(word) > 3:
        return word[:-2]
    if word.endswith("es") and len(word) > 3:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def terms(text: str) -> list[str]:
    result: list[str] = []
    for token in WORD_RE.findall(text):
        cleaned = token.strip(PUNCT_STRIP).lower()
        if cleaned:
            result.append(cleaned)
    return result


def key_terms(text: str) -> set[str]:
    result = {token for token in terms(text) if token not in STOPWORDS}
    for match in re.findall(r"\b\d+(?:\.\d+)*\b|\b(?:not|never|no|without)\b", text.lower()):
        result.add(match)
    return result


def scope_blind_terms(text: str, scope: str) -> set[str]:
    """Remove terms already guaranteed by the store's scope filter.

    Repeating a project name in every record is useful to a reader but carries
    no ranking information once retrieval is already isolated to that project.
    """

    return key_terms(text) - key_terms(scope)


HEADER_NOISE_RE = re.compile(r"^\[D\d+:\d+ \| [^\]]+\]\s*(?:[A-Za-z0-9_]+:\s*)?")
TURN_HEADER_RE = re.compile(
    r"^\[(?:[^\]]*\|[^\]]*|\d{4}-\d{2}-\d{2}[^\]]*|\d{1,2}:\d{2}[^\]]*|session_\d+[^\]]*)\]\s*",
    re.IGNORECASE,
)
SPEAKER_PREFIX_RE = re.compile(
    r"^(?:User|Assistant|Observation|Correction|Fact|Verified observation|Verified procedure|Procedure candidate|Learned Habitus behavioral pattern|User preference|Personal preference|Agent behavioral tendency|[A-Z][a-z]{1,20}):\s*"
)


def clean_embedding_payload(text: str, scope: str = "") -> str:
    cleaned = text.strip()
    while True:
        match = TURN_HEADER_RE.match(cleaned)
        if not match:
            break
        cleaned = cleaned[match.end():].strip()

    cleaned = SPEAKER_PREFIX_RE.sub("", cleaned).strip()

    if scope:
        for term in key_terms(scope):
            cleaned = re.sub(rf"\b{re.escape(term)}\b", "", cleaned, flags=re.IGNORECASE)

    cleaned = " ".join(cleaned.split())
    return cleaned if cleaned else text.strip()


def scope_blind_text(text: str, scope: str) -> str:
    clean_text = HEADER_NOISE_RE.sub("", text)
    scope_terms = key_terms(scope)
    return " ".join(token for token in terms(clean_text) if token not in scope_terms)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    ln = math.sqrt(sum(a * a for a in left))
    rn = math.sqrt(sum(b * b for b in right))
    return dot / (ln * rn) if ln and rn else 0.0


class EmbeddingProvider(Protocol):
    identity: str

    def embed(self, text: str) -> list[float]: ...


class HashingSemanticEmbedder:
    """Portable deterministic fallback with a small auditable concept lexicon.

    It is suitable for tests and offline operation. It is not represented as a
    learned sentence embedding model; production evaluation can plug in Ollama.
    """

    DEFAULT_CONCEPTS = {
        "where": "location",
        "venue": "location",
        "place": "location",
        "located": "location",
        "save": "write",
        "create": "write",
        "wrote": "write",
        "read": "inspect",
        "open": "inspect",
        "view": "inspect",
        "postgres": "postgresql",
        "db": "database",
        "remove": "delete",
        "erase": "delete",
        "likes": "preference",
        "prefers": "preference",
        "favorite": "preference",
        "error": "failure",
        "failed": "failure",
    }

    def __init__(self, dimensions: int = 384, concepts: dict[str, str] | None = None) -> None:
        self.dimensions = dimensions
        self.concepts = {**self.DEFAULT_CONCEPTS, **(concepts or {})}
        concept_hash = hashlib.sha256(json.dumps(self.concepts, sort_keys=True).encode()).hexdigest()[:12]
        self.identity = f"hashing-semantic-v1:{dimensions}:{concept_hash}"

    def embed(self, text: str) -> list[float]:
        toks = [
            self.concepts.get(token, token)
            for token in terms(text)
            if token not in STOPWORDS
        ]
        features = toks + [f"{a}::{b}" for a, b in zip(toks, toks[1:])]
        vector = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OllamaEmbeddingProvider:
    """Optional local semantic provider using available acceleration."""

    def __init__(
        self,
        model: str = "qwen3-embedding:0.6b",
        endpoint: str = "http://127.0.0.1:11434",
        timeout: float | None = None,
        num_gpu: int = -1,
    ) -> None:
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.base_url = self.endpoint
        self.timeout = timeout
        self.num_gpu = num_gpu
        self.identity = f"ollama:{model}:num_gpu={num_gpu}"

    def embed(self, text: str) -> list[float]:
        safe_text = text.strip() if text and text.strip() else "empty text"
        payload = json.dumps(
            {"model": self.model, "input": safe_text, "keep_alive": -1,
             "truncate": True, "options": {"num_gpu": self.num_gpu}}
        ).encode()
        request = urllib.request.Request(
            self.endpoint + "/api/embed", data=payload, headers={"Content-Type": "application/json"}
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read())
                embeddings = data.get("embeddings") or []
                if embeddings:
                    return [float(value) for value in embeddings[0]]
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(0.2)
        raise RuntimeError("Ollama returned no embedding")

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        safe_texts = [t.strip() if t and t.strip() else "empty text" for t in texts]
        payload = json.dumps(
            {"model": self.model, "input": safe_texts, "keep_alive": -1,
             "truncate": True, "options": {"num_gpu": self.num_gpu}}
        ).encode()
        request = urllib.request.Request(
            self.endpoint + "/api/embed", data=payload, headers={"Content-Type": "application/json"}
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read())
                embeddings = data.get("embeddings") or []
                if embeddings:
                    return [[float(v) for v in vector] for vector in embeddings]
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(0.2)
        raise RuntimeError("Ollama returned no embeddings")


@dataclass(slots=True)
class RetrievalContext:
    query: str
    scope: str
    now: float
    records: list[MemoryRecord]
    store: LoomStore
    task: TaskState | None = None
    historical: bool = False
    embedding_provider: EmbeddingProvider | None = None


class RetrievalLane(Protocol):
    name: str

    def nominate(self, context: RetrievalContext) -> list[Candidate]: ...


def semantic_index_deferred(context: RetrievalContext, record: MemoryRecord) -> bool:
    """A current tool return is already present in task state and model tool history."""
    return bool(
        context.task
        and record.metadata.get("deferred_semantic_for_task") == context.task.id
    )


def section_for(record: MemoryRecord) -> str:
    if record.kind in {RecordKind.USER, RecordKind.ASSISTANT}:
        return "continuity"
    if record.kind == RecordKind.PREFERENCE:
        return "personal"
    if record.kind == RecordKind.PROCEDURE:
        return "procedures"
    if record.kind in {RecordKind.TOOL_RESULT, RecordKind.OBSERVATION}:
        return "observations"
    if record.kind in {RecordKind.FACT, RecordKind.EPISODE, RecordKind.CORRECTION, RecordKind.CONFLICT}:
        return "evidence"
    if record.kind == RecordKind.THOUGHT:
        return "private"
    return "evidence"


def render_record(record: MemoryRecord, *, historical: bool = False) -> str:
    source = f"record:{record.id}"
    effective = f" effective:{record.effective_at:g}" if record.effective_at is not None else ""
    if record.kind == RecordKind.ASSISTANT:
        prefix = "Earlier assistant statement (unverified)"
    elif record.kind == RecordKind.USER:
        prefix = "Conversation"
    elif record.kind == RecordKind.CORRECTION:
        prefix = "Correction"
    elif record.kind == RecordKind.CONFLICT:
        prefix = "Unresolved conflict"
    elif record.kind == RecordKind.PREFERENCE:
        if record.metadata.get("origin") == "habitus_promoted_pattern":
            prefix = "Learned Habitus behavioral pattern"
        elif record.metadata.get("preference_owner") == "agent":
            prefix = "Agent behavioral tendency"
        elif record.metadata.get("preference_owner") == "human":
            prefix = "User preference"
        else:
            prefix = "Personal preference"
    elif record.kind == RecordKind.PROCEDURE:
        prefix = "Verified procedure" if record.evidence_status == EvidenceStatus.TOOL_VERIFIED else "Procedure candidate"
    elif record.kind in {RecordKind.TOOL_RESULT, RecordKind.OBSERVATION}:
        prefix = "Verified observation" if record.evidence_status == EvidenceStatus.TOOL_VERIFIED else "Observation"
    elif record.kind == RecordKind.FACT:
        prefix = "Supported claim" if record.source_ids else "Claim"
    else:
        prefix = record.kind.value.replace("_", " ").title()
    supersedes = f"; supersedes:{record.supersedes_id}" if record.supersedes_id else ""
    history = "; historical" if historical else ""
    relative_dates = record.metadata.get("resolved_relative_dates") or {}
    temporal = (
        " [system-derived relative dates: "
        + ", ".join(f"{term}={value}" for term, value in sorted(relative_dates.items()))
        + "]"
        if relative_dates
        else ""
    )
    return f"- {prefix}: {record.content}{temporal} [{source}{effective}{supersedes}{history}]"


def candidate_for(record: MemoryRecord, lane: str, score: float, reason: str) -> Candidate:
    native_sec = section_for(record)
    if native_sec in {"personal", "continuity"}:
        sec = native_sec
    elif lane in {"y_cipher_depth", "subjective"}:
        sec = "multi_hop"
    else:
        sec = native_sec
    return Candidate(
        id=f"record:{record.id}",
        text=render_record(record),
        section=sec,
        source_record_ids=(record.id,),
        lane_scores={lane: max(0.0, min(1.0, score))},
        reasons=[reason],
        evidence_status=record.evidence_status,
        effective_at=record.effective_at,
        importance=record.importance,
        metadata={
            "kind": record.kind.value,
            "source_ids": list(record.source_ids),
            "supersedes_id": record.supersedes_id,
            **record.metadata,
        },
    )


class SemanticLane:
    name = "semantic"

    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider

    def _embedding(self, context: RetrievalContext, record: MemoryRecord) -> list[float]:
        provider_key = self.provider.identity + ":clean-payload-v2"
        cached = context.store.get_embedding(record.id, provider_key)
        if cached is not None:
            return cached
        vector = self.provider.embed(clean_embedding_payload(record.content, context.scope))
        context.store.put_embedding(record.id, provider_key, vector)
        return vector

    def nominate(self, context: RetrievalContext) -> list[Candidate]:
        query_vector = self.provider.embed(scope_blind_text(context.query, context.scope))
        ranked: list[tuple[float, MemoryRecord]] = []
        for record in context.records:
            if (
                record.kind in {RecordKind.THOUGHT, RecordKind.TOOL_REQUEST}
                or semantic_index_deferred(context, record)
            ):
                continue
            score = max(0.0, cosine(query_vector, self._embedding(context, record)))
            ranked.append((score, record))
        ranked.sort(key=lambda item: (-item[0], -item[1].created_at, item[1].id))
        return [candidate_for(record, self.name, score, "semantic similarity") for score, record in ranked]


class ExactLane:
    name = "exact"

    def nominate(self, context: RetrievalContext) -> list[Candidate]:
        query_terms = scope_blind_terms(context.query, context.scope)
        if not query_terms:
            return []
        stemmed_query = {stem_token(t) for t in query_terms}
        results: list[tuple[float, MemoryRecord]] = []
        for record in context.records:
            if record.kind in {RecordKind.THOUGHT, RecordKind.TOOL_REQUEST}:
                continue
            cleaned_content = TURN_HEADER_RE.sub("", record.content)
            record_terms = scope_blind_terms(cleaned_content, context.scope)
            stemmed_record = {stem_token(t) for t in record_terms}
            exact_overlap = query_terms & record_terms
            stem_overlap = stemmed_query & stemmed_record
            overlap = exact_overlap | stem_overlap
            score = len(overlap) / max(1, len(query_terms))
            rare = {term for term in overlap if any(c.isdigit() for c in term) or len(term) >= 8}
            negation = overlap & {"not", "never", "no", "without"}
            score = min(1.0, score + 0.15 * len(rare) + 0.1 * len(negation))
            if overlap:
                results.append((score, record))
        results.sort(key=lambda item: (-item[0], -item[1].created_at, item[1].id))
        return [candidate_for(record, self.name, score, "exact terms, identifiers or values") for score, record in results]


class RecencyLane:
    name = "recent"

    def __init__(
        self, *, half_life_seconds: float = 3600.0, always_keep: int = 6
    ) -> None:
        self.half_life_seconds = half_life_seconds
        self.always_keep = always_keep

    def nominate(self, context: RetrievalContext) -> list[Candidate]:
        eligible = [
            record
            for record in context.records
            if record.kind
            in {RecordKind.USER, RecordKind.ASSISTANT, RecordKind.OBSERVATION, RecordKind.TOOL_RESULT}
        ]
        eligible.sort(key=lambda item: (-item.created_at, item.id))
        relevance_terms = scope_blind_terms(
            " ".join(
                [
                    context.query,
                    context.task.goal if context.task else "",
                    " ".join(context.task.focus_stack) if context.task else "",
                    " ".join(context.task.open_questions) if context.task else "",
                ]
            ),
            context.scope,
        )
        output = []
        for index, record in enumerate(eligible):
            age = max(0.0, context.now - record.created_at)
            freshness = math.exp(-math.log(2) * age / max(1.0, self.half_life_seconds))
            overlap = len(relevance_terms & scope_blind_terms(record.content, context.scope)) / max(1, len(relevance_terms))
            # Preserve a very small immediate tail for conversational deixis. Older
            # recent material must also relate to the current task or input.
            tail = index < self.always_keep
            score = freshness * ((0.38 if tail else 0.03) + 0.62 * overlap)
            candidate = candidate_for(
                record,
                self.name,
                min(1.0, score),
                "immediate conversational tail" if tail else "recency-ranked conversational record",
            )
            # The immediate tail is not competing on topical relevance -- it is
            # there so deixis resolves ("what did you say?", "do that again"),
            # and such a query has no lexical overlap with its own answer by
            # construction.  Mark it so a relevance floor does not cut it; it
            # still pays token budget, and always_keep still bounds it.
            candidate.metadata["conversational_tail"] = tail
            output.append(candidate)
        return output


class SubjectiveLane:
    name = "subjective"

    def __init__(
        self, provider: EmbeddingProvider, *, minimum_importance: float | None = None
    ) -> None:
        self.provider = provider
        # None means every record is nominated and ranking decides.  A hard
        # importance floor silently hid a 0.54 record from a query it answered.
        self.minimum_importance = minimum_importance

    def nominate(self, context: RetrievalContext) -> list[Candidate]:
        query_vector = self.provider.embed(scope_blind_text(context.query, context.scope))
        personal_intent = bool(
            key_terms(context.query)
            & {"prefer", "preference", "style", "format", "deliverable", "personal", "favorite"}
        )
        results: list[tuple[float, MemoryRecord]] = []
        for record in context.records:
            if record.kind == RecordKind.THOUGHT or semantic_index_deferred(context, record):
                continue
            if (
                self.minimum_importance is not None
                and record.importance < self.minimum_importance
            ):
                continue
            provider_key = self.provider.identity + ":scope-blind-v1"
            cached = context.store.get_embedding(record.id, provider_key)
            if cached is None:
                cached = self.provider.embed(scope_blind_text(record.content, context.scope))
                context.store.put_embedding(record.id, provider_key, cached)
            relevance = max(0.0, cosine(query_vector, cached))
            score = record.importance * (0.05 + 0.95 * relevance)
            if personal_intent and record.kind == RecordKind.PREFERENCE:
                score = max(score, 0.70 * record.importance)
            results.append((score, record))
        results.sort(key=lambda item: (-item[0], -item[1].created_at, item[1].id))
        return [candidate_for(record, self.name, score, "subjective importance with query relevance") for score, record in results]


class ProcedureLane:
    name = "procedure"

    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider

    def nominate(self, context: RetrievalContext) -> list[Candidate]:
        query_vector = self.provider.embed(scope_blind_text(context.query, context.scope))
        output: list[tuple[float, MemoryRecord]] = []
        for record in context.records:
            if record.kind != RecordKind.PROCEDURE:
                continue
            provider_key = self.provider.identity + ":scope-blind-v1"
            vector = context.store.get_embedding(record.id, provider_key)
            if vector is None:
                vector = self.provider.embed(scope_blind_text(record.content, context.scope))
                context.store.put_embedding(record.id, provider_key, vector)
            relevance = max(0.0, cosine(query_vector, vector))
            reliability = float(record.metadata.get("reliability", record.confidence))
            score = min(1.0, 0.82 * relevance + 0.18 * reliability)
            output.append((score, record))
        output.sort(key=lambda item: (-item[0], -item[1].created_at, item[1].id))
        return [candidate_for(record, self.name, score, "procedure relevance and reliability") for score, record in output]


class CorrectionLane:
    name = "correction"

    def __init__(self, provider: EmbeddingProvider, *, minimum: float | None = None) -> None:
        self.provider = provider
        # Corrections carry the heaviest lane weight; a similarity floor here
        # could drop the very record that supersedes a stale answer.
        self.minimum = minimum

    def nominate(self, context: RetrievalContext) -> list[Candidate]:
        query_vector = self.provider.embed(scope_blind_text(context.query, context.scope))
        output = []
        for record in context.records:
            if record.kind not in {RecordKind.CORRECTION, RecordKind.CONFLICT}:
                continue
            provider_key = self.provider.identity + ":scope-blind-v1"
            vector = context.store.get_embedding(record.id, provider_key)
            if vector is None:
                vector = self.provider.embed(scope_blind_text(record.content, context.scope))
                context.store.put_embedding(record.id, provider_key, vector)
            score = max(0.0, cosine(query_vector, vector))
            if self.minimum is None or score >= self.minimum or scope_blind_terms(context.query, context.scope) & scope_blind_terms(record.content, context.scope):
                candidate = candidate_for(record, self.name, max(score, 0.5), "correction or conflict closure")
                candidate.hard = score >= 0.18
                output.append(candidate)
        output.sort(key=lambda item: (-max(item.lane_scores.values()), -(item.effective_at or 0), item.id))
        return output


def default_lanes(provider: EmbeddingProvider) -> list[RetrievalLane]:
    return [
        SemanticLane(provider),
        ExactLane(),
        RecencyLane(),
        SubjectiveLane(provider),
        ProcedureLane(provider),
        CorrectionLane(provider),
    ]


def is_historical_query(query: str) -> bool:
    return bool(
        HISTORICAL_RE.search(query)
        or re.search(r"\bdid\b.{0,80}\buse to\b", query, re.IGNORECASE)
    )


def is_future_query(query: str) -> bool:
    return bool(FUTURE_RE.search(query))


class BM25IdentityReranker:
    """2nd-Stage Reranker combining BM25 lexical term frequency/IDF with Regex Identity Intent matching."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._identity_intents = {
            "profession": re.compile(r"\b(job|work|working|worked|career|profession|hired|position|role|occupation|employed|degree|studied|architect|banker|engineer|store|studio|company|firm|agency)\b", re.IGNORECASE),
            "location": re.compile(r"\b(from|live|lived|relocated|moved|born|city|state|resid|house|apartment|bay|downtown)\b", re.IGNORECASE),
            "relationship": re.compile(r"\b(married|husband|wife|partner|spouse|dating|single|together|brother|sister|father|mother|son|daughter|family)\b", re.IGNORECASE),
            "interests": re.compile(r"\b(hobby|hobbies|enjoys|likes|passion|favorite|pastime|sport|painting|running|music|art)\b", re.IGNORECASE),
            "owns": re.compile(r"\b(drives|owns|bought|opened|started|car|business|store|studio|shop)\b", re.IGNORECASE),
        }

    def score_candidates(
        self,
        context: RetrievalContext,
        records: list[MemoryRecord],
    ) -> dict[str, float]:
        if not records:
            return {}

        cleaned_docs: dict[str, list[str]] = {}
        for rec in records:
            cleaned_text = clean_embedding_payload(rec.content, context.scope)
            doc_terms = [t.casefold() for t in terms(cleaned_text) if t.casefold() not in STOPWORDS]
            cleaned_docs[rec.id] = doc_terms

        N = len(records)
        avgdl = sum(len(doc) for doc in cleaned_docs.values()) / max(1, N)

        query_tokens = [t.casefold() for t in terms(context.query) if t.casefold() not in STOPWORDS]
        if not query_tokens:
            return {rec.id: 0.0 for rec in records}

        df: dict[str, int] = {}
        for q_token in set(query_tokens):
            df[q_token] = sum(1 for doc in cleaned_docs.values() if q_token in doc)

        bm25_scores: dict[str, float] = {}
        for rec in records:
            doc = cleaned_docs[rec.id]
            doc_len = len(doc)
            score = 0.0
            for q_token in query_tokens:
                n_q = df.get(q_token, 0)
                if n_q == 0:
                    continue
                idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
                tf = doc.count(q_token)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, avgdl)))
                score += idf * (numerator / max(1e-5, denominator))
            bm25_scores[rec.id] = score

        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
        if max_bm25 > 0:
            bm25_scores = {rid: s / max_bm25 for rid, s in bm25_scores.items()}

        q_lower = context.query.lower()
        active_intents = [
            category for category, pattern in self._identity_intents.items()
            if pattern.search(q_lower)
        ]

        query_entities = set(re.findall(r"\b[A-Z][a-z]{1,15}\b", context.query))
        query_entities_casefold = {e.casefold() for e in query_entities if e.casefold() not in STOPWORDS}

        raw_final_scores: dict[str, float] = {}
        for rec in records:
            base_score = bm25_scores.get(rec.id, 0.0)
            cleaned_text = clean_embedding_payload(rec.content, context.scope)
            doc_lower = cleaned_text.lower()

            # Extract speaker/entity identity from record header or content
            speaker_match = re.search(r"\|\s*([A-Z][a-z]{1,20})\s*\]:", rec.content) or re.search(r"\b([A-Z][a-z]{1,20}):", rec.content)
            record_speaker = speaker_match.group(1).casefold() if speaker_match else ""

            # Check if record entity identity matches any query entity
            is_matching_entity = False
            if record_speaker and record_speaker in query_entities_casefold:
                is_matching_entity = True
            elif any(re.search(rf"\b{re.escape(ent)}\b", doc_lower) for ent in query_entities_casefold):
                is_matching_entity = True

            identity_boost = 1.0
            if is_matching_entity and active_intents:
                for category in active_intents:
                    if self._identity_intents[category].search(doc_lower):
                        identity_boost = max(identity_boost, 2.5)

            # High base score if exact entity match + intent match even if BM25 lexical score is low
            if is_matching_entity and identity_boost > 1.0:
                raw_final_scores[rec.id] = max(0.85, base_score * identity_boost)
            else:
                raw_final_scores[rec.id] = max(0.0, base_score * identity_boost)

        scale = max(1.0, max(raw_final_scores.values(), default=1.0))
        return {
            record_id: min(1.0, score / scale)
            for record_id, score in raw_final_scores.items()
        }
