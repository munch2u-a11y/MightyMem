from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable

from .capabilities import CapabilityEntry, CapabilitySelector, capability_schema
from .models import Candidate, ContextPacket, EvidenceStatus, ManifestEntry, RecordKind, TaskState
from .retrieval import (
    BM25IdentityReranker,
    EmbeddingProvider,
    HashingSemanticEmbedder,
    RetrievalContext,
    RetrievalLane,
    candidate_for,
    default_lanes,
    is_future_query,
    is_historical_query,
    key_terms,
    render_record,
    stem_token,
)
from .store import LoomStore
from .tokenization import RegexTokenCounter, TokenCounter


COMPILER_VERSION = "context-loom-0.2"
DIRECT_ANSWER_SURFACE_SCHEMA = "habitus-direct-answer-surface-v2"
DIRECT_ANSWER_SURFACE_PROTOCOL = (
    "single leading or terminal ANSWER field for a concise response"
)
DIRECT_ANSWER_RESPONSE_INSTRUCTION = (
    "When you can give a concise answer to the current question, output exactly "
    "ANSWER: <answer> and nothing else. If the evidence is insufficient or conflicts, explain "
    "that normally without an ANSWER field."
)


class ContextBudgetExceeded(ValueError):
    pass


@dataclass(slots=True)
class AdaptiveBudgetPolicy:
    enabled: bool = True
    base_tokens: int = 800
    min_tokens: int = 750
    max_tokens: int = 1450
    depth_scaling_step: int = 250
    multi_hop_boost: int = 300


@dataclass(slots=True)
class CompilerConfig:
    model_context_tokens: int = 2048
    reserved_output_tokens: int = 448
    adaptive_budget: AdaptiveBudgetPolicy = field(default_factory=AdaptiveBudgetPolicy)
    # None shows every eligible capability; the token budget is the only
    # thing that decides how many actually fit.
    max_visible_capabilities: int | None = None
    # Product adapters that pass native tool schemas separately should disable
    # their duplicate prose rendering while retaining selection and receipts.
    render_capability_schemas: bool = True
    mode_instruction: str = (
        "Use the current input and admitted evidence. Treat observations, reports, inferences, "
        "and earlier assistant statements according to their labels. Do not claim an action "
        "succeeded without its outcome-specific verification."
    )
    # Product adapters may provide their answer protocol as a separate model
    # instruction so formatting does not consume the retrieval packet budget.
    response_instruction: str = ""
    lane_weights: dict[str, float] = field(
        default_factory=lambda: {
            "bm25_identity": 1.45,
            "correction": 1.35,
            "multi_hop_graph": 1.25,
            "y_cipher_depth": 1.25,
            "exact": 1.15,
            "habitus_graph": 1.10,
            "semantic": 1.05,
            "summary": 0.95,
            "topical": 0.85,
            "procedure": 0.9,
            "recent": 0.62,
            "subjective": 0.42,
        }
    )
    section_limits: dict[str, int] = field(default_factory=lambda: {"private": 0})
    # Optional deployment policy. By default every eligible candidate competes
    # for the real context-token budget.
    minimum_utility: float | None = None
    learned_utility_weight: float = 0.18
    include_unverified_inferences: bool = False

    def __post_init__(self) -> None:
        if self.model_context_tokens is None:
            object.__setattr__(self, "model_context_tokens", 131072)
        if self.model_context_tokens <= self.reserved_output_tokens:
            raise ValueError("model context must exceed reserved output")
        if self.max_visible_capabilities is not None and self.max_visible_capabilities < 1:
            raise ValueError("at least one capability must be visible")

    @property
    def max_input_tokens(self) -> int:
        return self.model_context_tokens - self.reserved_output_tokens

    def calculate_adaptive_target(self, max_depth: int = 1, is_multi_hop: bool = False) -> int:
        if not self.adaptive_budget.enabled:
            return self.max_input_tokens
        depth_boost = self.adaptive_budget.depth_scaling_step * max(0, max_depth - 1)
        intent_boost = self.adaptive_budget.multi_hop_boost if is_multi_hop else 0
        target = self.adaptive_budget.base_tokens + depth_boost + intent_boost
        return max(self.adaptive_budget.min_tokens, min(self.adaptive_budget.max_tokens, target))


class ContextCompiler:
    """Builds a new evidence and capability packet before every model call."""

    SECTION_TITLES = {
        "work": "ACTIVE WORK",
        "evidence": "DIRECT SEMANTIC & COSINE OVERLAP",
        "multi_hop": "SUBJECTIVE RELEVANCE & MULTI-HOP VAULTS",
        "topical": "TOPICAL ORIENTATION & DRIFT",
        "summary": "ROLLING FACTUAL SUMMARY",
        "continuity": "RECENCY & CONVERSATIONAL CONTINUITY",
        "observations": "LATEST VERIFIED OBSERVATIONS",
        "personal": "BOUNDED PERSONAL CONTINUITY",
        "procedures": "RELEVANT VERIFIED PROCEDURES",
        "private": "PRIVATE STATE",
    }
    SECTION_ORDER = (
        "topical",
        "summary",
        "work",
        "evidence",
        "multi_hop",
        "personal",
        "procedures",
        "continuity",
    )

    def __init__(
        self,
        store: LoomStore,
        *,
        config: CompilerConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        lanes: Iterable[RetrievalLane] | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.store = store
        self.config = config or CompilerConfig()
        self.embedding_provider = embedding_provider or HashingSemanticEmbedder()
        self.lanes = list(lanes) if lanes is not None else default_lanes(self.embedding_provider)
        self.token_counter = token_counter or RegexTokenCounter()
        self.capabilities = CapabilitySelector(
            store, max_visible=self.config.max_visible_capabilities
        )
        self.bm25_identity_reranker = BM25IdentityReranker()

    @staticmethod
    def _task_candidate(task: TaskState) -> Candidate:
        lines = [f"Goal: {task.goal}", f"Stage: {task.stage}; status: {task.status}"]
        if task.focus_stack:
            lines.append("Focus stack: " + " -> ".join(task.focus_stack))
        if task.open_questions:
            lines.append("Unresolved questions: " + " | ".join(task.open_questions))
        if task.constraints:
            lines.append("Constraints: " + " | ".join(task.constraints))
        if task.completion_conditions:
            lines.append("Completion requires: " + " | ".join(task.completion_conditions))
        return Candidate(
            id=f"task:{task.id}",
            text="\n".join(lines),
            section="work",
            source_record_ids=(),
            lane_scores={"focus": 1.0},
            reasons=["active structured task state"],
            hard=True,
            evidence_status=EvidenceStatus.SYSTEM_DERIVED,
            metadata={"task_id": task.id},
        )

    def _trust_allowed(self, candidate: Candidate) -> tuple[bool, str]:
        kind = RecordKind(candidate.metadata["kind"]) if candidate.metadata.get("kind") else None
        if kind == RecordKind.THOUGHT or candidate.section == "private":
            return False, "private thought is not language-visible evidence"
        if kind == RecordKind.ASSISTANT:
            return candidate.section == "continuity", "assistant statement allowed only as labeled continuity"
        if kind == RecordKind.PROCEDURE and candidate.evidence_status != EvidenceStatus.TOOL_VERIFIED:
            return False, "procedure lacks verified trajectory evidence"
        if candidate.evidence_status in {EvidenceStatus.MODEL_INFERRED, EvidenceStatus.UNVERIFIED}:
            if self.config.include_unverified_inferences:
                return True, "unverified material enabled by configuration"
            return False, "unverified inference excluded"
        if candidate.evidence_status == EvidenceStatus.SYSTEM_DERIVED:
            support = candidate.metadata.get("support_assessment")
            source_ids = candidate.metadata.get("source_ids") or []
            if kind in {RecordKind.FACT, RecordKind.CORRECTION} and not (
                source_ids and support in {"entailed", "extractive", "state_machine"}
            ):
                return False, "derived claim lacks an assessed supporting source"
        return True, "evidence status admitted"

    optional_selection_policy = "weighted score-discounted reciprocal lane rank within exact token budget"
    optional_admission_reason = "highest fused utility within budget"

    def _order_optional_candidates(self, candidates):
        """Composition hook; the default preserves native admission ordering."""
        return sorted(candidates, key=lambda c: (
            -(self._utility(c) / max(1.0, math.sqrt(self.token_counter.count(c.text) / 24.0))),
            -self._utility(c), c.id,
        ))

    def _candidate_admitted(self, candidate, added_tokens):
        """Allow a compositor to account for actual accepted context space."""
        pass

    def _utility(self, candidate: Candidate) -> float:
        utility = sum(
            self.config.lane_weights.get(lane, 0.5)
            * score
            / max(1, candidate.lane_ranks.get(lane, 1))
            for lane, score in candidate.lane_scores.items()
        )
        utility += self.config.learned_utility_weight * candidate.learned_utility
        return utility

    @staticmethod
    def _assign_lane_ranks(candidates: dict[str, Candidate]) -> None:
        """Attach tied ranks so differently calibrated lanes combine coherently."""
        lane_names = {
            lane
            for candidate in candidates.values()
            for lane, score in candidate.lane_scores.items()
            if score > 0
        }
        for candidate in candidates.values():
            candidate.lane_ranks.clear()
        for lane in lane_names:
            ordered = sorted(
                (
                    candidate
                    for candidate in candidates.values()
                    if candidate.lane_scores.get(lane, 0.0) > 0
                ),
                key=lambda candidate: (-candidate.lane_scores[lane], candidate.id),
            )
            previous_score: float | None = None
            tied_rank = 0
            for position, candidate in enumerate(ordered, start=1):
                score = candidate.lane_scores[lane]
                if previous_score is None or score != previous_score:
                    tied_rank = position
                    previous_score = score
                candidate.lane_ranks[lane] = tied_rank

    def _merge_nominations(self, nominations: Iterable[Candidate]) -> dict[str, Candidate]:
        merged: dict[str, Candidate] = {}
        for candidate in nominations:
            if candidate.id in merged:
                merged[candidate.id].merge(candidate)
            else:
                merged[candidate.id] = candidate
        for candidate in merged.values():
            if len(candidate.source_record_ids) == 1:
                candidate.learned_utility = self.store.learned_utility(candidate.source_record_ids[0])
        return merged

    def _apply_version_policy(
        self,
        candidates: dict[str, Candidate],
        *,
        historical: bool,
        as_of: float,
    ) -> tuple[dict[str, Candidate], list[ManifestEntry]]:
        exclusions: list[ManifestEntry] = []
        if historical:
            for candidate in candidates.values():
                if candidate.source_record_ids:
                    record = self.store.get_record(candidate.source_record_ids[0])
                    if record:
                        candidate.text = render_record(record, historical=True)
            return candidates, exclusions
        retained: dict[str, Candidate] = {}
        replacements: list[Candidate] = []
        for candidate_id, candidate in candidates.items():
            if not candidate.source_record_ids:
                retained[candidate_id] = candidate
                continue
            record_id = candidate.source_record_ids[0]
            latest = self.store.latest_version(record_id, as_of=as_of)
            if latest is not None and latest.id != record_id:
                exclusions.append(
                    ManifestEntry(
                        candidate_id=candidate_id,
                        source_record_ids=candidate.source_record_ids,
                        lanes=dict(candidate.lane_scores),
                        admitted=False,
                        reason=f"superseded by record:{latest.id} for a current-state query",
                        token_cost=self.token_counter.count(candidate.text),
                        utility=self._utility(candidate),
                    )
                )
                replacement = candidate_for(
                    latest,
                    "correction",
                    max(0.85, max(candidate.lane_scores.values(), default=0.0)),
                    "current version closure",
                )
                replacement.hard = candidate.hard
                replacements.append(replacement)
                continue
            retained[candidate_id] = candidate
        for replacement in replacements:
            if replacement.id in retained:
                retained[replacement.id].merge(replacement)
            else:
                retained[replacement.id] = replacement
        return retained, exclusions

    def _expand_historical_corrections(
        self, candidates: dict[str, Candidate], *, historical: bool, as_of: float
    ) -> None:
        if not historical:
            return
        additions: list[Candidate] = []
        expanded_ids: set[str] = set()
        for candidate in list(candidates.values()):
            if not candidate.source_record_ids:
                continue
            record = self.store.get_record(candidate.source_record_ids[0])
            if record is None:
                continue
            chain: list = []
            cursor = record
            seen: set[str] = set()
            while cursor.supersedes_id and cursor.id not in seen:
                seen.add(cursor.id)
                previous = self.store.get_record(cursor.supersedes_id)
                if previous is None or previous.created_at > as_of:
                    break
                cursor = previous
            while cursor is not None and cursor.id not in expanded_ids:
                expanded_ids.add(cursor.id)
                chain.append(cursor)
                cursor = self.store.superseding_record(cursor.id, as_of=as_of)
            for version in chain:
                closure = candidate_for(version, "correction", 0.85, "historical version-chain closure")
                closure.text = render_record(version, historical=True)
                closure.hard = candidate.hard
                additions.append(closure)
        for candidate in additions:
            if candidate.id in candidates:
                candidates[candidate.id].merge(candidate)
            else:
                candidates[candidate.id] = candidate

    def _render(
        self,
        *,
        mode: str,
        sections: dict[str, list[str]],
        capabilities: list,
        current_input: str,
    ) -> str:
        blocks = [f"[MODE: {mode.upper()}]\n{self.config.mode_instruction}"]
        if capabilities and self.config.render_capability_schemas:
            tools = [capability_schema(capability) for capability in capabilities]
            blocks.append("[AVAILABLE ACTIONS]\n" + json.dumps(tools, ensure_ascii=False, sort_keys=True))
        for section in self.SECTION_ORDER:
            values = sections.get(section) or []
            if values:
                blocks.append(f"[{self.SECTION_TITLES[section]}]\n" + "\n".join(values))
        if self.config.response_instruction:
            blocks.append("[RESPONSE CONTRACT]\n" + self.config.response_instruction)
        blocks.append("[CURRENT INPUT]\n" + current_input)
        return "\n\n".join(blocks)

    def compile(
        self,
        current_input: str,
        *,
        scope: str = "default",
        task_id: str | None = None,
        current_event_id: str | None = None,
        mode: str = "reply",
        now: float | None = None,
    ) -> ContextPacket:
        if not current_input.strip():
            raise ValueError("current input cannot be empty")
        created_at = time.time() if now is None else now
        task = self.store.get_task(task_id) if task_id else None
        if task_id and task is None:
            raise ValueError(f"unknown task {task_id}")
        if task and task.scope != scope:
            raise ValueError("task scope does not match compilation scope")
        historical = is_historical_query(current_input)
        future_query = is_future_query(current_input)
        records = self.store.list_records(
            scope=scope, as_of=created_at, exclude_ids=([current_event_id] if current_event_id else [])
        )
        context = RetrievalContext(
            query=current_input,
            scope=scope,
            now=created_at,
            records=records,
            store=self.store,
            task=task,
            historical=historical,
            embedding_provider=self.embedding_provider,
        )
        nominations: list[Candidate] = []
        for lane in self.lanes:
            nominations.extend(lane.nominate(context))
        candidates = self._merge_nominations(nominations)
        candidates, manifest = self._apply_version_policy(
            candidates, historical=historical, as_of=created_at
        )
        self._expand_historical_corrections(candidates, historical=historical, as_of=created_at)

        # Apply 2nd-Stage BM25 & Identity Intent Reranking across nominated candidates
        if candidates:
            cand_records = []
            for cand in candidates.values():
                if cand.source_record_ids:
                    rec = self.store.get_record(cand.source_record_ids[0])
                    if rec is not None:
                        cand_records.append(rec)
            if cand_records:
                rerank_scores = self.bm25_identity_reranker.score_candidates(context, cand_records)
                for cand in candidates.values():
                    if cand.source_record_ids:
                        rid = cand.source_record_ids[0]
                        if rid in rerank_scores and rerank_scores[rid] > 0:
                            cand.lane_scores["bm25_identity"] = rerank_scores[rid]

        if task:
            candidates[f"task:{task.id}"] = self._task_candidate(task)
            if task.last_observation_id:
                observation = self.store.get_record(task.last_observation_id)
                if observation and observation.scope == scope and observation.created_at <= created_at:
                    item = candidate_for(observation, "recent", 1.0, "task's latest verified observation")
                    item.hard = True
                    if item.id in candidates:
                        candidates[item.id].merge(item)
                    else:
                        candidates[item.id] = item

        self._assign_lane_ranks(candidates)

        selected_capabilities, capability_manifest = self.capabilities.select(
            query=current_input, scope=scope, task=task
        )
        sections: dict[str, list[str]] = defaultdict(list)
        base_rendered = self._render(
            mode=mode,
            sections=sections,
            capabilities=selected_capabilities,
            current_input=current_input,
        )
        # Determine adaptive max input budget based on candidate depth and query intent
        max_depth = 1
        for cand in candidates.values():
            if "y_cipher_depth" in cand.lane_scores or "y_depth" in cand.metadata:
                depth = int(cand.metadata.get("y_depth", 1))
                if depth > max_depth:
                    max_depth = depth

        query_lower = current_input.casefold()
        is_multi_hop = len(key_terms(current_input)) >= 3 or any(
            kw in query_lower for kw in ("both", "and", "all", "relationship", "between", "common", "visited")
        )
        effective_max_tokens = min(
            self.config.max_input_tokens,
            self.config.calculate_adaptive_target(max_depth=max_depth, is_multi_hop=is_multi_hop),
        )

        base_tokens = self.token_counter.count(base_rendered)
        if base_tokens > effective_max_tokens:
            raise ContextBudgetExceeded(
                f"fixed packet content requires {base_tokens} tokens; "
                f"input budget is {effective_max_tokens}"
            )

        trusted: list[Candidate] = []
        for candidate in candidates.values():
            if (
                candidate.effective_at is not None
                and candidate.effective_at > created_at
                and not future_query
            ):
                manifest.append(
                    ManifestEntry(
                        candidate_id=candidate.id,
                        source_record_ids=candidate.source_record_ids,
                        lanes=dict(candidate.lane_scores),
                        admitted=False,
                        reason="record is not effective yet",
                        token_cost=self.token_counter.count(candidate.text),
                        utility=self._utility(candidate),
                        lane_ranks=dict(candidate.lane_ranks),
                    )
                )
                continue
            applicability = str(candidate.metadata.get("applicability", "active")).lower()
            query_kt = key_terms(current_input)
            if applicability in {"rejected", "retracted", "invalid"} and applicability not in query_kt and stem_token(applicability) not in query_kt:
                manifest.append(
                    ManifestEntry(
                        candidate_id=candidate.id,
                        source_record_ids=candidate.source_record_ids,
                        lanes=dict(candidate.lane_scores),
                        admitted=False,
                        reason=f"record applicability is {applicability!r}",
                        token_cost=self.token_counter.count(candidate.text),
                        utility=self._utility(candidate),
                        lane_ranks=dict(candidate.lane_ranks),
                    )
                )
                continue
            allowed, reason = self._trust_allowed(candidate)
            utility = self._utility(candidate)
            if not allowed:
                manifest.append(
                    ManifestEntry(
                        candidate_id=candidate.id,
                        source_record_ids=candidate.source_record_ids,
                        lanes=dict(candidate.lane_scores),
                        admitted=False,
                        reason=reason,
                        token_cost=self.token_counter.count(candidate.text),
                        utility=utility,
                        lane_ranks=dict(candidate.lane_ranks),
                    )
                )
                continue
            if candidate.section != "work" and self.config.section_limits.get(candidate.section) == 0:
                manifest.append(
                    ManifestEntry(
                        candidate_id=candidate.id,
                        source_record_ids=candidate.source_record_ids,
                        lanes=dict(candidate.lane_scores),
                        admitted=False,
                        reason=f"section {candidate.section!r} is disabled",
                        token_cost=self.token_counter.count(candidate.text),
                        utility=utility,
                        lane_ranks=dict(candidate.lane_ranks),
                    )
                )
                continue
            trusted.append(candidate)

        hard = sorted(
            (candidate for candidate in trusted if candidate.hard),
            key=lambda c: (0 if c.section == "work" else 1, -self._utility(c), c.id),
        )
        soft = self._order_optional_candidates(
            candidate for candidate in trusted if not candidate.hard
        )
        selected_ids: set[str] = set()
        section_counts: dict[str, int] = defaultdict(int)
        additive = type(self) is ContextCompiler and type(self.token_counter) is RegexTokenCounter
        used_tokens = base_tokens
        section_header_tokens: dict[str, int] = {}

        def try_admit(candidate: Candidate, required: bool) -> None:
            nonlocal used_tokens
            utility = self._utility(candidate)
            token_cost = self.token_counter.count(candidate.text)
            if candidate.id in selected_ids:
                return
            limit = self.config.section_limits.get(candidate.section)
            if not required and limit is not None and section_counts[candidate.section] >= limit:
                manifest.append(
                    ManifestEntry(
                        candidate.id,
                        candidate.source_record_ids,
                        dict(candidate.lane_scores),
                        False,
                        "section item limit",
                        token_cost,
                        utility,
                        dict(candidate.lane_ranks),
                    )
                )
                return
            if (not required
                    and self.config.minimum_utility is not None
                    and not candidate.metadata.get("conversational_tail")
                    and utility < self.config.minimum_utility):
                manifest.append(
                    ManifestEntry(
                        candidate.id,
                        candidate.source_record_ids,
                        dict(candidate.lane_scores),
                        False,
                        "below minimum utility",
                        token_cost,
                        utility,
                        dict(candidate.lane_ranks),
                    )
                )
                return
            if additive and candidate.section in self.SECTION_ORDER:
                header_cost = 0
                if not sections.get(candidate.section):
                    if candidate.section not in section_header_tokens:
                        section_header_tokens[candidate.section] = self.token_counter.count(
                            f"[{self.SECTION_TITLES[candidate.section]}]"
                        )
                    header_cost = section_header_tokens[candidate.section]
                total = used_tokens + token_cost + header_cost
            else:
                trial = {key: list(value) for key, value in sections.items()}
                trial.setdefault(candidate.section, []).append(candidate.text)
                rendered = self._render(
                    mode=mode,
                    sections=trial,
                    capabilities=selected_capabilities,
                    current_input=current_input,
                )
                total = self.token_counter.count(rendered)
            if total > effective_max_tokens:
                if required:
                    raise ContextBudgetExceeded(
                        f"mandatory candidate {candidate.id} would require {total} tokens; "
                        f"input budget is {effective_max_tokens}"
                    )
                manifest.append(
                    ManifestEntry(
                        candidate.id,
                        candidate.source_record_ids,
                        dict(candidate.lane_scores),
                        False,
                        "total context budget",
                        token_cost,
                        utility,
                        dict(candidate.lane_ranks),
                    )
                )
                return
            self._candidate_admitted(candidate, total - used_tokens)
            used_tokens = total
            sections[candidate.section].append(candidate.text)
            section_counts[candidate.section] += 1
            selected_ids.add(candidate.id)
            manifest.append(
                ManifestEntry(
                    candidate.id,
                    candidate.source_record_ids,
                    dict(candidate.lane_scores),
                    True,
                    "mandatory" if required else self.optional_admission_reason,
                    token_cost,
                    utility,
                    dict(candidate.lane_ranks),
                )
            )

        for candidate in hard:
            try_admit(candidate, True)

        for candidate in soft:
            try_admit(candidate, False)

        rendered = self._render(
            mode=mode,
            sections=sections,
            capabilities=selected_capabilities,
            current_input=current_input,
        )
        token_count = self.token_counter.count(rendered)
        if additive and token_count != used_tokens:
            raise AssertionError("incremental context count differs from the full rendered packet")
        if token_count > self.config.max_input_tokens:
            raise AssertionError("compiler emitted an over-budget packet")
        fingerprint = json.dumps(
            {
                "scope": scope,
                "event": current_event_id,
                "task": task_id,
                "mode": mode,
                "created_at": created_at,
                "rendered": rendered,
                "compiler": COMPILER_VERSION,
                "lanes": [lane.name for lane in self.lanes],
                "manifest": [asdict(item) for item in manifest],
            },
            sort_keys=True,
        )
        packet_id = "ctx_" + hashlib.sha256(fingerprint.encode()).hexdigest()[:24]
        packet = ContextPacket(
            id=packet_id,
            created_at=created_at,
            scope=scope,
            current_input=current_input,
            current_event_id=current_event_id,
            task_id=task_id,
            mode=mode,
            sections={key: list(value) for key, value in sections.items()},
            capabilities=selected_capabilities,
            manifest=sorted(manifest, key=lambda item: (not item.admitted, item.candidate_id)),
            token_count=token_count,
            max_input_tokens=self.config.max_input_tokens,
            reserved_output_tokens=self.config.reserved_output_tokens,
            compiler_version=COMPILER_VERSION,
            rendered=rendered,
            metadata={
                "historical_query": historical,
                "future_query": future_query,
                "token_counter": self.token_counter.identity,
                "embedding_provider": self.embedding_provider.identity,
                "capability_manifest": [asdict(item) for item in capability_manifest],
                "capability_schemas_rendered_in_packet": self.config.render_capability_schemas,
                "candidate_count": len(candidates),
                "admitted_candidate_count": len(selected_ids),
                "selection_policy": self.optional_selection_policy,
            },
        )
        self.store.save_packet(packet)
        return packet
