"""Embedding intent cortex: route an utterance to an affordance by hidden state.

This replaces ``HabitusLoomAgent._stage_and_capability``, which routes by
substring match::

    if any(word in lowered for word in ("run ", "execute ", "launch ", ...)):
        return "act", "run-program"

Substring routing has two structural failures. It cannot fire on an utterance
that carries the intent without the word ("is the parser still passing?" should
run the suite), and it fires *harder* on a negated utterance than a neutral one
("do not run the migration" contains "run "). The second is a safety problem:
the word that forbids the act is invisible to the router, while the verb that
names it still scores.

Here the utterance is embedded with the same local embedding model the agent
already runs, and compared against per-affordance prototype centroids built from
exemplar phrasings. The exemplars play the same authoring role the ``keywords``
tuple plays on each ``Capability`` -- a human lists what the affordance sounds
like -- but they are compared in representation space, so paraphrase and
negation carry.

The contract is identical to the function it replaces: ``(stage, capability_id)``,
always populated, falling back to ``("understand", "search-memory")``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence


class SupportsEmbedMany(Protocol):
    """Either calling convention in this codebase is accepted.

    ``LocalModels.embed(texts)`` takes a sequence and returns a matrix;
    ``OllamaEmbeddingProvider.embed(text)`` takes one string. ``_embed_batch``
    adapts whichever is installed.
    """

    def embed(self, text): ...


# Stage each affordance is exercised from, matching default_capabilities() phases.
CAPABILITY_STAGE = {
    "list-files": "inspect",
    "read-file": "inspect",
    "write-file": "act",
    "run-program": "act",
    "search-memory": "understand",
    "report-result": "report",
}

# The route taken when nothing is clearly indicated. Identical to the substring
# router's fallback, so installing the cortex cannot change default behaviour.
NEUTRAL_ROUTE = ("understand", "search-memory")

# "Do not act" is a first-class class, not the absence of a match. Without it a
# nearest-prototype classifier must assign every utterance to some affordance,
# which is exactly how the substring router ends up firing on a refusal.
ABSTAIN = "no-action"


PROTOTYPES: dict[str, tuple[str, ...]] = {
    "list-files": (
        "list the files in the project folder",
        "show me the directory contents",
        "what is in that folder",
        "is there anything in the build directory",
        "let me see what state the checkout is in",
        "enumerate the entries under src",
    ),
    "read-file": (
        "read the config file and show me what is in it",
        "open the script and show me its contents",
        "what is actually inside that yaml right now",
        "I need the exact bytes of that file",
        "show me the real contents rather than a summary",
        "print the contents of the readme",
    ),
    "write-file": (
        "write the new config to settings json",
        "save these notes to a file",
        "update the changelog with this entry",
        "make sure that gets persisted to disk",
        "put that somewhere I can find it later",
        "create a file containing those items",
    ),
    "run-program": (
        "run the test suite",
        "execute the deploy script",
        "launch the indexer program",
        "check whether that module still compiles",
        "is the parser still passing after the change",
        "kick that off and tell me how it exits",
        "start the local server",
    ),
    "search-memory": (
        "what did we decide about the retry policy",
        "who suggested moving to that architecture",
        "remind me which client we agreed to prioritise",
        "did I ever mention a preference about databases",
        "recall the deployment window we agreed on",
        "has that come up between us before",
    ),
    ABSTAIN: (
        # explicit refusal -- the verb is present and must not fire
        "do not run the migration yet",
        "don't write anything to that file",
        "no need to list the directory, I already looked",
        "skip the file read this time",
        "never launch that program without asking me first",
        "hold off on saving that for now",
        # deferred -- real intent, wrong moment
        "later I'll need you to write that to a file",
        "once the review lands, run the full suite",
        "when I give the word, execute the rollback",
        # discussed rather than requested
        "if I ran the tests now would that tell us anything",
        "what would happen if you wrote to that file directly",
        "I used to run that script by hand every morning",
        # ordinary conversation
        "thanks, that makes sense",
        "I'm not sure I follow, can you rephrase that",
        "why does that matter",
        "explain that last part again",
    ),
}


def _normalise(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1e-9
    return [value / norm for value in vector]


def _centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("cannot build a centroid from no vectors")
    width = len(vectors[0])
    totals = [0.0] * width
    for vector in vectors:
        for index, value in enumerate(vector):
            totals[index] += value
    return _normalise([value / len(vectors) for value in totals])


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


@dataclass
class IntentDecision:
    stage: str
    capability_id: str
    label: str
    confidence: float
    margin: float
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def acted(self) -> bool:
        return self.label not in {ABSTAIN, "search-memory"}


class EmbeddingIntentCortex:
    """Nearest-prototype affordance routing over embedding hidden states.

    ``margin`` is the minimum gap between the winning class and the runner-up.
    Below it the utterance is treated as unclear and takes the neutral route,
    so an ambiguous turn degrades to memory search rather than to a side effect.
    """

    def __init__(
        self,
        embedder: SupportsEmbedMany,
        *,
        prototypes: dict[str, tuple[str, ...]] | None = None,
        margin: float = 0.01,
    ) -> None:
        self.embedder = embedder
        self.prototypes = dict(prototypes or PROTOTYPES)
        self.margin = float(margin)
        self._centroids: dict[str, list[float]] = {}
        self._build()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        embedder = self.embedder
        if hasattr(embedder, "embed_many"):
            raw = embedder.embed_many(texts)
        else:
            # LocalModels.embed takes the whole batch and returns a matrix.
            raw = embedder.embed(texts)
        return [_normalise([float(x) for x in row]) for row in raw]

    def _build(self) -> None:
        labels: list[str] = []
        texts: list[str] = []
        for label, examples in self.prototypes.items():
            for example in examples:
                labels.append(label)
                texts.append(example)
        vectors = self._embed_batch(texts)
        grouped: dict[str, list[list[float]]] = {}
        for label, vector in zip(labels, vectors):
            grouped.setdefault(label, []).append(vector)
        self._centroids = {label: _centroid(vs) for label, vs in grouped.items()}

    def classify(self, text: str) -> IntentDecision:
        query = self._embed_batch([text])[0]
        scores = {label: _cosine(query, centroid)
                  for label, centroid in self._centroids.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_label, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - runner_up

        if best_label == ABSTAIN or margin < self.margin:
            stage, capability_id = NEUTRAL_ROUTE
            label = best_label if best_label == ABSTAIN else "unclear"
        else:
            capability_id = best_label
            stage = CAPABILITY_STAGE[capability_id]
            label = best_label
        return IntentDecision(stage, capability_id, label, best_score, margin, scores)

    def stage_and_capability(self, text: str) -> tuple[str, str]:
        """Drop-in replacement for HabitusLoomAgent._stage_and_capability."""
        decision = self.classify(text)
        return decision.stage, decision.capability_id
