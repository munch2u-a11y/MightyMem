"""Opt-in bulk sensory association learning from attributed source documents.

This is an ordered, coarse feature learner, not language-model pretraining.
The frozen RWKV embedding space supplies Habitus's existing sensory receptors.
Observed transitions supply exposure counts, never correctness/reward labels.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from .graph import GraphRuntime, PREFERENCE_NODE_IDS, SENSORY_FEATURE_KIND
from .habitus_stream import _NumericSpace
from .pipeline import BaseAgenticMemoryRAG
from .types import ConceptNode, GraphSide, InputTrunk, MemoryRecord, RecordType


SCHEMA = "habitus-corpus-associations-v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def feature_id(bank, code, level):
    bits = int(math.log2(level))
    return f"sensory-fiber:hear:neutral:{bank:02d}:b{bits:02d}:{code:0{max(1, math.ceil(bits / 4))}x}"


class ByteTokenizer:
    """Exact longest-prefix RWKV byte tokenizer without a model dependency.

    Reserved/special strings in external text remain ordinary UTF-8 bytes.
    Rank monotonicity is checked, as in the pinned RWKV reference tokenizer.
    """

    def __init__(self, path):
        raw = Path(path).read_bytes()
        self.sha256 = sha(raw)
        config = json.loads(raw)
        model = config["model"]
        pre = config["pre_tokenizer"]
        if (model["type"] != "WordPiece" or model["continuing_subword_prefix"] != ""
                or model["unk_token"] != "<|endoftext|>" or config["normalizer"] is not None
                or pre != {"type": "ByteLevel", "add_prefix_space": False,
                           "trim_offsets": True, "use_regex": False}):
            raise ValueError("requires exact RWKV byte tokenizer semantics")
        values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        characters = list(values)
        missing = [value for value in range(256) if value not in values]
        values += missing
        characters += list(range(256, 256 + len(missing)))
        decoder = dict(zip(map(chr, characters), values))
        vocab = model["vocab"]
        self.vocab_size = len(vocab)
        if set(vocab.values()) != set(range(len(vocab))) or vocab.get("<|endoftext|>") != 0:
            raise ValueError("invalid RWKV vocabulary indices")
        reverse = {}
        for text, index in vocab.items():
            if index == 0 or text == f"\ue000{index}\ue001":
                continue
            token = bytes(decoder[character] for character in text)
            if not token or token in reverse:
                raise ValueError("empty or repeated vocabulary bytes")
            reverse[token] = index
        if any(bytes([value]) not in reverse for value in range(256)):
            raise ValueError("missing singleton byte tokens")
        self.children, self.terminal = [{}], [None]
        for token, index in reverse.items():
            if any(reverse.get(token[:end], -1) > index for end in range(1, len(token))):
                raise ValueError("ranks violate longest-prefix semantics")
            node = 0
            for byte in token:
                child = self.children[node].get(byte)
                if child is None:
                    child = len(self.children)
                    self.children[node][byte] = child
                    self.children.append({})
                    self.terminal.append(None)
                node = child
            self.terminal[node] = index

    def encode(self, text):
        data = text.encode("utf-8")
        children, terminal, length = self.children, self.terminal, len(data)
        output, position = [], 0
        while position < length:
            node, cursor, best_end, best = 0, position, position, None
            while cursor < length:
                node = children[node].get(data[cursor])
                if node is None:
                    break
                cursor += 1
                if terminal[node] is not None:
                    best, best_end = terminal[node], cursor
            if best is None:
                raise ValueError("singleton byte invariant violated")
            output.append(best)
            position = best_end
        return np.asarray(output, dtype=np.uint32)


def receptor_codes(embeddings, *, banks=6, signature_bits=8, samples=32):
    """Vectorize the existing Habitus receptor planes; retain their exact sums."""
    if not 1 <= banks <= 6 or not 1 <= signature_bits <= 8:
        raise ValueError("invalid receptor configuration")
    dimension = embeddings.shape[1]
    mind = BaseAgenticMemoryRAG(":memory:", embedder=_NumericSpace(dimension, SCHEMA))
    try:
        planes = mind.graph._breadth_planes(banks=banks, signature_bits=signature_bits, samples=samples)
        bits = [signature_bits if banks == 1 else min(3, signature_bits) + round(
            bank * (signature_bits - min(3, signature_bits)) / (banks - 1)) for bank in range(banks)]
        codes = np.zeros((len(embeddings), banks), dtype=np.uint16)
        for bank, count in enumerate(bits):
            for bit, plane in enumerate(planes[bank][:count]):
                projection = np.zeros(len(embeddings), dtype=np.float64)
                for coordinate, sign in plane:
                    projection += embeddings[:, coordinate].astype(np.float64) * sign
                codes[:, bank] |= ((projection >= 0).astype(np.uint16) << bit)
        return codes, tuple(1 << count for count in bits)
    finally:
        mind.close()


class Associations:
    def __init__(self, codes, levels):
        self.codes = np.asarray(codes)
        self.levels = tuple(int(level) for level in levels)
        if (self.codes.ndim != 2 or self.codes.shape[1] != len(self.levels)
                or any(level < 2 or level > 256 for level in self.levels)
                or any(np.any(self.codes[:, bank] >= level) for bank, level in enumerate(self.levels))
                or not np.issubdtype(self.codes.dtype, np.integer) or np.any(self.codes < 0)):
            raise ValueError("invalid sensory codes")
        self.ordered = [np.zeros((level, level), dtype=np.uint64) for level in self.levels]
        self.shuffled = [np.zeros_like(value) for value in self.ordered]
        self.frequency = np.zeros(len(codes), dtype=np.uint64)
        self.documents = self.tokens = self.transitions = 0

    def observe(self, documents):
        """Each (token array, content hash) is an independently bounded document.

        The caller owns durable content deduplication and held-out admission.
        A content-seeded within-document permutation preserves token frequency.
        """
        docs, controls = [], []
        for ids, content_hash in documents:
            ids = np.asarray(ids)
            if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
                    or np.any(ids <= 0) or np.any(ids >= len(self.codes))):
                raise ValueError("invalid or reserved source token")
            rng = np.random.default_rng(int(content_hash[:16], 16))
            docs.append(ids)
            controls.append(rng.permutation(ids))
        if not docs:
            return
        all_ids = np.concatenate(docs)
        self.frequency += np.bincount(all_ids.astype(np.int64), minlength=len(self.codes)).astype(np.uint64)
        for corpus, counts in ((docs, self.ordered), (controls, self.shuffled)):
            left = np.concatenate([ids[:-1] for ids in corpus]).astype(np.int64)
            right = np.concatenate([ids[1:] for ids in corpus]).astype(np.int64)
            for bank, level in enumerate(self.levels):
                pair = self.codes[left, bank].astype(np.int64) * level + self.codes[right, bank]
                counts[bank] += np.bincount(pair, minlength=level * level).reshape(level, level).astype(np.uint64)
        self.documents += len(docs)
        self.tokens += len(all_ids)
        self.transitions += sum(max(0, len(ids) - 1) for ids in docs)

    def probabilities(self, control=False, alpha=0.5):
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("smoothing must be positive")
        return [(value.astype(np.float64) + alpha) / (value.sum(axis=1, keepdims=True) + alpha * len(value))
                for value in (self.shuffled if control else self.ordered)]

    def evaluate(self, documents):
        """Read-only feature prediction; not an assessment of subjective coherence."""
        pairs = sum(max(0, len(ids) - 1) for ids in documents)
        result = {"transitions": pairs, "documents": len(documents), "banks": []}
        for bank, (level, ordered, shuffled) in enumerate(zip(
                self.levels, self.probabilities(), self.probabilities(True))):
            unigram = np.bincount(self.codes[:, bank], weights=self.frequency, minlength=level) + 0.5
            unigram /= unigram.sum()
            losses = dict(ordered=0.0, shuffled=0.0, unigram=0.0, untrained=0.0)
            for ids in documents:
                codes = self.codes[ids, bank]
                for name, matrix in (("ordered", ordered), ("shuffled", shuffled)):
                    losses[name] -= float(np.log(matrix[codes[:-1], codes[1:]]).sum())
                losses["unigram"] -= float(np.log(unigram[codes[1:]]).sum())
                losses["untrained"] += max(0, len(codes) - 1) * math.log(level)
            result["banks"].append({"level": level, "nats_per_transition": {
                key: value / max(1, pairs) for key, value in losses.items()}})
        return result

    def dump(self):
        data = io.BytesIO()
        np.savez_compressed(data, frequency=self.frequency,
            totals=np.asarray([self.documents, self.tokens, self.transitions], dtype=np.uint64),
            **{f"ordered_{bank}": value for bank, value in enumerate(self.ordered)},
            **{f"shuffled_{bank}": value for bank, value in enumerate(self.shuffled)})
        return data.getvalue()

    def restore(self, data):
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            ordered = [archive[f"ordered_{bank}"] for bank in range(len(self.levels))]
            shuffled = [archive[f"shuffled_{bank}"] for bank in range(len(self.levels))]
            frequency, totals = archive["frequency"], archive["totals"]
            if (totals.shape != (3,) or totals.dtype != np.uint64
                    or frequency.shape != self.frequency.shape or frequency.dtype != np.uint64
                    or any(value.shape != (level, level) or value.dtype != np.uint64
                           for group in (ordered, shuffled) for value, level in zip(group, self.levels))
                    or int(frequency.sum()) != int(totals[1])
                    or any(int(value.sum()) != int(totals[2]) for value in (*ordered, *shuffled))):
                raise ValueError("invalid association checkpoint counts")
            self.ordered, self.shuffled, self.frequency = ordered, shuffled, frequency
            self.documents, self.tokens, self.transitions = map(int, totals)

    def centroids(self, embeddings):
        if embeddings.shape[0] != len(self.codes):
            raise ValueError("embedding vocabulary mismatch")
        result = []
        for bank, level in enumerate(self.levels):
            centers = np.zeros((level, embeddings.shape[1]), dtype=np.float64)
            for code in range(level):
                ids = np.flatnonzero((self.codes[:, bank] == code) & (self.frequency > 0))
                if len(ids):
                    # Only observed source tokens shape this receptor's centroid.
                    centers[code] = self.frequency[ids].astype(np.float64) @ embeddings[ids].astype(np.float64)
            norms = np.linalg.norm(centers, axis=1, keepdims=True)
            result.append(centers / np.maximum(1e-12, norms))
        return result


def export_graph(path, associations, centers, manifest, *, control=False):
    """Materialize learned conditional routes in a new, native Habitus database.

    Batch evidence references exact source receipts/counts; it does not invent
    one canonical lived experience or positive outcome per training token.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    dimension = centers[0].shape[1]
    mind = BaseAgenticMemoryRAG(path, embedder=_NumericSpace(dimension, SCHEMA))
    probabilities = associations.probabilities(control)
    evidence = {**manifest, "schema": SCHEMA, "control": control,
                "tokens": associations.tokens, "documents": associations.documents,
                "transitions": associations.transitions, "levels": associations.levels,
                "dimension": dimension, "counts_sha256": sha(associations.dump()),
                "source_role": "external reading", "reward_events": 0,
                "rule": "directed within-bank token adjacency, additive smoothing 0.5",
                "embedding_space": "frozen RWKV input embeddings"}
    record_id = "corpus-exposure:" + sha(canonical(evidence))
    def node_id(bank, code):
        return feature_id(bank, code, associations.levels[bank])
    try:
        mind.graph.recency_strength = 0
        with mind.store.transaction():
            mind.store.add_record(MemoryRecord(record_id, None, RecordType.OBSERVATION,
                "attributed-pretraining-corpus", datetime.now(timezone.utc).isoformat(),
                "external-corpus-receipt-sha256:" + sha(canonical(evidence)),
                (0.0,) * dimension, provenance=evidence,
                metadata={"external_action_verified": False, "aggregate_exposure": True}))
            for bank, vectors in enumerate(centers):
                for code, vector in enumerate(vectors):
                    node = node_id(bank, code)
                    vault = "lower-vault:" + node
                    mind.store.add_concept(ConceptNode(node, node, SENSORY_FEATURE_KIND,
                        tuple(map(float, vector)), (), vault, 0, 0))
                    mind.store.add_to_vault(vault, record_id, node)
            edges = []
            parent = PREFERENCE_NODE_IDS[(InputTrunk.HEAR, "NEUTRAL")]
            for bank, probability in enumerate(probabilities):
                for source in range(len(probability)):
                    target_id = node_id(bank, source)
                    edges.append((GraphRuntime.edge_id(GraphSide.INPUT, parent, target_id),
                        "input", parent, target_id, 1.0, -math.log(sum(associations.levels)), 0.0, None, 0, 0))
                    for target, value in enumerate(probability[source]):
                        source_id, target_id = node_id(bank, source), node_id(bank, target)
                        edges.append((GraphRuntime.edge_id(GraphSide.INPUT, source_id, target_id),
                            "input", source_id, target_id, 1.0, math.log(float(value)), 0.0, None, 0, 0))
            mind.store.connection.executemany("""INSERT INTO edges(edge_id,side,source_id,target_id,
                delta_y,log_strength,conflict_penalty,last_active_time,created_pulse,archived)
                VALUES(?,?,?,?,?,?,?,?,?,?)""", edges)
            mind.store.connection.executemany("INSERT INTO edge_evidence(edge_id,record_id,relation) VALUES(?,?,?)",
                [(edge[0], record_id, "observed_sequence") for edge in edges])
            mind.store.set_metadata("corpus_manifest", canonical(evidence).decode())
        errors = mind.graph.validate_invariants()
        if errors:
            raise ValueError("corpus graph invariants: " + repr(errors))
        if mind.store.connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("corpus graph has broken evidence references")
        return {**evidence, "edges": len(edges), "graph_invariants": errors}
    finally:
        mind.close()


def read_graph(path):
    """Read the numeric routing field from the authoritative saved graph."""
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        manifest = json.loads(connection.execute(
            "SELECT value FROM metadata WHERE key='corpus_manifest'").fetchone()[0])
        if manifest["schema"] != SCHEMA or connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("invalid corpus graph")
        centers, probabilities = [], []
        for bank, level in enumerate(manifest["levels"]):
            vectors = np.zeros((level, manifest["dimension"]), dtype=np.float64)
            probability = np.zeros((level, level), dtype=np.float64)
            targets = {feature_id(bank, code, level): code for code in range(level)}
            for code in range(level):
                node = feature_id(bank, code, level)
                vectors[code] = json.loads(connection.execute(
                    "SELECT embedding_json FROM concepts WHERE concept_id=?", (node,)).fetchone()[0])
                rows = connection.execute("SELECT target_id,log_strength FROM edges WHERE source_id=? AND side='input' AND archived=0", (node,)).fetchall()
                if len(rows) != level:
                    raise ValueError("incomplete corpus transition row")
                for target, strength in rows:
                    if target not in targets:
                        raise ValueError("unexpected corpus transition target")
                    probability[code, targets[target]] = math.exp(strength)
            if (not np.isfinite(vectors).all() or not np.isfinite(probability).all()
                    or not np.allclose(probability.sum(axis=1), 1, atol=1e-12)):
                raise ValueError("invalid corpus numeric readout")
            centers.append(vectors)
            probabilities.append(probability)
        return manifest, centers, probabilities
    finally:
        connection.close()
