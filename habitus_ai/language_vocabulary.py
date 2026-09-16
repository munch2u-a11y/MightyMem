"""A lossless, locally learned byte-pair vocabulary with no pretrained assets.

Merges stay inside individual word, whitespace, or punctuation segments. The
artifact contains bounded lexical pieces, not conversation examples or response
templates. All 256 bytes remain available for unseen text. Only the CLI trainer
reads a teaching corpus; encoding is stateless and admits one current string.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Sequence


SCHEMA = "habitus.language-vocabulary.v1"
SEGMENTER = "unicode-word-space-punctuation-v1"
SEGMENTS = re.compile(r"\s+|[^\W\d_]+|\d+|[^\w\s]+|_", re.UNICODE)


def _replace_pair(units: tuple[int, ...], pair: tuple[int, int], replacement: int) -> tuple[int, ...]:
    result = []
    index = 0
    while index < len(units):
        if index + 1 < len(units) and (units[index], units[index + 1]) == pair:
            result.append(replacement)
            index += 2
        else:
            result.append(units[index])
            index += 1
    return tuple(result)


class LanguageVocabulary:
    __slots__ = ("_maximum_piece_bytes", "_merges", "_pieces", "_ranks", "training")

    def __init__(
        self, merges: Sequence[tuple[int, int]] = (), *, maximum_piece_bytes: int = 16,
        training: dict[str, Any] | None = None,
    ) -> None:
        if not 2 <= maximum_piece_bytes <= 64 or len(merges) > 16128:
            raise ValueError("vocabulary exceeds its supported bounds")
        self._maximum_piece_bytes = maximum_piece_bytes
        self._merges = tuple(tuple(pair) for pair in merges)
        pieces = [bytes((value,)) for value in range(256)]
        ranks = {}
        for rank, pair in enumerate(self.merges):
            if len(pair) != 2 or any(type(value) is not int or not 0 <= value < len(pieces) for value in pair):
                raise ValueError("vocabulary merge references an unavailable piece")
            if pair in ranks:
                raise ValueError("duplicate vocabulary merge")
            piece = pieces[pair[0]] + pieces[pair[1]]
            if len(piece) > maximum_piece_bytes:
                raise ValueError("vocabulary piece exceeds its byte bound")
            ranks[pair] = rank
            pieces.append(piece)
        self._pieces = tuple(pieces)
        self._ranks = MappingProxyType(ranks)
        self.training = dict(training or {})

    @property
    def maximum_piece_bytes(self) -> int:
        return self._maximum_piece_bytes

    @property
    def merges(self) -> tuple[tuple[int, int], ...]:
        return self._merges

    @property
    def pieces(self) -> tuple[bytes, ...]:
        return self._pieces

    @property
    def ranks(self) -> Any:
        return self._ranks

    def __len__(self) -> int:
        return len(self.pieces)

    def encode(self, current_text: str) -> tuple[int, ...]:
        if not isinstance(current_text, str):
            raise ValueError("vocabulary input must be text")
        result = []
        for segment in SEGMENTS.findall(current_text):
            units = tuple(segment.encode("utf-8"))
            while len(units) > 1:
                candidates = ((self.ranks[pair], pair) for pair in zip(units, units[1:]) if pair in self.ranks)
                selected = min(candidates, default=None)
                if selected is None:
                    break
                rank, pair = selected
                units = _replace_pair(units, pair, 256 + rank)
            result.extend(units)
        return tuple(result)

    def decode_bytes(self, symbols: Iterable[int]) -> bytes:
        values = tuple(symbols)
        if any(type(value) is not int or not 0 <= value < len(self) for value in values):
            raise ValueError("symbol ID is outside the vocabulary")
        return b"".join(self.pieces[value] for value in values)

    def decode(self, symbols: Iterable[int]) -> str:
        return self.decode_bytes(symbols).decode("utf-8")

    def _identity(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "segmenter": SEGMENTER,
                "maximum_piece_bytes": self.maximum_piece_bytes, "merges": [list(pair) for pair in self.merges]}

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self._identity(), sort_keys=True).encode()).hexdigest()

    def payload(self) -> dict[str, Any]:
        return {**self._identity(), "vocabulary_sha256": self.sha256, "training": self.training}

    def save(self, path: Path) -> None:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(self.payload(), stream, indent=2, sort_keys=True)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> LanguageVocabulary:
        if payload.get("schema") != SCHEMA or payload.get("segmenter") != SEGMENTER:
            raise ValueError("unsupported native vocabulary format")
        vocabulary = cls(payload["merges"], maximum_piece_bytes=payload["maximum_piece_bytes"],
                         training=payload.get("training", {}))
        if vocabulary.sha256 != payload.get("vocabulary_sha256"):
            raise ValueError("vocabulary checksum mismatch")
        return vocabulary

    @classmethod
    def load(cls, path: Path) -> LanguageVocabulary:
        return cls.from_payload(json.loads(path.read_text(encoding="utf-8")))


def fit_training_dialogues(
    dialogues: Sequence[Any], *, vocabulary_size: int = 1024, maximum_piece_bytes: int = 16,
) -> LanguageVocabulary:
    if not 256 <= vocabulary_size <= 16384 or not 2 <= maximum_piece_bytes <= 64:
        raise ValueError("requested vocabulary is outside the supported bounds")
    training = sorted((item for item in dialogues if item.split == "train"), key=lambda item: item.dialogue_id)
    if not training:
        raise ValueError("vocabulary learning requires training-split dialogues")
    words: Counter[tuple[int, ...]] = Counter()
    fingerprint = hashlib.sha256()
    for item in training:
        fingerprint.update(json.dumps({"id": item.dialogue_id,
                                       "turns": [(turn.user, turn.assistant) for turn in item.turns]},
                                      sort_keys=True).encode())
        for turn in item.turns:
            for text in (turn.user, turn.assistant):
                words.update(tuple(segment.encode("utf-8")) for segment in SEGMENTS.findall(text))
    pieces = [bytes((value,)) for value in range(256)]
    merges = []
    while len(pieces) < vocabulary_size:
        counts: Counter[tuple[int, int]] = Counter()
        for units, frequency in words.items():
            for pair in zip(units, units[1:]):
                if len(pieces[pair[0]]) + len(pieces[pair[1]]) <= maximum_piece_bytes:
                    counts[pair] += frequency
        if not counts:
            break
        pair = min(counts, key=lambda value: (-counts[value], value))
        if counts[pair] < 2:
            break
        replacement = len(pieces)
        merged_words: Counter[tuple[int, ...]] = Counter()
        for units, frequency in words.items():
            merged_words[_replace_pair(units, pair, replacement)] += frequency
        words = merged_words
        merges.append(pair)
        pieces.append(pieces[pair[0]] + pieces[pair[1]])
    return LanguageVocabulary(merges, maximum_piece_bytes=maximum_piece_bytes, training={
        "kind": "training_split_byte_pair_counts", "pretrained_tokenizer_loaded": False,
        "train_dialogue_ids": [item.dialogue_id for item in training],
        "train_examples_sha256": fingerprint.hexdigest(), "requested_vocabulary_size": vocabulary_size,
    })


def main() -> int:
    from .communication_training import load_curriculum

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--vocabulary-size", type=int, default=1024)
    parser.add_argument("--maximum-piece-bytes", type=int, default=16)
    args = parser.parse_args()
    dialogues = load_curriculum(args.corpus)
    vocabulary = fit_training_dialogues(dialogues, vocabulary_size=args.vocabulary_size,
                                        maximum_piece_bytes=args.maximum_piece_bytes)
    metrics = {}
    for split in ("train", "validation", "test"):
        byte_count = symbol_count = message_count = 0
        for item in dialogues:
            if item.split != split:
                continue
            for turn in item.turns:
                for text in (turn.user, turn.assistant):
                    symbols = vocabulary.encode(text)
                    if vocabulary.decode(symbols) != text:
                        raise RuntimeError("vocabulary failed an exact text roundtrip")
                    byte_count += len(text.encode("utf-8"))
                    symbol_count += len(symbols)
                    message_count += 1
        metrics[split] = {"text_bytes": byte_count, "symbols": symbol_count, "messages": message_count,
                          "bytes_per_symbol": byte_count / symbol_count if symbol_count else None}
    args.output_directory.mkdir(parents=True, exist_ok=False)
    vocabulary.save(args.output_directory / "vocabulary.json")
    report = {"vocabulary_sha256": vocabulary.sha256, "vocabulary_size": len(vocabulary),
              "split_metrics": metrics, "all_roundtrips_exact": True,
              "scope": "lexical_segmentation_not_a_language_competence_result",
              "pretrained_assets_loaded": False,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (args.output_directory / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
