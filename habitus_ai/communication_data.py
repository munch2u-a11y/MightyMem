"""Prepare attributed human dialogue examples; never used by inference.

OpenAssistant/oasst1 and oasst2 are published under Apache-2.0. Their authorship
flags are annotations, not an independent guarantee about every contributor.
No source text is executed, rewritten by a model, or treated as lived memory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

from .communication_training import TeachingDialogue, TeachingTurn, curriculum_hash, load_curriculum


OASST_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
OASST_SOURCE_SHA256 = "2a9a8fd343e9b28e04a895a669d3253f82d93e9c174d440199ae19d5fafbdff7"
OASST_LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
OASST_URL = (
    "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/"
    + OASST_REVISION + "/2023-04-12_oasst_ready.trees.jsonl.gz"
)
RELEASES = {
    "oasst1": {
        "revision": OASST_REVISION, "source_sha256": OASST_SOURCE_SHA256,
        "filename": "2023-04-12_oasst_ready.trees.jsonl.gz",
        "license_reference": f"https://huggingface.co/datasets/OpenAssistant/oasst1/blob/{OASST_REVISION}/LICENSE",
    },
    "oasst2": {
        "revision": "179dd21fc55192153d94adb0e0ce8f69e222bf75",
        "source_sha256": "7a886a16ccfc1173c4f00a6897523e3c95b2785a86ee44a18a98f4f2807ee29b",
        "filename": "2023-11-05_oasst2_ready.trees.jsonl.gz",
        "dataset_card_sha256": "52bd472cf3cae04a4b21ecd5316f929741c47d991d4763b041c3a94de072dc31",
        "license_reference": "https://huggingface.co/datasets/OpenAssistant/oasst2/blob/179dd21fc55192153d94adb0e0ce8f69e222bf75/README.md",
    },
}


@dataclass(frozen=True)
class DialogueFilter:
    maximum_event_bytes: int | None = None
    maximum_dialogue_bytes: int | None = None
    maximum_turns: int | None = None
    minimum_quality: float = 0.65
    split_seed: int = 1701


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rejection_reason(message: Mapping[str, Any], role: str, policy: DialogueFilter) -> str | None:
    if message.get("role") != role:
        return "role_order"
    if message.get("lang") != "en":
        return "non_english"
    if message.get("synthetic") is not False or message.get("model_name"):
        return "synthetic_or_unknown_authorship"
    if message.get("review_result") is not True or message.get("deleted") is not False:
        return "not_reviewed_or_deleted"
    text = message.get("text")
    if not isinstance(text, str) or not text.strip():
        return "empty_text"
    if (
        policy.maximum_event_bytes is not None
        and len(text.encode("utf-8")) > policy.maximum_event_bytes
    ):
        return "event_byte_limit"
    labels = message.get("labels") or {}
    for name in ("pii", "spam", "lang_mismatch"):
        annotation = labels.get(name) or {}
        if annotation.get("count", 0) < 1 or annotation.get("value") != 0:
            return "flagged_or_unreviewed_" + name
    for name in ("hate_speech", "sexual_content", "not_appropriate", "fails_task"):
        if (labels.get(name) or {}).get("value", 0) > 0:
            return "flagged_" + name
    quality = labels.get("quality") or {}
    if quality.get("count", 0) < 1 or quality.get("value", 0) < policy.minimum_quality:
        return "low_or_unrated_quality"
    if role == "assistant" and message.get("rank") not in (0, None):
        return "nonpreferred_reply"
    return None


def root_split(root_text: str, seed: int) -> str:
    # Identical normalized root questions stay together even if the source
    # assigned them different tree IDs. All branches of a tree share this key.
    normalized = " ".join(root_text.casefold().split())
    value = int(hashlib.sha256(f"{seed}\0{normalized}".encode("utf-8")).hexdigest()[:12], 16) % 100
    return "test" if value < 5 else "validation" if value < 10 else "train"


def prepare_dialogues(
    trees: Iterable[Mapping[str, Any]], policy: DialogueFilter, *, dataset: str = "oasst1",
) -> tuple[tuple[TeachingDialogue, ...], dict[str, Any]]:
    if dataset not in RELEASES:
        raise ValueError("unknown pinned dialogue release")
    if any(
        value is not None and value < 1
        for value in (
            policy.maximum_event_bytes,
            policy.maximum_dialogue_bytes,
            policy.maximum_turns,
        )
    ):
        raise ValueError("dialogue limits must be positive")
    if not 0 <= policy.minimum_quality <= 1:
        raise ValueError("minimum quality must be within [0, 1]")
    rejected: Counter[str] = Counter()
    selected: list[TeachingDialogue] = []
    seen_dialogues: set[str] = set()
    tree_count = 0
    for tree in trees:
        tree_count += 1
        if tree.get("tree_state") != "ready_for_export":
            rejected["tree_not_ready"] += 1
            continue
        root = tree["prompt"]
        tree_id = str(tree["message_tree_id"])
        candidates: list[tuple[TeachingTurn, ...]] = []
        candidate_ids: list[tuple[str, ...]] = []

        def visit(message: Mapping[str, Any], path: tuple[Mapping[str, Any], ...], byte_count: int) -> None:
            expected_role = "prompter" if len(path) % 2 == 0 else "assistant"
            reason = rejection_reason(message, expected_role, policy)
            if reason is not None:
                rejected[reason] += 1
                return  # never splice around a missing or rejected parent
            text = str(message["text"])
            next_count = byte_count + len(text.encode("utf-8")) + 1
            if (
                policy.maximum_dialogue_bytes is not None
                and next_count > policy.maximum_dialogue_bytes
            ):
                rejected["dialogue_byte_limit"] += 1
                return
            extended = (*path, message)
            if expected_role == "assistant":
                turns = tuple(TeachingTurn(str(extended[i]["text"]), str(extended[i + 1]["text"]))
                              for i in range(0, len(extended), 2))
                candidates.append(turns)
                candidate_ids.append(tuple(str(node["message_id"]) for node in extended))
                if policy.maximum_turns is not None and len(turns) >= policy.maximum_turns:
                    return
            for child in message.get("replies", ()):
                if child.get("parent_id") not in (None, message["message_id"]):
                    rejected["parent_mismatch"] += 1
                    continue
                visit(child, extended, next_count)

        visit(root, (), 0)
        if not candidates:
            rejected["no_complete_approved_turn"] += 1
            continue
        # Prefer a complete multi-turn path. Stable IDs resolve equal-length
        # alternatives without concatenating mutually exclusive replies.
        index = min(range(len(candidates)), key=lambda i: (-len(candidates[i]), candidate_ids[i]))
        turns = candidates[index]
        signature = hashlib.sha256(json.dumps([asdict(turn) for turn in turns], sort_keys=True).encode()).hexdigest()
        if signature in seen_dialogues:
            rejected["duplicate_complete_dialogue"] += 1
            continue
        seen_dialogues.add(signature)
        selected.append(TeachingDialogue(
            f"{dataset}:{tree_id}", root_split(str(root["text"]), policy.split_seed), "human_dialogue", turns,
            provenance={
                "dataset": f"OpenAssistant/{dataset}", "revision": RELEASES[dataset]["revision"],
                "tree_id": tree_id, "message_ids": list(candidate_ids[index]),
                "source_authorship_annotation": "non_synthetic",
                "license": "Apache-2.0", "text_modified": False,
                "training_example_not_lived_experience": True,
            },
        ))
    selected.sort(key=lambda item: item.dialogue_id)
    return tuple(selected), {
        "source_trees": tree_count, "accepted_dialogues": len(selected),
        "split_counts": dict(Counter(item.split for item in selected)),
        "turn_counts": dict(Counter(len(item.turns) for item in selected)),
        "rejected_counts": dict(rejected),
        "accepted_text_bytes": sum(len(turn.user.encode()) + len(turn.assistant.encode())
                                   for item in selected for turn in item.turns),
    }


def align_prior_splits(
    dialogues: Sequence[TeachingDialogue], prior: Sequence[TeachingDialogue],
) -> tuple[tuple[TeachingDialogue, ...], dict[str, int]]:
    """A later source revision cannot move a previously held-out tree to train."""
    def keys(item: TeachingDialogue) -> set[str]:
        result = {"root:" + " ".join(item.turns[0].user.casefold().split())}
        if item.provenance.get("tree_id"):
            result.add("tree:" + item.provenance["tree_id"])
        result.update("message:" + value for value in item.provenance.get("message_ids", ()))
        return result

    ownership: dict[str, str] = {}
    for item in prior:
        for key in keys(item):
            if key in ownership and ownership[key] != item.split:
                raise ValueError("prior corpus already has conflicting split ownership")
            ownership[key] = item.split
    aligned = []
    matched = changed = 0
    for item in dialogues:
        matches = {ownership[key] for key in keys(item) if key in ownership}
        if len(matches) > 1:
            raise ValueError("source revision connects conflicting prior splits")
        if matches:
            split = matches.pop()
            matched += 1
            changed += split != item.split
            item = replace(item, split=split)
        aligned.append(item)
    # Also catch a new normalized duplicate that would otherwise straddle the
    # inherited partition and the release's default hashed partition.
    new_ownership: dict[str, str] = {}
    for item in aligned:
        for key in keys(item):
            if key in new_ownership and new_ownership[key] != item.split:
                raise ValueError("aligned release contains conflicting split ownership")
            new_ownership[key] = item.split
    return tuple(aligned), {"matched_prior_dialogues": matched, "changed_split_assignments": changed}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(RELEASES), default="oasst1")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--dataset-card", type=Path)
    parser.add_argument("--prior-corpus", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--maximum-event-bytes", type=int, default=None)
    parser.add_argument("--maximum-dialogue-bytes", type=int, default=None)
    parser.add_argument("--maximum-turns", type=int, default=None)
    parser.add_argument("--minimum-quality", type=float, default=0.65)
    args = parser.parse_args(argv)
    release = RELEASES[args.dataset]
    if file_sha256(args.source) != release["source_sha256"]:
        parser.error("source checksum differs from the pinned export")
    if file_sha256(args.license) != OASST_LICENSE_SHA256:
        parser.error("license checksum differs from the supported Apache-2.0 text")
    if release.get("dataset_card_sha256"):
        if args.dataset_card is None or file_sha256(args.dataset_card) != release["dataset_card_sha256"]:
            parser.error("this release requires its pinned dataset card and license declaration")
    policy = DialogueFilter(args.maximum_event_bytes, args.maximum_dialogue_bytes,
                           args.maximum_turns, args.minimum_quality)
    with gzip.open(args.source, "rt", encoding="utf-8") as source:
        dialogues, counts = prepare_dialogues((json.loads(line) for line in source), policy, dataset=args.dataset)
    prior_alignment = None
    if args.prior_corpus:
        prior = load_curriculum(args.prior_corpus)
        dialogues, alignment_counts = align_prior_splits(dialogues, prior)
        prior_alignment = {**alignment_counts, "prior_curriculum_sha256": curriculum_hash(prior)}
        counts["split_counts"] = dict(Counter(item.split for item in dialogues))
    args.output_directory.mkdir(parents=True, exist_ok=False)
    corpus_path = args.output_directory / "dialogues.jsonl"
    with corpus_path.open("x", encoding="utf-8") as destination:
        for dialogue in dialogues:
            destination.write(json.dumps(asdict(dialogue), ensure_ascii=False, sort_keys=True) + "\n")
    shutil.copyfile(args.license, args.output_directory / "LICENSE")
    if args.dataset_card:
        shutil.copyfile(args.dataset_card, args.output_directory / "SOURCE_README.md")
    manifest = {
        "dataset": f"OpenAssistant/{args.dataset}", "revision": release["revision"],
        "source_url": f"https://huggingface.co/datasets/OpenAssistant/{args.dataset}/resolve/{release['revision']}/{release['filename']}",
        "source_sha256": release["source_sha256"],
        "license": "Apache-2.0", "license_sha256": OASST_LICENSE_SHA256,
        "license_reference": release["license_reference"],
        "dataset_card_sha256": file_sha256(args.dataset_card) if args.dataset_card else None,
        "prior_split_alignment": prior_alignment,
        "filters": asdict(policy), "counts": counts,
        "curriculum_sha256": curriculum_hash(dialogues),
        "artifact_sha256": file_sha256(corpus_path),
        "importer_sha256": file_sha256(Path(__file__)),
        "modifications": "Filtered complete dialogue prefixes; text unchanged; one branch per tree; deterministic grouped splits.",
        "limitations": "Human-origin flags are source annotations. Quality filters do not guarantee factual accuracy or prove authorship.",
    }
    with (args.output_directory / "manifest.json").open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, sort_keys=True)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
