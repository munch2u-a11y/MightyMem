from __future__ import annotations

import pytest

from context_loom import Capability, EvidenceStatus, MemoryRecord, RecordKind
from context_loom.store import IdentityConflict, LoomStore


def record(record_id: str, content: str, *, scope: str = "alpha", at: float = 10) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        kind=RecordKind.USER,
        content=content,
        created_at=at,
        scope=scope,
        evidence_status=EvidenceStatus.USER_REPORTED,
    )


def test_record_ids_are_idempotent_but_cannot_be_reused() -> None:
    with LoomStore() as store:
        first = record("stable", "canonical text")
        assert store.append_record(first) == first
        assert store.append_record(first).content == "canonical text"
        with pytest.raises(IdentityConflict):
            store.append_record(record("stable", "changed text"))


def test_correction_must_target_an_existing_record_in_same_scope() -> None:
    with LoomStore() as store:
        store.append_record(record("old", "old state"))
        with pytest.raises(ValueError, match="another scope"):
            store.append_record(
                MemoryRecord(
                    id="bad-correction",
                    kind=RecordKind.CORRECTION,
                    content="new state",
                    created_at=20,
                    scope="beta",
                    supersedes_id="old",
                    evidence_status=EvidenceStatus.USER_REPORTED,
                )
            )
        with pytest.raises(ValueError, match="already exist"):
            store.append_record(
                MemoryRecord(
                    id="orphan",
                    kind=RecordKind.CORRECTION,
                    content="new state",
                    created_at=20,
                    scope="alpha",
                    supersedes_id="missing",
                    evidence_status=EvidenceStatus.USER_REPORTED,
                )
            )


def test_latest_version_walks_a_correction_chain() -> None:
    with LoomStore() as store:
        store.append_record(record("v1", "one", at=10))
        for identifier, parent, text, at in (("v2", "v1", "two", 20), ("v3", "v2", "three", 30)):
            store.append_record(
                MemoryRecord(
                    id=identifier,
                    kind=RecordKind.CORRECTION,
                    content=text,
                    created_at=at,
                    scope="alpha",
                    supersedes_id=parent,
                    evidence_status=EvidenceStatus.USER_REPORTED,
                )
            )
        assert store.latest_version("v1").id == "v3"
        assert store.latest_version("v1", as_of=25).id == "v2"
        assert store.latest_version("v1", as_of=5) is None


def test_future_effective_correction_does_not_replace_current_state() -> None:
    with LoomStore() as store:
        store.append_record(record("current", "current state", at=10))
        store.append_record(
            MemoryRecord(
                id="planned", kind=RecordKind.CORRECTION, content="planned state",
                created_at=20, effective_at=100, scope="alpha", supersedes_id="current",
                evidence_status=EvidenceStatus.USER_REPORTED,
            )
        )
        assert store.latest_version("current", as_of=50).id == "current"
        assert store.latest_version("current", as_of=110).id == "planned"


def test_as_of_and_scope_filters_prevent_temporal_and_project_leakage() -> None:
    with LoomStore() as store:
        store.append_record(record("alpha-now", "visible", at=10))
        store.append_record(record("alpha-future", "future", at=30))
        store.append_record(record("beta-now", "other", scope="beta", at=10))
        assert [item.id for item in store.list_records(scope="alpha", as_of=20)] == ["alpha-now"]


def test_database_reopens_with_canonical_state(tmp_path) -> None:
    path = tmp_path / "persistent.sqlite"
    with LoomStore(path) as store:
        store.append_record(record("persisted", "survives restart"))
        assert store.integrity_check() == "ok"
    with LoomStore(path) as reopened:
        assert reopened.get_record("persisted").content == "survives restart"
        assert reopened.integrity_check() == "ok"


def test_embedding_cache_is_bound_to_record_content() -> None:
    with LoomStore() as store:
        store.append_record(record("embedded", "unchanging"))
        store.put_embedding("embedded", "provider-v1", [0.1, 0.2])
        assert store.get_embedding("embedded", "provider-v1") == [0.1, 0.2]
        assert store.get_embedding("embedded", "provider-v2") is None


def test_capability_contract_ids_cannot_silently_change() -> None:
    base = Capability("reader", "read", "read a file", {"type": "object"})
    changed = Capability("reader", "read", "read everything", {"type": "object"})
    with LoomStore() as store:
        store.register_capability(base)
        store.register_capability(base)
        with pytest.raises(IdentityConflict):
            store.register_capability(changed)


@pytest.mark.parametrize("importance", [-0.01, 1.01])
def test_record_importance_is_bounded(importance: float) -> None:
    with pytest.raises(ValueError, match="importance"):
        MemoryRecord("bad", RecordKind.USER, "text", 1, importance=importance)


def test_unknown_embedding_target_is_rejected() -> None:
    with LoomStore() as store, pytest.raises(ValueError, match="unknown record"):
        store.put_embedding("missing", "provider", [1.0])
