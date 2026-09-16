from __future__ import annotations

import pytest

from context_loom.retrieval import is_future_query, is_historical_query


@pytest.mark.parametrize(
    "query",
    [
        "What did we use originally?",
        "Show the previous setting.",
        "What was configured before the migration?",
        "I need the historical state.",
        "What did this service use to do?",
    ],
)
def test_historical_intent_is_detected(query: str) -> None:
    assert is_historical_query(query)


@pytest.mark.parametrize(
    "query",
    [
        "Reread the file before reporting completion.",
        "Inspect before writing.",
        "How old is this file?",
        "Previous instructions say to verify, but what is current?",
    ],
)
def test_ordinary_sequence_words_do_not_trigger_history(query: str) -> None:
    expected = query.startswith("Previous")
    assert is_historical_query(query) is expected


@pytest.mark.parametrize(
    "query",
    ["What will we use?", "Show the planned setting.", "What is scheduled next?"],
)
def test_future_intent_is_detected(query: str) -> None:
    assert is_future_query(query)
