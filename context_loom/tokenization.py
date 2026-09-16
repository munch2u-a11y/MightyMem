from __future__ import annotations

import json
import re
from typing import Any, Protocol


TOKEN_RE = re.compile(r"\w+(?:[-'][\w]+)*|[^\w\s]", re.UNICODE)


class TokenCounter(Protocol):
    identity: str

    def count(self, text: str) -> int: ...


class RegexTokenCounter:
    """Deterministic token counter for Context Loom compiler budgets.

    Production adapters should supply the actual language model tokenizer. The
    compiler remains exact with respect to whichever counter is configured.
    """

    identity = "regex-v1"

    def count(self, text: str) -> int:
        return len(TOKEN_RE.findall(text))


def schema_tokens(counter: TokenCounter, schema: dict[str, Any]) -> int:
    return counter.count(json.dumps(schema, ensure_ascii=False, sort_keys=True))
