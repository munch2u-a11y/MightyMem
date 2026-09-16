from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Mapping, Protocol, Sequence, runtime_checkable


ChatMessage = Mapping[str, str]


class ModelUnavailableError(RuntimeError):
    pass


@runtime_checkable
class ChatModel(Protocol):
    def generate(self, messages: Sequence[ChatMessage]) -> str: ...


class OllamaChatModel:
    """Small standard-library adapter for Ollama's local chat endpoint."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float | None = None,
        temperature: float = 0.7,
        context_tokens: int | None = None,
        num_gpu: int | None = None,
        maximum_tokens: int | None = None,
        think: bool | None = False,
    ):
        self.model = str(model).strip()
        if not self.model:
            raise ValueError("Ollama model name cannot be empty")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = (
            None if timeout_seconds is None else float(timeout_seconds)
        )
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("an explicit model timeout must be positive")
        self.temperature = max(0.0, float(temperature))
        self.context_tokens = (
            None if context_tokens is None else int(context_tokens)
        )
        if self.context_tokens is not None and self.context_tokens < 1:
            raise ValueError("an explicit model context must be positive")
        self.num_gpu = None if num_gpu is None else int(num_gpu)
        if self.num_gpu is not None and self.num_gpu < -1:
            raise ValueError("num_gpu must be -1, zero, or a positive layer count")
        self.maximum_tokens = (
            None if maximum_tokens is None else int(maximum_tokens)
        )
        if self.maximum_tokens is not None and self.maximum_tokens < 1:
            raise ValueError("an explicit generation limit must be positive")
        self.think = None if think is None else bool(think)

    def _native_context_tokens(self) -> int:
        request = urllib.request.Request(
            f"{self.base_url}/api/show",
            data=json.dumps({"model": self.model}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        values = [
            int(value)
            for key, value in result.get("model_info", {}).items()
            if str(key).endswith(".context_length") and int(value) > 0
        ]
        if not values:
            raise ModelUnavailableError(
                f"Ollama did not report a native context length for {self.model!r}"
            )
        return max(values)

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        if self.context_tokens is None:
            try:
                self.context_tokens = self._native_context_tokens()
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
                raise ModelUnavailableError(
                    f"Ollama model inspection failed for {self.model!r} at {self.base_url}: {error}"
                ) from error
        options: dict[str, float | int] = {"temperature": self.temperature}
        if self.context_tokens is not None:
            options["num_ctx"] = self.context_tokens
        if self.maximum_tokens is not None:
            options["num_predict"] = self.maximum_tokens
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        request_payload = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "stream": False,
            "keep_alive": -1,
            "options": options,
        }
        if self.think is not None:
            request_payload["think"] = self.think
        payload = json.dumps(request_payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise ModelUnavailableError(
                f"Ollama chat failed for {self.model!r} at {self.base_url}: {error}"
            ) from error
        if result.get("done_reason") == "length":
            raise ModelUnavailableError(
                "Ollama reached its physical context boundary; partial output was rejected"
            )
        content = str(result.get("message", {}).get("content", "")).strip()
        if not content:
            detail = result.get("error") or "the model returned no ordinary response content"
            raise ModelUnavailableError(str(detail))
        return content
