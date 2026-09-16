"""Multi-Channel Message Router mapping channel messages to user profile memory scopes."""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from .base import BaseChannelAdapter, ChannelMessage
from .profiles import UserProfile, UserProfileStore
from ..engine import ContextLoom

logger = logging.getLogger(__name__)


def _try_local_llm_completion(prompt: str, ollama_url: str = "http://127.0.0.1:11434", model: str = "granite4.2:8b") -> Optional[str]:
    """Attempt a local LLM completion via Ollama HTTP API if server is running."""
    try:
        url = f"{ollama_url}/api/generate"
        payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            response_text = data.get("response", "").strip()
            if response_text:
                return response_text
    except Exception:
        pass
    return None



class ChannelRouter:
    """Routes incoming multi-channel messages to user-isolated ContextLoom memory scopes."""

    def __init__(
        self,
        loom: ContextLoom,
        profiles: UserProfileStore,
        agent_runner: Optional[Callable[[str, str, UserProfile], str]] = None,
        ollama_url: str = "http://127.0.0.1:11434",
        model_name: str = "granite4.2:8b",
    ) -> None:
        self.loom = loom
        self.profiles = profiles
        self.agent_runner = agent_runner
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.adapters: Dict[str, BaseChannelAdapter] = {}

    def register_adapter(self, adapter: BaseChannelAdapter) -> None:
        """Register a channel adapter and set its message processing handler."""
        self.adapters[adapter.channel_id] = adapter
        adapter.set_message_handler(self.handle_incoming_message)

    def _synthesize_response(self, message_text: str, scope: str, profile: UserProfile) -> str:
        """Synthesize a natural language assistant response using JIT context and local LLM."""
        packet = self.loom.compiler.compile(message_text, scope=scope)
        
        # 1. Attempt local LLM completion on the compiled JIT context header
        llm_reply = _try_local_llm_completion(packet.rendered, ollama_url=self.ollama_url, model=self.model_name)
        if llm_reply:
            return llm_reply

        # 2. Fallback: Formulate a clean, grounded natural language response without leaking raw prompt headers
        clean_text = message_text.lower().strip()
        
        # Handle greetings
        if re.match(r"^(hi|hello|hey|greetings|howdy|sup)\b", clean_text):
            return f"Hello {profile.display_name}! I've noted that into your personal memory context. How can I help you today?"

        # Extract admitted evidence facts if query matches
        records = self.loom.store.list_records(scope=scope)
        matched_facts = [r.content for r in records if any(word in r.content.lower() for word in clean_text.split() if len(word) > 3)]
        if matched_facts:
            return f"Based on your stored context ({profile.display_name}): {matched_facts[-1]}"

        return f"Acknowledged, {profile.display_name}. I've remembered: \"{message_text}\""

    async def handle_incoming_message(self, message: ChannelMessage) -> str:
        """Process an incoming ChannelMessage, isolated by user profile memory scope."""
        profile = self.profiles.get_or_create_profile(
            channel_id=message.channel_id,
            external_user_id=message.external_user_id,
            display_name=message.display_name,
        )

        scope = profile.memory_scope

        # Ingest user statement into user-isolated scope
        self.loom.remember_user_statement(
            content=f"[{profile.display_name}]: {message.text}",
            scope=scope,
            created_at=message.timestamp,
        )

        # Generate agent reply (either via custom agent_runner or synthesized response)
        if self.agent_runner:
            reply_text = self.agent_runner(message.text, scope, profile)
        else:
            reply_text = self._synthesize_response(message.text, scope, profile)

        # Ingest assistant statement into user-isolated scope
        self.loom.remember_assistant_statement(
            content=reply_text,
            scope=scope,
            created_at=time.time(),
        )

        return reply_text

    async def start_all(self) -> None:
        """Start listening on all registered channel adapters."""
        for adapter in self.adapters.values():
            await adapter.start()

    async def stop_all(self) -> None:
        """Stop all registered channel adapters."""
        for adapter in self.adapters.values():
            await adapter.stop()
