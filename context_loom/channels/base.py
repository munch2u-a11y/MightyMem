"""Base definitions and channel adapter interface for multi-channel communications."""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, Optional


@dataclass
class ChannelMessage:
    """Standardized multi-channel incoming message representation."""

    channel_id: str
    external_user_id: str
    text: str
    display_name: str = "User"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseChannelAdapter(abc.ABC):
    """Abstract interface for external communications channel adapters."""

    def __init__(self, channel_id: str) -> None:
        self.channel_id = channel_id
        self._message_handler: Optional[
            Callable[[ChannelMessage], Coroutine[Any, Any, str]]
        ] = None

    def set_message_handler(
        self, handler: Callable[[ChannelMessage], Coroutine[Any, Any, str]]
    ) -> None:
        """Register the channel router message processing callback."""
        self._message_handler = handler

    @abc.abstractmethod
    async def start(self) -> None:
        """Start listening for incoming messages (e.g. polling loop or webhook)."""
        pass

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop listening and close channel adapter connections."""
        pass

    @abc.abstractmethod
    async def send_reply(self, external_user_id: str, text: str) -> None:
        """Send an outbound reply message back to the external user."""
        pass
