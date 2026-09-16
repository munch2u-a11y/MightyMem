"""Multi-channel communications and backend user profile identity system for MightyMem."""
from __future__ import annotations

from .base import BaseChannelAdapter, ChannelMessage
from .discord import DiscordChannelAdapter
from .piper_voice import PiperVoiceChannelAdapter
from .profiles import UserProfile, UserProfileStore
from .router import ChannelRouter
from .telegram import TelegramChannelAdapter

__all__ = [
    "BaseChannelAdapter",
    "ChannelMessage",
    "UserProfile",
    "UserProfileStore",
    "ChannelRouter",
    "DiscordChannelAdapter",
    "PiperVoiceChannelAdapter",
    "TelegramChannelAdapter",
]
