"""Context Loom public API."""

from .channels import (
    BaseChannelAdapter,
    ChannelMessage,
    ChannelRouter,
    DiscordChannelAdapter,
    PiperVoiceChannelAdapter,
    TelegramChannelAdapter,
    UserProfile,
    UserProfileStore,
)
from .compiler import CompilerConfig, ContextCompiler
from .credentials import get_credential, load_credentials, save_credential
from .engine import ContextLoom

from .models import (
    Capability,
    ContextPacket,
    EvidenceStatus,
    MemoryRecord,
    RecordKind,
    TaskState,
    ToolReceipt,
)
from .reflection import TopicalReflectionEngine
from .retrieval import OllamaEmbeddingProvider
from .runtime import BoundedFileRuntime, RuntimeResult, ToolExecutionError
from .store import LoomStore

__all__ = [
    "BaseChannelAdapter",
    "Capability",
    "BoundedFileRuntime",
    "ChannelMessage",
    "ChannelRouter",
    "CompilerConfig",
    "ContextCompiler",
    "ContextLoom",
    "ContextPacket",
    "DiscordChannelAdapter",
    "EvidenceStatus",
    "LoomStore",
    "MemoryRecord",
    "OllamaEmbeddingProvider",
    "PiperVoiceChannelAdapter",
    "RecordKind",
    "RuntimeResult",
    "TaskState",
    "TelegramChannelAdapter",
    "ToolExecutionError",
    "ToolReceipt",
    "TopicalReflectionEngine",
    "UserProfile",
    "UserProfileStore",
    "get_credential",
    "load_credentials",
    "save_credential",
]

