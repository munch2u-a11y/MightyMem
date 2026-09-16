"""Unit tests for multi-channel comms routing and backend user profile store."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from context_loom.channels import (
    BaseChannelAdapter,
    ChannelMessage,
    ChannelRouter,
    DiscordChannelAdapter,
    PiperVoiceChannelAdapter,
    TelegramChannelAdapter,
    UserProfileStore,
)
from context_loom.engine import ContextLoom


class MockChannelAdapter(BaseChannelAdapter):
    """Mock channel adapter for unit testing multi-user message routing."""

    def __init__(self, channel_id: str = "mock_telegram") -> None:
        super().__init__(channel_id=channel_id)
        self.sent_replies: list[tuple[str, str]] = []
        self.is_started = False

    async def start(self) -> None:
        self.is_started = True

    async def stop(self) -> None:
        self.is_started = False

    async def send_reply(self, external_user_id: str, text: str) -> None:
        self.sent_replies.append((external_user_id, text))

    async def simulate_incoming(self, external_user_id: str, display_name: str, text: str) -> str:
        msg = ChannelMessage(
            channel_id=self.channel_id,
            external_user_id=external_user_id,
            display_name=display_name,
            text=text,
        )
        if self._message_handler:
            reply = await self._message_handler(msg)
            await self.send_reply(external_user_id, reply)
            return reply
        return ""


def test_user_profile_store_creation_and_mapping(tmp_path: Path) -> None:
    db_path = tmp_path / "profiles.sqlite"
    store = UserProfileStore(db_path)

    # 1. Create profile for Telegram User 1001 (Alice)
    p1 = store.get_or_create_profile(
        channel_id="telegram",
        external_user_id="1001",
        display_name="Alice",
    )
    assert p1.profile_id.startswith("usr_")
    assert p1.memory_scope == f"user_{p1.profile_id}"
    assert p1.display_name == "Alice"
    assert p1.channel_id == "telegram"

    # 2. Idempotent fetch returns existing profile with same memory scope
    p1_again = store.get_or_create_profile(
        channel_id="telegram",
        external_user_id="1001",
        display_name="Alice Updated",
    )
    assert p1_again.profile_id == p1.profile_id
    assert p1_again.memory_scope == p1.memory_scope
    assert p1_again.display_name == "Alice Updated"

    # 3. Create profile for Telegram User 1002 (Bob)
    p2 = store.get_or_create_profile(
        channel_id="telegram",
        external_user_id="1002",
        display_name="Bob",
    )
    assert p2.profile_id != p1.profile_id
    assert p2.memory_scope != p1.memory_scope

    # 4. Lookup by scope
    fetched_p1 = store.get_profile_by_scope(p1.memory_scope)
    assert fetched_p1 is not None
    assert fetched_p1.external_user_id == "1001"

    # 5. List profiles
    all_profiles = store.list_profiles()
    assert len(all_profiles) == 2
    store.close()


def test_channel_router_multi_user_memory_isolation(tmp_path: Path) -> None:
    async def _test() -> None:
        loom = ContextLoom(tmp_path / "loom.sqlite")
        profiles = UserProfileStore(tmp_path / "profiles.sqlite")
        router = ChannelRouter(loom=loom, profiles=profiles)

        adapter = MockChannelAdapter(channel_id="telegram")
        router.register_adapter(adapter)

        await router.start_all()
        assert adapter.is_started

        # Alice tells the agent her favorite programming language is Python
        await adapter.simulate_incoming(
            external_user_id="alice_tg_101",
            display_name="Alice",
            text="My favorite programming language is Python.",
        )

        # Bob tells the agent his favorite programming language is Rust
        await adapter.simulate_incoming(
            external_user_id="bob_tg_102",
            display_name="Bob",
            text="My favorite programming language is Rust.",
        )

        # Verify Alice's memory scope has Alice's statement
        alice_profile = profiles.get_or_create_profile("telegram", "alice_tg_101")
        alice_records = loom.store.list_records(scope=alice_profile.memory_scope)
        alice_content = [r.content for r in alice_records]
        assert any("Python" in c for c in alice_content)
        assert not any("Rust" in c for c in alice_content)

        # Verify Bob's memory scope has Bob's statement
        bob_profile = profiles.get_or_create_profile("telegram", "bob_tg_102")
        bob_records = loom.store.list_records(scope=bob_profile.memory_scope)
        bob_content = [r.content for r in bob_records]
        assert any("Rust" in c for c in bob_content)
        assert not any("Python" in c for c in bob_content)

        await router.stop_all()
        assert not adapter.is_started
        profiles.close()

    asyncio.run(_test())


def test_telegram_adapter_instantiation() -> None:
    adapter = TelegramChannelAdapter(bot_token="123456:TEST_TOKEN")
    assert adapter.channel_id == "telegram"
    assert adapter.bot_token == "123456:TEST_TOKEN"


def test_discord_adapter_instantiation_and_simulation(tmp_path: Path) -> None:
    async def _test() -> None:
        loom = ContextLoom(tmp_path / "loom.sqlite")
        profiles = UserProfileStore(tmp_path / "profiles.sqlite")
        router = ChannelRouter(loom=loom, profiles=profiles)

        adapter = DiscordChannelAdapter(bot_token="MTIzNDU2Nzg5.TEST.DISCORD_TOKEN", channel_id_target="1234567890123456789")
        assert adapter.channel_id_target == "1234567890123456789"
        router.register_adapter(adapter)


        # Simulate incoming Discord message
        reply = await adapter.simulate_incoming(
            external_user_id="discord_chan_999",
            display_name="DiscordAlice",
            text="hello from discord",
        )

        assert "DiscordAlice" in reply or "Hello" in reply
        assert "[MODE: REPLY]" not in reply
        assert "[AVAILABLE ACTIONS]" not in reply

        profiles.close()

    asyncio.run(_test())


def test_piper_voice_adapter_instantiation_and_simulation(tmp_path: Path) -> None:
    async def _test() -> None:
        loom = ContextLoom(tmp_path / "loom.sqlite")
        profiles = UserProfileStore(tmp_path / "profiles.sqlite")
        router = ChannelRouter(loom=loom, profiles=profiles)

        adapter = PiperVoiceChannelAdapter(piper_model_path="en_US-lessac-medium.onnx")
        router.register_adapter(adapter)

        reply = await adapter.simulate_voice_input(
            external_user_id="voice_user_1",
            display_name="VoiceAlice",
            transcript_text="what is my favorite programming language?",
        )


        assert "[MODE: REPLY]" not in reply
        assert "[AVAILABLE ACTIONS]" not in reply
        profiles.close()

    asyncio.run(_test())


def test_channel_router_does_not_leak_raw_system_headers(tmp_path: Path) -> None:
    async def _test() -> None:
        loom = ContextLoom(tmp_path / "loom.sqlite")
        profiles = UserProfileStore(tmp_path / "profiles.sqlite")
        router = ChannelRouter(loom=loom, profiles=profiles)

        adapter = MockChannelAdapter(channel_id="telegram")
        router.register_adapter(adapter)

        reply = await adapter.simulate_incoming(
            external_user_id="user_777",
            display_name="Stranger",
            text="hey, what's up?",
        )

        # Verify output is a clean natural conversational reply, not raw system header tags
        assert "[MODE: REPLY]" not in reply
        assert "[AVAILABLE ACTIONS]" not in reply
        assert "[RECENCY & CONVERSATIONAL CONTINUITY]" not in reply
        assert "Hello Stranger" in reply or "acknowledged" in reply.lower() or "noted" in reply.lower()

        profiles.close()

    asyncio.run(_test())



