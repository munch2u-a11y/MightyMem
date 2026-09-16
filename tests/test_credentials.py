"""Unit tests for MightyMem safe credentials store and env loader."""
from __future__ import annotations

import os
from pathlib import Path
import pytest

from context_loom.credentials import (
    get_credential,
    get_credentials_file_paths,
    load_credentials,
    parse_env_file,
    save_credential,
)
from context_loom.channels import (
    TelegramChannelAdapter,
    DiscordChannelAdapter,
    PiperVoiceChannelAdapter,
)


def test_parse_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Comment line\n"
        "TELEGRAM_BOT_TOKEN=123456:SECRET\n"
        "   # Another comment\n"
        "DISCORD_BOT_TOKEN=\"MTIzNDU2.QUOTED_TOKEN\"\n"
        "PIPER_MODEL_PATH='models/piper.onnx'\n",
        encoding="utf-8",
    )

    parsed = parse_env_file(env_file)
    assert parsed["TELEGRAM_BOT_TOKEN"] == "123456:SECRET"
    assert parsed["DISCORD_BOT_TOKEN"] == "MTIzNDU2.QUOTED_TOKEN"
    assert parsed["PIPER_MODEL_PATH"] == "models/piper.onnx"


def test_save_and_get_credential(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target_env = tmp_path / "mightymem_test.env"
    monkeypatch.setenv("MIGHTYMEM_ENV_FILE", str(target_env))

    saved_path = save_credential("TELEGRAM_BOT_TOKEN", "test_tg_token_123", target_file=target_env)
    assert saved_path == target_env
    assert target_env.exists()

    val = get_credential("TELEGRAM_BOT_TOKEN")
    assert val == "test_tg_token_123"

    # Verify file permissions mode on Linux is 0o600 (owner read/write only)
    if os.name == "posix":
        mode = oct(target_env.stat().st_mode & 0o777)
        assert mode == "0o600"


def test_channel_adapter_credential_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target_env = tmp_path / "mightymem_test.env"
    monkeypatch.setenv("MIGHTYMEM_ENV_FILE", str(target_env))
    save_credential("TELEGRAM_BOT_TOKEN", "tg_fallback_token_888", target_file=target_env)
    save_credential("DISCORD_BOT_TOKEN", "dc_fallback_token_999", target_file=target_env)
    save_credential("PIPER_MODEL_PATH", "custom_piper.onnx", target_file=target_env)

    tg_adapter = TelegramChannelAdapter()
    assert tg_adapter.bot_token == "tg_fallback_token_888"

    dc_adapter = DiscordChannelAdapter()
    assert dc_adapter.bot_token == "dc_fallback_token_999"

    piper_adapter = PiperVoiceChannelAdapter()
    assert piper_adapter.piper_model_path == "custom_piper.onnx"
