"""Local Spoken Voice Channel Adapter using Piper for TTS Synthesis & Local STT Transcription."""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseChannelAdapter, ChannelMessage
from ..credentials import get_credential

logger = logging.getLogger(__name__)


def transcribe_audio_file(audio_path: Path | str) -> str:
    """Transcribe an audio file to text using local STT engines (Whisper / Vosk / CLI) if installed."""
    path = Path(audio_path)
    if not path.exists():
        logger.warning("STT Audio file does not exist: %s", path)
        return ""

    # 1. Attempt OpenAI whisper CLI if available
    whisper_cmd = shutil.which("whisper")
    if whisper_cmd:
        try:
            res = subprocess.run(
                [whisper_cmd, str(path), "--model", "base", "--output_format", "txt"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            txt_file = path.with_suffix(".txt")
            if txt_file.exists():
                text = txt_file.read_text(encoding="utf-8").strip()
                txt_file.unlink(missing_ok=True)
                if text:
                    return text
        except Exception as err:
            logger.debug("Local whisper CLI transcription failed: %s", err)

    # 2. Attempt whisper-cpp CLI if available
    whisper_cpp = shutil.which("whisper-cli") or shutil.which("main")
    if whisper_cpp:
        try:
            res = subprocess.run(
                [whisper_cpp, "-f", str(path), "--no-timestamps"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            text = res.stdout.strip()
            if text:
                return text
        except Exception as err:
            logger.debug("Whisper.cpp CLI transcription failed: %s", err)

    logger.info("No local STT binary (whisper/vosk) found; audio file path ingested as reference.")
    return f"[Audio input: {path.name}]"


class PiperVoiceChannelAdapter(BaseChannelAdapter):
    """Local Spoken Voice Channel Adapter (Piper TTS for speech synthesis + Local STT for intake)."""

    def __init__(
        self,
        piper_model_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        super().__init__(channel_id="piper-voice")
        self.piper_model_path = piper_model_path or get_credential("PIPER_MODEL_PATH", "piper_en_US.onnx")
        self.output_dir = Path(output_dir or tempfile.gettempdir()) / "mightymem_voice"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._running = False
        self._poll_task: Optional[asyncio.Task[None]] = None

    def synthesize_speech(self, text: str, output_wav: Path) -> bool:
        """Synthesize text into a WAV audio file using local Piper CLI (TTS)."""
        try:
            cmd = ["piper", "--model", self.piper_model_path, "--output_file", str(output_wav)]
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            _, err = proc.communicate(input=text)
            if proc.returncode == 0 and output_wav.exists():
                return True
            logger.warning("Piper CLI synthesis returned exit code %d: %s", proc.returncode, err)
        except Exception as e:
            logger.info("Local Piper CLI synthesis not available (%s); falling back to text voice payload.", e)
        return False

    async def send_reply(self, external_user_id: str, text: str) -> None:
        """Send voice speech reply (synthesizing WAV audio if Piper TTS installed, otherwise text voice message)."""
        wav_path = self.output_dir / f"reply_{int(asyncio.get_event_loop().time())}.wav"
        success = await asyncio.to_thread(self.synthesize_speech, text, wav_path)
        if success:
            logger.info("Generated Piper TTS voice reply at %s for user %s", wav_path, external_user_id)
        else:
            logger.info("Voice channel reply to [%s]: %s", external_user_id, text)

    async def process_audio_file_input(self, audio_file: Path | str, external_user_id: str, display_name: str) -> str:
        """Transcribe incoming WAV/audio file via local STT and route through channel message router."""
        transcript_text = await asyncio.to_thread(transcribe_audio_file, audio_file)
        return await self.simulate_voice_input(external_user_id, display_name, transcript_text)

    async def simulate_voice_input(self, external_user_id: str, display_name: str, transcript_text: str) -> str:
        """Process an incoming voice transcript into the channel message router and return spoken TTS reply."""
        msg = ChannelMessage(
            channel_id=self.channel_id,
            external_user_id=external_user_id,
            display_name=display_name,
            text=transcript_text,
            metadata={"input_mode": "voice_stt"},
        )
        if self._message_handler:
            reply = await self._message_handler(msg)
            if reply:
                await self.send_reply(external_user_id, reply)
            return reply
        return ""

    async def start(self) -> None:
        """Start the Piper Voice channel adapter."""
        self._running = True
        logger.info("Piper Voice Channel Adapter online (output_dir: %s)", self.output_dir)

    async def stop(self) -> None:
        """Stop the Piper Voice channel adapter."""
        self._running = False
        logger.info("Piper Voice Channel Adapter stopped.")
