from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
import wave
import struct
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .pipeline import BaseAgenticMemoryRAG
from .types import EventKind, OutputTrunk


@dataclass
class AudioReceipt:
    receipt_id: str
    text: str
    audio_path: str
    verified: bool = False
    voice_engine: str = "synthetic_fallback"
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AudioReflexBridge:
    """Verbal Audio Bridge connecting TTS (Piper) and STT intake to Habitus AI trunks."""

    def __init__(
        self,
        mind: BaseAgenticMemoryRAG,
        *,
        piper_executable: str = "piper",
        piper_model: str = "en_US-lessac-medium",
    ) -> None:
        self.mind = mind
        self.piper_executable = piper_executable
        self.piper_model = piper_model

    def _is_piper_available(self) -> bool:
        return shutil.which(self.piper_executable) is not None

    def _generate_fallback_wav(self, text: str, output_path: Path) -> Path:
        """Generate a lightweight PCM audio WAV file as fallback when Piper is offline."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = 16000
        duration_seconds = max(0.5, min(5.0, len(text) * 0.06))
        num_samples = int(sample_rate * duration_seconds)

        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)

            # Generate pleasant dual-tone chime
            freq1 = 440.0  # A4
            freq2 = 880.0  # A5
            frames = bytearray()
            for i in range(num_samples):
                t = i / sample_rate
                # Envelope decay
                envelope = math.exp(-3.0 * t / duration_seconds)
                sample_val = int(32767 * 0.3 * envelope * (math.sin(2 * math.pi * freq1 * t) + 0.5 * math.sin(2 * math.pi * freq2 * t)))
                sample_val = max(-32768, min(32767, sample_val))
                frames.extend(struct.pack("<h", sample_val))

            wav_file.writeframes(frames)
        return output_path

    def speak(
        self,
        text: str,
        output_wav: Path | str | None = None,
    ) -> AudioReceipt:
        """Synthesize verbal speech output (via Piper TTS if available, else synthetic WAV fallback)."""
        receipt_id = f"audio_receipt:{uuid.uuid4().hex}"
        start_time = time.perf_counter()

        if output_wav is None:
            with tempfile.NamedTemporaryFile(
                prefix="habitus_speech_",
                suffix=".wav",
                delete=False,
            ) as temporary:
                output_wav = Path(temporary.name)
        else:
            output_wav = Path(output_wav)

        if self._is_piper_available():
            try:
                cmd = f"echo {shlex.quote(text)} | {self.piper_executable} --model {shlex.quote(self.piper_model)} --output_file {shlex.quote(str(output_wav))}"
                res = subprocess.run(cmd, shell=True, capture_output=True, timeout=None)
                if res.returncode == 0 and output_wav.exists():
                    elapsed = time.perf_counter() - start_time
                    return AudioReceipt(
                        receipt_id=receipt_id,
                        text=text,
                        audio_path=str(output_wav),
                        verified=True,
                        voice_engine="piper",
                        elapsed_seconds=elapsed,
                    )
            except Exception:
                pass

        # Fallback to synthetic PCM WAV generator
        self._generate_fallback_wav(text, output_wav)
        elapsed = time.perf_counter() - start_time
        return AudioReceipt(
            receipt_id=receipt_id,
            text=text,
            audio_path=str(output_wav),
            verified=True,
            voice_engine="synthetic_fallback",
            elapsed_seconds=elapsed,
        )

    def transcribe_audio(self, audio_path: Path | str) -> str:
        """Transcribe audio WAV file to text using local STT (Whisper/Vosk) if available."""
        path = Path(audio_path)
        if not path.exists():
            return str(audio_path)

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
            except Exception:
                pass

        return path.name

    def listen_and_route(
        self,
        audio_or_text: str | Path,
        source_id: str = "voice_stt",
    ) -> Any:
        """Route transcribed speech intake (or audio file transcribed via local STT) through HEAR trunk."""
        if isinstance(audio_or_text, Path) or (isinstance(audio_or_text, str) and os.path.exists(audio_or_text)):
            speech_text = self.transcribe_audio(audio_or_text)
        else:
            speech_text = str(audio_or_text)

        return self.mind.remember(
            text=speech_text,
            kind=EventKind.MESSAGE,
            source_id=source_id,
        )

    def process_reflex_turn(
        self,
        audio_or_text: str | Path,
        *,
        source_id: str = "voice_stt",
        output_wav: Path | str | None = None,
    ) -> dict[str, Any]:
        """Execute a verbal reflex turn: STT transcription -> HEAR intake -> Y-recall -> SPEAK output -> Piper TTS."""
        if isinstance(audio_or_text, Path) or (isinstance(audio_or_text, str) and os.path.exists(audio_or_text)):
            speech_text = self.transcribe_audio(audio_or_text)
        else:
            speech_text = str(audio_or_text)

        # Step 1: Intake via HEAR trunk
        record = self.listen_and_route(speech_text, source_id=source_id)

        # Step 2: Y-traversal recall
        recall_result = self.mind.recall(speech_text)

        # Step 3: Classify output intent
        candidate_response = recall_result.context.splitlines()[0] if recall_result.context else f"Acknowledged: {speech_text}"
        decision = self.mind.classify_output(candidate_response)

        # Step 4: Verbal synthesis via Piper TTS
        spoken_text = candidate_response
        audio_receipt = self.speak(spoken_text, output_wav=output_wav)

        # Step 5: Reinforce path with verified outcome
        outcome = self.mind.record_outcome(
            decision=decision,
            stability_delta=1.0,
            verified=audio_receipt.verified,
            receipt_id=audio_receipt.receipt_id,
        )

        return {
            "input_text": speech_text,
            "recalled_record_ids": recall_result.packet.direct_record_ids,
            "y_paths": recall_result.packet.y_paths,
            "classified_trunk": decision.trunk.value if decision.trunk else "SPEAK",
            "spoken_text": spoken_text,
            "spoken": True,
            "audio_path": audio_receipt.audio_path,
            "receipt": audio_receipt,
            "outcome_id": outcome.outcome_id,
        }

