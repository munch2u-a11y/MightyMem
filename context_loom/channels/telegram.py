"""Telegram Channel Adapter for MightyMem using standard Python library HTTP primitives."""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from .base import BaseChannelAdapter, ChannelMessage
from ..credentials import get_credential

logger = logging.getLogger(__name__)


def _chunk_text(text: str, max_chunk_size: int = 4000) -> List[str]:
    """Split text into chunks of max_chunk_size without breaking lines where possible."""
    if len(text) <= max_chunk_size:
        return [text]

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for line in text.splitlines(keepends=True):
        if current_length + len(line) > max_chunk_size and current_chunk:
            chunks.append("".join(current_chunk))
            current_chunk = []
            current_length = 0

        # Handle exceptionally long single lines
        while len(line) > max_chunk_size:
            head = line[:max_chunk_size]
            line = line[max_chunk_size:]
            chunks.append(head)

        if line:
            current_chunk.append(line)
            current_length += len(line)

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks


class TelegramChannelAdapter(BaseChannelAdapter):
    """Telegram Bot API channel adapter for multi-user chat routing."""

    def __init__(self, bot_token: Optional[str] = None, poll_interval: float = 1.0) -> None:
        super().__init__(channel_id="telegram")
        resolved_token = bot_token or get_credential("TELEGRAM_BOT_TOKEN", "")
        if not resolved_token:
            raise ValueError(
                "Telegram Bot Token missing. Provide bot_token or set TELEGRAM_BOT_TOKEN in ~/.mightymem/.env"
            )
        self.bot_token = resolved_token
        self.poll_interval = poll_interval
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
        self._running = False
        self._last_update_id = 0
        self._poll_task: Optional[asyncio.Task[None]] = None

    def _call_api(self, method: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a HTTP POST request against Telegram Bot API."""
        url = f"{self.api_url}/{method}"
        data = json.dumps(payload or {}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                res_body = resp.read().decode("utf-8")
                return json.loads(res_body)
        except Exception as err:
            logger.error("Telegram API call failed [%s]: %s", method, err)
            return {"ok": False, "error": str(err)}

    async def send_reply(self, external_user_id: str, text: str) -> None:
        """Send an outbound text message to a Telegram chat ID with chunking and Markdown fallback."""
        chunks = _chunk_text(text, max_chunk_size=4000)
        for chunk in chunks:
            # 1. Attempt sending with Markdown formatting
            payload = {
                "chat_id": external_user_id,
                "text": chunk,
                "parse_mode": "Markdown",
            }
            res = await asyncio.to_thread(self._call_api, "sendMessage", payload)
            
            # 2. Fallback to plain text if Telegram rejects unbalanced Markdown
            if not res.get("ok"):
                logger.info("Markdown parse failed for Telegram chunk; retrying as plain text...")
                payload.pop("parse_mode", None)
                await asyncio.to_thread(self._call_api, "sendMessage", payload)

    async def _poll_loop(self) -> None:
        """Asynchronous polling loop fetching Telegram updates."""
        while self._running:
            try:
                payload = {
                    "offset": self._last_update_id + 1,
                    "timeout": 10,
                }
                res = await asyncio.to_thread(self._call_api, "getUpdates", payload)
                if res.get("ok") and isinstance(res.get("result"), list):
                    for update in res["result"]:
                        update_id = update.get("update_id", 0)
                        self._last_update_id = max(self._last_update_id, update_id)

                        msg = update.get("message")
                        if not msg or "text" not in msg:
                            continue

                        chat = msg.get("chat", {})
                        chat_id = str(chat.get("id"))
                        from_user = msg.get("from", {})
                        display_name = from_user.get("first_name") or from_user.get("username") or "User"
                        text = msg.get("text", "")

                        channel_msg = ChannelMessage(
                            channel_id=self.channel_id,
                            external_user_id=chat_id,
                            text=text,
                            display_name=display_name,
                            timestamp=msg.get("date", 0),
                            metadata={"update_id": update_id, "chat_type": chat.get("type")},
                        )

                        if self._message_handler:
                            reply_text = await self._message_handler(channel_msg)
                            if reply_text:
                                await self.send_reply(chat_id, reply_text)

            except Exception as err:
                logger.error("Error in Telegram polling loop: %s", err)

            await asyncio.sleep(self.poll_interval)

    async def start(self) -> None:
        """Start the background Telegram update polling loop."""
        if self._running:
            return
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("Telegram Channel Adapter started.")

    async def stop(self) -> None:
        """Stop the Telegram update polling loop."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        logger.info("Telegram Channel Adapter stopped.")
