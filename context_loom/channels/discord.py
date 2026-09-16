"""Discord Channel Adapter for MightyMem using standard Python library HTTP primitives."""
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


class DiscordChannelAdapter(BaseChannelAdapter):
    """Discord Bot API channel adapter for multi-user chat routing."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        channel_id_target: Optional[str] = None,
        poll_interval: float = 2.0,
    ) -> None:
        super().__init__(channel_id="discord")
        resolved_token = bot_token or get_credential("DISCORD_BOT_TOKEN", "")
        if not resolved_token:
            raise ValueError(
                "Discord Bot Token missing. Provide bot_token or set DISCORD_BOT_TOKEN in ~/.mightymem/.env"
            )
        self.bot_token = resolved_token
        self.channel_id_target = channel_id_target or get_credential("DISCORD_CHANNEL_ID", "")
        self.poll_interval = poll_interval
        self.api_url = "https://discord.com/api/v10"
        self._running = False
        self._last_message_id: Optional[str] = None
        self._poll_task: Optional[asyncio.Task[None]] = None

    def _call_api(self, endpoint: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None) -> Any:
        """Execute HTTP request against Discord Bot API v10."""
        url = f"{self.api_url}{endpoint}"
        headers = {
            "Authorization": f"Bot {self.bot_token}",
            "Content-Type": "application/json",
            "User-Agent": "MightyMem/1.0",
        }
        data = json.dumps(payload).encode("utf-8") if payload else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                res_body = resp.read().decode("utf-8")
                return json.loads(res_body) if res_body else {"ok": True}
        except Exception as err:
            logger.error("Discord API call failed [%s %s]: %s", method, endpoint, err)
            return {"ok": False, "error": str(err)}

    async def send_reply(self, external_user_id: str, text: str) -> None:
        """Send an outbound text message to a Discord channel or user DM ID."""
        endpoint = f"/channels/{external_user_id}/messages"
        payload = {"content": text}
        await asyncio.to_thread(self._call_api, endpoint, "POST", payload)

    async def simulate_incoming(self, external_user_id: str, display_name: str, text: str) -> str:
        """Simulate an incoming Discord message for testing or direct dispatching."""
        msg = ChannelMessage(
            channel_id=self.channel_id,
            external_user_id=external_user_id,
            display_name=display_name,
            text=text,
        )
        if self._message_handler:
            reply = await self._message_handler(msg)
            if reply:
                await self.send_reply(external_user_id, reply)
            return reply
        return ""

    async def _poll_loop(self) -> None:
        """Asynchronous polling loop fetching Discord channel messages via REST API."""
        while self._running:
            if self.channel_id_target:
                try:
                    endpoint = f"/channels/{self.channel_id_target}/messages?limit=10"
                    if self._last_message_id:
                        endpoint += f"&after={self._last_message_id}"
                        
                    res = await asyncio.to_thread(self._call_api, endpoint, "GET")
                    if isinstance(res, list):
                        # Sort messages by ID ascending (chronological order)
                        for msg_data in sorted(res, key=lambda m: int(m.get("id", 0))):
                            msg_id = str(msg_data.get("id", ""))
                            if not self._last_message_id or int(msg_id) > int(self._last_message_id):
                                self._last_message_id = msg_id

                            author = msg_data.get("author", {})
                            if author.get("bot"):
                                continue

                            content = msg_data.get("content", "").strip()
                            if not content:
                                continue

                            display_name = author.get("username", "DiscordUser")
                            user_id = str(author.get("id", self.channel_id_target))

                            channel_msg = ChannelMessage(
                                channel_id=self.channel_id,
                                external_user_id=user_id,
                                text=content,
                                display_name=display_name,
                                metadata={"discord_message_id": msg_id, "channel_id": self.channel_id_target},
                            )

                            if self._message_handler:
                                reply_text = await self._message_handler(channel_msg)
                                if reply_text:
                                    await self.send_reply(self.channel_id_target, reply_text)
                except Exception as err:
                    logger.error("Error in Discord polling loop: %s", err)

            await asyncio.sleep(self.poll_interval)

    async def start(self) -> None:
        """Start the Discord adapter."""
        if self._running:
            return
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info(
            "Discord Channel Adapter started (target channel: %s)",
            self.channel_id_target or "Unconfigured",
        )


    async def stop(self) -> None:
        """Stop the Discord adapter."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        logger.info("Discord Channel Adapter stopped.")
