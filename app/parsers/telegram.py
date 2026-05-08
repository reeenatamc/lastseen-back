import json
from datetime import datetime

from app.parsers.base import BaseParser, ParsedChat, ParsedMessage


class TelegramParser(BaseParser):
    """Parses Telegram JSON exports (Settings → Export chat history → JSON)."""

    platform = "telegram"

    def can_parse(self, raw: str) -> bool:
        try:
            data = json.loads(raw)
            return "messages" in data and "type" in data
        except (json.JSONDecodeError, TypeError):
            return False

    def parse(self, raw: str) -> ParsedChat:
        data = json.loads(raw)
        messages: list[ParsedMessage] = []
        participants: set[str] = set()

        for msg in data.get("messages", []):
            if msg.get("type") != "message":
                continue

            sender = msg.get("from") or msg.get("actor") or "Unknown"
            participants.add(sender)
            timestamp = datetime.fromisoformat(msg["date"])
            text = msg.get("text", "")
            if isinstance(text, list):
                text = "".join(
                    t if isinstance(t, str) else t.get("text", "") for t in text
                )

            media_type = msg.get("media_type") or msg.get("file")
            is_media = media_type is not None

            messages.append(
                ParsedMessage(
                    timestamp=timestamp,
                    sender=sender,
                    content=str(text),
                    is_media=is_media,
                    media_type=str(media_type) if media_type else None,
                )
            )

        return ParsedChat(
            platform=self.platform,
            participants=sorted(participants),
            messages=messages,
            metadata={"chat_name": data.get("name", "")},
        )
