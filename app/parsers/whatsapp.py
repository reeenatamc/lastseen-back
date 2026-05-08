import re
from datetime import datetime

from app.parsers.base import BaseParser, ParsedChat, ParsedMessage

# Matches both 12h and 24h WhatsApp export formats
_LINE_RE = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\s-\s([^:]+):\s(.+)$"
)
_MEDIA_TOKENS = {"<Media omitted>", "<archivo adjunto omitido>", "image omitted"}


class WhatsAppParser(BaseParser):
    platform = "whatsapp"

    def can_parse(self, raw: str) -> bool:
        return bool(_LINE_RE.match(raw.splitlines()[0].strip())) if raw else False

    def parse(self, raw: str) -> ParsedChat:
        messages: list[ParsedMessage] = []
        participants: set[str] = set()

        for line in raw.splitlines():
            match = _LINE_RE.match(line.strip())
            if not match:
                # continuation of previous message
                if messages:
                    messages[-1].content += f"\n{line}"
                continue

            date_str, time_str, sender, content = match.groups()
            timestamp = self._parse_timestamp(date_str, time_str)
            is_media = content.strip() in _MEDIA_TOKENS
            participants.add(sender.strip())

            messages.append(
                ParsedMessage(
                    timestamp=timestamp,
                    sender=sender.strip(),
                    content=content.strip(),
                    is_media=is_media,
                )
            )

        return ParsedChat(
            platform=self.platform,
            participants=sorted(participants),
            messages=messages,
        )

    def _parse_timestamp(self, date_str: str, time_str: str) -> datetime:
        time_str = time_str.strip()
        for fmt in (
            "%d/%m/%Y, %H:%M:%S",
            "%d/%m/%Y, %H:%M",
            "%m/%d/%Y, %H:%M:%S",
            "%m/%d/%Y, %H:%M",
            "%m/%d/%y, %I:%M:%S %p",
            "%m/%d/%y, %I:%M %p",
        ):
            try:
                return datetime.strptime(f"{date_str}, {time_str}", fmt)
            except ValueError:
                continue
        raise ValueError(f"Unrecognised WhatsApp timestamp: {date_str}, {time_str}")
