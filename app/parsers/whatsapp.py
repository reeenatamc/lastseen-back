import re
from datetime import datetime

from app.parsers.base import BaseParser, ParsedChat, ParsedMessage

# Android format: "dd/mm/yy, hh:mm - Sender: message"
_ANDROID_RE = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][\.\s]*[Mm]\.?)?)\s-\s([^:]+):\s(.+)$"
)

# iOS format: "[dd/mm/yy, hh:mm:ss] Sender: message"
_IOS_RE = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][\.\s]*[Mm]\.?)?)\]\s([^:]+):\s(.+)$"
)

_MEDIA_TOKENS = {
    "<Media omitted>",
    "<archivo adjunto omitido>",
    "image omitted",
    "video omitted",
    "audio omitted",
    "sticker omitted",
    "GIF omitted",
    "document omitted",
}

# Lines that are WhatsApp system messages, not chat content
_SYSTEM_PATTERNS = re.compile(
    r"(cifrados de extremo|end-to-end encrypted|Messages and calls|created group|added you|left|changed the subject|changed this group)",
    re.IGNORECASE,
)


def _normalize_time(t: str) -> str:
    """Normalize Spanish/Portuguese AM/PM to standard: 'a. m.' → 'AM'."""
    t = t.strip()
    t = re.sub(r"a[\.\s]*\s*m\.?", "AM", t, flags=re.IGNORECASE)
    t = re.sub(r"p[\.\s]*\s*m\.?", "PM", t, flags=re.IGNORECASE)
    return t.strip()


class WhatsAppParser(BaseParser):
    platform = "whatsapp"

    def can_parse(self, raw: str) -> bool:
        if not raw:
            return False
        for line in raw.splitlines()[:15]:
            line = line.strip()
            if not line or _SYSTEM_PATTERNS.search(line):
                continue
            if _ANDROID_RE.match(line) or _IOS_RE.match(line):
                return True
        return False

    def parse(self, raw: str) -> ParsedChat:
        messages: list[ParsedMessage] = []
        participants: set[str] = set()

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue

            match = _ANDROID_RE.match(line) or _IOS_RE.match(line)
            if not match:
                if messages and not _SYSTEM_PATTERNS.search(line):
                    messages[-1].content += f"\n{line}"
                continue

            date_str, time_str, sender, content = match.groups()
            sender = sender.strip()

            # Skip system messages masquerading as sender lines
            if _SYSTEM_PATTERNS.search(content) and len(content) > 80:
                continue

            try:
                timestamp = self._parse_timestamp(date_str, time_str)
            except ValueError:
                continue

            is_media = content.strip() in _MEDIA_TOKENS
            participants.add(sender)

            messages.append(
                ParsedMessage(
                    timestamp=timestamp,
                    sender=sender,
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
        time_str = _normalize_time(time_str)
        for fmt in (
            "%d/%m/%Y, %H:%M:%S",
            "%d/%m/%Y, %H:%M",
            "%d/%m/%y, %H:%M:%S",
            "%d/%m/%y, %H:%M",
            "%m/%d/%Y, %H:%M:%S",
            "%m/%d/%Y, %H:%M",
            "%m/%d/%y, %I:%M:%S %p",
            "%m/%d/%y, %I:%M %p",
            "%d/%m/%y, %I:%M:%S %p",
            "%d/%m/%y, %I:%M %p",
            "%d/%m/%Y, %I:%M:%S %p",
            "%d/%m/%Y, %I:%M %p",
        ):
            try:
                return datetime.strptime(f"{date_str}, {time_str}", fmt)
            except ValueError:
                continue
        raise ValueError(f"Unrecognised WhatsApp timestamp: {date_str}, {time_str}")
