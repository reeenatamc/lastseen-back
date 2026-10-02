import re
from datetime import datetime

from app.parsers.base import (
    MAX_MESSAGE_CHARS,
    BaseParser,
    ParsedChat,
    ParsedMessage,
    check_message_count,
    join_message,
)

# iMessage exports via third-party tools (e.g. iExplorer, Decipher)
# Format: [YYYY-MM-DD HH:MM:SS] Sender: content
_LINE_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s([^:]+):\s(.+)$")


class IMessageParser(BaseParser):
    platform = "imessage"

    def can_parse(self, raw: str) -> bool:
        if not raw:
            return False
        first_line = raw.splitlines()[0].strip()
        return bool(_LINE_RE.match(first_line))

    def parse(self, raw: str) -> ParsedChat:
        messages: list[ParsedMessage] = []
        participants: set[str] = set()

        # Continuation lines of the last message, joined once when it is complete
        extra: list[str] = []
        extra_chars = 0

        def flush() -> None:
            nonlocal extra, extra_chars
            if messages:
                messages[-1].content = join_message(messages[-1].content, extra)
            extra, extra_chars = [], 0

        for line in raw.splitlines():
            match = _LINE_RE.match(line.strip())
            if not match:
                if messages and extra_chars < MAX_MESSAGE_CHARS:
                    extra.append(line)
                    extra_chars += len(line) + 1
                continue
            flush()

            ts_str, sender, content = match.groups()
            timestamp = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            participants.add(sender.strip())

            messages.append(
                ParsedMessage(
                    timestamp=timestamp,
                    sender=sender.strip(),
                    content=content.strip(),
                )
            )
            check_message_count(len(messages))

        flush()
        return ParsedChat(
            platform=self.platform,
            participants=sorted(participants),
            messages=messages,
        )
