import re
from datetime import datetime

from app.parsers.base import BaseParser, ParsedChat, ParsedMessage

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

        for line in raw.splitlines():
            match = _LINE_RE.match(line.strip())
            if not match:
                if messages:
                    messages[-1].content += f"\n{line}"
                continue

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

        return ParsedChat(
            platform=self.platform,
            participants=sorted(participants),
            messages=messages,
        )
