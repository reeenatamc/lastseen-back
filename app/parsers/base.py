from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

# A single message longer than this is a pasted document or an attack, not conversation:
# tone and timing do not need more, and unbounded text inflates memory and LLM cost
MAX_MESSAGE_CHARS = 4000
# Hard cap on messages per chat so one upload cannot exhaust worker memory
MAX_MESSAGES = 200_000


def join_message(first: str, continuation: list[str]) -> str:
    """Join a message's first line with its continuation lines, truncated to the cap.

    Parsers collect continuation lines in a list and join once; repeated `+=`
    on a growing string is quadratic.
    """
    if not continuation:
        return first[:MAX_MESSAGE_CHARS]
    return "\n".join([first, *continuation])[:MAX_MESSAGE_CHARS]


def check_message_count(count: int) -> None:
    if count > MAX_MESSAGES:
        raise ValueError("chat_too_large")


@dataclass
class ParsedMessage:
    timestamp: datetime
    sender: str
    content: str
    is_media: bool = False
    media_type: str | None = None


@dataclass
class ParsedChat:
    platform: str
    participants: list[str]
    messages: list[ParsedMessage]
    metadata: dict = field(default_factory=dict)

    @property
    def total_messages(self) -> int:
        return len(self.messages)


class BaseParser(ABC):
    platform: str = ""

    @abstractmethod
    def can_parse(self, raw: str) -> bool:
        """Return True if this parser recognises the raw input format."""

    @abstractmethod
    def parse(self, raw: str) -> ParsedChat:
        """Parse raw chat export text into a normalised ParsedChat."""
