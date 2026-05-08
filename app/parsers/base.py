from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


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
