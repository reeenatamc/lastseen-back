from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.parsers.base import ParsedChat


@dataclass
class AnalysisResult:
    analyzer: str
    data: dict[str, Any] = field(default_factory=dict)


class BaseAnalyzer(ABC):
    name: str = ""

    @abstractmethod
    def analyze(self, chat: ParsedChat) -> AnalysisResult:
        """Run analysis on a parsed chat and return structured results."""
