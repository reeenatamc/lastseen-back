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
    def analyze(self, chat: ParsedChat, context: dict | None = None) -> AnalysisResult:
        """
        Run analysis on a parsed chat and return structured results.

        `context` carries the accumulated results of analyzers that ran before
        this one in the pipeline. Use it to build on prior work without
        re-processing the chat (e.g. NarrativeAnalyzer reads temporal + sentiment).
        """
