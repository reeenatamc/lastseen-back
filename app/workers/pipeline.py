import logging
from datetime import date
from typing import Literal

import app.models  # noqa: F401 — ensure all mappers are registered before any query
from app.analyzers.base import BaseAnalyzer
from app.analyzers.conflict import ConflictAnalyzer
from app.analyzers.narrative import NarrativeAnalyzer
from app.analyzers.sentiment import SentimentAnalyzer
from app.analyzers.temporal import TemporalAnalyzer
from app.parsers.base import BaseParser, ParsedChat
from app.parsers.imessage import IMessageParser
from app.parsers.telegram import TelegramParser
from app.parsers.whatsapp import WhatsAppParser

logger = logging.getLogger(__name__)

_PARSERS: list[BaseParser] = [
    WhatsAppParser(),
    TelegramParser(),
    IMessageParser(),
]

# Analyzers run in order; each receives the accumulated results of all
# prior analyzers via `context` so later ones can build on earlier work.
_ANALYZERS: list[BaseAnalyzer] = [
    TemporalAnalyzer(),
    SentimentAnalyzer(),
    ConflictAnalyzer(),
    NarrativeAnalyzer(),
]

# Analyzers that need no external API. The free preview runs only these:
# sentiment and narrative call an LLM (Gemini) and cost money per run, so they
# are reserved for paid ("full") analyses.
_NO_LLM_ANALYZERS = frozenset({"temporal", "conflict"})

Tier = Literal["preview", "full"]


def _select_parser(content: str, platform: str) -> BaseParser:
    for parser in _PARSERS:
        if parser.platform == platform and parser.can_parse(content):
            return parser
    for parser in _PARSERS:
        if parser.can_parse(content):
            return parser
    raise ValueError(f"No parser found for platform '{platform}'")


def _filter_by_date(
    chat: ParsedChat, date_from: date | None, date_to: date | None
) -> ParsedChat:
    """Keep only messages whose calendar day falls within [date_from, date_to].

    Both limits are inclusive and compared by day, so a message at 23:59 on
    `date_to` stays in. A None limit leaves that side open.
    """
    if date_from is None and date_to is None:
        return chat

    kept = [
        m
        for m in chat.messages
        if (date_from is None or m.timestamp.date() >= date_from)
        and (date_to is None or m.timestamp.date() <= date_to)
    ]
    if not kept:
        raise ValueError("No messages in the selected date range")

    return ParsedChat(
        platform=chat.platform,
        # Recomputed so people who only spoke outside the range don't linger
        participants=sorted({m.sender for m in kept}),
        messages=kept,
        metadata=chat.metadata,
    )


def run_pipeline(
    *,
    analysis_id: int | None,
    content: str,
    platform: str,
    language: str = "auto",
    date_from: str | None = None,
    date_to: str | None = None,
    tier: Tier = "full",
) -> dict:
    if analysis_id is not None:
        _mark_processing(analysis_id)

    try:
        parser = _select_parser(content, platform)
        parsed_chat = parser.parse(content)

        # Filter before analyzers so every metric reflects only the range
        start = date.fromisoformat(date_from) if date_from else None
        end = date.fromisoformat(date_to) if date_to else None
        parsed_chat = _filter_by_date(parsed_chat, start, end)

        # `_meta` carries pipeline-level flags (e.g. language) into analyzer
        # context without polluting saved results — popped before persistence.
        results: dict = {"_meta": {"language": language}}
        for analyzer in _ANALYZERS:
            if tier == "preview" and analyzer.name not in _NO_LLM_ANALYZERS:
                continue
            result = analyzer.analyze(parsed_chat, context=results)
            results[result.analyzer] = result.data
        results.pop("_meta", None)

        if analysis_id is not None:
            _save_result(analysis_id, results, refund=_is_incomplete(results, tier))

        output = {
            "platform": parsed_chat.platform,
            "total_messages": parsed_chat.total_messages,
            "participants": parsed_chat.participants,
            "analysis": results,
            "tier": tier,
        }
        if start is not None or end is not None:
            output["date_filter"] = {"from": date_from, "to": date_to}
        return output
    except Exception as exc:
        code = _error_code(exc)
        # Type and code only: exception text can quote chat content
        logger.error("Pipeline failed: code=%s exc=%s", code, type(exc).__name__)
        if analysis_id is not None:
            _mark_failed(analysis_id, code)
        raise


def _error_code(exc: BaseException) -> str:
    """Map an exception to a fixed public error code (never the exception text).

    The text can contain parts of the chat or internal details, and it is
    stored on the analysis and shown to users.
    """
    message = str(exc)
    if isinstance(exc, ValueError):
        if message.startswith("No parser found"):
            return "parse_failed"
        if message == "No messages in the selected date range":
            return "empty_date_range"
        if message == "chat_too_large":
            return "chat_too_large"
    return "internal_error"


def _is_incomplete(results: dict, tier: str) -> bool:
    """True when a full-tier run lost one of the analyzers the credit pays for.

    Analyzers report failures as {"error": ...} instead of raising, so the
    pipeline still completes; the report is delivered as is, but it is not
    what was sold.
    """
    if tier != "full":
        return False
    return any(
        "error" in (results.get(analyzer.name) or {})
        for analyzer in _ANALYZERS
        if analyzer.name not in _NO_LLM_ANALYZERS
    )


def _save_result(analysis_id: int, results: dict, *, refund: bool = False) -> None:
    from app.core.database import SyncSession
    from app.models.analysis import Analysis, AnalysisStatus
    from app.services.credits import refund_analysis_credit

    with SyncSession() as db:
        analysis = db.get(Analysis, analysis_id)
        if analysis is None:
            raise ValueError(f"Analysis {analysis_id} not found in DB")
        analysis.status = AnalysisStatus.completed
        analysis.result = results
        if refund:
            # Incomplete paid report: return the credit AND re-lock it, otherwise the
            # paid sections that did compute would be delivered for free
            refund_analysis_credit(db, analysis_id)
            analysis.unlocked = False
        db.commit()


def _mark_processing(analysis_id: int) -> None:
    from app.core.database import SyncSession
    from app.models.analysis import Analysis, AnalysisStatus

    with SyncSession() as db:
        analysis = db.get(Analysis, analysis_id)
        if analysis:
            analysis.status = AnalysisStatus.processing
            db.commit()


def _mark_failed(analysis_id: int, error_code: str) -> None:
    from app.core.database import SyncSession
    from app.models.analysis import Analysis, AnalysisStatus
    from app.services.credits import refund_analysis_credit

    with SyncSession() as db:
        analysis = db.get(Analysis, analysis_id)
        if analysis:
            analysis.status = AnalysisStatus.failed
            analysis.error = error_code
            # Give back any credit spent on this upload, in the same transaction
            refund_analysis_credit(db, analysis_id)
            db.commit()
