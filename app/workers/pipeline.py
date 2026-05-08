import app.models  # noqa: F401 — ensure all mappers are registered before any query
from app.analyzers.base import BaseAnalyzer
from app.analyzers.sentiment import SentimentAnalyzer
from app.analyzers.temporal import TemporalAnalyzer
from app.parsers.base import BaseParser, ParsedChat
from app.parsers.imessage import IMessageParser
from app.parsers.telegram import TelegramParser
from app.parsers.whatsapp import WhatsAppParser

_PARSERS: list[BaseParser] = [
    WhatsAppParser(),
    TelegramParser(),
    IMessageParser(),
]

_ANALYZERS: list[BaseAnalyzer] = [
    TemporalAnalyzer(),
    SentimentAnalyzer(),
]


def _select_parser(content: str, platform: str) -> BaseParser:
    for parser in _PARSERS:
        if parser.platform == platform and parser.can_parse(content):
            return parser
    for parser in _PARSERS:
        if parser.can_parse(content):
            return parser
    raise ValueError(f"No parser found for platform '{platform}'")


def run_pipeline(*, analysis_id: int | None, content: str, platform: str) -> dict:
    parser = _select_parser(content, platform)
    parsed_chat = parser.parse(content)

    results: dict = {}
    for analyzer in _ANALYZERS:
        result = analyzer.analyze(parsed_chat)
        results[result.analyzer] = result.data

    if analysis_id is not None:
        try:
            _save_result(analysis_id, results)
        except Exception as exc:
            _mark_failed(analysis_id, str(exc))
            raise

    return {
        "platform": parsed_chat.platform,
        "total_messages": parsed_chat.total_messages,
        "participants": parsed_chat.participants,
        "analysis": results,
    }


def _save_result(analysis_id: int, results: dict) -> None:
    from app.core.database import SyncSession
    from app.models.analysis import Analysis, AnalysisStatus

    with SyncSession() as db:
        analysis = db.get(Analysis, analysis_id)
        if analysis is None:
            raise ValueError(f"Analysis {analysis_id} not found in DB")
        analysis.status = AnalysisStatus.completed
        analysis.result = results
        db.commit()


def _mark_failed(analysis_id: int, error: str) -> None:
    from app.core.database import SyncSession
    from app.models.analysis import Analysis, AnalysisStatus

    with SyncSession() as db:
        analysis = db.get(Analysis, analysis_id)
        if analysis:
            analysis.status = AnalysisStatus.failed
            analysis.error = error[:500]
            db.commit()
