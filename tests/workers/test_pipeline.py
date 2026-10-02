from datetime import date, datetime

import pytest

from app.parsers.base import ParsedChat, ParsedMessage
from app.workers import pipeline
from app.workers.pipeline import _filter_by_date

# ── Helpers ───────────────────────────────────────────────────────────────────


def _msg(sender: str, dt: datetime, content: str = "hi") -> ParsedMessage:
    return ParsedMessage(timestamp=dt, sender=sender, content=content)


def _make_chat() -> ParsedChat:
    msgs = [
        _msg("Alice", datetime(2024, 1, 1, 0, 0)),
        _msg("Bob", datetime(2024, 1, 2, 12, 0)),
        _msg("Alice", datetime(2024, 1, 3, 23, 59)),
        _msg("Carol", datetime(2024, 1, 4, 0, 0)),
    ]
    return ParsedChat(
        platform="telegram",
        participants=["Alice", "Bob", "Carol"],
        messages=msgs,
        metadata={"chat_name": "test"},
    )


# ── _filter_by_date ───────────────────────────────────────────────────────────


def test_no_limits_returns_same_object():
    chat = _make_chat()
    assert _filter_by_date(chat, None, None) is chat


def test_only_date_from():
    result = _filter_by_date(_make_chat(), date(2024, 1, 3), None)
    assert [m.timestamp.day for m in result.messages] == [3, 4]


def test_only_date_to():
    result = _filter_by_date(_make_chat(), None, date(2024, 1, 2))
    assert [m.timestamp.day for m in result.messages] == [1, 2]


def test_both_limits_include_full_boundary_days():
    result = _filter_by_date(_make_chat(), date(2024, 1, 1), date(2024, 1, 3))
    days = [m.timestamp for m in result.messages]
    # 00:00 on date_from and 23:59 on date_to are in; 00:00 next day is out
    assert datetime(2024, 1, 1, 0, 0) in days
    assert datetime(2024, 1, 3, 23, 59) in days
    assert datetime(2024, 1, 4, 0, 0) not in days
    assert len(result.messages) == 3


def test_participants_recalculated_and_sorted():
    result = _filter_by_date(_make_chat(), date(2024, 1, 2), date(2024, 1, 3))
    assert result.participants == ["Alice", "Bob"]


def test_metadata_and_platform_preserved():
    result = _filter_by_date(_make_chat(), date(2024, 1, 1), date(2024, 1, 2))
    assert result.platform == "telegram"
    assert result.metadata == {"chat_name": "test"}


def test_empty_range_raises():
    with pytest.raises(ValueError, match="No messages in the selected date range"):
        _filter_by_date(_make_chat(), date(2025, 1, 1), date(2025, 1, 31))


# ── tiers ─────────────────────────────────────────────────────────────────────

_WHATSAPP = "\n".join(
    f"[1/{d}/26, 10:{m:02d}:00] {name}: mensaje {d}-{m}"
    for d in range(1, 6)
    for m, name in ((0, "Ana"), (5, "Beto"))
)


def _fail(*args, **kwargs):
    raise AssertionError("LLM analyzer must not run in preview tier")


def test_preview_tier_skips_llm_analyzers(monkeypatch):
    for analyzer in pipeline._ANALYZERS:
        if analyzer.name in ("sentiment", "narrative"):
            monkeypatch.setattr(analyzer, "analyze", _fail)

    out = pipeline.run_pipeline(
        analysis_id=None, content=_WHATSAPP, platform="whatsapp", tier="preview"
    )
    assert out["tier"] == "preview"
    assert set(out["analysis"]) == {"temporal", "conflict"}


def test_full_tier_runs_every_analyzer(monkeypatch):
    calls = []

    class _Fake:
        def __init__(self, name):
            self.name = name

        def analyze(self, chat, context=None):
            calls.append(self.name)
            from app.analyzers.base import AnalysisResult
            return AnalysisResult(analyzer=self.name, data={})

    monkeypatch.setattr(
        pipeline, "_ANALYZERS", [_Fake(n) for n in ("temporal", "sentiment", "conflict", "narrative")]
    )
    out = pipeline.run_pipeline(analysis_id=None, content=_WHATSAPP, platform="whatsapp")
    assert out["tier"] == "full"
    assert calls == ["temporal", "sentiment", "conflict", "narrative"]


# ── Incomplete paid reports ───────────────────────────────────────────────────

_FULL_OK = {
    "temporal": {"overview": {}},
    "sentiment": {"per_person": {}},
    "conflict": {"episodes": []},
    "narrative": {"resumen": "ok"},
}


def test_full_report_with_every_analyzer_is_complete():
    assert pipeline._is_incomplete(_FULL_OK, "full") is False


@pytest.mark.parametrize("failed", ["sentiment", "narrative"])
def test_full_report_missing_a_paid_analyzer_is_incomplete(failed):
    results = {**_FULL_OK, failed: {"error": "unavailable"}}
    assert pipeline._is_incomplete(results, "full") is True


def test_error_in_a_free_analyzer_does_not_count():
    results = {**_FULL_OK, "conflict": {"error": "insufficient_data"}}
    assert pipeline._is_incomplete(results, "full") is False


def test_preview_is_never_incomplete():
    assert pipeline._is_incomplete({"temporal": {}, "conflict": {}}, "preview") is False


# ── error codes ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "exc, code",
    [
        (ValueError("No parser found for platform 'whatsapp'"), "parse_failed"),
        (ValueError("No messages in the selected date range"), "empty_date_range"),
        (ValueError("chat_too_large"), "chat_too_large"),
        (ValueError("Analysis 3 not found in DB"), "internal_error"),
        (KeyError("secret chat text"), "internal_error"),
        (RuntimeError("postgresql://user:pw@host/db failed"), "internal_error"),
    ],
)
def test_error_code_is_fixed_and_hides_text(exc, code):
    assert pipeline._error_code(exc) == code
