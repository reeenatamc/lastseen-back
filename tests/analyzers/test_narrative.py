"""
Narrative analyzer tests.

The Anthropic client is mocked so tests run without a real API key.
The mock returns a deterministic, valid narrative JSON, letting us verify
payload construction, context wiring, and error-handling paths in isolation.
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from app.analyzers.narrative import NarrativeAnalyzer, _build_payload, _fmt_seconds, _short_date
from app.parsers.base import ParsedChat, ParsedMessage

# ── Fixtures ──────────────────────────────────────────────────────────────────

BASE = datetime(2024, 1, 1, 10, 0)

MOCK_NARRATIVE = {
    "resumen": "Alice y Bob compartieron meses de conversación fluida antes de que el silencio se instalara.",
    "dinamica": "Alice sostuvo casi toda la iniciativa, especialmente hacia el final.",
    "punto_de_quiebre": "2024-Q2",
    "estado_actual": "La conversación está casi inactiva.",
    "reflexion": "Hay relaciones que mueren de silencio, no de pelea.",
}

MOCK_CONTEXT = {
    "temporal": {
        "overview": {
            "participants": ["Alice", "Bob"],
            "total_messages": 500,
            "share_per_person": {"Alice": 0.6, "Bob": 0.4},
            "date_range": {
                "start": "2024-01-01T10:00:00",
                "end": "2024-06-30T10:00:00",
                "total_days": 181,
            },
        },
        "response_time": {
            "per_person": {
                "Alice": {"mean_seconds": 300},
                "Bob": {"mean_seconds": 7200},
            }
        },
        "initiative_balance": {
            "share": {"Alice": 0.7, "Bob": 0.3},
            "total_conversations": 40,
        },
        "message_length": {
            "per_person": {
                "Alice": {"mean_chars": 45},
                "Bob": {"mean_chars": 12},
            }
        },
        "response_decay": {
            "decay_score": 0.72,
            "trend": "deteriorating",
            "turning_point": "2024-Q2",
        },
    },
    "sentiment": {
        "per_person": {
            "Alice": {"dominant": "positive", "avg_score": 0.4},
            "Bob": {"dominant": "negative", "avg_score": -0.3},
        },
        "emotional_drift": {"score": 0.5, "direction": "Alice_positive_Bob_negative"},
    },
}


def _make_chat() -> ParsedChat:
    msgs = [
        ParsedMessage(timestamp=BASE, sender="Alice", content="hey"),
        ParsedMessage(timestamp=BASE + timedelta(minutes=5), sender="Bob", content="k"),
    ]
    return ParsedChat(platform="whatsapp", participants=["Alice", "Bob"], messages=msgs)


def _mock_client(narrative: dict = MOCK_NARRATIVE):
    """Return a mock Anthropic client that returns the given narrative."""
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = str(narrative).replace("'", '"')

    # Use actual json.dumps to produce valid JSON
    import json
    mock_block.text = json.dumps(narrative)

    mock_response = MagicMock()
    mock_response.content = [mock_block]

    mock_messages = MagicMock()
    mock_messages.create.return_value = mock_response

    mock_client = MagicMock()
    mock_client.messages = mock_messages
    return mock_client


# ── _build_payload ────────────────────────────────────────────────────────────

def test_build_payload_includes_participants():
    chat = _make_chat()
    payload = _build_payload(chat, MOCK_CONTEXT)
    assert payload["participantes"] == ["Alice", "Bob"]


def test_build_payload_includes_period():
    chat = _make_chat()
    payload = _build_payload(chat, MOCK_CONTEXT)
    assert payload["periodo"]["dias_total"] == 181


def test_build_payload_includes_decay():
    chat = _make_chat()
    payload = _build_payload(chat, MOCK_CONTEXT)
    assert payload["deterioro"]["tendencia"] == "deteriorating"
    assert payload["deterioro"]["score_0_a_1"] == pytest.approx(0.72)


def test_build_payload_includes_sentiment_when_available():
    chat = _make_chat()
    payload = _build_payload(chat, MOCK_CONTEXT)
    assert "sentimiento" in payload
    assert payload["sentimiento"]["Alice"]["tono_dominante"] == "positive"


def test_build_payload_omits_sentiment_when_missing():
    chat = _make_chat()
    context = {**MOCK_CONTEXT, "sentiment": {}}
    payload = _build_payload(chat, context)
    assert "sentimiento" not in payload


def test_build_payload_no_raw_content():
    """Ensure no message content leaks into the payload."""
    chat = _make_chat()
    payload = _build_payload(chat, MOCK_CONTEXT)
    payload_str = str(payload)
    assert "hey" not in payload_str
    assert "k" not in payload_str


# ── _fmt_seconds ──────────────────────────────────────────────────────────────

def test_fmt_seconds_minutes():
    assert _fmt_seconds(300) == "5 minutos"


def test_fmt_seconds_hours():
    assert "horas" in _fmt_seconds(7200)


def test_fmt_seconds_days():
    assert "días" in _fmt_seconds(90000)


def test_fmt_seconds_none():
    assert _fmt_seconds(None) is None


# ── _short_date ───────────────────────────────────────────────────────────────

def test_short_date_formats_correctly():
    assert _short_date("2024-01-15T10:00:00") == "15/01/2024"


def test_short_date_none():
    assert _short_date(None) is None


# ── Full analyzer ─────────────────────────────────────────────────────────────

def test_analyzer_returns_narrative_fields():
    chat = _make_chat()

    with patch("app.analyzers.narrative.anthropic.Anthropic", return_value=_mock_client()):
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.ANTHROPIC_API_KEY = "sk-test"
            result = NarrativeAnalyzer().analyze(chat, context=MOCK_CONTEXT)

    assert result.analyzer == "narrative"
    assert set(result.data.keys()) == {
        "resumen", "dinamica", "punto_de_quiebre", "estado_actual", "reflexion"
    }


def test_analyzer_narrative_content():
    chat = _make_chat()

    with patch("app.analyzers.narrative.anthropic.Anthropic", return_value=_mock_client()):
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.ANTHROPIC_API_KEY = "sk-test"
            result = NarrativeAnalyzer().analyze(chat, context=MOCK_CONTEXT)

    assert result.data["punto_de_quiebre"] == "2024-Q2"
    assert len(result.data["resumen"]) > 0


def test_analyzer_no_api_key_returns_error():
    chat = _make_chat()
    with patch("app.core.config.settings") as mock_settings:
        mock_settings.ANTHROPIC_API_KEY = None
        mock_settings.GEMINI_API_KEY = None  # both must be absent
        result = NarrativeAnalyzer().analyze(chat, context=MOCK_CONTEXT)

    assert result.data["error"] == "not_configured"


def test_analyzer_no_context_returns_error():
    chat = _make_chat()
    with patch("app.core.config.settings") as mock_settings:
        mock_settings.ANTHROPIC_API_KEY = "sk-test"
        result = NarrativeAnalyzer().analyze(chat, context=None)

    assert result.data["error"] == "insufficient_context"


def test_analyzer_api_error_returns_error():
    chat = _make_chat()
    failing_client = MagicMock()
    failing_client.messages.create.side_effect = Exception("connection timeout")

    with patch("app.analyzers.narrative.anthropic.Anthropic", return_value=failing_client):
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.ANTHROPIC_API_KEY = "sk-test"
            result = NarrativeAnalyzer().analyze(chat, context=MOCK_CONTEXT)

    assert "error" in result.data
    assert "connection timeout" in result.data["error"]


def test_analyzer_context_passed_to_payload():
    """NarrativeAnalyzer must use context, not re-analyze the chat."""
    chat = _make_chat()
    captured = {}

    def capture_call(**kwargs):
        captured["messages"] = kwargs.get("messages", [])
        import json
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = json.dumps(MOCK_NARRATIVE)
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        return mock_response

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = capture_call

    with patch("app.analyzers.narrative.anthropic.Anthropic", return_value=mock_client):
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.ANTHROPIC_API_KEY = "sk-test"
            NarrativeAnalyzer().analyze(chat, context=MOCK_CONTEXT)

    user_message = captured["messages"][0]["content"]
    assert "deteriorating" in user_message   # from context["temporal"]
    assert "positive" in user_message        # from context["sentiment"]
