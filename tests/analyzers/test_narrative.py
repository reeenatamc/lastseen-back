"""
Narrative analyzer tests.

The Anthropic client is mocked so tests run without a real API key.
The mock returns a deterministic, valid narrative JSON, letting us verify
payload construction, context wiring, and error-handling paths in isolation.
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from app.analyzers.narrative import (
    _MAX_EPISODES,
    _MAX_TIMELINE_POINTS,
    NarrativeAnalyzer,
    _build_payload,
    _conflict_summary,
    _fmt_seconds,
    _short_date,
    _weekly_timeline,
)
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


# ── closing phase, weekly timeline and conflicts ─────────────────────────────

def _week_entry(i: int, health: float = 0.8, per_day: float = 200.0) -> dict:
    start = (BASE + timedelta(weeks=i)).strftime("%Y-%m-%d")
    return {"period_start": start, "health": health, "messages_per_day": per_day}


def _episode(day: int, mentions: int = 3, blocked: bool = False) -> dict:
    date = (BASE + timedelta(days=day)).strftime("%Y-%m-%d")
    return {
        "start": date,
        "end": date,
        "mentions": mentions,
        "per_person": {"Alice": mentions},
        "categories": {"breakup": mentions},
        "missed_calls": 0,
        "blocked": blocked,
        "severity": "high" if blocked or mentions >= 8 else "medium",
    }


def _context_with(**temporal_decay) -> dict:
    temporal = {
        **MOCK_CONTEXT["temporal"],
        "response_decay": {**MOCK_CONTEXT["temporal"]["response_decay"], **temporal_decay},
    }
    return {**MOCK_CONTEXT, "temporal": temporal}


def test_build_payload_omits_closing_phase_when_not_detected():
    context = _context_with(closing_phase={"detected": False, "reason": "insufficient_baseline"})
    payload = _build_payload(_make_chat(), context)
    assert "tramo_final" not in payload


def test_build_payload_includes_closing_phase_when_detected():
    context = _context_with(closing_phase={
        "detected": True,
        "start": "2024-06-10",
        "volume_ratio": 0.31,
        "max_silence_days": 2.8,
        "baseline_max_silence_days": 0.6,
        "window_weeks": 3,
    })
    payload = _build_payload(_make_chat(), context)
    assert payload["tramo_final"]["inicio"] == "10/06/2024"
    assert payload["tramo_final"]["volumen_respecto_a_lo_habitual"] == pytest.approx(0.31)
    assert payload["tramo_final"]["silencio_mas_largo_dias"] == pytest.approx(2.8)


def test_build_payload_includes_weekly_timeline():
    context = _context_with(evolution=[_week_entry(i) for i in range(4)])
    payload = _build_payload(_make_chat(), context)
    assert len(payload["evolucion_semanal"]) == 4
    assert payload["evolucion_semanal"][0]["semana"] == "01/01/2024"


def test_weekly_timeline_is_thinned_and_keeps_last_week():
    evolution = [_week_entry(i, health=round(i / 100, 2)) for i in range(100)]
    timeline = _weekly_timeline(evolution)
    assert len(timeline) <= _MAX_TIMELINE_POINTS
    assert timeline[0]["salud_0_a_1"] == pytest.approx(0.0)
    assert timeline[-1]["salud_0_a_1"] == pytest.approx(0.99)


def test_build_payload_reports_initiative_confidence():
    temporal = {
        **MOCK_CONTEXT["temporal"],
        "initiative_balance": {
            **MOCK_CONTEXT["temporal"]["initiative_balance"],
            "confidence": {"level": "low", "reason": "continuous_thread"},
        },
    }
    payload = _build_payload(_make_chat(), {**MOCK_CONTEXT, "temporal": temporal})
    assert payload["iniciativa"]["confianza"] == "low"


def _context_with_confidence(level: str) -> dict:
    temporal = {
        **MOCK_CONTEXT["temporal"],
        "initiative_balance": {
            **MOCK_CONTEXT["temporal"]["initiative_balance"],
            "confidence": {"level": level},
        },
    }
    return {**MOCK_CONTEXT, "temporal": temporal}


def test_build_payload_low_confidence_initiative_has_no_distribution():
    payload = _build_payload(_make_chat(), _context_with_confidence("low"))
    assert payload["iniciativa"] == {"confianza": "low"}


def test_build_payload_non_low_confidence_keeps_distribution():
    payload = _build_payload(_make_chat(), _context_with_confidence("high"))
    assert payload["iniciativa"]["confianza"] == "high"
    assert payload["iniciativa"]["distribucion"] == {"Alice": 0.7, "Bob": 0.3}
    assert payload["iniciativa"]["conversaciones_totales"] == 40


def test_build_payload_includes_charged_tone_when_present():
    sentiment = {
        **MOCK_CONTEXT["sentiment"],
        "per_person": {
            "Alice": {
                "dominant": "neutral",
                "avg_score": 0.1,
                "charged": {"share": 0.3, "positive": 0.333, "negative": 0.667},
            },
            "Bob": {"dominant": "negative", "avg_score": -0.3},
        },
    }
    payload = _build_payload(_make_chat(), {**MOCK_CONTEXT, "sentiment": sentiment})
    assert payload["sentimiento"]["Alice"]["con_carga_emocional"] == {
        "proporcion": 0.3,
        "calidos": 0.333,
        "tensos": 0.667,
    }
    assert "con_carga_emocional" not in payload["sentimiento"]["Bob"]


def test_build_payload_includes_conflicts():
    conflict = {
        "episodes": [_episode(10), _episode(150, mentions=9, blocked=True)],
        "system_events": {"blocks": ["2024-05-30T22:10:00"]},
        "recent": {"ratio": 9.5},
    }
    payload = _build_payload(_make_chat(), {**MOCK_CONTEXT, "conflict": conflict})
    conflicts = payload["conflictos"]
    assert conflicts["episodios_totales"] == 2
    assert conflicts["episodios"][1]["bloqueo"] is True
    assert conflicts["episodios"][1]["gravedad"] == "high"
    assert conflicts["bloqueos"] == ["30/05/2024"]
    assert conflicts["lenguaje_de_ruptura_reciente_vs_habitual"] == pytest.approx(9.5)


def test_build_payload_omits_conflicts_when_none_or_errored():
    chat = _make_chat()
    assert "conflictos" not in _build_payload(chat, MOCK_CONTEXT)
    errored = {**MOCK_CONTEXT, "conflict": {"error": "insufficient_data"}}
    assert "conflictos" not in _build_payload(chat, errored)
    quiet = {**MOCK_CONTEXT, "conflict": {"episodes": [], "system_events": {"blocks": []}}}
    assert "conflictos" not in _build_payload(chat, quiet)


def test_conflict_summary_keeps_heaviest_episodes_in_date_order():
    episodes = [_episode(day, mentions=3) for day in range(0, 60, 2)]   # 30 light ones
    episodes[4] = _episode(8, mentions=20)
    episodes[20] = _episode(40, mentions=4, blocked=True)
    summary = _conflict_summary({"episodes": episodes, "system_events": {}})

    assert summary["episodios_totales"] == 30
    assert len(summary["episodios"]) == _MAX_EPISODES
    starts = [e["inicio"] for e in summary["episodios"]]
    assert _short_date(episodes[4]["start"]) in starts
    assert _short_date(episodes[20]["start"]) in starts
    iso = [datetime.strptime(s, "%d/%m/%Y") for s in starts]
    assert iso == sorted(iso)


def test_build_payload_includes_recent_sentiment():
    sentiment = {
        **MOCK_CONTEXT["sentiment"],
        "recent": {
            "window_days": 14,
            "start": "2024-06-16",
            "shift": "more_negative",
            "per_person": {
                "Alice": {"recent_avg": -0.2, "baseline_avg": 0.3, "delta": -0.5},
                "Bob": {"recent_avg": -0.4, "baseline_avg": -0.1, "delta": -0.3},
            },
        },
    }
    payload = _build_payload(_make_chat(), {**MOCK_CONTEXT, "sentiment": sentiment})
    recent = payload["sentimiento"]["tramo_reciente"]
    assert recent["cambio"] == "more_negative"
    assert recent["por_persona"]["Alice"]["score_reciente"] == pytest.approx(-0.2)


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
    assert result.data["error"] == "llm_failed"  # no exception text in the result


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
