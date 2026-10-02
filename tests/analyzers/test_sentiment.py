"""
Sentiment analyzer tests.

The HuggingFace model is mocked so tests run without downloading weights.
The mock returns deterministic scores that simulate realistic sentiment patterns,
letting us verify aggregation logic, sampling, and drift detection independently.
"""
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from app.analyzers.sentiment import (
    SentimentAnalyzer,
    _resolve_backend,
    _emotional_drift,
    _evolution,
    _evolution_weekly,
    _per_person,
    _recent_shift,
    _sample,
)
from app.analyzers.sentiment_llm import SentimentLLMError
from app.analyzers.temporal import _week
from app.core.config import settings
from app.parsers.base import ParsedChat, ParsedMessage

# ── Fixtures ──────────────────────────────────────────────────────────────────

BASE = datetime(2024, 1, 1, 10, 0)


@pytest.fixture(autouse=True)
def _local_backend(monkeypatch):
    """Existing tests exercise the local models (mocked); Gemini tests opt in explicitly."""
    monkeypatch.setattr(settings, "SENTIMENT_BACKEND", "local")


def _msg(sender: str, content: str, dt: datetime) -> ParsedMessage:
    return ParsedMessage(timestamp=dt, sender=sender, content=content)


def _make_chat(msgs: list[ParsedMessage]) -> ParsedChat:
    return ParsedChat(
        platform="whatsapp",
        participants=sorted({m.sender for m in msgs}),
        messages=msgs,
    )


def _mock_pipe(texts: list[str]) -> list[list[dict]]:
    """
    Deterministic mock: score depends on first character of the text.
    'p' → positive, 'n' → negative, anything else → neutral.
    """
    results = []
    for text in texts:
        first = text[0].lower() if text else "x"
        if first == "p":
            results.append([
                {"label": "positive", "score": 0.85},
                {"label": "neutral",  "score": 0.10},
                {"label": "negative", "score": 0.05},
            ])
        elif first == "n":
            results.append([
                {"label": "positive", "score": 0.05},
                {"label": "neutral",  "score": 0.10},
                {"label": "negative", "score": 0.85},
            ])
        else:
            results.append([
                {"label": "positive", "score": 0.20},
                {"label": "neutral",  "score": 0.65},
                {"label": "negative", "score": 0.15},
            ])
    return results


def _patched_analyzer():
    """Returns a SentimentAnalyzer with the HuggingFace pipeline mocked."""
    return patch("app.analyzers.sentiment._get_multilingual_pipe", return_value=_mock_pipe)


# ── per_person ────────────────────────────────────────────────────────────────

def test_per_person_dominant_positive():
    msgs = [_msg("Alice", "positive message", BASE + timedelta(minutes=i)) for i in range(5)]
    scores = [0.8] * 5
    result = _per_person(msgs, scores)
    assert result["Alice"]["dominant"] == "positive"
    assert result["Alice"]["avg_score"] == pytest.approx(0.8)


def test_per_person_dominant_negative():
    msgs = [_msg("Bob", "negative message", BASE + timedelta(minutes=i)) for i in range(5)]
    scores = [-0.7] * 5
    result = _per_person(msgs, scores)
    assert result["Bob"]["dominant"] == "negative"
    assert result["Bob"]["avg_score"] == pytest.approx(-0.7)


def test_per_person_shares_sum_to_one():
    msgs = [_msg("Alice", f"msg{i}", BASE + timedelta(minutes=i)) for i in range(10)]
    scores = [0.9, -0.8, 0.1, 0.7, -0.3, 0.05, -0.9, 0.6, 0.0, -0.1]
    result = _per_person(msgs, scores)
    total = result["Alice"]["positive"] + result["Alice"]["neutral"] + result["Alice"]["negative"]
    assert total == pytest.approx(1.0, abs=0.01)


def test_per_person_charged_splits_tone_among_charged_messages():
    # 10 messages: 1 positive, 2 negative, 7 neutral -> share 0.3, split 1/3 vs 2/3.
    msgs = [_msg("Alice", f"m{i}", BASE + timedelta(minutes=i)) for i in range(10)]
    scores = [0.9, -0.5, -0.8] + [0.0] * 7
    result = _per_person(msgs, scores)["Alice"]
    assert result["dominant"] == "neutral"
    assert result["charged"] == {"share": 0.3, "positive": 0.333, "negative": 0.667}


def test_per_person_charged_none_when_only_neutral():
    msgs = [_msg("Alice", f"m{i}", BASE + timedelta(minutes=i)) for i in range(5)]
    result = _per_person(msgs, [0.0, 0.1, -0.1, 0.2, -0.2])["Alice"]
    assert result["charged"] == {"share": 0.0, "positive": None, "negative": None}


def test_per_person_existing_keys_unchanged_by_charged():
    msgs = [_msg("Alice", f"m{i}", BASE + timedelta(minutes=i)) for i in range(4)]
    result = _per_person(msgs, [0.8, -0.6, 0.0, 0.0])["Alice"]
    assert result["positive"] == 0.25
    assert result["neutral"] == 0.5
    assert result["negative"] == 0.25
    assert result["dominant"] == "neutral"
    assert result["avg_score"] == pytest.approx(0.05)


# ── evolution ─────────────────────────────────────────────────────────────────

def test_evolution_has_period_key():
    msgs = [_msg("Alice", "msg", BASE + timedelta(days=i)) for i in range(3)]
    scores = [0.5, 0.3, -0.1]
    result = _evolution(msgs, scores)
    assert all("period" in entry for entry in result)


def test_evolution_groups_by_quarter():
    msgs = [
        _msg("Alice", "q1", datetime(2024, 1, 15)),
        _msg("Alice", "q2", datetime(2024, 4, 15)),
        _msg("Alice", "q3", datetime(2024, 7, 15)),
    ]
    scores = [0.5, -0.3, 0.1]
    result = _evolution(msgs, scores)
    periods = [e["period"] for e in result]
    assert "2024-Q1" in periods
    assert "2024-Q2" in periods
    assert "2024-Q3" in periods


# ── emotional_drift ───────────────────────────────────────────────────────────

def test_drift_zero_when_aligned():
    msgs = (
        [_msg("Alice", "msg", BASE + timedelta(days=i)) for i in range(0, 6)]
        + [_msg("Bob", "msg", BASE + timedelta(days=i)) for i in range(6, 12)]
    )
    # Both equally positive
    scores = [0.6] * 12
    result = _emotional_drift(msgs, scores)
    assert result["score"] == pytest.approx(0.0, abs=0.05)


def test_drift_high_when_opposite():
    msgs = (
        [_msg("Alice", "p", BASE + timedelta(days=i*7)) for i in range(8)]
        + [_msg("Bob",   "n", BASE + timedelta(days=i*7)) for i in range(8)]
    )
    # Alice strongly positive, Bob strongly negative
    scores = [0.9] * 8 + [-0.9] * 8
    result = _emotional_drift(msgs, scores)
    assert result["score"] > 0.5


def test_drift_direction_label():
    msgs = (
        [_msg("Alice", "p", BASE + timedelta(days=i*7)) for i in range(4)]
        + [_msg("Bob",   "n", BASE + timedelta(days=i*7)) for i in range(4)]
    )
    scores = [0.8] * 4 + [-0.8] * 4
    result = _emotional_drift(msgs, scores)
    assert "Alice" in result["direction"]
    assert "positive" in result["direction"]


def test_drift_score_bounds():
    msgs = (
        [_msg("Alice", "x", BASE + timedelta(days=i)) for i in range(10)]
        + [_msg("Bob",   "x", BASE + timedelta(days=i)) for i in range(10)]
    )
    scores = [0.5, -0.5] * 10
    result = _emotional_drift(msgs, scores)
    assert 0.0 <= result["score"] <= 1.0


# ── sampling ──────────────────────────────────────────────────────────────────

def test_sample_returns_all_when_under_limit():
    msgs = [_msg("Alice", "msg", BASE + timedelta(minutes=i)) for i in range(100)]
    assert len(_sample(msgs, 200)) == 100


def test_sample_caps_at_max():
    msgs = [_msg("Alice", "msg", BASE + timedelta(minutes=i)) for i in range(5000)]
    sampled = _sample(msgs, 2000)
    assert len(sampled) <= 2000


def test_sample_preserves_order():
    msgs = [_msg("Alice", str(i), BASE + timedelta(minutes=i)) for i in range(5000)]
    sampled = _sample(msgs, 2000)
    timestamps = [m.timestamp for m in sampled]
    assert timestamps == sorted(timestamps)


def _spread(n: int, weeks: int, sender: str = "Alice") -> list[ParsedMessage]:
    """n messages spread evenly over `weeks` weeks starting at BASE (a Monday)."""
    span = weeks * 7 * 24 * 60 - 1
    return [
        _msg(sender, str(i), BASE + timedelta(minutes=i * span // max(n - 1, 1)))
        for i in range(n)
    ]


def test_sample_regression_keeps_end_of_chat():
    msgs = _spread(28505, 21)
    sampled = _sample(msgs, 2000)
    assert len(sampled) <= 2000
    last_idx = msgs.index(sampled[-1])
    assert last_idx >= len(msgs) - 14


def test_sample_low_volume_final_week_gets_floor():
    msgs = []
    for w in range(10):
        msgs += [_msg("Alice", "x", BASE + timedelta(weeks=w, seconds=i)) for i in range(2000)]
    msgs += [_msg("Alice", "x", BASE + timedelta(weeks=10, seconds=i)) for i in range(60)]
    sampled = _sample(msgs, 2000)
    last_week = _week(msgs[-1].timestamp)
    assert sum(1 for m in sampled if _week(m.timestamp) == last_week) >= 40
    assert len(sampled) <= 2000


def test_sample_chronological_and_unique():
    msgs = _spread(9000, 15)
    sampled = _sample(msgs, 2000)
    ts = [m.timestamp for m in sampled]
    assert ts == sorted(ts)
    assert len({id(m) for m in sampled}) == len(sampled)


def test_sample_short_chat_returned_intact():
    msgs = _spread(100, 3)
    assert _sample(msgs, 2000) is msgs


def test_sample_floors_exceeding_budget():
    # 100 weeks x 100 msgs, budget 500: floor 40 x 100 weeks would be 4000.
    msgs = []
    for w in range(100):
        msgs += [_msg("Alice", "x", BASE + timedelta(weeks=w, minutes=i)) for i in range(100)]
    sampled = _sample(msgs, 500)
    assert len(sampled) <= 500
    assert {_week(m.timestamp) for m in sampled} == {_week(m.timestamp) for m in msgs}
    assert sampled[-1] is msgs[-1]


# ── weekly series ─────────────────────────────────────────────────────────────

def test_evolution_weekly_threshold_and_period_start():
    monday = datetime(2026, 9, 21, 9, 0)  # ISO week 2026-W39
    msgs = (
        [_msg("Alice", "a", monday + timedelta(hours=i)) for i in range(5)]
        + [_msg("Bob", "b", monday + timedelta(hours=i)) for i in range(4)]
    )
    scores = [0.5] * 5 + [-0.5] * 4
    result = _evolution_weekly(msgs, scores)
    assert len(result) == 1
    entry = result[0]
    assert entry["period"] == "2026-W39"
    assert entry["period_start"] == "2026-09-21"
    assert entry["Alice"] == pytest.approx(0.5)
    assert "Bob" not in entry
    assert entry["n"] == 9


# ── recent stretch ────────────────────────────────────────────────────────────

def _two_phase(recent_score: float, days: int = 60):
    end = datetime(2026, 9, 30, 12, 0)
    msgs, scores = [], []
    for d in range(days):
        for sender in ("Alice", "Bob"):
            ts = end - timedelta(days=days - 1 - d, minutes=1 if sender == "Alice" else 0)
            msgs.append(_msg(sender, "x", ts))
            scores.append(recent_score if d >= days - 14 else 0.3)
    return msgs, scores


def test_recent_shift_none_on_short_chat():
    msgs, scores = _two_phase(-0.5, days=20)
    assert _recent_shift(msgs, scores) is None


def test_recent_shift_none_with_few_points():
    msgs = [_msg("Alice", "x", datetime(2026, 8, 1) + timedelta(days=i)) for i in range(40)]
    assert _recent_shift(msgs, [0.2] * 40) is None  # only ~14 msgs in window


def test_recent_shift_more_negative():
    msgs, scores = _two_phase(-0.5)
    result = _recent_shift(msgs, scores)
    assert result["shift"] == "more_negative"
    assert result["window_days"] == 14
    alice = result["per_person"]["Alice"]
    assert alice["delta"] == pytest.approx(-0.8)
    assert alice["recent_negative_share"] == 1.0
    assert alice["baseline_negative_share"] == 0.0


def test_recent_shift_stable():
    msgs, scores = _two_phase(0.3)
    assert _recent_shift(msgs, scores)["shift"] == "stable"


# ── full analyzer (mocked model) ─────────────────────────────────────────────

def test_full_analyzer_output_structure():
    msgs = (
        [_msg("Alice", f"positive msg {i}", BASE + timedelta(days=i)) for i in range(20)]
        + [_msg("Bob",   f"neutral msg {i}",   BASE + timedelta(days=i)) for i in range(20)]
    )
    chat = _make_chat(msgs)

    with _patched_analyzer():
        result = SentimentAnalyzer().analyze(chat)

    assert result.analyzer == "sentiment"
    assert "per_person" in result.data
    assert "evolution" in result.data
    assert "emotional_drift" in result.data
    assert "sample_size" in result.data
    assert set(result.data["per_person"].keys()) == {"Alice", "Bob"}


def test_full_analyzer_per_person_keys():
    msgs = [_msg("Alice", "positive", BASE + timedelta(minutes=i)) for i in range(10)]
    chat = _make_chat(msgs + [_msg("Bob", "neutral", BASE + timedelta(minutes=11))])

    with _patched_analyzer():
        result = SentimentAnalyzer().analyze(chat)

    alice = result.data["per_person"]["Alice"]
    assert set(alice.keys()) == {
        "positive", "neutral", "negative", "dominant", "avg_score", "charged"
    }


def test_analyzer_insufficient_data():
    chat = _make_chat([_msg("Alice", "hi", BASE)])
    with _patched_analyzer():
        result = SentimentAnalyzer().analyze(chat)
    assert result.data.get("error") == "insufficient_data"


def test_analyzer_skips_media():
    msgs = [
        ParsedMessage(timestamp=BASE, sender="Alice", content="", is_media=True),
        ParsedMessage(timestamp=BASE + timedelta(minutes=1), sender="Bob", content="", is_media=True),
    ]
    chat = _make_chat(msgs)
    with _patched_analyzer():
        result = SentimentAnalyzer().analyze(chat)
    assert result.data.get("error") == "insufficient_data"


# ── language routing ────────────────────────────────────────────────────────

def test_spanish_routes_to_pysentimiento():
    """When context language is 'es', the Spanish backends are invoked
    and `emotions_per_person` is populated."""
    msgs = [
        _msg("Alice", "qué lindo día", BASE + timedelta(minutes=i)) for i in range(6)
    ] + [
        _msg("Bob", "estoy muy triste", BASE + timedelta(minutes=i + 6)) for i in range(6)
    ]
    chat = _make_chat(msgs)

    sentiment_outputs = [type("O", (), {"probas": {"POS": 0.9, "NEU": 0.05, "NEG": 0.05}})() for _ in range(12)]
    emotion_outputs = [type("O", (), {"output": "joy"})() for _ in range(12)]

    with patch("app.analyzers.sentiment._get_es_sentiment") as mock_sent, \
         patch("app.analyzers.sentiment._get_es_emotion") as mock_emo:
        mock_sent.return_value.predict.side_effect = lambda batch: sentiment_outputs[:len(batch)]
        mock_emo.return_value.predict.side_effect = lambda batch: emotion_outputs[:len(batch)]

        result = SentimentAnalyzer().analyze(chat, context={"_meta": {"language": "es"}})

    assert "emotions_per_person" in result.data
    assert result.data["language"] == "es"
    assert "pysentimiento" in result.data["model"]
    mock_sent.assert_called_once()
    mock_emo.assert_called_once()


def test_non_spanish_uses_multilingual_only():
    """When language != 'es', pysentimiento is NOT loaded and no emotions block is added."""
    msgs = [_msg("Alice", "positive msg", BASE + timedelta(minutes=i)) for i in range(6)]
    chat = _make_chat(msgs + [_msg("Bob", "neutral msg", BASE + timedelta(minutes=7))])

    with patch("app.analyzers.sentiment._get_es_sentiment") as mock_sent, \
         patch("app.analyzers.sentiment._get_es_emotion") as mock_emo, \
         _patched_analyzer():
        result = SentimentAnalyzer().analyze(chat, context={"_meta": {"language": "en"}})

    assert "emotions_per_person" not in result.data
    assert result.data["language"] == "en"
    mock_sent.assert_not_called()
    mock_emo.assert_not_called()


def test_full_analyzer_includes_weekly_and_recent():
    msgs = []
    for d in range(60):
        msgs.append(_msg("Alice", "positive msg", BASE + timedelta(days=d, hours=1)))
        msgs.append(_msg("Bob", "neutral msg", BASE + timedelta(days=d, hours=2)))
    with _patched_analyzer():
        result = SentimentAnalyzer().analyze(_make_chat(msgs))
    assert result.data["weekly"]
    assert result.data["recent"]["shift"] == "stable"


def test_full_analyzer_omits_recent_on_short_chat():
    msgs = [_msg("Alice", "positive", BASE + timedelta(minutes=i)) for i in range(10)]
    with _patched_analyzer():
        result = SentimentAnalyzer().analyze(_make_chat(msgs))
    assert "weekly" in result.data
    assert "recent" not in result.data


# ── backend selection ─────────────────────────────────────────────────────────

class _Cfg:
    def __init__(self, backend: str, key: str | None):
        self.SENTIMENT_BACKEND = backend
        self.GEMINI_API_KEY = key


@pytest.mark.parametrize("backend, key, expected", [
    ("auto", "k", "gemini"),
    ("auto", None, "local"),
    ("gemini", "k", "gemini"),
    ("gemini", None, "local"),   # nothing to call Gemini with
    ("local", "k", "local"),
    ("local", None, "local"),
])
def test_resolve_backend(backend, key, expected):
    assert _resolve_backend(_Cfg(backend, key)) == expected


def _gemini_chat() -> ParsedChat:
    msgs = [_msg("Alice", f"positive {i}", BASE + timedelta(minutes=i)) for i in range(6)]
    msgs += [_msg("Bob", f"neutral {i}", BASE + timedelta(minutes=i + 6)) for i in range(6)]
    return _make_chat(msgs)


def _use_gemini(monkeypatch):
    monkeypatch.setattr(settings, "SENTIMENT_BACKEND", "gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-key")


def test_gemini_branch_scores_and_skips_local_models(monkeypatch):
    _use_gemini(monkeypatch)
    calls = {}

    def fake_score(texts, *, api_key, model, generate=None):
        calls["texts"] = texts
        calls["model"] = model
        return [0.5] * len(texts), ["joy"] * len(texts)

    monkeypatch.setattr("app.analyzers.sentiment_llm.score_messages", fake_score)
    with patch("app.analyzers.sentiment._get_es_sentiment") as es, \
         patch("app.analyzers.sentiment._get_multilingual_pipe") as multi:
        result = SentimentAnalyzer().analyze(_gemini_chat(), context={"_meta": {"language": "es"}})

    es.assert_not_called()
    multi.assert_not_called()
    assert result.data["backend"] == "gemini"
    assert result.data["model"] == f"gemini:{settings.GEMINI_SENTIMENT_MODEL}"
    assert result.data["language"] == "es"
    assert len(calls["texts"]) == 12
    assert result.data["per_person"]["Alice"]["avg_score"] == pytest.approx(0.5)
    assert result.data["emotions_per_person"]["Bob"]["dominant"] == "joy"


def test_gemini_failure_falls_back_to_local(monkeypatch):
    _use_gemini(monkeypatch)

    def boom(*args, **kwargs):
        raise SentimentLLMError("quota")

    monkeypatch.setattr("app.analyzers.sentiment_llm.score_messages", boom)
    with _patched_analyzer():
        result = SentimentAnalyzer().analyze(_gemini_chat(), context={"_meta": {"language": "en"}})

    assert result.data["backend"] == "local"
    assert result.data["model"] == "distilbert-multilingual"


def test_gemini_failure_and_no_local_models_reports_unavailable(monkeypatch):
    _use_gemini(monkeypatch)

    def boom(*args, **kwargs):
        raise SentimentLLMError("quota")

    monkeypatch.setattr("app.analyzers.sentiment_llm.score_messages", boom)
    with patch("app.analyzers.sentiment._get_multilingual_pipe", side_effect=ImportError("no transformers")):
        result = SentimentAnalyzer().analyze(_gemini_chat(), context={"_meta": {"language": "en"}})

    assert result.data == {"error": "sentiment_unavailable"}


def test_local_backend_without_models_reports_unavailable():
    with patch("app.analyzers.sentiment._get_multilingual_pipe", side_effect=ImportError("no torch")):
        result = SentimentAnalyzer().analyze(_gemini_chat(), context={"_meta": {"language": "en"}})
    assert result.data == {"error": "sentiment_unavailable"}


def test_local_result_reports_backend():
    with _patched_analyzer():
        result = SentimentAnalyzer().analyze(_gemini_chat(), context={"_meta": {"language": "en"}})
    assert result.data["backend"] == "local"
