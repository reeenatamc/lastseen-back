"""
Sentiment analyzer tests.

The HuggingFace model is mocked so tests run without downloading weights.
The mock returns deterministic scores that simulate realistic sentiment patterns,
letting us verify aggregation logic, sampling, and drift detection independently.
"""
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from app.analyzers.sentiment import SentimentAnalyzer, _per_person, _evolution, _emotional_drift, _sample
from app.parsers.base import ParsedChat, ParsedMessage

# ── Fixtures ──────────────────────────────────────────────────────────────────

BASE = datetime(2024, 1, 1, 10, 0)


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
    return patch("app.analyzers.sentiment._get_pipe", return_value=_mock_pipe)


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
    assert set(alice.keys()) == {"positive", "neutral", "negative", "dominant", "avg_score"}


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
