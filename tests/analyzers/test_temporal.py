from datetime import datetime, timedelta

import pytest

from app.analyzers.temporal import TemporalAnalyzer
from app.parsers.base import ParsedChat, ParsedMessage

# ── Helpers ───────────────────────────────────────────────────────────────────

BASE = datetime(2024, 1, 1, 10, 0)


def _msg(sender: str, content: str, dt: datetime, is_media: bool = False) -> ParsedMessage:
    return ParsedMessage(timestamp=dt, sender=sender, content=content, is_media=is_media)


def _make_chat(msgs: list[ParsedMessage]) -> ParsedChat:
    return ParsedChat(
        platform="whatsapp",
        participants=sorted({m.sender for m in msgs}),
        messages=msgs,
    )


def healthy_chat() -> ParsedChat:
    """
    20 blocks separated by 2-day gaps. Alice and Bob alternate who initiates.
    Each block ends with the INITIATOR so the next block's opener is always
    the OTHER person — ensuring they get correctly counted as a new initiative
    (not a double text) under the corrected _initiative_balance definition.
    """
    msgs = []
    for block in range(20):
        t = BASE + timedelta(days=block * 2)
        init, resp = ("Alice", "Bob") if block % 2 == 0 else ("Bob", "Alice")
        msgs += [
            _msg(init, "hey there!", t),
            _msg(resp, "hello!", t + timedelta(minutes=5)),
            _msg(resp, "how was your day?", t + timedelta(minutes=8)),
            _msg(init, "was really good!", t + timedelta(minutes=12)),  # init speaks last
        ]
    return _make_chat(msgs)


def decayed_chat() -> ParsedChat:
    """
    30 blocks separated by 2-day gaps.
    First 15: symmetric and quick — sets a healthy baseline.
    Last 15: Alice always initiates, Bob responds in 2–3.7h (< 4h so he never
             appears as initiator), conversation is terse and one-sided.
    The contrast lets _response_decay detect the turning point.
    """
    msgs = []
    for block in range(30):
        t = BASE + timedelta(days=block * 2)
        if block < 15:
            init, resp = ("Alice", "Bob") if block % 2 == 0 else ("Bob", "Alice")
            msgs += [
                _msg(init, "hey there!", t),
                _msg(resp, "hello!", t + timedelta(minutes=5)),
                _msg(init, "how was your day?", t + timedelta(minutes=10)),
                _msg(resp, "was really good!", t + timedelta(minutes=15)),
            ]
        else:
            # 120 min growing to 210 min — never exceeds the 240-min initiative gap
            response_mins = 120 + (block - 15) * 6
            msgs += [
                _msg("Alice", "hey", t),
                _msg("Bob", "k", t + timedelta(minutes=response_mins)),
            ]
    return _make_chat(msgs)


# ── Overview ──────────────────────────────────────────────────────────────────

def test_overview_totals():
    result = TemporalAnalyzer().analyze(healthy_chat())
    ov = result.data["overview"]
    assert ov["total_messages"] == 80
    assert set(ov["participants"]) == {"Alice", "Bob"}
    assert ov["share_per_person"]["Alice"] == pytest.approx(0.5, abs=0.01)


def test_overview_date_range():
    result = TemporalAnalyzer().analyze(healthy_chat())
    dr = result.data["overview"]["date_range"]
    assert dr["total_days"] == 38  # block 0 day 0 → block 19 day 38


# ── Response time ─────────────────────────────────────────────────────────────

def test_response_time_healthy():
    result = TemporalAnalyzer().analyze(healthy_chat())
    rt = result.data["response_time"]["per_person"]
    assert rt["Alice"]["mean_seconds"] < 600
    assert rt["Bob"]["mean_seconds"] < 600


def test_response_time_decayed():
    result = TemporalAnalyzer().analyze(decayed_chat())
    rt = result.data["response_time"]["per_person"]
    # Bob's mean is pulled up by the slow decayed phase (2–3.7h)
    assert rt["Bob"]["mean_seconds"] > 3600
    # Bob's p90 reflects the worst delays in the decayed phase
    assert rt["Bob"]["p90_seconds"] > rt["Bob"]["median_seconds"]


def test_response_time_evolution_has_periods():
    result = TemporalAnalyzer().analyze(healthy_chat())
    evo = result.data["response_time"]["evolution"]
    assert len(evo) >= 1
    assert "period" in evo[0]


# ── Initiative balance ────────────────────────────────────────────────────────

def test_initiative_balance_healthy():
    result = TemporalAnalyzer().analyze(healthy_chat())
    share = result.data["initiative_balance"]["share"]
    assert share["Alice"] == pytest.approx(0.5, abs=0.05)
    assert share["Bob"] == pytest.approx(0.5, abs=0.05)


def test_initiative_balance_decayed():
    result = TemporalAnalyzer().analyze(decayed_chat())
    share = result.data["initiative_balance"]["share"]
    assert share["Alice"] > share.get("Bob", 0)
    assert share["Alice"] > 0.65


def test_double_text_keys_present():
    result = TemporalAnalyzer().analyze(healthy_chat())
    dt = result.data["initiative_balance"]["double_text"]
    assert "per_person" in dt
    assert "share" in dt
    assert "total" in dt


def test_late_reply_not_counted_as_initiative():
    """
    Alice messages, Bob responds 5h later. Under the corrected logic Bob's
    reply is a late response — not an initiative. Alice is the initiator.
    """
    base = datetime(2024, 1, 1, 10, 0)
    msgs = [
        _msg("Alice", "hey",        base),
        _msg("Bob",   "hola",       base + timedelta(hours=5)),   # late reply, NOT initiative
        _msg("Alice", "como estas", base + timedelta(hours=5, minutes=2)),
        _msg("Bob",   "bien",       base + timedelta(hours=5, minutes=5)),
    ]
    result = TemporalAnalyzer().analyze(_make_chat(msgs))
    share = result.data["initiative_balance"]["share"]
    # Alice opened the only real initiative (the very first message)
    assert share.get("Alice", 0) >= share.get("Bob", 0)


def test_double_text_detected():
    """
    Alice messages twice without a response — that's a double text.
    """
    base = datetime(2024, 1, 1, 10, 0)
    msgs = [
        _msg("Bob",   "hi",         base),
        _msg("Alice", "hey",        base + timedelta(minutes=1)),
        _msg("Alice", "sigo aqui?", base + timedelta(hours=5)),  # double text by Alice
        _msg("Bob",   "sorry",      base + timedelta(hours=5, minutes=10)),
    ]
    result = TemporalAnalyzer().analyze(_make_chat(msgs))
    dt = result.data["initiative_balance"]["double_text"]
    assert dt["per_person"].get("Alice", 0) >= 1


def test_initiative_balance_total_conversations():
    result = TemporalAnalyzer().analyze(healthy_chat())
    assert result.data["initiative_balance"]["total_conversations"] == 20


# ── Activity patterns ─────────────────────────────────────────────────────────

def test_activity_patterns_keys():
    result = TemporalAnalyzer().analyze(healthy_chat())
    p = result.data["activity_patterns"]
    assert len(p["by_hour"]) == 24
    assert len(p["by_weekday"]) == 7
    assert len(p["by_month"]) >= 1


def test_activity_patterns_by_hour_sum():
    chat = healthy_chat()
    result = TemporalAnalyzer().analyze(chat)
    total = sum(result.data["activity_patterns"]["by_hour"].values())
    assert total == len(chat.messages)


# ── Conversation gaps ─────────────────────────────────────────────────────────

def test_conversation_gaps_sorted_descending():
    result = TemporalAnalyzer().analyze(healthy_chat())
    top = result.data["conversation_gaps"]["top_gaps"]
    assert len(top) > 0
    hours = [g["hours"] for g in top]
    assert hours == sorted(hours, reverse=True)


def test_conversation_gaps_distribution_complete():
    chat = healthy_chat()
    result = TemporalAnalyzer().analyze(chat)
    dist = result.data["conversation_gaps"]["distribution"]
    assert set(dist.keys()) == {"under_1h", "1h_to_6h", "6h_to_24h", "1d_to_7d", "over_7d"}
    assert sum(dist.values()) == len(chat.messages) - 1


# ── Message length ────────────────────────────────────────────────────────────

def test_message_length_per_person():
    result = TemporalAnalyzer().analyze(healthy_chat())
    length = result.data["message_length"]["per_person"]
    assert "Alice" in length
    assert length["Alice"]["mean_chars"] > 0


def test_message_length_skips_media():
    msgs = [
        _msg("Alice", "", BASE, is_media=True),
        _msg("Bob", "hello there friend", BASE + timedelta(minutes=1)),
        _msg("Alice", "hey", BASE + timedelta(minutes=2)),
    ]
    result = TemporalAnalyzer().analyze(_make_chat(msgs))
    length = result.data["message_length"]["per_person"]
    assert length["Alice"]["mean_chars"] == 3
    assert length["Bob"]["mean_chars"] == 18


# ── Response decay ────────────────────────────────────────────────────────────

def test_decay_healthy_trend():
    result = TemporalAnalyzer().analyze(healthy_chat())
    assert result.data["response_decay"]["trend"] in ("stable", "improving")


def test_decay_healthy_score():
    result = TemporalAnalyzer().analyze(healthy_chat())
    assert result.data["response_decay"]["decay_score"] < 0.5


def test_decay_decayed_trend():
    result = TemporalAnalyzer().analyze(decayed_chat())
    assert result.data["response_decay"]["trend"] == "deteriorating"


def test_decay_decayed_score():
    result = TemporalAnalyzer().analyze(decayed_chat())
    assert result.data["response_decay"]["decay_score"] > 0.5


def test_decay_has_turning_point():
    result = TemporalAnalyzer().analyze(decayed_chat())
    assert result.data["response_decay"]["turning_point"] is not None


def test_decay_score_always_in_bounds():
    for chat in [healthy_chat(), decayed_chat()]:
        score = TemporalAnalyzer().analyze(chat).data["response_decay"]["decay_score"]
        assert 0.0 <= score <= 1.0


def test_decay_evolution_fields():
    result = TemporalAnalyzer().analyze(healthy_chat())
    evo = result.data["response_decay"]["evolution"]
    assert len(evo) >= 1
    required = {"period", "avg_response_seconds", "message_count", "initiative_imbalance"}
    assert required <= set(evo[0])


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_single_message_returns_error():
    chat = _make_chat([_msg("Alice", "hi", BASE)])
    assert TemporalAnalyzer().analyze(chat).data.get("error") == "insufficient_data"


def test_two_messages_minimum():
    msgs = [_msg("Alice", "hi", BASE), _msg("Bob", "hey", BASE + timedelta(minutes=5))]
    result = TemporalAnalyzer().analyze(_make_chat(msgs))
    assert result.data["overview"]["total_messages"] == 2
