from datetime import datetime, timedelta

import pytest

from app.analyzers.temporal import (
    TemporalAnalyzer,
    _initiative_balance,
    _message_length,
)
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
    20 blocks, 2-day gaps. Alice opens, Bob closes (opener ≠ last_speaker).
    Under the new initiative rule Alice gets genuine initiative on every block:
    each block closes with Bob → next Alice open ≠ Bob close, opener(Alice) ≠ last(Bob)
    → genuine initiative. No late replies, no double texts.
    Healthy dynamic: consistent leader (Alice) + reliable responder (Bob).
    """
    msgs = []
    for block in range(20):
        t = BASE + timedelta(days=block * 2)
        msgs += [
            _msg("Alice", "hey there!",       t),
            _msg("Bob",   "hello!",            t + timedelta(minutes=5)),
            _msg("Alice", "how was your day?", t + timedelta(minutes=10)),
            _msg("Bob",   "was really good!",  t + timedelta(minutes=15)),
        ]
    return _make_chat(msgs)


def decayed_chat() -> ParsedChat:
    """
    30 blocks separated by 2-day gaps.
    First 15: symmetric and quick — sets a healthy baseline.
    Last 15: Alice reaches out (twice, into the void) and Bob ghosts for
             hours, growing worse (4h → 11h). This is decay as the metric
             now defines it — one-sided multi-hour abandonment, not a mild
             slowdown — so neglect_score, not avg response time, is what
             drives the drop.
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
            # Bob's ghost grows from 5h to 12h; the handoff that counts is
            # Alice's last reach-out → Bob's reply (≈ ghost − 1h, always > 3h).
            ghost_hours = 5 + (block - 15) * 0.5
            msgs += [
                _msg("Alice", "hey", t),
                _msg("Alice", "you there?", t + timedelta(hours=1)),
                _msg("Bob", "k", t + timedelta(hours=ghost_hours)),
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
    ib = result.data["initiative_balance"]
    # Structure is present
    assert "per_person" in ib
    assert "abandoned_open" in ib
    assert "late_reply" in ib
    assert "double_text" in ib
    # No double texts in a healthy chat
    assert ib["double_text"]["total"] == 0


def test_initiative_balance_decayed():
    result = TemporalAnalyzer().analyze(decayed_chat())
    ib = result.data["initiative_balance"]
    # Alice has more initiative OR more abandoned_opens than Bob —
    # she carries more of the conversational effort
    alice_effort = (
        ib["per_person"].get("Alice", 0)
        + ib["abandoned_open"]["per_person"].get("Alice", 0)
    )
    bob_effort = (
        ib["per_person"].get("Bob", 0)
        + ib["abandoned_open"]["per_person"].get("Bob", 0)
    )
    assert alice_effort > bob_effort


def test_initiative_confidence_low_for_continuous_thread():
    # 40 messages 10 min apart: the 95th-pct gap is well below the 1 h block
    # floor, so "conversations" are an arbitrary cut → low confidence.
    msgs = [
        _msg("Alice" if i % 2 == 0 else "Bob", "x", BASE + timedelta(minutes=10 * i))
        for i in range(40)
    ]
    ib = _initiative_balance(_make_chat(msgs).messages)
    assert ib["confidence"]["level"] == "low"
    assert ib["confidence"]["reason"] == "continuous_thread"


def test_initiative_confidence_ok_for_spaced_chat():
    ib = _initiative_balance(healthy_chat().messages)
    assert ib["confidence"]["level"] == "ok"
    assert ib["confidence"]["reason"] is None


def test_initiative_structure_has_late_reply():
    result = TemporalAnalyzer().analyze(healthy_chat())
    ib = result.data["initiative_balance"]
    assert "late_reply" in ib
    assert "per_person" in ib["late_reply"]
    assert "share" in ib["late_reply"]
    assert "total" in ib["late_reply"]


def test_initiative_structure_has_double_text():
    result = TemporalAnalyzer().analyze(healthy_chat())
    dt = result.data["initiative_balance"]["double_text"]
    assert "per_person" in dt
    assert "share" in dt
    assert "total" in dt


def test_late_reply_not_counted_as_initiative():
    """
    Block opened AND closed by Alice (her message went unanswered).
    Bob responding after a 5h gap is a LATE REPLY, not an initiative.
    """
    base = datetime(2024, 1, 1, 10, 0)
    msgs = [
        # Block 1: Alice opens, Alice closes (Bob never responded)
        _msg("Alice", "hey",          base),
        _msg("Alice", "estas ahi?",   base + timedelta(minutes=5)),
        # Block 2: Bob finally responds (late reply)
        _msg("Bob",   "perdon",       base + timedelta(hours=6)),
        _msg("Alice", "ok",           base + timedelta(hours=6, minutes=2)),
    ]
    ib = _initiative_balance(_make_chat(msgs).messages)
    assert ib["abandoned_open"]["per_person"].get("Alice", 0) >= 1
    assert ib["late_reply"]["per_person"].get("Bob", 0) >= 1
    assert ib["per_person"].get("Bob", 0) == 0
    assert ib["per_person"].get("Alice", 0) == 0


def test_genuine_initiative_requires_natural_prior_and_last_word():
    """
    Alice opens block 2 after a natural exchange in block 1.
    For it to count as genuine initiative, Alice must also have the last
    word in block 2 (stayed until the end).
    """
    base = datetime(2024, 1, 1, 10, 0)

    # Block 1: Alice opens, Bob closes (natural exchange, opener ≠ last)
    # Block 2: Alice opens AND Alice closes → genuine sustained initiative
    msgs_sustained = [
        _msg("Alice", "hey",       base),
        _msg("Bob",   "hola",      base + timedelta(minutes=5)),
        _msg("Alice", "que tal?",  base + timedelta(minutes=10)),
        _msg("Bob",   "bien",      base + timedelta(minutes=15)),
        _msg("Alice", "sigues?",   base + timedelta(hours=6)),
        _msg("Bob",   "si",        base + timedelta(hours=6, minutes=3)),
        _msg("Alice", "buenoo",    base + timedelta(hours=6, minutes=5)),  # Alice last
    ]
    # Call pure function directly — bypasses the MIN_MESSAGES threshold
    ib = _initiative_balance(_make_chat(msgs_sustained).messages)
    assert ib["per_person"].get("Alice", 0) >= 1

    msgs_abandoned = [
        _msg("Alice", "hey",       base),
        _msg("Bob",   "hola",      base + timedelta(minutes=5)),
        _msg("Alice", "que tal?",  base + timedelta(minutes=10)),
        _msg("Bob",   "bien",      base + timedelta(minutes=15)),
        _msg("Alice", "sigues?",   base + timedelta(hours=6)),
        _msg("Bob",   "si",        base + timedelta(hours=6, minutes=3)),
    ]
    ib2 = _initiative_balance(_make_chat(msgs_abandoned).messages)
    assert ib2["abandoned_open"]["per_person"].get("Alice", 0) >= 1


def test_double_text_detected():
    base = datetime(2024, 1, 1, 10, 0)
    msgs = [
        _msg("Bob",   "hi",         base),
        _msg("Alice", "hey",        base + timedelta(minutes=1)),
        _msg("Alice", "sigo aqui?", base + timedelta(hours=5)),
        _msg("Bob",   "sorry",      base + timedelta(hours=5, minutes=10)),
    ]
    ib = _initiative_balance(_make_chat(msgs).messages)
    assert ib["double_text"]["per_person"].get("Alice", 0) >= 1


def test_initiative_sustained_requires_last_word():
    base = datetime(2024, 1, 1, 10, 0)

    abandoned = [
        _msg("Alice", "hey",     base),
        _msg("Bob",   "hola",    base + timedelta(minutes=5)),
        _msg("Alice", "que tal", base + timedelta(minutes=10)),
        _msg("Bob",   "bien",    base + timedelta(minutes=15)),
        _msg("Bob",   "extra",   base + timedelta(hours=6)),
    ]
    ib = _initiative_balance(_make_chat(abandoned).messages)
    assert ib["abandoned_open"]["per_person"].get("Alice", 0) >= 1

    sustained = [
        _msg("Alice", "hey",     base),
        _msg("Bob",   "hola",    base + timedelta(minutes=5)),
        _msg("Bob",   "que tal", base + timedelta(minutes=8)),
        _msg("Alice", "bien",    base + timedelta(minutes=12)),
        _msg("Bob",   "extra",   base + timedelta(hours=6)),
    ]
    ib2 = _initiative_balance(_make_chat(sustained).messages)
    assert ib2["per_person"].get("Alice", 0) >= 1


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
    # Call pure function directly — bypasses MIN_MESSAGES threshold
    length = _message_length(_make_chat(msgs).messages)["per_person"]
    assert length["Alice"]["mean_chars"] == 3
    assert length["Bob"]["mean_chars"] == 18


# ── Response decay ────────────────────────────────────────────────────────────

def test_decay_healthy_trend():
    result = TemporalAnalyzer().analyze(healthy_chat())
    assert result.data["response_decay"]["trend"] in ("stable", "improving")


def test_decay_healthy_score():
    result = TemporalAnalyzer().analyze(healthy_chat())
    # Alice holds 100% initiative in the redesigned fixture, which adds slight
    # imbalance to the decay formula — threshold relaxed accordingly.
    assert result.data["response_decay"]["decay_score"] < 0.6


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


# ── Closing phase ─────────────────────────────────────────────────────────────

def _daily_chat(
    weeks: int,
    per_day=lambda week: 8,
    skip_days: frozenset = frozenset(),
    last_day: int | None = None,
    first_day: int = 0,
) -> ParsedChat:
    """
    Messages every day from `first_day` to `last_day` (default: end of `weeks`),
    `per_day(week)` messages 10 min apart, alternating senders. BASE is a Monday,
    so day // 7 is the ISO week index.
    """
    last = weeks * 7 - 1 if last_day is None else last_day
    msgs = []
    for day in range(first_day, last + 1):
        if day in skip_days:
            continue
        t = BASE + timedelta(days=day)
        for k in range(per_day(day // 7)):
            sender = "Alice" if (day + k) % 2 == 0 else "Bob"
            msgs.append(_msg(sender, "hola", t + timedelta(minutes=10 * k)))
    return _make_chat(msgs)


def _evo(mpd: list[float], silence: list[float | None] | None = None) -> list[dict]:
    """Synthetic weekly evolution entries (only the fields _closing_phase reads)."""
    silence = silence if silence is not None else [0.5] * len(mpd)
    return [
        {
            "period_start": (BASE + timedelta(weeks=i)).strftime("%Y-%m-%d"),
            "messages_per_day": v,
            "silence_gap_days": s,
        }
        for i, (v, s) in enumerate(zip(mpd, silence))
    ]


def test_closing_constant_volume_not_detected():
    rd = TemporalAnalyzer().analyze(_daily_chat(12)).data["response_decay"]
    assert rd["closing_phase"]["detected"] is False
    assert rd["closing_phase"]["start"] is None
    assert rd["trend"] == "stable"
    assert rd["trend_basis"] == "thirds"


def test_closing_volume_drop_detected():
    chat = _daily_chat(12, per_day=lambda w: 2 if w >= 9 else 8)
    rd = TemporalAnalyzer().analyze(chat).data["response_decay"]
    cp = rd["closing_phase"]
    assert cp["detected"] is True
    assert cp["volume_ratio"] == pytest.approx(0.25, abs=0.01)
    assert rd["trend"] == "deteriorating"
    assert rd["trend_basis"] == "closing_phase"
    window_start = (BASE + timedelta(weeks=9)).strftime("%Y-%m-%d")
    assert rd["turning_point"] is not None
    assert window_start <= rd["turning_point"] <= rd["evolution"][-1]["period_start"]


def test_closing_detected_by_silence():
    # Same daily volume, but Mon-Wed of the penultimate week are silent.
    chat = _daily_chat(12, skip_days=frozenset({70, 71, 72}))
    cp = TemporalAnalyzer().analyze(chat).data["response_decay"]["closing_phase"]
    assert cp["detected"] is True
    assert cp["volume_ratio"] > 0.5
    assert cp["max_silence_days"] >= 3.0


def test_closing_truncated_last_week_not_detected():
    # Export cut on the Tuesday of the last week: same daily rate, fewer days.
    chat = _daily_chat(12, last_day=78)
    rd = TemporalAnalyzer().analyze(chat).data["response_decay"]
    assert rd["evolution"][-1]["message_count"] == 16
    assert rd["closing_phase"]["detected"] is False
    assert rd["trend_basis"] == "thirds"


def test_closing_short_chat_insufficient_baseline():
    rd = TemporalAnalyzer().analyze(_daily_chat(6)).data["response_decay"]
    assert rd["closing_phase"] == {"detected": False, "reason": "insufficient_baseline"}


def test_messages_per_day_partial_weeks():
    # Starts Wednesday (5 covered days), ends Tuesday (2 covered days).
    rd = TemporalAnalyzer().analyze(_daily_chat(4, first_day=2, last_day=29)).data["response_decay"]
    evo = rd["evolution"]
    assert evo[0]["message_count"] == 40 and evo[0]["messages_per_day"] == 8.0
    assert evo[1]["messages_per_day"] == 8.0
    assert evo[-1]["message_count"] == 16 and evo[-1]["messages_per_day"] == 8.0


def test_closing_phase_insufficient_boundary():
    from app.analyzers.temporal import _closing_phase

    assert _closing_phase(_evo([10] * 6))["reason"] == "insufficient_baseline"
    assert "reason" not in _closing_phase(_evo([10] * 7))


def test_closing_phase_volume_ratio_boundary():
    from app.analyzers.temporal import _closing_phase

    at = _closing_phase(_evo([10] * 4 + [5, 5, 5]))
    assert at["detected"] is True and at["volume_ratio"] == 0.5
    above = _closing_phase(_evo([10] * 4 + [5.1, 5.1, 5.1]))
    assert above["detected"] is False and above["start"] is None


def test_closing_phase_silence_min_days_boundary():
    from app.analyzers.temporal import _closing_phase

    base = [0.5] * 4
    assert _closing_phase(_evo([10] * 7, base + [0.5, 0.5, 1.5]))["detected"] is True
    assert _closing_phase(_evo([10] * 7, base + [0.5, 0.5, 1.4]))["detected"] is False


def test_closing_phase_silence_factor_boundary():
    from app.analyzers.temporal import _closing_phase

    base = [1.0] * 4
    assert _closing_phase(_evo([10] * 7, base + [0.5, 0.5, 2.0]))["detected"] is True
    assert _closing_phase(_evo([10] * 7, base + [0.5, 0.5, 1.9]))["detected"] is False


def test_closing_phase_none_silence_treated_as_zero():
    from app.analyzers.temporal import _closing_phase

    cp = _closing_phase(_evo([10] * 7, [None] * 4 + [None, None, 1.5]))
    assert cp["detected"] is True
    assert cp["baseline_max_silence_days"] == 0.0
    assert cp["max_silence_days"] == 1.5


def test_closing_phase_start_ratio_boundary():
    from app.analyzers.temporal import _closing_phase

    # Window [8, 7, 0]: mean 5 -> ratio 0.5. Week 1 (8) is above 0.7*base, week 2 (7) is at it.
    cp = _closing_phase(_evo([10] * 4 + [8, 7, 0]))
    assert cp["detected"] is True
    assert cp["start"] == (BASE + timedelta(weeks=5)).strftime("%Y-%m-%d")
    assert cp["window_weeks"] == 3


def test_closing_phase_start_by_silence_week():
    from app.analyzers.temporal import _closing_phase

    cp = _closing_phase(_evo([10] * 7, [0.5] * 4 + [0.5, 2.0, 0.5]))
    assert cp["detected"] is True
    assert cp["start"] == (BASE + timedelta(weeks=5)).strftime("%Y-%m-%d")


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_single_message_returns_error():
    chat = _make_chat([_msg("Alice", "hi", BASE)])
    assert TemporalAnalyzer().analyze(chat).data.get("error") == "insufficient_data"


def test_two_messages_returns_insufficient():
    """2 messages is far below the minimum threshold — must return an error."""
    msgs = [_msg("Alice", "hi", BASE), _msg("Bob", "hey", BASE + timedelta(minutes=5))]
    result = TemporalAnalyzer().analyze(_make_chat(msgs))
    assert result.data.get("error") == "insufficient_data"
