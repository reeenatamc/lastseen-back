from datetime import datetime, timedelta

import pytest

from app.analyzers.conflict import ConflictAnalyzer
from app.analyzers.lexicons.rupture_es import match_categories
from app.parsers.base import ParsedChat, ParsedMessage

# ── Helpers ───────────────────────────────────────────────────────────────────

BASE = datetime(2026, 9, 1, 10, 0)  # a Tuesday


def _msg(
    sender: str,
    content: str,
    dt: datetime,
    is_media: bool = False,
    media_type: str | None = None,
) -> ParsedMessage:
    return ParsedMessage(
        timestamp=dt, sender=sender, content=content, is_media=is_media, media_type=media_type
    )


def _make_chat(msgs: list[ParsedMessage]) -> ParsedChat:
    msgs = sorted(msgs, key=lambda m: m.timestamp)
    return ParsedChat(
        platform="whatsapp",
        participants=sorted({m.sender for m in msgs}),
        messages=msgs,
    )


def _filler(days: int = 30, per_day: int = 4) -> list[ParsedMessage]:
    msgs = []
    for d in range(days):
        for i in range(per_day):
            sender = "Alice" if i % 2 == 0 else "Bob"
            msgs.append(_msg(sender, "todo bien por aca", BASE + timedelta(days=d, minutes=i)))
    return msgs


def _rupture(day: int, n: int, sender: str = "Alice") -> list[ParsedMessage]:
    return [
        _msg(sender, "terminemos ya", BASE + timedelta(days=day, hours=5, minutes=i))
        for i in range(n)
    ]


def _run(msgs: list[ParsedMessage]) -> dict:
    return ConflictAnalyzer().analyze(_make_chat(msgs)).data


# ── Lexicon ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,category", [
    ("terminamos", "breakup"),
    ("Terminemos esto", "breakup"),
    ("termíname si quieres", "breakup"),
    ("me terminaste ayer", "breakup"),
    ("no quiero terminarte", "breakup"),
    ("voy a terminar con vos", "breakup"),
    ("me kieres terminar?", "breakup"),
    ("me vas a dejar", "breakup"),
    ("kieres dejarme", "breakup"),
    ("ya no kiero nada", "breakup"),
    ("no kiero saber mas de ti", "breakup"),
    ("no quiero terminar", "breakup"),
    ("segura que quieres terminar amor?", "breakup"),
    ("no veo otra solución que terminar", "breakup"),
    ("para que no te sientas mal por dejarme de querer", "breakup"),
    ("buscate a alguien q te de todo", "breakup"),
    ("mejor te buscas a alguien", "breakup"),
    ("te voy a bloquear", "block_threat"),
    ("me bloqueaste", "block_threat"),
    ("no me bloquees", "block_threat"),
    ("te voy a blokear", "block_threat"),
    ("estoy blokeada", "block_threat"),
    ("podemos darnos un tiempo", "time_out"),
    ("dame tiempo", "time_out"),
    ("necesito tiempo", "time_out"),
    ("kiero un tiempo", "time_out"),
    ("pedí unos días", "time_out"),
    ("siento que necesitas tiempo", "time_out"),
    ("debemos darle un poco de tiempo a nuestras cosas", "time_out"),
    ("ese será el tiempo q nos demos", "time_out"),
    ("no me hables", "withdrawal"),
    ("ya no te voy a escribir", "withdrawal"),
    ("no te voy a contestar", "withdrawal"),
    ("déjame en paz", "withdrawal"),
    ("solo te voy a dejar de responder", "withdrawal"),
])
def test_lexicon_positive(text, category):
    assert category in match_categories(text)


@pytest.mark.parametrize("text", [
    "terminé el turno",
    "voy a terminar la tesis",
    "ya termino",
    "hola como estas",
    "nos vemos manana",
    "dame el cargador",
    "quiero terminar la tesis hoy",
    "kiero terminar de comer",
    "ahora q tienes tiempo",
    "cuando haya tiempo, no problem",
    "hace un tiempo que no vamos",
    "te voy a dejar de molestar",
    "perdóname, el bloqueo mental q m dio",
])
def test_lexicon_negative(text):
    assert match_categories(text) == set()


# ── Analyzer ──────────────────────────────────────────────────────────────────

def test_insufficient_data():
    msgs = [_msg("Alice", "hola", BASE + timedelta(minutes=i)) for i in range(49)]
    assert _run(msgs) == {"error": "insufficient_data"}


def test_media_does_not_count_toward_minimum():
    msgs = [_msg("Alice", "hola", BASE + timedelta(minutes=i)) for i in range(40)]
    msgs += [
        _msg("Bob", "image omitted", BASE + timedelta(hours=1, minutes=i), True, "image")
        for i in range(30)
    ]
    assert _run(msgs) == {"error": "insufficient_data"}


def test_no_conflict_chat_has_no_episodes():
    data = _run(_filler())
    assert data["lexicon"] == "es"
    assert data["episodes"] == []
    assert data["mentions"]["total"] == 0
    assert data["mentions"]["rate_per_1000"] == 0.0
    assert data["system_events"]["blocks"] == []


def test_three_mentions_in_a_day_is_medium_episode():
    data = _run(_filler() + _rupture(10, 3))
    assert len(data["episodes"]) == 1
    ep = data["episodes"][0]
    assert ep["start"] == ep["end"] == "2026-09-11"
    assert ep["mentions"] == 3
    assert ep["severity"] == "medium"
    assert ep["blocked"] is False
    assert ep["per_person"] == {"Alice": 3}
    assert ep["categories"] == {"breakup": 3}


def test_two_mentions_is_not_an_episode():
    assert _run(_filler() + _rupture(10, 2))["episodes"] == []


def _missed_calls(day: int, n: int, sender: str = "Bob") -> list[ParsedMessage]:
    return [
        _msg(
            sender,
            "Missed voice call. Tap to call back",
            BASE + timedelta(days=day, hours=6, minutes=i),
            True,
            "missed_call",
        )
        for i in range(n)
    ]


def test_burst_of_missed_calls_is_an_episode_without_any_rupture_words():
    data = _run(_filler() + _missed_calls(10, 8))
    assert len(data["episodes"]) == 1
    ep = data["episodes"][0]
    assert ep["mentions"] == 0
    assert ep["missed_calls"] == 8
    assert ep["severity"] == "medium"


def test_a_few_missed_calls_are_not_an_episode():
    assert _run(_filler() + _missed_calls(10, 7))["episodes"] == []


def test_block_event_makes_high_episode():
    block = _msg("Bob", "You blocked this contact", BASE + timedelta(days=12, hours=3), True, "block")
    unblock = _msg("Bob", "You unblocked this contact", BASE + timedelta(days=13), True, "unblock")
    data = _run(_filler() + [block, unblock])
    assert len(data["episodes"]) == 1
    ep = data["episodes"][0]
    assert ep["blocked"] is True
    assert ep["severity"] == "high"
    assert data["system_events"]["blocks"] == [block.timestamp.isoformat()]
    assert data["system_events"]["unblocks"] == [unblock.timestamp.isoformat()]


def test_high_severity_by_mentions():
    ep = _run(_filler() + _rupture(10, 8))["episodes"][0]
    assert ep["severity"] == "high"
    assert ep["blocked"] is False


def test_consecutive_days_merge_and_distant_days_do_not():
    msgs = _filler() + _rupture(5, 3) + _rupture(7, 3, "Bob") + _rupture(20, 3)
    msgs.append(_msg("Bob", "Missed voice call. Tap to call back",
                     BASE + timedelta(days=6), True, "missed_call"))
    eps = _run(msgs)["episodes"]
    assert len(eps) == 2
    assert (eps[0]["start"], eps[0]["end"]) == ("2026-09-06", "2026-09-08")
    assert eps[0]["mentions"] == 6
    assert eps[0]["per_person"] == {"Alice": 3, "Bob": 3}
    assert eps[0]["missed_calls"] == 1
    assert eps[1]["start"] == "2026-09-21"


def test_system_events_counts():
    msgs = _filler()
    msgs += [
        _msg("Alice", "You deleted this message.", BASE + timedelta(days=2, hours=2), True, "deleted"),
        _msg("Bob", "Missed voice call", BASE + timedelta(days=3, hours=2), True, "missed_call"),
        _msg("Bob", "Missed video call", BASE + timedelta(days=4, hours=2), True, "missed_call"),
    ]
    ev = _run(msgs)["system_events"]
    assert ev["deleted_messages"] == {"Alice": 1}
    assert ev["missed_calls"] == {"total": 2, "per_person": {"Bob": 2}}


def test_output_contains_no_message_text():
    secret = "terminemos ya xyzsecreto"
    msgs = _filler() + [
        _msg("Alice", secret, BASE + timedelta(days=10, hours=5, minutes=i)) for i in range(4)
    ]
    data = _run(msgs)
    assert data["mentions"]["total"] == 4
    dumped = repr(data)
    assert "xyzsecreto" not in dumped
    assert "terminemos" not in dumped


def test_recent_rate_against_baseline():
    # 5 quiet weeks, then rupture language concentrated in the last 3
    msgs = _filler(days=56, per_day=4)
    for day in (37, 38, 45, 46, 52, 53):
        msgs += _rupture(day, 1)
    data = _run(msgs)
    recent = data["recent"]
    assert recent["window_weeks"] == 3
    assert recent["baseline_rate_per_1000"] == 0.0
    assert recent["ratio"] is None
    assert recent["rate_per_1000"] > 0

    # A non-zero baseline yields a ratio
    msgs += _rupture(2, 1) + _rupture(9, 1) + _rupture(16, 1) + _rupture(23, 1) + _rupture(30, 1)
    recent = _run(msgs)["recent"]
    assert recent["baseline_rate_per_1000"] > 0
    assert recent["ratio"] is not None and recent["ratio"] > 0


def test_weekly_shape():
    weekly = _run(_filler(days=14) + _rupture(3, 1))["weekly"]
    assert weekly[0]["period"] == "2026-W36"
    assert weekly[0]["period_start"] == "2026-08-31"
    assert set(weekly[0]) == {"period", "period_start", "mentions", "rate_per_1000"}
