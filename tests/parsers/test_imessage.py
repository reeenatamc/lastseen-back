import time

import pytest

from app.parsers.base import MAX_MESSAGE_CHARS
from app.parsers.imessage import IMessageParser

HEAD = "[2026-01-09 10:00:00] Ana: start\n"


def test_continuation_lines_are_joined():
    chat = IMessageParser().parse(HEAD + "second\nthird\n[2026-01-09 10:01:00] Beto: ok")
    assert chat.messages[0].content == "start\nsecond\nthird"
    assert len(chat.messages) == 2


def test_many_continuation_lines_parse_fast():
    raw = HEAD + "continuation line\n" * 200_000
    t0 = time.perf_counter()
    chat = IMessageParser().parse(raw)
    assert time.perf_counter() - t0 < 2
    assert len(chat.messages[0].content) == MAX_MESSAGE_CHARS


def test_too_many_messages_raises(monkeypatch):
    import app.parsers.base as base

    monkeypatch.setattr(base, "MAX_MESSAGES", 2)
    raw = "\n".join(f"[2026-01-09 10:0{i}:00] Ana: m{i}" for i in range(4))
    with pytest.raises(ValueError, match="chat_too_large"):
        IMessageParser().parse(raw)
