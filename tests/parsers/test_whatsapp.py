import pytest

from app.parsers.whatsapp import WhatsAppParser

SAMPLE = """\
12/25/2023, 10:00 - Alice: Hello!
12/25/2023, 10:01 - Bob: Hey there
12/25/2023, 10:02 - Alice: How are you?
"""


def test_can_parse():
    parser = WhatsAppParser()
    assert parser.can_parse(SAMPLE)


def test_parse_message_count():
    parser = WhatsAppParser()
    chat = parser.parse(SAMPLE)
    assert chat.total_messages == 3


def test_parse_participants():
    parser = WhatsAppParser()
    chat = parser.parse(SAMPLE)
    assert set(chat.participants) == {"Alice", "Bob"}


def test_parse_sender_and_content():
    parser = WhatsAppParser()
    chat = parser.parse(SAMPLE)
    assert chat.messages[0].sender == "Alice"
    assert chat.messages[0].content == "Hello!"


# ── Service lines ─────────────────────────────────────────────────────────────

def _one(content: str):
    raw = f"[20/9/26, 10:17:05 PM] are: ‎{content}\n"
    chat = WhatsAppParser().parse(raw)
    assert chat.total_messages == 1
    return chat.messages[0]


@pytest.mark.parametrize("content,media_type", [
    ("<Media omitted>", "media"),
    ("image omitted", "image"),
    ("video omitted", "video"),
    ("audio omitted", "audio"),
    ("sticker omitted", "sticker"),
    ("GIF omitted", "gif"),
    ("document omitted", "document"),
    ("video note omitted", "media"),
    ("contact card omitted", "media"),
    ("Voice call. 16 min", "call"),
    ("Video call. 3 sec", "call"),
    ("Voice call. 1 hr", "call"),
    ("Llamada de voz. 5 min", "call"),
    ("Videollamada. 2 min", "call"),
    ("Missed voice call. Tap to call back", "missed_call"),
    ("Missed video call. Tap to call back", "missed_call"),
    ("Voice call. No answer", "missed_call"),
    ("Video call. No answer", "missed_call"),
    ("Llamada de voz perdida", "missed_call"),
    ("Videollamada perdida", "missed_call"),
    ("Llamada perdida", "missed_call"),
    ("Llamada de voz. Sin respuesta", "missed_call"),
    ("You deleted this message.", "deleted"),
    ("This message was deleted.", "deleted"),
    ("Eliminaste este mensaje.", "deleted"),
    ("Se eliminó este mensaje.", "deleted"),
    ("You blocked this person", "block"),
    ("You blocked this contact", "block"),
    ("Bloqueaste a este contacto", "block"),
    ("Bloqueaste a esta persona", "block"),
    ("You unblocked this person", "unblock"),
    ("Desbloqueaste a este contacto", "unblock"),
    ("Desbloqueaste a esta persona", "unblock"),
])
def test_service_line_classification(content, media_type):
    msg = _one(content)
    assert msg.is_media is True
    assert msg.media_type == media_type


def test_edited_suffix_is_stripped_and_stays_text():
    for suffix in ("<This message was edited>", "<Se editó este mensaje.>"):
        msg = _one(f"see you at nine {suffix}")
        assert msg.content == "see you at nine"
        assert msg.is_media is False
        assert msg.media_type is None


def test_normal_messages_untouched():
    for text in ("Hello!", "I missed your call yesterday", "Voice call me later"):
        msg = _one(text)
        assert msg.content == text
        assert msg.is_media is False
        assert msg.media_type is None


# ── Limits ────────────────────────────────────────────────────────────────────

def test_many_continuation_lines_parse_fast():
    import time

    raw = "[1/9/26, 10:00:00] Ana: start\n" + "continuation line\n" * 200_000
    t0 = time.perf_counter()
    chat = WhatsAppParser().parse(raw)
    assert time.perf_counter() - t0 < 2
    assert len(chat.messages) == 1


def test_long_message_is_truncated():
    from app.parsers.base import MAX_MESSAGE_CHARS

    raw = "[1/9/26, 10:00:00] Ana: " + "x" * 10_000 + "\n" + "y" * 10_000
    chat = WhatsAppParser().parse(raw)
    assert len(chat.messages[0].content) == MAX_MESSAGE_CHARS


def test_too_many_messages_raises(monkeypatch):
    import app.parsers.base as base

    monkeypatch.setattr(base, "MAX_MESSAGES", 3)
    raw = "\n".join(f"[1/9/26, 10:0{i}:00] Ana: m{i}" for i in range(5))
    with pytest.raises(ValueError, match="chat_too_large"):
        WhatsAppParser().parse(raw)
