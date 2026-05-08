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
