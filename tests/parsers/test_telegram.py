import json

import pytest

from app.parsers.telegram import TelegramParser

SAMPLE = {
    "name": "Test Chat",
    "type": "personal_chat",
    "messages": [
        {"id": 1, "type": "message", "date": "2023-12-25T10:00:00", "from": "Alice", "text": "Hello!"},
        {"id": 2, "type": "message", "date": "2023-12-25T10:01:00", "from": "Bob", "text": "Hey"},
        {"id": 3, "type": "service", "date": "2023-12-25T10:02:00", "actor": "Alice", "text": "joined"},
    ],
}


def test_can_parse():
    parser = TelegramParser()
    assert parser.can_parse(json.dumps(SAMPLE))


def test_ignores_service_messages():
    parser = TelegramParser()
    chat = parser.parse(json.dumps(SAMPLE))
    assert chat.total_messages == 2


def test_metadata():
    parser = TelegramParser()
    chat = parser.parse(json.dumps(SAMPLE))
    assert chat.metadata["chat_name"] == "Test Chat"
