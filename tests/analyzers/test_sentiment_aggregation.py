from datetime import datetime

import pytest

from app.analyzers.sentiment import _per_person
from app.parsers.base import ParsedMessage


def _msg(sender: str) -> ParsedMessage:
    return ParsedMessage(timestamp=datetime(2024, 1, 1), sender=sender, content="hi")


def test_per_person_groups_scores_by_sender():
    result = _per_person([_msg("a"), _msg("b"), _msg("a")], [0.5, -1.0, 0.25])
    assert set(result) == {"a", "b"}


def test_per_person_rejects_scores_out_of_step_with_messages():
    # A model returning fewer scores than messages used to be truncated silently
    with pytest.raises(ValueError):
        _per_person([_msg("a"), _msg("b")], [0.5])
