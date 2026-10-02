"""LLM sentiment scorer tests: a fake `generate` replaces Gemini, no network."""
import json

import pytest

from app.analyzers import sentiment_llm
from app.analyzers.sentiment_llm import (
    SentimentLLMError,
    _build_prompt,
    _parse_batch,
    score_messages,
)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(sentiment_llm, "_RETRY_BASE_DELAY", 0.0)


def _raw(*items: dict) -> str:
    return json.dumps(list(items))


def _batch_from_prompt(prompt: str) -> list[dict]:
    """The JSON list the model receives sits after the last blank line of the prompt."""
    return json.loads(prompt.rsplit("\n\n", 1)[1])


def _score_by_text(prompt: str) -> str:
    """Fake model: score = number in the text / 1000, emotion joy."""
    items = _batch_from_prompt(prompt)
    return json.dumps([
        {"i": it["i"], "s": int(it["t"].split("-")[1]) / 1000, "e": "joy"} for it in items
    ])


# ── _parse_batch ──────────────────────────────────────────────────────────────

def test_parse_valid():
    scores, emotions, missing = _parse_batch(
        _raw({"i": 0, "s": 0.5, "e": "joy"}, {"i": 1, "s": -0.25, "e": "anger"}), 2
    )
    assert scores == [0.5, -0.25]
    assert emotions == ["joy", "anger"]
    assert missing == 0


def test_parse_repeated_index_first_wins():
    scores, emotions, missing = _parse_batch(
        _raw({"i": 0, "s": 0.5, "e": "joy"}, {"i": 0, "s": -0.9, "e": "anger"}), 1
    )
    assert (scores, emotions, missing) == ([0.5], ["joy"], 0)


def test_parse_out_of_range_dropped():
    scores, _, missing = _parse_batch(
        _raw({"i": 5, "s": 0.5, "e": "joy"}, {"i": -1, "s": 0.5, "e": "joy"}), 2
    )
    assert scores == [0.0, 0.0]
    assert missing == 2


def test_parse_missing_default_to_neutral():
    scores, emotions, missing = _parse_batch(_raw({"i": 1, "s": 0.4, "e": "joy"}), 3)
    assert scores == [0.0, 0.4, 0.0]
    assert emotions == ["others", "joy", "others"]
    assert missing == 2


def test_parse_scores_clamped():
    scores, _, _ = _parse_batch(
        _raw({"i": 0, "s": 7, "e": "joy"}, {"i": 1, "s": -3.2, "e": "sadness"}), 2
    )
    assert scores == [1.0, -1.0]


def test_parse_unknown_emotion_becomes_others():
    _, emotions, missing = _parse_batch(
        _raw({"i": 0, "s": 0.1, "e": "love"}, {"i": 1, "s": 0.1}), 2
    )
    assert emotions == ["others", "others"]
    assert missing == 0


def test_parse_non_numeric_score_counts_as_missing():
    scores, _, missing = _parse_batch(_raw({"i": 0, "s": "high", "e": "joy"}), 1)
    assert scores == [0.0]
    assert missing == 1


def test_parse_broken_json_raises():
    with pytest.raises(ValueError):
        _parse_batch("{not json", 2)


def test_parse_non_list_raises():
    with pytest.raises(ValueError):
        _parse_batch('{"i": 0}', 2)


# ── prompt ────────────────────────────────────────────────────────────────────

def test_prompt_redacts_truncates_and_has_only_index_and_text():
    texts = ["llámame al 099 123 4567 o a@b.com", "x" * 1000]
    items = _batch_from_prompt(_build_prompt(texts))
    assert items[0] == {"i": 0, "t": "llámame al [number] o [email]"}
    assert set(items[1]) == {"i", "t"}
    assert len(items[1]["t"]) == 300


# ── score_messages ────────────────────────────────────────────────────────────

def test_order_kept_across_batches(monkeypatch):
    monkeypatch.setattr(sentiment_llm, "_BATCH_SIZE", 10)
    texts = [f"m-{i}" for i in range(35)]
    scores, emotions = score_messages(texts, api_key="k", model="m", generate=_score_by_text)
    assert scores == [i / 1000 for i in range(35)]
    assert emotions == ["joy"] * 35


def test_prompt_received_is_redacted_and_has_no_names():
    seen: list[str] = []

    def fake(prompt: str) -> str:
        seen.append(prompt)
        return _raw({"i": 0, "s": 0.1, "e": "joy"}, {"i": 1, "s": 0.0, "e": "others"})

    texts = ["mi cédula es 1712345678", "escribe a ana@mail.com o mira www.x.com"]
    score_messages(texts, api_key="k", model="m", generate=fake)
    assert len(seen) == 1
    payload = seen[0]
    for leaked in ("1712345678", "ana@mail.com", "www.x.com"):
        assert leaked not in payload
    for marker in ("[number]", "[email]", "[link]"):
        assert marker in payload


def test_retry_then_success():
    calls = {"n": 0}

    def flaky(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("503")
        return _raw({"i": 0, "s": 0.6, "e": "joy"})

    scores, _ = score_messages(["hola"], api_key="k", model="m", generate=flaky)
    assert scores == [0.6]
    assert calls["n"] == 3


def test_broken_json_is_retried():
    answers = iter(["oops", _raw({"i": 0, "s": 0.2, "e": "joy"})])
    scores, _ = score_messages(
        ["hola"], api_key="k", model="m", generate=lambda p: next(answers)
    )
    assert scores == [0.2]


def test_total_failure_raises():
    calls = {"n": 0}

    def broken(prompt: str) -> str:
        calls["n"] += 1
        raise RuntimeError("quota")

    with pytest.raises(SentimentLLMError):
        score_messages(["hola"], api_key="k", model="m", generate=broken)
    assert calls["n"] == 3  # first try + _MAX_RETRIES


def test_too_many_missing_raises():
    def partial(prompt: str) -> str:
        return _raw({"i": 0, "s": 0.5, "e": "joy"})  # 1 of 5 scored

    with pytest.raises(SentimentLLMError):
        score_messages([f"m{i}" for i in range(5)], api_key="k", model="m", generate=partial)


def test_few_missing_tolerated():
    def nearly_all(prompt: str) -> str:
        return _raw(*({"i": i, "s": 0.5, "e": "joy"} for i in range(19)))  # 1 of 20 missing

    scores, emotions = score_messages(
        [f"m{i}" for i in range(20)], api_key="k", model="m", generate=nearly_all
    )
    assert scores[19] == 0.0 and emotions[19] == "others"


def test_empty_input():
    assert score_messages([], api_key="k", model="m", generate=lambda p: "") == ([], [])


def test_prepare_bounds_input_before_redacting():
    import time

    from app.analyzers import sentiment_llm

    hostile = ("a@" * 20 + "www." * 20) * 2000
    t0 = time.perf_counter()
    out = sentiment_llm._prepare(hostile)
    assert time.perf_counter() - t0 < 0.5
    assert len(out) <= sentiment_llm._MAX_CHARS
