"""
Per-message hybrid sentiment scoring (Spanish, couple-chat register).

`hybrid_score()` is the public entry point. Given a raw text and the base
model's sentiment score, it returns a fused score in [-1, +1] that incorporates:

  • lexicon_score(text)   — NRC/AFINN Spanish (7.6k words).
  • intimate_score(text)  — hand-curated couple-chat vocabulary.
  • emoji_score(text)     — Emoji Sentiment Ranking 1.0.

Adaptive weights: each side-layer abstains (returns None) when it has no
signal for the message; the remaining components keep their proportional
weights. So a message with only the base model and no lexicon hits gets the
base model's score, unchanged. A message rich in intimate vocabulary and ❤️
gets all four layers fused.

The orthographic-emphasis multiplier from the intimate layer is applied to
the final fused score (not just to its own contribution) because emphasis
expresses the intensity of the whole utterance.
"""
from __future__ import annotations

import re
from typing import Iterable

from .intimate_es import _WORD_RE, intimate_score
from .loader import load_emoji_lexicon, load_spanish_lexicon

# Base weights when ALL layers fire. When a side-layer abstains its weight
# is removed and the remaining weights are renormalised.
#
# Calibration note: we deliberately give the intimate hand-curated layer a
# similar weight to the ML base. The ML model has a real but mild systematic
# bias against couple-chat affection (Twitter training set), and the intimate
# layer is the highest-precision signal for messages where it fires. When the
# intimate layer is silent, only the ML drives the score (no false uplift).
_W_BASE = 0.38        # pysentimiento (ML)
_W_LEXICON = 0.20     # NRC/AFINN ES dictionary
_W_INTIMATE = 0.32    # hand-curated couple-chat vocab
_W_EMOJI = 0.10       # emoji_sentiment ranking

# Emoji-sentiment data is from 2015; many post-2018 affect-heavy emojis
# (pleading face, hearts in new colors, etc.) aren't in the ranking. This
# small overlay fills the most common modern affect-heavy ones with valence
# values calibrated to be consistent with the existing 2015 entries.
# All values in [-1, +1]; positive emojis match the ranking's mean for the
# "hearts" cluster (~+0.7).
_EMOJI_OVERRIDES: dict[str, float] = {
    "🩷": 0.75,    # pink heart (U+1FA77, 2022)
    "🩵": 0.70,    # light blue heart
    "🩶": 0.55,    # grey heart (more melancholic)
    "🤍": 0.70,    # white heart
    "🖤": 0.30,    # black heart (used playfully in love chats too)
    "🤎": 0.55,    # brown heart
    "🧡": 0.70,    # orange heart
    "💗": 0.75,    # sparkling heart
    "💖": 0.80,    # heart with stars
    "💝": 0.75,    # heart with ribbon
    "💞": 0.75,    # revolving hearts
    "💕": 0.75,    # two hearts
    "💓": 0.70,    # beating heart
    "💘": 0.70,    # heart with arrow
    "💟": 0.65,    # heart decoration
    "💌": 0.65,    # love letter
    "🥺": 0.10,    # pleading face — used affectionately in couple chats
    "🥹": 0.30,    # face holding back tears (often happy-tearful in this register)
    "🫶": 0.65,    # heart hands
    "🫦": 0.20,    # biting lip — flirty
    "🥰": 0.85,    # smiling face with hearts
    "😍": 0.80,    # heart eyes (already in ranking actually, but reaffirm)
    "😘": 0.75,    # kiss face
    "😚": 0.60,    # kissing closed eyes
    "🤗": 0.55,    # hugging face
    "👩‍❤️‍👩": 0.80,    # couple-with-heart (women)
    "👨‍❤️‍👨": 0.80,    # couple-with-heart (men)
    "💑": 0.75,    # couple with heart
    "💏": 0.75,    # kiss
}


def lexicon_score(text: str) -> float | None:
    """
    Score a message against the merged Spanish lexicon (NRC + AFINN).

    Returns the mean valence of all matched tokens, in [-1, +1], or None when
    no lexicon token was matched (so the caller can drop this layer cleanly).
    """
    if not text or not text.strip():
        return None
    lex = load_spanish_lexicon()
    seen: set[str] = set()
    valences: list[float] = []
    for tok in _WORD_RE.findall(text.lower()):
        if tok in seen:
            continue
        seen.add(tok)
        entry = lex.get(tok)
        if entry is None or not entry.has_polarity:
            continue
        valences.append(entry.valence)
    if not valences:
        return None
    return round(sum(valences) / len(valences), 4)


def emoji_score(text: str) -> float | None:
    """
    Score the emojis in a message using Kralj Novak (2015) ranking, with a
    modern-emoji overlay (see _EMOJI_OVERRIDES) for emojis added after 2015.

    Returns the mean valence of matched emojis, in [-1, +1], or None when no
    emoji from either source is present in the message.
    """
    if not text:
        return None
    lex = load_emoji_lexicon()
    valences: list[float] = []
    # Walk the string char-by-char, skipping VS-16 (U+FE0F) so "❤️" matches "❤".
    for ch in text:
        if ch == "️":
            continue
        v = _EMOJI_OVERRIDES.get(ch)
        if v is None:
            v = lex.get(ch)
        if v is not None:
            valences.append(v)
    if not valences:
        return None
    return round(sum(valences) / len(valences), 4)


def hybrid_score(text: str, base_score: float) -> float:
    """
    Combine the base ML score with the side layers into a single final score
    in [-1, +1].

    `base_score` is whatever the model says (pysentimiento gives POS-NEG in
    [-1, +1]). All side layers are computed here; layers that abstain are
    dropped and weights renormalised.

    The orthographic-emphasis multiplier (e.g. "AMOOOR!!!") amplifies the
    *magnitude* of the final score by up to 1.4×, capped at ±1.0.
    """
    lex = lexicon_score(text)
    intim, emphasis = intimate_score(text)
    emo = emoji_score(text)

    components: list[tuple[float, float]] = [(base_score, _W_BASE)]
    if lex is not None:
        components.append((lex, _W_LEXICON))
    if intim is not None:
        components.append((intim, _W_INTIMATE))
    if emo is not None:
        components.append((emo, _W_EMOJI))

    total_w = sum(w for _, w in components)
    fused = sum(s * w for s, w in components) / total_w
    fused *= emphasis
    return round(max(-1.0, min(1.0, fused)), 4)


# Exposed for the analyzer's diagnostic / debug paths (not used in the hot
# loop, but useful for inspecting why a score was assigned).
def explain(text: str, base_score: float) -> dict:
    """Per-component breakdown for a single message (debugging only)."""
    lex = lexicon_score(text)
    intim, emphasis = intimate_score(text)
    emo = emoji_score(text)
    final = hybrid_score(text, base_score)
    return {
        "text": text[:120],
        "base_pysent": round(base_score, 4),
        "lexicon": lex,
        "intimate": intim,
        "emoji": emo,
        "emphasis_mult": round(emphasis, 3),
        "final_hybrid": final,
    }
