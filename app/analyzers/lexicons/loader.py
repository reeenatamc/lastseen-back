"""
Lazy, cached loaders for the bundled lexicon CSVs.

The CSVs ship inside this package. Both loaders parse once on first call and
return module-level singletons; subsequent calls are O(1).

Files
-----
- lexicon_es.csv   : Spanish merged lexicon (NRC translation + AFINN translation).
                      Columns: word, valence, positive, negative, joy, sadness,
                      anger, fear, trust, anticipation, surprise, disgust.
                      Sources: see CITATIONS.md.
- emoji_sentiment.csv : Emoji Sentiment Ranking 1.0 (Kralj Novak et al. 2015).
                      Columns: Emoji, Unicode codepoint, Occurrences, Position,
                      Negative, Neutral, Positive, Unicode name, Unicode block.
                      CC BY-SA 4.0.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

_LEXICON_DIR = Path(__file__).parent


# ── Spanish word lexicon (NRC ES + AFINN ES merged) ─────────────────────────

@dataclass(frozen=True)
class LexEntry:
    """One row from the merged Spanish lexicon."""
    valence: float            # in [-1, +1]
    positive: bool
    negative: bool
    joy: bool
    sadness: bool
    anger: bool
    fear: bool
    trust: bool
    anticipation: bool
    surprise: bool
    disgust: bool

    @property
    def has_polarity(self) -> bool:
        """True if the row carries any polarity signal at all."""
        return self.valence != 0.0 or self.positive or self.negative


_spanish_lexicon: dict[str, LexEntry] | None = None


def load_spanish_lexicon() -> dict[str, LexEntry]:
    """
    Return the merged Spanish word-level lexicon. Lazily loaded on first call.
    Keys are lowercased words (multi-word entries kept verbatim).
    """
    global _spanish_lexicon
    if _spanish_lexicon is not None:
        return _spanish_lexicon

    path = _LEXICON_DIR / "lexicon_es.csv"
    out: dict[str, LexEntry] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            word = row["word"].strip().lower()
            if not word:
                continue
            out[word] = LexEntry(
                valence=float(row["valence"]),
                positive=row["positive"] == "1",
                negative=row["negative"] == "1",
                joy=row["joy"] == "1",
                sadness=row["sadness"] == "1",
                anger=row["anger"] == "1",
                fear=row["fear"] == "1",
                trust=row["trust"] == "1",
                anticipation=row["anticipation"] == "1",
                surprise=row["surprise"] == "1",
                disgust=row["disgust"] == "1",
            )
    _spanish_lexicon = out
    return out


# ── Emoji sentiment (Kralj Novak et al. 2015) ──────────────────────────────

_emoji_lexicon: dict[str, float] | None = None


def load_emoji_lexicon() -> dict[str, float]:
    """
    Return {emoji_char: valence in [-1, +1]} from Emoji Sentiment Ranking 1.0.

    Valence formula: (positive_count - negative_count) / total_count.
    """
    global _emoji_lexicon
    if _emoji_lexicon is not None:
        return _emoji_lexicon

    path = _LEXICON_DIR / "emoji_sentiment.csv"
    out: dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            emoji = row["Emoji"]
            try:
                neg = int(row["Negative"])
                pos = int(row["Positive"])
                total = int(row["Occurrences"])
            except (ValueError, KeyError):
                continue
            if not emoji or total <= 0:
                continue
            out[emoji] = round((pos - neg) / total, 4)
    _emoji_lexicon = out
    return out
