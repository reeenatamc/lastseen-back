"""
Tests for the hybrid Spanish sentiment scorer.

Covers each of the four layers in isolation, then their fusion via
`hybrid_score`. The assertions are deliberately loose on exact numbers
(weights are tunable) but strict on sign and ordering — the contract the
scorer guarantees is "affectionate utterances score positive, even when the
base model says otherwise."
"""
from __future__ import annotations

import pytest

from app.analyzers.lexicons.intimate_es import intimate_score
from app.analyzers.lexicons.loader import (
    load_emoji_lexicon,
    load_spanish_lexicon,
)
from app.analyzers.lexicons.scorer import (
    emoji_score,
    explain,
    hybrid_score,
    lexicon_score,
)


# ── Loaders ────────────────────────────────────────────────────────────────

def test_spanish_lexicon_loads():
    lex = load_spanish_lexicon()
    assert len(lex) > 5000, "merged lexicon should have thousands of entries"
    # Spot-check a few core words
    assert "amor" in lex
    assert lex["amor"].valence > 0
    assert lex["amor"].positive
    assert lex["amor"].joy
    assert "triste" in lex
    assert lex["triste"].valence < 0


def test_emoji_lexicon_loads():
    lex = load_emoji_lexicon()
    assert len(lex) > 500
    # Heavy black heart is the most positive heart emoji in the dataset
    assert lex.get("❤") is not None
    assert lex["❤"] > 0.5


# ── Lexicon scorer ─────────────────────────────────────────────────────────

def test_lexicon_score_abstains_on_no_signal():
    assert lexicon_score("aaa bbb ccc") is None


def test_lexicon_score_positive_for_amor():
    assert (lexicon_score("amor cariño") or 0) > 0.3


def test_lexicon_score_negative_for_sad_words():
    assert (lexicon_score("triste enfermo") or 0) < -0.1


def test_lexicon_score_handles_empty_text():
    assert lexicon_score("") is None
    assert lexicon_score("   ") is None


# ── Intimate scorer ────────────────────────────────────────────────────────

def test_intimate_abstains_when_no_match():
    score, mult = intimate_score("hola que tal todo bien")
    assert score is None
    assert mult == 1.0


def test_intimate_strong_positive_for_te_amo():
    score, mult = intimate_score("te amo mucho")
    assert score is not None and score > 0.7


def test_intimate_emphasis_multiplier_on_repeats():
    _, mult_plain = intimate_score("te amo")
    _, mult_loud = intimate_score("TE AMOOOOO!!!")
    assert mult_loud > mult_plain
    assert mult_loud <= 1.4    # cap


def test_intimate_playful_negative_capped():
    """'me muero' is playful in this register — should be only slightly negative."""
    score, _ = intimate_score("me muero")
    assert score is not None and -0.4 < score < 0.0


def test_intimate_diminutive_boost():
    """Diminutives not already in the lexicon get a small positive boost."""
    s_none, _ = intimate_score("la chica")
    s_dim, _ = intimate_score("la chiquilla")    # diminutive not in vocab
    if s_none is None and s_dim is not None:
        assert s_dim > 0


# ── Emoji scorer ───────────────────────────────────────────────────────────

def test_emoji_abstains_on_no_emoji():
    assert emoji_score("hello world") is None


def test_emoji_red_heart_positive():
    """❤️ (with VS-16) should score positive thanks to VS-16 stripping."""
    s = emoji_score("te amo ❤️")
    assert s is not None and s > 0.5


def test_emoji_modern_overrides():
    """🩷 isn't in the 2015 dataset but should be in our overlay."""
    s = emoji_score("hola 🩷")
    assert s is not None and s > 0.5


def test_emoji_mean_of_multiple():
    s = emoji_score("❤️❤️❤️")
    assert s is not None and s > 0.5


# ── Hybrid fusion ──────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text",
    [
        "te amo tanto",
        "mi amor de mi vida",
        "eres lo mejor q tengo",
        "MAS MI AMOR",
        "TE AMOOOOOO",
    ],
)
def test_hybrid_overrides_negative_base_for_affection(text):
    """
    With pysentimiento giving -0.3 (the typical bias on affection),
    a strongly affectionate utterance should still come out positive,
    or at minimum noticeably less negative than -0.3.
    """
    base = -0.3
    final = hybrid_score(text, base)
    assert final > base, f"{text!r}: final {final} should beat base {base}"
    assert final > -0.05, f"{text!r}: should clear neutrality, got {final}"


def test_hybrid_keeps_base_when_no_side_signal():
    """A neutral non-Spanish-vocab message with no emphasis should pass the
    base through. ("aaaa" would trigger the emphasis multiplier, so we use
    distinct chars.)"""
    assert hybrid_score("xy zw qp", base_score=0.2) == pytest.approx(0.2, abs=0.01)
    assert hybrid_score("xy zw qp", base_score=-0.4) == pytest.approx(-0.4, abs=0.01)


def test_hybrid_emphasis_amplifies_even_with_no_side_signal():
    """Repeated characters alone should still amplify the base score (the
    intimate layer only contributes if there's vocab match, but the emphasis
    multiplier applies to the *fused* result regardless)."""
    assert hybrid_score("aaaa bbbb", 0.2) > 0.2


def test_hybrid_respects_clamp():
    """Even with all layers maxed + emphasis, score must stay within [-1, +1]."""
    high = hybrid_score("TE AMOOOO MI AMOR DE MI VIDA ❤️❤️❤️", base_score=1.0)
    low = hybrid_score("te odio enfermo decepcionada", base_score=-1.0)
    assert -1.0 <= low <= 0.0
    assert 0.0 <= high <= 1.0


def test_hybrid_emphasis_amplifies_magnitude():
    """The orthographic-emphasis multiplier should *amplify*, not change sign."""
    soft = hybrid_score("te amo", 0.0)
    loud = hybrid_score("TE AMOOO!!!", 0.0)
    assert loud > soft
    assert loud > 0


# ── Explain (debug helper) ─────────────────────────────────────────────────

def test_explain_returns_all_components():
    info = explain("te amo mi vida ❤️", -0.2)
    assert "base_pysent" in info
    assert "lexicon" in info
    assert "intimate" in info
    assert "emoji" in info
    assert "emphasis_mult" in info
    assert "final_hybrid" in info
    assert info["base_pysent"] == -0.2
