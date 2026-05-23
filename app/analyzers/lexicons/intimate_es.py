"""
Hand-curated Spanish lexicon for intimate / couple-chat register.

Covers what NRC and AFINN miss for love messaging:
  - Affectionate vocatives ("amor", "cariño", "bebé", "mi vida"…)
  - Hyperbolic-but-playful expressions ("convulsiono", "me muero", "te re mil
    pasaste") — these are NEGATIVE-coded by tweet-trained models but in couple
    chats they read as playful affection.
  - Strong love declarations ("te amo", "mi todo", "eres lo mejor", "amaré")
  - Diminutive suffixes (-ito, -ita, -cito, -cita, -uelo, -uela, -illa, -illo) —
    these are warmth markers in Spanish; multiplies the valence of a token.
  - Orthographic emphasis ("amoooor", "TE AMOOO", "!!!!", "❤️❤️❤️") —
    intensity multiplier on the magnitude.
  - "Faux-negative" expressions specific to couple chats: panic-as-affection,
    drama-as-flirt, jealousy-as-cuteness.

Conventions
-----------
- Multi-word entries are stored verbatim; they are looked up as substrings,
  case-folded, after stripping diacritics on accented vowels. Single-word
  entries are looked up against word-level tokens.
- All valence values are in [-1.0, +1.0]. Magnitudes >0.7 are reserved for
  unambiguous declarations.
- `playful_negative` entries cap their negative contribution at -0.3 instead of
  letting the base model assign worse scores — they are *not* genuinely
  negative in this register.

This file is meant to be tunable. Add entries as you read more chats.
"""
from __future__ import annotations

import re

# ── Affectionate vocatives & terms of endearment ────────────────────────────
# Words/phrases that almost always carry warmth when addressed at someone.
VOCATIVES: dict[str, float] = {
    # Core
    "amor": 0.65,
    "amorcito": 0.75,
    "amorcita": 0.75,
    "amorcillo": 0.7,
    "mi amor": 0.8,
    "mi amorcito": 0.85,
    "mi amorcita": 0.85,
    "miamor": 0.8,
    "miamorcito": 0.85,
    "miamorcita": 0.85,
    "cariño": 0.6,
    "cariñito": 0.7,
    "mi cariño": 0.7,
    "mi cariñito": 0.8,
    # Body / heart
    "corazón": 0.6,
    "corazoncito": 0.75,
    "mi corazón": 0.75,
    "mi corazoncito": 0.85,
    # Life / world
    "mi vida": 0.8,
    "mi todo": 0.85,
    "mi cielo": 0.7,
    "mi sol": 0.7,
    "mi luna": 0.65,
    "mi mundo": 0.8,
    "mi universo": 0.8,
    "mi reina": 0.7,
    "mi rey": 0.7,
    # Family-style endearments
    "bebé": 0.55,
    "bebito": 0.65,
    "bebita": 0.65,
    "nene": 0.45,
    "nena": 0.45,
    "gordi": 0.5,
    "gordito": 0.5,
    "gordita": 0.5,
    # Royal / precious metaphors
    "princesa": 0.6,
    "princesita": 0.7,
    "príncipe": 0.55,
    "principito": 0.65,
    "preciosa": 0.65,
    "preciosísima": 0.75,
    "precioso": 0.6,
    "hermosa": 0.6,
    "hermoso": 0.55,
    "hermosita": 0.65,
    "hermosísima": 0.75,
    "bella": 0.55,
    "bello": 0.5,
    "bellísima": 0.7,
    "linda": 0.5,
    "lindo": 0.45,
    "lindita": 0.6,
    "lindito": 0.55,
    "preciosita": 0.7,
    "guapa": 0.5,
    "guapo": 0.45,
    # Diminutive nicknames common in Latin American Spanish
    "chiquita": 0.55,
    "chiquito": 0.5,
    "chiquitita": 0.65,
    "chiquititita": 0.7,
    "pequeñita": 0.55,
    "pequeñito": 0.5,
    "cushita": 0.55,
    "cuchita": 0.55,
    "cuchurrumin": 0.7,
    "pollito": 0.55,
    "pollita": 0.55,
    "ratoncita": 0.5,
    "ratoncito": 0.45,
    "muñeca": 0.55,
    "muñequita": 0.65,
    # Playful affectionate pet-names
    "boba": 0.4,    # used playfully between partners
    "bobita": 0.5,
    "bobo": 0.35,
    "bobito": 0.45,
    "tontita": 0.4,
    "tontito": 0.35,
    "loquita": 0.4,
    "loquito": 0.4,
    "bavita": 0.4,    # phonetic spelling, common in voseo regions
    "bavito": 0.35,
}

# ── Strong love declarations (high valence anchors) ────────────────────────
DECLARATIONS: dict[str, float] = {
    "te amo": 0.9,
    "yo te amo": 0.95,
    "te amo mucho": 0.95,
    "te amo demasiado": 0.95,
    "te amo tanto": 0.95,
    "te amooo": 0.95,
    "te amoo": 0.95,
    "teamo": 0.85,
    "ti amo": 0.85,
    "te quiero": 0.7,
    "te quiero mucho": 0.8,
    "te adoro": 0.85,
    "amor de mi vida": 0.9,
    "amor mío": 0.85,
    "amor mio": 0.85,
    "mi amor de mi vida": 0.95,
    "amorcito de mi vida": 0.95,
    "eres lo mejor": 0.85,
    "eres lo mejor que tengo": 0.95,
    "lo mejor que tengo": 0.9,
    "lo mejor que me ha pasado": 0.95,
    "lo mejor que me pasó": 0.95,
    "la mujer de mi vida": 0.95,
    "el hombre de mi vida": 0.95,
    "la persona de mi vida": 0.95,
    "amaré": 0.7,
    "jamás amaré a alguien": 0.85,
    "nunca amaré a alguien": 0.85,
    "te extraño": 0.5,   # mild positive — missing implies caring
    "te extraño mucho": 0.65,
    "te extraño tanto": 0.7,
    "me haces feliz": 0.85,
    "me haces muy feliz": 0.9,
    "te necesito": 0.55,
    "no puedo sin ti": 0.7,
    "sin ti no": 0.5,
    "para siempre": 0.6,
    "siempre contigo": 0.65,
    "siempre tuyo": 0.7,
    "siempre tuya": 0.7,
}

# ── Playful drama (faux-negative; do NOT score deeply negative) ──────────
# Tweet-trained models tend to read these as anger/sadness. In couple chats
# they are flirty exaggeration. We assign a small NEGATIVE valence (so they
# slightly damp the avg, reflecting some tension) but cap how negative they
# can go to override the model's harsh reading.
PLAYFUL_NEGATIVE: dict[str, float] = {
    "me muero": -0.15,
    "convulsiono": -0.15,
    "me odias": -0.2,
    "ya no me quieres": -0.15,
    "me quieres dejar": -0.2,
    "ya no me amas": -0.2,
    "ya no me hablas": -0.15,
    "no me extrañas": -0.1,
    "no me extrañaste": -0.1,
    "ya no me das atención": -0.15,
    "no me das bola": -0.1,
    "no me das atención": -0.1,
    "no me paras bola": -0.1,
    "párame bola": -0.05,
    "no me amas": -0.2,
    "no me quieres": -0.2,
    "es porque me quieres dejar": -0.2,
    "wtf amor": -0.1,
    "wtf miamor": -0.1,
    "wtfff": -0.1,
    "te re mil pasaste": -0.05,    # mock-scolding, mostly affectionate
    "te pasaste": -0.05,
    "te pasaste de": -0.05,
    "boba": -0.05,    # mock-insult; net affectionate, see VOCATIVES override
}

# ── Genuinely negative expressions (relationship-specific) ──────────
# These DO indicate distress and should keep their negative score.
GENUINE_NEGATIVE: dict[str, float] = {
    "te odio": -0.6,
    "ya no te aguanto": -0.6,
    "no te aguanto": -0.55,
    "estoy harta": -0.55,
    "estoy harto": -0.55,
    "no te soporto": -0.6,
    "se acabó": -0.6,
    "se acabo": -0.6,
    "ya no quiero": -0.5,
    "déjame en paz": -0.55,
    "dejame en paz": -0.55,
    "me lastimaste": -0.6,
    "me dolió": -0.5,
    "me dolio": -0.5,
    "estoy triste": -0.5,
    "estoy mal": -0.5,
    "estoy llorando": -0.55,
    "no me hables": -0.55,
    "no quiero hablar": -0.5,
}

# ── Affectionate phrasing & general positive ────────────────────────
GENERAL_POSITIVE: dict[str, float] = {
    "amar": 0.55,        # NRC gap
    "amaste": 0.55,
    "amada": 0.55,
    "amado": 0.55,
    "querida": 0.5,
    "querido": 0.5,
    "feliz contigo": 0.75,
    "muy feliz": 0.75,
    "felices": 0.6,
    "felizmente": 0.6,
    "abrazo": 0.5,
    "abrazote": 0.6,
    "besos": 0.55,
    "besito": 0.55,
    "besitos": 0.6,
    "besote": 0.6,
    "ternura": 0.6,
    "tierno": 0.55,
    "tierna": 0.55,
    "tiernísima": 0.7,
    "ricura": 0.6,
    "lindísima": 0.7,
    "lindísimo": 0.65,
    "encantadora": 0.65,
    "encantador": 0.6,
    "gracias amor": 0.65,
}

# ── Genuine negative (basic gap-fill) ───────────────────────────────
GENERAL_NEGATIVE: dict[str, float] = {
    "odiar": -0.55,        # NRC gap
    "odio": -0.55,
    "odias": -0.55,
    "odié": -0.55,
    "celos": -0.3,
    "celosa": -0.25,
    "celoso": -0.25,
    "molestia": -0.35,
    "molesta": -0.3,
    "molesto": -0.3,
    "enojada": -0.45,
    "enojado": -0.45,
    "decepción": -0.55,
    "decepcionada": -0.55,
    "decepcionado": -0.55,
    "abandonada": -0.6,
    "abandonado": -0.6,
}

# ── All multi-word phrases, merged into one substring-lookup table ──────
# The source dicts (DECLARATIONS, PLAYFUL_NEGATIVE, GENUINE_NEGATIVE, multi-word
# VOCATIVES) stay separate above for readability; the playful/genuine split
# does NOT affect scoring (each phrase already carries its calibrated valence
# directly — playful entries have a deliberately capped negative magnitude).
_PHRASES: dict[str, float] = {}
for phrase, val in DECLARATIONS.items():
    _PHRASES[phrase.lower()] = val
for phrase, val in PLAYFUL_NEGATIVE.items():
    _PHRASES[phrase.lower()] = val
for phrase, val in GENUINE_NEGATIVE.items():
    _PHRASES[phrase.lower()] = val
for phrase, val in VOCATIVES.items():
    if " " in phrase:
        _PHRASES[phrase.lower()] = val

# ── Single-token vocab (vocatives + general) ─────────────────────────────
_TOKENS: dict[str, float] = {}
for word, val in VOCATIVES.items():
    if " " not in word:
        _TOKENS[word.lower()] = val
for word, val in GENERAL_POSITIVE.items():
    if " " not in word:
        _TOKENS[word.lower()] = val
for word, val in GENERAL_NEGATIVE.items():
    if " " not in word:
        _TOKENS[word.lower()] = val

# ── Diminutive suffix patterns ───────────────────────────────────────────
# Matches words ending in canonical Spanish diminutive suffixes. These are
# warmth markers regardless of base word; we apply a +0.15 boost (additive,
# capped) only if the word is NOT already in our intimate lexicon (to avoid
# double-counting "amorcito" which is already in VOCATIVES).
_DIMINUTIVE_RE = re.compile(r"\b\w{4,}(?:cit[oa]|illit[oa]|cit[oa]s|illit[oa]s|illa|illas|illo|illos|uelo|uela|ito|ita|itos|itas)\b", re.IGNORECASE)

# ── Orthographic emphasis ───────────────────────────────────────────────
# Multiplier on the magnitude of the score when a message has heavy emphasis.
# Capped so that mild emphasis (one !) doesn't multiply; only repeated emphasis.
_REPEAT_CHAR_RE = re.compile(r"([a-záéíóúñü])\1{2,}", re.IGNORECASE)        # "amoor" → match
_REPEAT_PUNCT_RE = re.compile(r"([!?¡¿])\1{2,}")                            # "!!!" → match
_ALL_CAPS_RE = re.compile(r"\b[A-ZÁÉÍÓÚÑÜ]{4,}\b")                          # "AMOR" → match

def _emphasis_multiplier(text: str) -> float:
    """
    Return a magnitude multiplier in [1.0, 1.4] based on orthographic
    emphasis cues. Compounds (e.g. repeated chars + all caps + !!!! all in
    one message) cap at 1.4 to avoid runaway scaling.
    """
    mult = 1.0
    if _REPEAT_CHAR_RE.search(text):
        mult += 0.10
    if _REPEAT_PUNCT_RE.search(text):
        mult += 0.10
    if _ALL_CAPS_RE.search(text):
        mult += 0.20
    return min(mult, 1.4)


# ── Word tokenization ───────────────────────────────────────────────────
# Matches Spanish-aware word tokens. \w is Unicode-aware in Python re.
_WORD_RE = re.compile(r"\b[\wáéíóúñüÁÉÍÓÚÑÜ]+\b", re.UNICODE)


def _score_phrases(text_lower: str) -> tuple[float, int]:
    """
    Returns (sum_of_phrase_valences, count_matched).
    Multi-word phrases are matched as substrings on the lowercased text.
    """
    total = 0.0
    count = 0
    for phrase, val in _PHRASES.items():
        if phrase in text_lower:
            total += val
            count += 1
    return total, count


def _score_tokens(text_lower: str) -> tuple[float, int]:
    """Returns (sum_of_token_valences, count_matched)."""
    total = 0.0
    count = 0
    seen: set[str] = set()  # don't count repeats within the same message
    for tok in _WORD_RE.findall(text_lower):
        if tok in seen:
            continue
        if tok in _TOKENS:
            total += _TOKENS[tok]
            count += 1
            seen.add(tok)
    return total, count


def _diminutive_count(text_lower: str) -> int:
    """Count tokens with diminutive suffixes that are NOT already in our lexicon."""
    count = 0
    for m in _DIMINUTIVE_RE.finditer(text_lower):
        tok = m.group(0)
        if tok not in _TOKENS:    # avoid double-counting "amorcito"
            count += 1
    return count


def intimate_score(text: str) -> tuple[float | None, float]:
    """
    Compute the intimate-vocab score for a message.

    Returns:
      (score, emphasis_multiplier)
        score: a valence in [-1, +1] or None if no signal was found.
        emphasis_multiplier: a magnitude multiplier in [1.0, 1.4] applied
            by the caller to the *fused* score, NOT just this layer's score.
    """
    if not text or not text.strip():
        return None, 1.0
    lower = text.lower()
    p_sum, p_cnt = _score_phrases(lower)
    t_sum, t_cnt = _score_tokens(lower)
    dim_cnt = _diminutive_count(lower)

    # Diminutives that aren't in our lexicon get a modest positive boost each
    # (capped at 3 to avoid runaway).
    dim_boost = min(dim_cnt, 3) * 0.12

    total_signal = p_cnt + t_cnt + (1 if dim_cnt else 0)
    if total_signal == 0:
        score = None
    else:
        # Average across matches so a single hit doesn't shout louder than the
        # message warrants; add diminutive boost as a separate additive term.
        score = (p_sum + t_sum) / max(p_cnt + t_cnt, 1) + dim_boost
        score = max(-1.0, min(1.0, score))

    return score, _emphasis_multiplier(text)
