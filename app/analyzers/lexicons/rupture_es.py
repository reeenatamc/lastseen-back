"""
Rupture-language lexicon for colloquial Spanish chat.

Patterns are matched against accent-folded, lower-cased text, so they are
written without accents and cover the usual chat spellings ("kiero", "blokear").
Only the category names ever leave this module; matched text is never kept.
"""
from __future__ import annotations

import re
import unicodedata

_FLAGS = re.IGNORECASE

# ── Categories ───────────────────────────────────────────────────────────────

# Neutral uses of "terminar" ("termine el turno", "voy a terminar la tesis",
# "ya termino") are deliberately absent: only forms that address the
# relationship or the other person are listed.
_CATEGORIES: dict[str, list[re.Pattern[str]]] = {
    "breakup": [
        re.compile(p, _FLAGS)
        for p in (
            r"\bterminamos\b",
            r"\bterminemos\b",
            r"\btermin[ae]me\b",
            r"\bme terminaste\b",
            r"\bterminar(?:me|te|nos)\b",
            r"\bterminar con (?:vos|conmigo|contigo|lo nuestro|la relacion)\b",
            r"\bme (?:vas|quieres|kieres)(?: a)? terminar\b",
            # "no quiero terminar" / "segura que quieres terminar?", but not
            # "quiero terminar la tesis": an object right after rules it out.
            r"\b(?:quiero|kiero|quieres|kieres) terminar\b(?!\s+(?:de|con|el|la|los|las|un|una|unos|unas|mi|mis|esto|eso)\b)",
            r"\b(?:solucion|opcion|salida|remedio) (?:que|es) terminar\b",
            r"\bme vas a dejar\b",
            r"\b(?:quieres|kieres|vas a) dejarme\b",
            r"\bdejar(?:me|te) de (?:querer|amar)\b",
            r"\b(?:buscate|te buscas|buscarte) a (?:alguien|otra|otro)\b",
            r"\bya no (?:quiero|kiero) nada\b",
            r"\bno (?:quiero|kiero) saber (?:mas|nada) de (?:vos|ti)\b",
        )
    ],
    # Any conjugation of bloquear/blokear (bloquear, bloqueaste, blokeada...),
    # except the idiom "bloqueo mental", which is about going blank.
    "block_threat": [re.compile(r"\bblo[qk]u?e[aeo]\w*(?! mental)", _FLAGS)],
    "time_out": [
        re.compile(p, _FLAGS)
        for p in (
            r"\b(?:darnos|darle|demonos) (?:un poco de |un )?tiempo\b",
            r"\bdame (?:un poco de )?tiempo\b",
            r"\bnecesit(?:o|as|amos) (?:un poco de |un |mas )?tiempo\b",
            r"\btiempo (?:q|que) nos demos\b",
            r"\b(?:quiero|kiero) un tiempo\b",
            r"\bpedi unos dias\b",
        )
    ],
    "withdrawal": [
        re.compile(p, _FLAGS)
        for p in (
            r"\bno me hables\b",
            r"\bya no te voy a (?:hablar|responder|contestar|escribir)\b",
            r"\bno te voy a (?:contestar|responder)\b",
            r"\bte voy a dejar de (?:hablar|responder|contestar|escribir)\b",
            r"\bdejame en paz\b",
        )
    ],
}


def _fold(text: str) -> str:
    """Lower-case and strip accents (the n-tilde is irrelevant to the patterns)."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def match_categories(text: str) -> set[str]:
    """Return the set of rupture categories present in `text`."""
    folded = _fold(text)
    return {
        category
        for category, patterns in _CATEGORIES.items()
        if any(p.search(folded) for p in patterns)
    }
