"""
Personal-data redaction applied to message text before it leaves the server.

Covers, replacing each match with a fixed marker:
  - e-mail addresses            -> "[email]"
  - URLs (http, https, www.)    -> "[link]"
  - any run of 6+ digits, even split by spaces, dots, hyphens or parentheses,
    with an optional leading "+" (phones, national IDs, tax IDs, bank
    accounts, card numbers)     -> "[number]"

Does NOT cover: proper names, addresses or any other free-text identifier.
Names written inside a message stay as they are; this is a best-effort
filter for structured identifiers, not anonymization.

Left untouched on purpose: times ("10:30"), short dates ("12/05"), small
amounts ("$25.50", "1500") and years, since none reaches 6 digits in a run.
"""
from __future__ import annotations

import re

# ── Patterns ──────────────────────────────────────────────────────────────────

# Every quantifier is bounded (RFC 5321: local part <= 64, label <= 63): an
# unbounded `[\w.+-]+` rescans the whole run from every start position, which is
# quadratic on long inputs without spaces. Bounded, the worst case is linear.
_EMAIL_RE = re.compile(r"[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63}){1,8}")

# Greedy `\S+` consumes the whole token once, so there is nothing to backtrack
# into (the earlier `\S*[^\s.,]` form retried from every "www." in the run).
# Trailing punctuation is trimmed afterwards ("mira www.x.com.").
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_URL_TRAILING = ".,;:!?)]\"'"
_URL_PREFIX_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)

# Candidate digit runs: single-character separators only, so "1500 2000 pesos"
# style prose is rare to join but "099 123 4567" and "0991-234-567" are caught.
# Whether a candidate is a number to hide is decided by counting its digits.
_NUMBER_RE = re.compile(r"(?<!\w)\+?\(?\d+\)?(?:[ .\-]\(?\d+\)?)*")

# 6 digits is below every phone/ID/account length we care about and above
# times, short dates, years and everyday amounts.
_MIN_DIGITS = 6

# Full dates written with separators ("12-05-2024") have 6+ digits but are not
# identifiers, so they are kept.
_DATE_RE = re.compile(r"\d{1,2}[-.]\d{1,2}[-.]\d{2,4}")


# ── Public API ────────────────────────────────────────────────────────────────

def redact(text: str) -> str:
    """Replace e-mails, links and long digit runs in `text` with fixed markers."""
    text = _EMAIL_RE.sub("[email]", text)
    text = _URL_RE.sub(_link_replacement, text)
    return _NUMBER_RE.sub(_number_replacement, text)


def _number_replacement(match: re.Match[str]) -> str:
    value = match.group(0)
    if _DATE_RE.fullmatch(value):
        return value
    digits = sum(ch.isdigit() for ch in value)
    return "[number]" if digits >= _MIN_DIGITS else value


def _link_replacement(match: re.Match[str]) -> str:
    value = match.group(0)
    body = value.rstrip(_URL_TRAILING)
    # Nothing after the prefix ("www.,"): not a link
    if len(body) <= _URL_PREFIX_RE.match(value).end():  # type: ignore[union-attr]
        return value
    return "[link]" + value[len(body):]
