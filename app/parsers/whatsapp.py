import re
from datetime import datetime

from app.parsers.base import (
    MAX_MESSAGE_CHARS,
    BaseParser,
    ParsedChat,
    ParsedMessage,
    check_message_count,
    join_message,
)

# WhatsApp (especially iOS) injects Unicode bidi / formatting marks around
# media and system lines, and uses a narrow no-break space as the time
# separator. Both break naive regex anchoring and token comparison, so every
# line is normalised before anything else touches it.
_BIDI_MARKS = dict.fromkeys(
    map(ord, "‎‏‪‫‬‭‮⁦⁧⁨⁩﻿"),
    None,
)
_EXOTIC_SPACES = {ord(c): " " for c in "    "}


def _normalize_line(line: str) -> str:
    return line.translate(_BIDI_MARKS).translate(_EXOTIC_SPACES).strip()


# Android format: "dd/mm/yy, hh:mm - Sender: message"
_ANDROID_RE = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][\.\s]*[Mm]\.?)?)\s-\s([^:]+):\s(.+)$"
)

# iOS format: "[dd/mm/yy, hh:mm:ss] Sender: message"
_IOS_RE = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][\.\s]*[Mm]\.?)?)\]\s([^:]+):\s(.+)$"
)

_DATE_HEAD_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$")

# Compared case-insensitively against mark-stripped content.
# Maps each attachment placeholder to its media_type.
_MEDIA_TOKENS = {
    "<media omitted>": "media",
    "<archivo adjunto omitido>": "media",
    "image omitted": "image",
    "video omitted": "video",
    "audio omitted": "audio",
    "sticker omitted": "sticker",
    "gif omitted": "gif",
    "document omitted": "document",
    "video note omitted": "media",
    "contact card omitted": "media",
    "imagen omitida": "image",
    "video omitido": "video",
    "audio omitido": "audio",
    "sticker omitido": "sticker",
    "gif omitido": "gif",
    "documento omitido": "document",
}
_MEDIA_ATTACH_RE = re.compile(r"^<?(attached|adjunto|attached file|archivo adjunto):", re.IGNORECASE)

# Service lines (calls, deletions, blocks). They stay in the message stream so
# timing metrics see them, but are flagged so text analyzers skip them.
_CALL_WORD = r"(?:voice call|video call|llamada de voz|videollamada|llamada de v[ií]deo)"
_MISSED_CALL_RE = re.compile(
    r"^(?:missed (?:voice|video) call\b.*"
    rf"|{_CALL_WORD}\.?\s*(?:no answer|sin respuesta)\b.*"
    r"|(?:llamada de voz|videollamada|llamada de v[ií]deo|llamada) perdida\b.*)$"
)
_CALL_RE = re.compile(rf"^{_CALL_WORD}\.?\s*\d+\s*[a-z]+\.?$")
_DELETED = {
    "you deleted this message",
    "this message was deleted",
    "eliminaste este mensaje",
    "se eliminó este mensaje",
}
_BLOCKED = {
    "you blocked this person",
    "you blocked this contact",
    "bloqueaste a este contacto",
    "bloqueaste a esta persona",
}
_UNBLOCKED = {
    "you unblocked this person",
    "you unblocked this contact",
    "desbloqueaste a este contacto",
    "desbloqueaste a esta persona",
}
# Trailing edit marker; the message itself is still ordinary text.
_EDITED_RE = re.compile(r"\s*<(?:this message was edited|se editó este mensaje\.?)>\s*$", re.IGNORECASE)


def _classify(lowered: str) -> str | None:
    """Return the media_type of a service/attachment line, or None for text."""
    if lowered in _MEDIA_TOKENS:
        return _MEDIA_TOKENS[lowered]
    if _MEDIA_ATTACH_RE.match(lowered):
        return "media"
    bare = lowered.rstrip(". ")
    if _MISSED_CALL_RE.match(lowered):
        return "missed_call"
    if _CALL_RE.match(lowered):
        return "call"
    if bare in _DELETED:
        return "deleted"
    if bare in _BLOCKED:
        return "block"
    if bare in _UNBLOCKED:
        return "unblock"
    return None

# Lines that are WhatsApp system messages, not chat content
_SYSTEM_PATTERNS = re.compile(
    r"(cifrados de extremo|end-to-end encrypted|Messages and calls|created group|"
    r"added you|changed the subject|changed this group)",
    re.IGNORECASE,
)
# Short iOS system lines the length guard above would otherwise let through
_SYSTEM_SHORT = re.compile(
    r"(\bis a contact\b|es un contacto|end-to-end encrypted|cifrados de extremo|"
    r"security code|código de seguridad)",
    re.IGNORECASE,
)


def _normalize_time(t: str) -> str:
    """Normalize Spanish/Portuguese AM/PM to standard: 'a. m.' → 'AM'."""
    t = t.strip()
    t = re.sub(r"a[\.\s]*\s*m\.?", "AM", t, flags=re.IGNORECASE)
    t = re.sub(r"p[\.\s]*\s*m\.?", "PM", t, flags=re.IGNORECASE)
    return t.strip()


def _detect_date_order(lines: list[str]) -> str:
    """
    Decide day-first vs month-first ONCE for the whole file, instead of
    guessing per line (which silently mixes interpretations on ambiguous
    dates). A token like 17/5/26 proves day-first; 12/25/23 proves
    month-first. Fully ambiguous files default to day-first — that is the
    locale default everywhere except a handful of month-first locales, and
    those always contain at least one day > 12 over a real conversation.
    """
    saw_day_first = False
    saw_month_first = False
    for line in lines:
        match = _ANDROID_RE.match(line) or _IOS_RE.match(line)
        if not match:
            continue
        d = _DATE_HEAD_RE.match(match.group(1))
        if not d:
            continue
        a, b = int(d.group(1)), int(d.group(2))
        if a > 12:
            saw_day_first = True
        if b > 12:
            saw_month_first = True
    if saw_month_first and not saw_day_first:
        return "mdy"
    return "dmy"


class WhatsAppParser(BaseParser):
    platform = "whatsapp"

    def can_parse(self, raw: str) -> bool:
        if not raw:
            return False
        for line in raw.splitlines()[:15]:
            line = _normalize_line(line)
            if not line or _SYSTEM_PATTERNS.search(line):
                continue
            if _ANDROID_RE.match(line) or _IOS_RE.match(line):
                return True
        return False

    def parse(self, raw: str) -> ParsedChat:
        messages: list[ParsedMessage] = []
        participants: set[str] = set()

        lines = [_normalize_line(ln) for ln in raw.splitlines()]
        order = _detect_date_order(lines)

        # Continuation lines of the last message, joined once when it is complete
        extra: list[str] = []
        extra_chars = 0

        def flush() -> None:
            nonlocal extra, extra_chars
            if messages:
                messages[-1].content = join_message(messages[-1].content, extra)
            extra, extra_chars = [], 0

        for line in lines:
            if not line:
                continue

            match = _ANDROID_RE.match(line) or _IOS_RE.match(line)
            if not match:
                # Past the cap the rest would be truncated anyway: stop collecting
                if messages and not _SYSTEM_PATTERNS.search(line) and extra_chars < MAX_MESSAGE_CHARS:
                    extra.append(line)
                    extra_chars += len(line) + 1
                continue

            date_str, time_str, sender, content = match.groups()
            sender = sender.strip()
            content = content.strip()

            # Skip system messages masquerading as sender lines
            if _SYSTEM_PATTERNS.search(content) and len(content) > 80:
                continue
            if _SYSTEM_SHORT.search(content):
                continue

            try:
                timestamp = self._parse_timestamp(date_str, time_str, order)
            except ValueError:
                continue

            flush()
            content = _EDITED_RE.sub("", content).strip()
            media_type = _classify(content.lower())
            participants.add(sender)

            messages.append(
                ParsedMessage(
                    timestamp=timestamp,
                    sender=sender,
                    content=content,
                    is_media=media_type is not None,
                    media_type=media_type,
                )
            )
            check_message_count(len(messages))

        flush()
        return ParsedChat(
            platform=self.platform,
            participants=sorted(participants),
            messages=messages,
        )

    def _parse_timestamp(self, date_str: str, time_str: str, order: str) -> datetime:
        d = _DATE_HEAD_RE.match(date_str)
        if not d:
            raise ValueError(f"Unrecognised date: {date_str}")
        a, b, y = int(d.group(1)), int(d.group(2)), int(d.group(3))
        day, month = (a, b) if order == "dmy" else (b, a)
        year = y if y >= 1000 else 2000 + y

        clock = _normalize_time(time_str)
        for fmt in ("%I:%M:%S %p", "%I:%M %p", "%H:%M:%S", "%H:%M"):
            try:
                t = datetime.strptime(clock, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Unrecognised time: {time_str}")

        # datetime() validates day/month — a line that contradicts the
        # file-wide order (e.g. an impossible month) raises and is skipped.
        return datetime(year, month, day, t.hour, t.minute, t.second)
