import re
from datetime import datetime

from app.parsers.base import BaseParser, ParsedChat, ParsedMessage

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
_MEDIA_TOKENS = {
    "<media omitted>",
    "<archivo adjunto omitido>",
    "image omitted",
    "video omitted",
    "audio omitted",
    "sticker omitted",
    "gif omitted",
    "document omitted",
    "imagen omitida",
    "video omitido",
    "audio omitido",
    "sticker omitido",
    "gif omitido",
    "documento omitido",
}
_MEDIA_ATTACH_RE = re.compile(r"^<?(attached|adjunto|attached file|archivo adjunto):", re.IGNORECASE)

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

        for line in lines:
            if not line:
                continue

            match = _ANDROID_RE.match(line) or _IOS_RE.match(line)
            if not match:
                if messages and not _SYSTEM_PATTERNS.search(line):
                    messages[-1].content += f"\n{line}"
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

            lowered = content.lower()
            is_media = lowered in _MEDIA_TOKENS or bool(_MEDIA_ATTACH_RE.match(lowered))
            participants.add(sender)

            messages.append(
                ParsedMessage(
                    timestamp=timestamp,
                    sender=sender,
                    content=content,
                    is_media=is_media,
                )
            )

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
