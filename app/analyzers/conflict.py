"""
Conflict analyzer — detects rupture language and WhatsApp system events.

Privacy rule: this module only ever emits counts, dates and participant names.
Message text is matched against the lexicon and discarded, never returned.
All metric functions are pure (no side effects, no I/O).
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta

from app.analyzers.base import AnalysisResult, BaseAnalyzer
from app.analyzers.lexicons.rupture_es import match_categories
from app.analyzers.temporal import _week, _week_start
from app.parsers.base import ParsedChat, ParsedMessage

# ── Constants ────────────────────────────────────────────────────────────────

_MIN_TEXT_MESSAGES = 50      # below this a rate per 1000 messages is noise
_EPISODE_MIN_MENTIONS = 3    # mentions in one day that make it a conflict day
_EPISODE_MIN_MISSED_CALLS = 8  # unanswered calls in one day that make it a conflict day
_EPISODE_HIGH_MENTIONS = 8   # mentions that make a whole episode "high" severity
_EPISODE_MAX_GAP_DAYS = 2    # conflict days at most one quiet day apart merge
_RECENT_WINDOW_WEEKS = 3     # "recent" = the last N ISO weeks of the chat


# ── Analyzer ─────────────────────────────────────────────────────────────────

class ConflictAnalyzer(BaseAnalyzer):
    name = "conflict"

    def analyze(self, chat: ParsedChat, context: dict | None = None) -> AnalysisResult:
        msgs = chat.messages
        text_msgs = [m for m in msgs if not m.is_media and m.content.strip()]
        if len(text_msgs) < _MIN_TEXT_MESSAGES:
            return AnalysisResult(analyzer=self.name, data={"error": "insufficient_data"})

        # One entry per mention: (message, categories). Text is dropped here.
        hits = [(m, cats) for m in text_msgs if (cats := match_categories(m.content))]
        weekly = _weekly(text_msgs, hits)

        return AnalysisResult(
            analyzer=self.name,
            data={
                "lexicon": "es",
                "mentions": _mentions(text_msgs, hits),
                "weekly": weekly,
                "episodes": _episodes(msgs, hits),
                "system_events": _system_events(msgs),
                "recent": _recent(weekly),
            },
        )


# ── Metric functions (pure) ───────────────────────────────────────────────────

def _rate(mentions: int, total: int) -> float:
    return round(mentions / total * 1000, 2) if total else 0.0


def _mentions(
    text_msgs: list[ParsedMessage],
    hits: list[tuple[ParsedMessage, set[str]]],
) -> dict:
    per_person: dict[str, int] = defaultdict(int)
    per_category: dict[str, int] = defaultdict(int)
    for m, cats in hits:
        per_person[m.sender] += 1
        for c in cats:
            per_category[c] += 1
    return {
        "total": len(hits),
        "per_person": dict(per_person),
        "per_category": dict(per_category),
        "rate_per_1000": _rate(len(hits), len(text_msgs)),
    }


def _weekly(
    text_msgs: list[ParsedMessage],
    hits: list[tuple[ParsedMessage, set[str]]],
) -> list[dict]:
    totals: dict[str, int] = defaultdict(int)
    mention_counts: dict[str, int] = defaultdict(int)
    for m in text_msgs:
        totals[_week(m.timestamp)] += 1
    for m, _ in hits:
        mention_counts[_week(m.timestamp)] += 1
    return [
        {
            "period": wk,
            "period_start": _week_start(wk),
            "mentions": mention_counts[wk],
            "rate_per_1000": _rate(mention_counts[wk], totals[wk]),
        }
        for wk in sorted(totals)
    ]


def _episodes(
    msgs: list[ParsedMessage],
    hits: list[tuple[ParsedMessage, set[str]]],
) -> list[dict]:
    """
    A conflict day has >= _EPISODE_MIN_MENTIONS mentions, a block event, or
    >= _EPISODE_MIN_MISSED_CALLS unanswered calls. Conflict days at most one
    quiet day apart merge into one episode.

    The missed-call rule exists because some fights leave almost no rupture
    vocabulary in the text: one person stops answering and the other keeps
    calling. A handful of unanswered calls is ordinary life; a burst of them
    in a single day is not.
    """
    day_mentions: dict[date, int] = defaultdict(int)
    day_people: dict[date, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    day_cats: dict[date, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    day_missed: dict[date, int] = defaultdict(int)
    block_days: set[date] = set()

    for m, cats in hits:
        d = m.timestamp.date()
        day_mentions[d] += 1
        day_people[d][m.sender] += 1
        for c in cats:
            day_cats[d][c] += 1
    for m in msgs:
        if m.media_type == "block":
            block_days.add(m.timestamp.date())
        elif m.media_type == "missed_call":
            day_missed[m.timestamp.date()] += 1

    conflict_days = sorted(
        {d for d, n in day_mentions.items() if n >= _EPISODE_MIN_MENTIONS}
        | {d for d, n in day_missed.items() if n >= _EPISODE_MIN_MISSED_CALLS}
        | block_days
    )

    groups: list[list[date]] = []
    for d in conflict_days:
        if groups and (d - groups[-1][-1]).days <= _EPISODE_MAX_GAP_DAYS:
            groups[-1].append(d)
        else:
            groups.append([d])

    episodes = []
    for group in groups:
        start, end = group[0], group[-1]
        # Mentions, people and missed calls cover every day inside the range,
        # including the quiet day that bridged two conflict days.
        span = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        people: dict[str, int] = defaultdict(int)
        cats: dict[str, int] = defaultdict(int)
        for d in span:
            for p, n in day_people[d].items():
                people[p] += n
            for c, n in day_cats[d].items():
                cats[c] += n
        mentions = sum(day_mentions[d] for d in span)
        blocked = any(d in block_days for d in span)
        episodes.append({
            "start": start.isoformat(),
            "end": end.isoformat(),
            "mentions": mentions,
            "per_person": dict(people),
            "categories": dict(cats),
            "missed_calls": sum(day_missed[d] for d in span),
            "blocked": blocked,
            "severity": "high" if mentions >= _EPISODE_HIGH_MENTIONS or blocked else "medium",
        })
    return episodes


def _system_events(msgs: list[ParsedMessage]) -> dict:
    # No author is reported for blocks: in iOS exports the block line is
    # attributed to the contact, not to the person who actually blocked, so
    # the sender field cannot be trusted for this event.
    blocks: list[str] = []
    unblocks: list[str] = []
    deleted: dict[str, int] = defaultdict(int)
    missed: dict[str, int] = defaultdict(int)
    for m in msgs:
        if m.media_type == "block":
            blocks.append(m.timestamp.isoformat())
        elif m.media_type == "unblock":
            unblocks.append(m.timestamp.isoformat())
        elif m.media_type == "deleted":
            deleted[m.sender] += 1
        elif m.media_type == "missed_call":
            missed[m.sender] += 1
    return {
        "blocks": blocks,
        "unblocks": unblocks,
        "deleted_messages": dict(deleted),
        "missed_calls": {"total": sum(missed.values()), "per_person": dict(missed)},
    }


def _recent(weekly: list[dict]) -> dict:
    """Rate over the last ISO weeks with messages vs the median of earlier weeks."""
    recent = weekly[-_RECENT_WINDOW_WEEKS:]
    earlier = weekly[:-_RECENT_WINDOW_WEEKS]
    # Mean of the window's weekly rates (each week weighs the same, so one
    # busy week cannot dominate the comparison with the weekly median).
    recent_rate = round(statistics.mean(w["rate_per_1000"] for w in recent), 2)
    baseline = (
        round(statistics.median(w["rate_per_1000"] for w in earlier), 2) if earlier else 0.0
    )
    return {
        "window_weeks": _RECENT_WINDOW_WEEKS,
        "rate_per_1000": recent_rate,
        "baseline_rate_per_1000": baseline,
        "ratio": round(recent_rate / baseline, 2) if baseline else None,
    }
