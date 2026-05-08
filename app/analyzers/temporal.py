"""
Temporal analyzer — measures the evolution of interaction dynamics over time.

All functions are pure (no side effects, no I/O) so they are trivially testable
and reusable by future analyzers (e.g. sentiment can call _split_into_blocks).
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime, timedelta

from app.analyzers.base import AnalysisResult, BaseAnalyzer
from app.parsers.base import ParsedChat, ParsedMessage

# ── Constants ────────────────────────────────────────────────────────────────

# A new conversation block starts when both sides have been silent for this long
_INITIATIVE_GAP = timedelta(hours=4)

# Beyond this window a delayed message is not counted as a "response"
_MAX_RESPONSE_WINDOW = timedelta(hours=24)

_TOP_GAPS = 5


# ── Analyzer ─────────────────────────────────────────────────────────────────

class TemporalAnalyzer(BaseAnalyzer):
    name = "temporal"

    def analyze(self, chat: ParsedChat) -> AnalysisResult:
        msgs = chat.messages
        if len(msgs) < 2:
            return AnalysisResult(analyzer=self.name, data={"error": "insufficient_data"})

        return AnalysisResult(
            analyzer=self.name,
            data={
                "overview": _overview(chat),
                "response_time": _response_time(msgs),
                "initiative_balance": _initiative_balance(msgs),
                "activity_patterns": _activity_patterns(msgs),
                "conversation_gaps": _conversation_gaps(msgs),
                "message_length": _message_length(msgs),
                "response_decay": _response_decay(msgs),
            },
        )


# ── Metric functions (pure) ───────────────────────────────────────────────────

def _overview(chat: ParsedChat) -> dict:
    msgs = chat.messages
    counts: dict[str, int] = defaultdict(int)
    for m in msgs:
        counts[m.sender] += 1
    total = len(msgs)

    return {
        "date_range": {
            "start": msgs[0].timestamp.isoformat(),
            "end": msgs[-1].timestamp.isoformat(),
            "total_days": (msgs[-1].timestamp - msgs[0].timestamp).days,
        },
        "total_messages": total,
        "participants": chat.participants,
        "messages_per_person": dict(counts),
        "share_per_person": {p: round(c / total, 3) for p, c in counts.items()},
    }


def _response_time(msgs: list[ParsedMessage]) -> dict:
    times: dict[str, list[float]] = defaultdict(list)
    by_quarter: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for i in range(1, len(msgs)):
        prev, curr = msgs[i - 1], msgs[i]
        if prev.sender == curr.sender:
            continue
        secs = (curr.timestamp - prev.timestamp).total_seconds()
        if 0 < secs <= _MAX_RESPONSE_WINDOW.total_seconds():
            times[curr.sender].append(secs)
            by_quarter[_quarter(curr.timestamp)][curr.sender].append(secs)

    quarters = sorted(by_quarter)

    return {
        "per_person": {
            p: {
                "mean_seconds": round(statistics.mean(v)),
                "median_seconds": round(statistics.median(v)),
                "p90_seconds": round(_p90(v)),
            }
            for p, v in times.items()
        },
        "evolution": [
            {
                "period": q,
                **{
                    p: round(statistics.mean(by_quarter[q][p]))
                    for p in times
                    if by_quarter[q].get(p)
                },
            }
            for q in quarters
        ],
    }


def _initiative_balance(msgs: list[ParsedMessage]) -> dict:
    blocks = _split_into_blocks(msgs)
    initiatives: dict[str, int] = defaultdict(int)
    by_quarter: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for block in blocks:
        sender = block[0].sender
        initiatives[sender] += 1
        by_quarter[_quarter(block[0].timestamp)][sender] += 1

    total = sum(initiatives.values())
    quarters = sorted(by_quarter)

    return {
        "total_conversations": total,
        "per_person": dict(initiatives),
        "share": {p: round(c / total, 3) for p, c in initiatives.items()},
        "evolution": [
            {
                "period": q,
                **{
                    p: round(by_quarter[q][p] / sum(by_quarter[q].values()), 3)
                    for p in initiatives
                    if by_quarter[q].get(p)
                },
            }
            for q in quarters
        ],
    }


def _activity_patterns(msgs: list[ParsedMessage]) -> dict:
    by_hour: dict[int, int] = defaultdict(int)
    by_weekday: dict[int, int] = defaultdict(int)
    by_month: dict[str, int] = defaultdict(int)

    for m in msgs:
        by_hour[m.timestamp.hour] += 1
        by_weekday[m.timestamp.weekday()] += 1
        by_month[_month(m.timestamp)] += 1

    return {
        "by_hour": {str(h): by_hour.get(h, 0) for h in range(24)},
        "by_weekday": {str(d): by_weekday.get(d, 0) for d in range(7)},
        "by_month": [
            {"period": m, "count": c} for m, c in sorted(by_month.items())
        ],
    }


def _conversation_gaps(msgs: list[ParsedMessage]) -> dict:
    gaps = []
    for i in range(1, len(msgs)):
        secs = (msgs[i].timestamp - msgs[i - 1].timestamp).total_seconds()
        if secs > 0:
            hours = secs / 3600
            gaps.append(
                {
                    "start": msgs[i - 1].timestamp.isoformat(),
                    "end": msgs[i].timestamp.isoformat(),
                    "hours": round(hours, 1),
                    "days": round(hours / 24, 1),
                }
            )

    top = sorted(gaps, key=lambda g: g["hours"], reverse=True)[:_TOP_GAPS]
    hours_list = [g["hours"] for g in gaps]

    return {
        "top_gaps": top,
        "distribution": {
            "under_1h": sum(1 for h in hours_list if h < 1),
            "1h_to_6h": sum(1 for h in hours_list if 1 <= h < 6),
            "6h_to_24h": sum(1 for h in hours_list if 6 <= h < 24),
            "1d_to_7d": sum(1 for h in hours_list if 24 <= h < 168),
            "over_7d": sum(1 for h in hours_list if h >= 168),
        },
    }


def _message_length(msgs: list[ParsedMessage]) -> dict:
    text_msgs = [m for m in msgs if not m.is_media and m.content.strip()]
    lengths: dict[str, list[int]] = defaultdict(list)
    by_quarter: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    for m in text_msgs:
        n = len(m.content)
        lengths[m.sender].append(n)
        by_quarter[_quarter(m.timestamp)][m.sender].append(n)

    quarters = sorted(by_quarter)

    return {
        "per_person": {
            p: {
                "mean_chars": round(statistics.mean(v)),
                "median_chars": round(statistics.median(v)),
            }
            for p, v in lengths.items()
            if v
        },
        "evolution": [
            {
                "period": q,
                **{
                    p: round(statistics.mean(by_quarter[q][p]))
                    for p in lengths
                    if by_quarter[q].get(p)
                },
            }
            for q in quarters
        ],
    }


def _response_decay(msgs: list[ParsedMessage]) -> dict:
    """
    Core LastSeen metric: detects progressive deterioration of reciprocity.

    decay_score: 0.0 (healthy) → 1.0 (fully decayed)
    turning_point: first month where health dropped and stayed down
    trend: "improving" | "stable" | "deteriorating"
    """
    # Build per-month snapshots in a single pass
    monthly_rt: dict[str, list[float]] = defaultdict(list)
    monthly_msgs: dict[str, int] = defaultdict(int)

    for m in msgs:
        monthly_msgs[_month(m.timestamp)] += 1

    for i in range(1, len(msgs)):
        prev, curr = msgs[i - 1], msgs[i]
        if prev.sender == curr.sender:
            continue
        secs = (curr.timestamp - prev.timestamp).total_seconds()
        if 0 < secs <= _MAX_RESPONSE_WINDOW.total_seconds():
            monthly_rt[_month(curr.timestamp)].append(secs)

    monthly_init: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for block in _split_into_blocks(msgs):
        monthly_init[_month(block[0].timestamp)][block[0].sender] += 1

    participants = sorted({m.sender for m in msgs})
    all_months = sorted(set(monthly_msgs) | set(monthly_rt) | set(monthly_init))

    if len(all_months) < 2:
        return {"trend": "insufficient_data"}

    # Build evolution timeline
    evolution = []
    for month in all_months:
        avg_rt = round(statistics.mean(monthly_rt[month])) if monthly_rt[month] else None

        total_init = sum(monthly_init[month].values())
        if total_init >= 2 and participants:
            shares = [monthly_init[month].get(p, 0) / total_init for p in participants]
            imbalance = round(abs(shares[0] - 0.5) * 2, 3)
        else:
            imbalance = 0.0

        evolution.append(
            {
                "period": month,
                "avg_response_seconds": avg_rt,
                "message_count": monthly_msgs[month],
                "initiative_imbalance": imbalance,
            }
        )

    # Health score per month: 1.0 = perfect, 0.0 = dead
    def _health(e: dict) -> float:
        rt_score = 1.0 - min((e["avg_response_seconds"] or 0) / _MAX_RESPONSE_WINDOW.total_seconds(), 1.0)
        return round(rt_score * 0.5 + (1.0 - e["initiative_imbalance"]) * 0.5, 3)

    health = [_health(e) for e in evolution]

    # Trend: compare first third vs last third
    third = max(1, len(health) // 3)
    early_health = statistics.mean(health[:third])
    late_health = statistics.mean(health[-third:])
    delta = late_health - early_health

    if delta < -0.15:
        trend = "deteriorating"
    elif delta > 0.10:
        trend = "improving"
    else:
        trend = "stable"

    # Decay score based on recent state
    decay_score = round(1.0 - statistics.mean(health[-third:]), 3)

    # Turning point: month with the biggest single-month health drop
    turning_point = None
    max_drop = 0.05  # minimum meaningful drop
    for i in range(1, len(health)):
        drop = health[i - 1] - health[i]
        if drop > max_drop:
            max_drop = drop
            turning_point = evolution[i]["period"]

    return {
        "trend": trend,
        "decay_score": max(0.0, min(1.0, decay_score)),
        "turning_point": turning_point,
        "evolution": evolution,
    }


# ── Shared helpers (importable by other analyzers) ────────────────────────────

def split_into_blocks(
    msgs: list[ParsedMessage],
    gap: timedelta = _INITIATIVE_GAP,
) -> list[list[ParsedMessage]]:
    """Split a message list into conversation blocks separated by `gap`."""
    if not msgs:
        return []
    blocks: list[list[ParsedMessage]] = [[msgs[0]]]
    for msg in msgs[1:]:
        if msg.timestamp - blocks[-1][-1].timestamp > gap:
            blocks.append([])
        blocks[-1].append(msg)
    return blocks


# ── Private helpers ───────────────────────────────────────────────────────────

def _split_into_blocks(msgs: list[ParsedMessage]) -> list[list[ParsedMessage]]:
    return split_into_blocks(msgs)


def _quarter(dt: datetime) -> str:
    return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"


def _month(dt: datetime) -> str:
    return f"{dt.year}-{dt.month:02d}"


def _p90(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    return sorted_v[min(int(len(sorted_v) * 0.9), len(sorted_v) - 1)]
