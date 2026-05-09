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

    def analyze(self, chat: ParsedChat, context: dict | None = None) -> AnalysisResult:
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
                "delayed_replies": _delayed_replies(msgs),
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

    def _person_stats(v: list[float]) -> dict:
        mean = statistics.mean(v)
        std = statistics.stdev(v) if len(v) >= 2 else 0.0
        # Consistency: % of responses that came within 1 hour.
        # 1.0 = almost always responds quickly.
        # 0.0 = almost always takes more than an hour.
        # Low score = hot/cold behavior (sometimes instant, sometimes hours).
        within_1h = sum(1 for s in v if s <= 3600) / len(v)
        return {
            "mean_seconds": round(mean),
            "median_seconds": round(statistics.median(v)),
            "p90_seconds": round(_p90(v)),
            "std_seconds": round(std),
            "consistency_score": round(within_1h, 3),
        }

    return {
        "per_person": {
            p: _person_stats(v)
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


def _block_was_sustained(block: list[ParsedMessage]) -> bool:
    """
    Returns True if the opener was still present at the end of the
    conversation they started.

    Rule: the opener's last message must be MORE RECENT than the other
    person's last message — i.e., the opener had the final word.

    If the other person had the last word, the opener faded and left
    the other person still writing/waiting.
    """
    opener = block[0].sender

    # Other person must have spoken at least once
    if not any(m.sender != opener for m in block):
        return False

    return block[-1].sender == opener


def _initiative_balance(msgs: list[ParsedMessage]) -> dict:
    """
    Classifies each gap > _INITIATIVE_GAP into four buckets:

    - initiative:      opened a conversation AND stayed to engage
                       (opener responded to the other person at least once).
    - abandoned_open:  opened a conversation but never replied after the
                       other person spoke — started but didn't sustain.
    - late_reply:      previous block was opened and closed by the same
                       person (unanswered) → other person finally responds.
    - double_text:     same person speaks again after their own last message.

    This captures the real emotional pattern: initiating only counts
    if you actually showed up for the conversation you started.
    """
    initiatives: dict[str, int] = defaultdict(int)
    abandoned_opens: dict[str, int] = defaultdict(int)
    late_replies: dict[str, int] = defaultdict(int)
    double_texts: dict[str, int] = defaultdict(int)
    by_quarter_init: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    blocks = _split_into_blocks(msgs)

    # First block — unconditional, classify by whether opener sustained it
    if blocks:
        opener = blocks[0][0].sender
        if _block_was_sustained(blocks[0]):
            initiatives[opener] += 1
            by_quarter_init[_quarter(blocks[0][0].timestamp)][opener] += 1
        else:
            abandoned_opens[opener] += 1

    for prev_block, curr_block in zip(blocks, blocks[1:]):
        block_opener  = prev_block[0].sender
        last_speaker  = prev_block[-1].sender
        first_speaker = curr_block[0].sender

        if first_speaker == last_speaker:
            double_texts[first_speaker] += 1

        elif block_opener == last_speaker:
            late_replies[first_speaker] += 1

        else:
            # Potential genuine initiative — but only counts if the opener
            # actually engaged with the conversation they started.
            if _block_was_sustained(curr_block):
                initiatives[first_speaker] += 1
                by_quarter_init[_quarter(curr_block[0].timestamp)][first_speaker] += 1
            else:
                abandoned_opens[first_speaker] += 1

    total_init = sum(initiatives.values())
    total_ab   = sum(abandoned_opens.values())
    total_lr   = sum(late_replies.values())
    total_dt   = sum(double_texts.values())
    quarters   = sorted(by_quarter_init)

    def _share(counts: dict[str, int], total: int) -> dict[str, float]:
        return {p: round(c / total, 3) for p, c in counts.items()} if total else {}

    return {
        "total_conversations": total_init,
        "per_person": dict(initiatives),
        "share": _share(initiatives, total_init),
        "abandoned_open": {
            "per_person": dict(abandoned_opens),
            "share": _share(abandoned_opens, total_ab),
            "total": total_ab,
        },
        "late_reply": {
            "per_person": dict(late_replies),
            "share": _share(late_replies, total_lr),
            "total": total_lr,
        },
        "double_text": {
            "per_person": dict(double_texts),
            "share": _share(double_texts, total_dt),
            "total": total_dt,
        },
        "evolution": [
            {
                "period": q,
                **{
                    p: round(by_quarter_init[q][p] / sum(by_quarter_init[q].values()), 3)
                    for p in initiatives
                    if by_quarter_init[q].get(p)
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


_DELAY_THRESHOLD = 3 * 3600  # 3 hours in seconds


def _response_decay(msgs: list[ParsedMessage]) -> dict:
    """
    Core LastSeen metric: detects progressive deterioration of reciprocity.

    Health score components (per month):
      - response_time  (40%) — how fast people respond
      - initiative     (30%) — how balanced the conversation-starting is
      - delay_rate     (30%) — how often someone made the other wait > 3h

    decay_score: 0.0 (healthy) → 1.0 (fully decayed)
    """
    monthly_rt: dict[str, list[float]] = defaultdict(list)
    monthly_msgs: dict[str, int] = defaultdict(int)
    monthly_turns: dict[str, int] = defaultdict(int)
    monthly_delayed: dict[str, int] = defaultdict(int)

    for m in msgs:
        monthly_msgs[_month(m.timestamp)] += 1

    for i in range(1, len(msgs)):
        prev, curr = msgs[i - 1], msgs[i]
        secs = (curr.timestamp - prev.timestamp).total_seconds()

        if prev.sender != curr.sender:
            # Cross-sender gap: track response time and delayed turns
            if 0 < secs <= _MAX_RESPONSE_WINDOW.total_seconds():
                monthly_rt[_month(curr.timestamp)].append(secs)
            monthly_turns[_month(curr.timestamp)] += 1
            if secs > _DELAY_THRESHOLD:
                monthly_delayed[_month(curr.timestamp)] += 1

    monthly_init: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for block in _split_into_blocks(msgs):
        monthly_init[_month(block[0].timestamp)][block[0].sender] += 1

    participants = sorted({m.sender for m in msgs})
    all_months = sorted(set(monthly_msgs) | set(monthly_rt) | set(monthly_init))

    if len(all_months) < 2:
        return {"trend": "insufficient_data"}

    evolution = []
    for month in all_months:
        avg_rt = round(statistics.mean(monthly_rt[month])) if monthly_rt[month] else None

        total_init = sum(monthly_init[month].values())
        if total_init >= 2 and participants:
            shares = [monthly_init[month].get(p, 0) / total_init for p in participants]
            imbalance = round(abs(shares[0] - 0.5) * 2, 3)
        else:
            imbalance = 0.0

        turns = monthly_turns[month]
        delay_rate = round(monthly_delayed[month] / turns, 3) if turns else 0.0

        evolution.append({
            "period": month,
            "avg_response_seconds": avg_rt,
            "message_count": monthly_msgs[month],
            "initiative_imbalance": imbalance,
            "delay_rate": delay_rate,
        })

    # Health score per month: 1.0 = perfect, 0.0 = dead
    # delay_rate acts as an additional penalty on top of the base score —
    # consistently making someone wait >3h chips away at relationship health.
    def _health(e: dict) -> float:
        rt_score = 1.0 - min((e["avg_response_seconds"] or 0) / _MAX_RESPONSE_WINDOW.total_seconds(), 1.0)
        base = rt_score * 0.5 + (1.0 - e["initiative_imbalance"]) * 0.5
        delay_penalty = e["delay_rate"] * 0.25
        return round(max(0.0, base - delay_penalty), 3)

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


def _delayed_replies(
    msgs: list[ParsedMessage],
    threshold_hours: float = 3.0,
) -> dict:
    """
    Counts how many times each person made the other wait more than
    threshold_hours before responding.

    Unit of measurement: one conversational TURN (consecutive messages from
    the same sender). If Person B takes > threshold_hours to respond to
    Person A's turn, that's one count for B — regardless of how many
    individual messages A sent in that turn.
    """
    threshold_secs = threshold_hours * 3600
    delayed: dict[str, int] = defaultdict(int)

    i = 0
    while i < len(msgs):
        sender = msgs[i].sender
        # Advance to end of this turn
        j = i + 1
        while j < len(msgs) and msgs[j].sender == sender:
            j += 1

        turn_end = msgs[j - 1]

        # Check if the next turn from the other person is > threshold away
        if j < len(msgs):
            gap_secs = (msgs[j].timestamp - turn_end.timestamp).total_seconds()
            if gap_secs > threshold_secs:
                delayed[msgs[j].sender] += 1

        i = j

    total = sum(delayed.values())
    return {
        "per_person": dict(delayed),
        "share": {
            p: round(c / total, 3) for p, c in delayed.items()
        } if total else {},
        "total": total,
        "threshold_hours": threshold_hours,
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
