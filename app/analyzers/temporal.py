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

# A "conversation block" ends when the pair goes quiet for longer than the
# block gap. A fixed 4 h is wrong for both extremes: hyper-active couples
# almost never pause 4 h (≈ everything is one block → initiative meaningless),
# slow texters always exceed 4 h (≈ every message is its own block). So the
# threshold adapts to THIS pair's rhythm — the 95th-percentile gap — clamped
# to a sane band, with a fixed fallback when there isn't enough data to adapt.
_INITIATIVE_GAP = timedelta(hours=4)             # fallback only (sample too small to adapt)
_ADAPTIVE_GAP_PCT = 0.95                         # a new conversation = a silence in the pair's top 5%
_ADAPTIVE_GAP_FLOOR = timedelta(hours=1)         # never split a conversation on a sub-hour lull
_ADAPTIVE_GAP_CEILING = timedelta(hours=6)       # a > 6 h silence always ends a conversation
_ADAPTIVE_GAP_MIN_SAMPLE = 30                    # need this many gaps before a rhythm is meaningful

# Beyond this window a delayed message is not counted as a "response"
_MAX_RESPONSE_WINDOW = timedelta(hours=24)

_TOP_GAPS = 5


# ── Analyzer ─────────────────────────────────────────────────────────────────

class TemporalAnalyzer(BaseAnalyzer):
    name = "temporal"

    MIN_MESSAGES = 50
    MIN_DAYS = 7

    def analyze(self, chat: ParsedChat, context: dict | None = None) -> AnalysisResult:
        msgs = chat.messages
        if len(msgs) < 2:
            return AnalysisResult(analyzer=self.name, data={"error": "insufficient_data"})

        total_days = (msgs[-1].timestamp - msgs[0].timestamp).days
        if len(msgs) < self.MIN_MESSAGES or total_days < self.MIN_DAYS:
            return AnalysisResult(
                analyzer=self.name,
                data={
                    "error": "insufficient_data",
                    "detail": f"El chat tiene {len(msgs)} mensajes en {total_days} días. "
                              f"Se necesitan al menos {self.MIN_MESSAGES} mensajes y "
                              f"{self.MIN_DAYS} días para un análisis significativo.",
                },
            )

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
    Classifies each gap larger than the adaptive block threshold (see
    _adaptive_gap) into four buckets:

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

    continuous = _is_continuous_thread(msgs)

    return {
        "confidence": {
            "level": "low" if continuous else "ok",
            "reason": "continuous_thread" if continuous else None,
        },
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


# ── response_decay tuning surface ─────────────────────────────────────────────
# The product-meaningful knobs of the core metric, grouped so they are easy to
# find and reason about. Changing any of these changes what "decay" means.

_DELAY_THRESHOLD = 3 * 3600          # seconds; a turn handoff slower than this starts to count as neglect
_RT_HALF_LIFE_SECONDS = 5400         # avg response time that scores 0.5 (90 min)
_RT_STEEPNESS = 1.2                  # logistic steepness for rt_score
_SILENCE_GRACE_DAYS = 1.0            # mutual silence at/below this scores 1.0
_SILENCE_DEAD_DAYS = 3.0             # mutual silence at/above this makes the week's silence 0.0
_NEGLECT_UNIT_HOURS = 12.0           # a single one-sided wait this long = 1.0 neglect unit
_NEGLECT_SATURATION = 3.0            # this many neglect units in a week drives neglect_score to 0
_HEALTH_W_RT = 0.25                  # responsiveness *when present*
_HEALTH_W_BALANCE = 0.15             # who carries the conversation-starting
_HEALTH_W_NEGLECT = 0.30             # one-sided abandonment (volume-robust)
_HEALTH_W_SILENCE = 0.30             # mutual silence — weights sum to 1.0
_TREND_DELTA = 0.12                  # symmetric trend threshold
_TURNING_POINT_MIN_DROP = 0.15       # min leading-vs-trailing health drop to flag a turning point
_CLOSING_WINDOW_WEEKS = 3            # the final stretch inspected for a fade-out
_CLOSING_MIN_BASELINE_WEEKS = 4      # weeks before the window needed to call anything "normal"
_CLOSING_VOLUME_RATIO = 0.5          # window volume at/below this share of baseline = fade-out
_CLOSING_SILENCE_MIN_DAYS = 1.5      # a window silence must be at least this long to count
_CLOSING_SILENCE_FACTOR = 2.0        # ...and at least this many times the longest earlier silence
_CLOSING_START_RATIO = 0.7           # first window week below this share of baseline marks the start


def _iter_turn_handoffs(msgs: list[ParsedMessage]):
    """
    Yield (responder, gap_seconds, response_timestamp) for every turn handoff.

    A *turn* is a maximal run of consecutive messages from the same sender.
    The gap is measured from the END of one turn to the START of the next
    (the other person's) turn. This is the single shared definition of a
    conversational turn — both _response_decay and _delayed_replies consume it
    so their numbers are directly comparable.
    """
    i = 0
    n = len(msgs)
    while i < n:
        sender = msgs[i].sender
        j = i + 1
        while j < n and msgs[j].sender == sender:
            j += 1
        if j < n:
            gap = (msgs[j].timestamp - msgs[j - 1].timestamp).total_seconds()
            yield msgs[j].sender, gap, msgs[j].timestamp
        i = j


def _rt_score(avg_rt: float | None) -> float:
    """Logistic on avg response time: seconds/minutes → ~1.0, 90 min → 0.5,
    ≥24 h → ~0. No response data for the month → 0.0 (silence is not health)."""
    if avg_rt is None:
        return 0.0
    return 1.0 / (1.0 + (avg_rt / _RT_HALF_LIFE_SECONDS) ** _RT_STEEPNESS)


def _silence_score(max_gap_days: float | None) -> float:
    """1.0 while contact stays roughly daily, linearly down to 0.0 once the
    longest mutual silence in the week reaches _SILENCE_DEAD_DAYS."""
    if max_gap_days is None:
        return 1.0
    span = _SILENCE_DEAD_DAYS - _SILENCE_GRACE_DAYS
    return max(0.0, min(1.0, 1.0 - max(0.0, max_gap_days - _SILENCE_GRACE_DAYS) / span))


def _neglect_score(units: float) -> float:
    """
    One-sided abandonment, volume-robust.

    rt_score / delay_rate are *averages or rates* — thousands of instant
    burst replies wash out the handful of multi-hour ghostings, so a chat
    where one person was repeatedly left waiting for hours still scores
    "fast". neglect counts the bad episodes directly instead of diluting
    them by total volume.

    Each turn handoff slower than _DELAY_THRESHOLD (but within the 24 h
    window — beyond that it is silence, handled separately) contributes
    `min(wait_hours / _NEGLECT_UNIT_HOURS, 1.0)` units, so a 12 h+ ghosting
    counts ~5× a 3 h one. _NEGLECT_SATURATION units in a week → 0.0.
    """
    return max(0.0, min(1.0, 1.0 - units / _NEGLECT_SATURATION))


def _closing_silence_hit(silence: float | None, baseline_max_silence: float) -> bool:
    """True when a silence is long in absolute terms and unusual for this chat."""
    s = silence or 0.0
    return s >= _CLOSING_SILENCE_MIN_DAYS and s >= _CLOSING_SILENCE_FACTOR * baseline_max_silence


def _closing_phase(evolution: list[dict]) -> dict:
    """
    Detects a fade-out in the last _CLOSING_WINDOW_WEEKS weeks.

    Thirds-based trend dilutes a short collapse at the end, and volume never
    enters the health formula. This looks only at the final window against the
    weeks before it, using evidence that survives an export cut mid-week:
    messages per covered day (not raw counts) and silences between real
    messages. Detected when the window's daily volume falls to
    _CLOSING_VOLUME_RATIO of the baseline median, OR its longest silence is
    both >= _CLOSING_SILENCE_MIN_DAYS and >= _CLOSING_SILENCE_FACTOR times the
    longest earlier silence. `start` is the first window week that already
    shows the drop (volume <= _CLOSING_START_RATIO of baseline, or a
    qualifying silence), falling back to the first window week.

    Result shapes: with too few baseline weeks to compare, returns
    {"detected": False, "reason": "insufficient_baseline"} (nothing to compare
    against, which is not the same as "stable"). With enough baseline, the full
    dict is returned and has no "reason" key, whether detected or not.
    """
    split = len(evolution) - _CLOSING_WINDOW_WEEKS
    if split < _CLOSING_MIN_BASELINE_WEEKS:
        return {"detected": False, "reason": "insufficient_baseline"}

    prior, window = evolution[:split], evolution[split:]
    base = statistics.median(e["messages_per_day"] for e in prior)
    window_mean = statistics.mean(e["messages_per_day"] for e in window)
    volume_ratio = window_mean / base if base > 0 else 1.0

    baseline_max = max((e["silence_gap_days"] or 0.0) for e in prior)
    window_max = max((e["silence_gap_days"] or 0.0) for e in window)

    detected = volume_ratio <= _CLOSING_VOLUME_RATIO or _closing_silence_hit(window_max, baseline_max)

    start = None
    if detected:
        start = window[0]["period_start"]
        for e in window:
            if (
                e["messages_per_day"] <= _CLOSING_START_RATIO * base
                or _closing_silence_hit(e["silence_gap_days"], baseline_max)
            ):
                start = e["period_start"]
                break

    return {
        "detected": detected,
        "start": start,
        "volume_ratio": round(volume_ratio, 3),
        "max_silence_days": round(window_max, 1),
        "baseline_max_silence_days": round(baseline_max, 1),
        "window_weeks": len(window),
    }


def _response_decay(msgs: list[ParsedMessage]) -> dict:
    """
    Core LastSeen metric: detects progressive deterioration of reciprocity.

    Aggregation is per ISO week (Mon–Sun). Weeks are calendar-aligned and
    sortable as strings; week-level resolution lets the metric speak on short,
    intense chats (a real 11-day conversation is ~2 weeks, not 1 month).

    Per-week health is a single weighted sum in [0, 1] (weights sum to 1):
      - rt_score      (0.25) — logistic on avg response time *when present*
      - balance       (0.15) — 1 − initiative imbalance
      - neglect_score (0.30) — one-sided multi-hour abandonment, volume-robust
      - silence_score (0.30) — penalises the longest mutual silence in the week

    rt_score and silence are deliberately blind to one-sided neglect: an
    average is dragged down by burst replies, and one person spamming an
    absent partner produces no mutual-silence gap at all. neglect_score is
    the signal that actually sees "they were left waiting for hours while
    reaching out" — it counts severity-weighted episodes, not a rate, so a
    high message volume cannot mask it.

    Any component with no data for the week contributes 0 (absence of activity
    is not health). There is deliberately no separate "inactive week" constant
    — a fully silent week falls out near 0 on its own. Silence is measured at
    timestamp resolution, so a 3-day gap inside an otherwise active week is
    still caught.

    decay_score: 0.0 (healthy) → 1.0 (fully decayed), from the recent third.

    Closing phase: the thirds comparison and the turning-point search both
    discount the end of the data, because the last weeks are where export
    artifacts live. But when someone exports a chat after it ended, the fade-out
    is exactly there. _closing_phase uses evidence robust to truncation (volume
    per covered day, silences between real messages); if it fires, trend becomes
    "deteriorating" (trend_basis "closing_phase") and, when the capped search
    found nothing, its start becomes the turning point.
    """
    weekly_rt: dict[str, list[float]] = defaultdict(list)
    weekly_msgs: dict[str, int] = defaultdict(int)
    weekly_turns: dict[str, int] = defaultdict(int)
    weekly_delayed: dict[str, int] = defaultdict(int)
    weekly_neglect: dict[str, float] = defaultdict(float)
    weekly_max_gap: dict[str, float] = {}

    for m in msgs:
        weekly_msgs[_week(m.timestamp)] += 1

    # Mutual silence: longest gap between *any* consecutive messages, attributed
    # to the week of the message that broke the silence. The trailing edge of
    # the data has no following message, so a chat that simply stops exporting
    # is never penalised here ("except if it's the end of the conversation").
    for i in range(1, len(msgs)):
        gap_days = (msgs[i].timestamp - msgs[i - 1].timestamp).total_seconds() / 86400
        wk = _week(msgs[i].timestamp)
        if gap_days > weekly_max_gap.get(wk, 0.0):
            weekly_max_gap[wk] = gap_days

    # Response time, delay and neglect all read the same turn-handoff stream.
    # neglect: a handoff in the 3 h–24 h window is a one-sided wait; its
    # severity is the wait length, saturating at _NEGLECT_UNIT_HOURS so a
    # single catastrophic ghosting cannot count as more than one full unit.
    for responder, gap, ts in _iter_turn_handoffs(msgs):
        wk = _week(ts)
        weekly_turns[wk] += 1
        if 0 < gap <= _MAX_RESPONSE_WINDOW.total_seconds():
            weekly_rt[wk].append(gap)
        if _DELAY_THRESHOLD < gap <= _MAX_RESPONSE_WINDOW.total_seconds():
            weekly_delayed[wk] += 1
            wait_hours = gap / 3600
            weekly_neglect[wk] += min(wait_hours / _NEGLECT_UNIT_HOURS, 1.0)

    weekly_init: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for block in _split_into_blocks(msgs):
        weekly_init[_week(block[0].timestamp)][block[0].sender] += 1

    participants = sorted({m.sender for m in msgs})
    all_weeks = sorted(set(weekly_msgs) | set(weekly_rt) | set(weekly_init))

    if len(all_weeks) < 2:
        return {"trend": "insufficient_data"}

    # Days of each week actually covered by the chat: Mon–Sun clipped to the
    # first/last message dates, inclusive, so a partial edge week is not
    # mistaken for a quiet one.
    first_day = msgs[0].timestamp.date()
    last_day = msgs[-1].timestamp.date()

    def _covered_days(week: str) -> int:
        monday = datetime.strptime(_week_start(week), "%Y-%m-%d").date()
        lo = max(monday, first_day)
        hi = min(monday + timedelta(days=6), last_day)
        return max(1, (hi - lo).days + 1)

    evolution = []
    for week in all_weeks:
        avg_rt = round(statistics.mean(weekly_rt[week])) if weekly_rt[week] else None

        total_init = sum(weekly_init[week].values())
        if total_init >= 2 and len(participants) == 2:
            share0 = weekly_init[week].get(participants[0], 0) / total_init
            imbalance = round(abs(share0 - 0.5) * 2, 3)
        else:
            # Not enough conversations started to judge balance → treat the
            # balance dimension as unhealthy, not as a free pass.
            imbalance = 1.0

        turns = weekly_turns[week]
        delay_rate = round(weekly_delayed[week] / turns, 3) if turns else 0.0
        neglect_units = round(weekly_neglect[week], 3)
        max_gap = weekly_max_gap.get(week)

        evolution.append({
            "period": week,
            "period_start": _week_start(week),
            "avg_response_seconds": avg_rt,
            "message_count": weekly_msgs[week],
            "messages_per_day": round(weekly_msgs[week] / _covered_days(week), 1),
            "initiative_imbalance": imbalance,
            "delay_rate": delay_rate,
            "neglect_units": neglect_units,
            "silence_gap_days": round(max_gap, 1) if max_gap is not None else None,
            "turns": turns,
        })

    def _health(e: dict) -> float:
        rt = _rt_score(e["avg_response_seconds"])
        balance = 1.0 - e["initiative_imbalance"]
        neglect = _neglect_score(e["neglect_units"])
        silence = _silence_score(e["silence_gap_days"])
        h = (
            rt * _HEALTH_W_RT
            + balance * _HEALTH_W_BALANCE
            + neglect * _HEALTH_W_NEGLECT
            + silence * _HEALTH_W_SILENCE
        )
        return round(max(0.0, min(1.0, h)), 3)

    for e in evolution:
        e["health"] = _health(e)
    health = [e["health"] for e in evolution]

    # Trend: recent third vs early third, symmetric threshold. The product's
    # pessimism lives in the health formula and silence decay, not here — this
    # classifier stays honest.
    third = max(1, len(health) // 3)
    delta = statistics.mean(health[-third:]) - statistics.mean(health[:third])
    if delta < -_TREND_DELTA:
        trend = "deteriorating"
    elif delta > _TREND_DELTA:
        trend = "improving"
    else:
        trend = "stable"

    decay_score = round(max(0.0, min(1.0, 1.0 - statistics.mean(health[-third:]))), 3)

    closing_phase = _closing_phase(evolution)
    trend_basis = "thirds"
    if closing_phase["detected"] and trend != "deteriorating":
        trend = "deteriorating"
        trend_basis = "closing_phase"

    # Turning point: the split that maximises (mean health before) − (mean
    # health after). Catches slow sustained declines, not just one-week
    # cliffs. Suppressed in the last 20% — that tail can hold export-boundary
    # artifacts (a cut-off week), so health alone is not trusted there. A real
    # fade-out at the end is covered separately by closing_phase, which relies
    # on volume per covered day and silences between real messages.
    # Reported as the week's start date (YYYY-MM-DD) so the narrative layer
    # and UI get something human, not an opaque ISO-week code.
    cutoff_idx = max(1, round(len(health) * 0.8))
    turning_point = None
    best_drop = _TURNING_POINT_MIN_DROP
    for i in range(1, cutoff_idx):
        drop = statistics.mean(health[:i]) - statistics.mean(health[i:])
        if drop > best_drop:
            best_drop = drop
            turning_point = evolution[i]["period_start"]

    if turning_point is None and closing_phase["detected"]:
        turning_point = closing_phase["start"]

    return {
        "trend": trend,
        "trend_basis": trend_basis,
        "decay_score": decay_score,
        "turning_point": turning_point,
        "closing_phase": closing_phase,
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

    Shares the _iter_turn_handoffs definition with response_decay's delay_rate
    so the two numbers are directly comparable.
    """
    threshold_secs = threshold_hours * 3600
    delayed: dict[str, int] = defaultdict(int)

    for responder, gap, _ in _iter_turn_handoffs(msgs):
        if gap > threshold_secs:
            delayed[responder] += 1

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

def _gap_p95_seconds(msgs: list[ParsedMessage]) -> float | None:
    """
    The 95th-percentile inter-message gap (seconds), or None when there are
    too few gaps to infer a rhythm. Single source of truth for both the
    adaptive block threshold and the continuous-thread check.
    """
    if len(msgs) < 2:
        return None
    gaps = sorted(
        (msgs[i].timestamp - msgs[i - 1].timestamp).total_seconds()
        for i in range(1, len(msgs))
        if msgs[i].timestamp >= msgs[i - 1].timestamp
    )
    if len(gaps) < _ADAPTIVE_GAP_MIN_SAMPLE:
        return None
    idx = min(int(len(gaps) * _ADAPTIVE_GAP_PCT), len(gaps) - 1)
    return gaps[idx]


def _adaptive_gap(msgs: list[ParsedMessage]) -> timedelta:
    """
    Block-split threshold tuned to this pair's rhythm: the 95th-percentile
    inter-message gap, clamped to [1 h, 6 h]. Falls back to the fixed
    _INITIATIVE_GAP when there are too few gaps to infer a rhythm — you
    cannot read a couple's cadence from a handful of messages.
    """
    p95 = _gap_p95_seconds(msgs)
    if p95 is None:
        return _INITIATIVE_GAP
    secs = min(
        max(p95, _ADAPTIVE_GAP_FLOOR.total_seconds()),
        _ADAPTIVE_GAP_CEILING.total_seconds(),
    )
    return timedelta(seconds=secs)


def _is_continuous_thread(msgs: list[ParsedMessage]) -> bool:
    """
    True when the pair is in essentially continuous contact: even their
    95th-percentile pause is below the block floor, so the adaptive gap is
    clamped *up* to the floor and "conversations separated by silence" is an
    arbitrary cut. In that regime the initiative breakdown is real but
    semantically thin — they never really stop talking — so it should be
    reported as low-confidence rather than read as precise.
    """
    p95 = _gap_p95_seconds(msgs)
    return p95 is not None and p95 < _ADAPTIVE_GAP_FLOOR.total_seconds()


def split_into_blocks(
    msgs: list[ParsedMessage],
    gap: timedelta | None = None,
) -> list[list[ParsedMessage]]:
    """
    Split a message list into conversation blocks. With `gap=None` (default)
    the threshold adapts to the pair's own rhythm (see _adaptive_gap); pass an
    explicit timedelta to force a fixed threshold.
    """
    if not msgs:
        return []
    if gap is None:
        gap = _adaptive_gap(msgs)
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


def _week(dt: datetime) -> str:
    """ISO week key, e.g. 2026-W19. ISO year handles the Dec/Jan boundary;
    zero-padded week keeps lexical sort == chronological sort."""
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _week_start(week_key: str) -> str:
    """Monday of an ISO week as YYYY-MM-DD (the human-facing label)."""
    iso_year, iso_week = week_key.split("-W")
    return datetime.fromisocalendar(int(iso_year), int(iso_week), 1).strftime("%Y-%m-%d")


def _p90(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    return sorted_v[min(int(len(sorted_v) * 0.9), len(sorted_v) - 1)]
