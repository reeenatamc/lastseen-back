"""
Sentiment analyzer — measures emotional tone and its evolution over time.

Backend (settings.SENTIMENT_BACKEND):
  - "gemini" → Gemini scores every message and labels its emotion (see
    `sentiment_llm`); personal data is redacted before anything is sent.
    No lexicon fusion: that correction is for tweet-trained models.
  - "local" → local models, routed by context["_meta"]["language"]:
      "es" → pysentimiento (BETO-based, Spanish sentiment + emotion)
      anything else / "auto" → lxyuan/distilbert multilingual baseline
  - "auto" (default) → gemini when GEMINI_API_KEY is set, local otherwise.
If Gemini fails the analyzer falls back to the local models; if those cannot
be imported either, it returns {"error": "sentiment_unavailable"}.
`data["backend"]` reports which one produced the scores.

All backends produce a per-message score in [-1, +1] so the downstream
aggregations (per_person / evolution / emotional_drift) stay backend-agnostic.
Gemini and the Spanish local backend additionally fill an `emotions` block
(joy/sadness/...). torch/transformers/pysentimiento are imported lazily so the
worker can start without them.

Sampling: at most MAX_SAMPLE messages are analyzed. When the chat exceeds
this limit, messages are sampled stratified by ISO week (see `_sample`) so
the whole timeline, including the most recent weeks, stays represented while
keeping inference time under ~30s on CPU.
"""
from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta

from app.analyzers.base import AnalysisResult, BaseAnalyzer
from app.analyzers.temporal import _week, _week_start
from app.parsers.base import ParsedChat, ParsedMessage

logger = logging.getLogger(__name__)

_MULTILINGUAL_MODEL_ID = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
_MAX_SAMPLE = 2000
_BATCH_SIZE = 32
_LANG_DETECT_SAMPLE = 30  # messages used to auto-detect language
_MIN_PER_WEEK = 40  # sampling floor per ISO week (enough points for a weekly mean)
_MIN_WEEK_POINTS = 5  # scored messages a person needs in a week to get a mean
_RECENT_WINDOW_DAYS = 14  # length of the "recent stretch" window
_RECENT_MIN_POINTS = 20  # scored messages the window needs to be reported
_RECENT_SHIFT_DELTA = 0.1  # mean per-person delta that counts as a shift
_NEGATIVE_THRESHOLD = -0.2  # score below this counts as negative (as in _per_person)
_POSITIVE_THRESHOLD = 0.2  # score above this counts as positive (mirror of the negative one)

_multilingual_pipe = None
_es_sentiment_analyzer = None
_es_emotion_analyzer = None


def _get_multilingual_pipe():
    global _multilingual_pipe
    if _multilingual_pipe is None:
        from transformers import pipeline
        _multilingual_pipe = pipeline(
            "sentiment-analysis",
            model=_MULTILINGUAL_MODEL_ID,
            top_k=None,
            truncation=True,
            max_length=128,
            device=-1,
        )
    return _multilingual_pipe


def _get_es_sentiment():
    global _es_sentiment_analyzer
    if _es_sentiment_analyzer is None:
        from pysentimiento import create_analyzer
        _es_sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
    return _es_sentiment_analyzer


def _get_es_emotion():
    global _es_emotion_analyzer
    if _es_emotion_analyzer is None:
        from pysentimiento import create_analyzer
        _es_emotion_analyzer = create_analyzer(task="emotion", lang="es")
    return _es_emotion_analyzer


def _detect_language(msgs: list) -> str:
    """Sample up to _LANG_DETECT_SAMPLE messages and guess the language.

    Returns the detected language code (e.g. "es") or "unknown" when no
    signal is available.

    Robustness:
      • Seeds langdetect's DetectorFactory so detection is *deterministic*
        across runs — langdetect's classifier is otherwise stochastic for
        short ambiguous inputs (chat openers like "amor"/"are" exhibit this).
      • Detects on the *concatenated* sample first. langdetect needs ≳20
        characters for reliable inference; per-message detection on chat
        openers (often 4–10 chars) is unreliable. The concatenation gives
        it a single long string with enough signal.
      • Falls back to per-message voting only if the concatenated pass
        fails outright, preserving behaviour for tiny chats.
    """
    try:
        from langdetect import DetectorFactory, LangDetectException, detect
    except ImportError:
        return "unknown"

    DetectorFactory.seed = 42    # make detection deterministic across runs

    texts = [m.content for m in msgs[:_LANG_DETECT_SAMPLE] if m.content.strip()]
    if not texts:
        return "unknown"

    # Primary: detect on concatenated text (much more stable on short msgs).
    # Capped at 2000 chars so langdetect stays fast even on dense samples.
    joined = " ".join(texts)[:2000]
    try:
        return detect(joined)
    except LangDetectException:
        pass

    # Fallback: per-message majority vote.
    votes: dict[str, int] = {}
    for text in texts:
        try:
            lang = detect(text)
            votes[lang] = votes.get(lang, 0) + 1
        except LangDetectException:
            pass
    if not votes:
        return "unknown"
    return max(votes, key=lambda k: votes[k])


# ── Analyzer ──────────────────────────────────────────────────────────────────

class SentimentAnalyzer(BaseAnalyzer):
    name = "sentiment"

    def analyze(self, chat: ParsedChat, context: dict | None = None) -> AnalysisResult:
        language = (context or {}).get("_meta", {}).get("language", "auto")

        text_msgs = [m for m in chat.messages if not m.is_media and m.content.strip()]
        if len(text_msgs) < 5:
            return AnalysisResult(analyzer=self.name, data={"error": "insufficient_data"})

        # Resolve "auto" → detect actual language from the messages.
        if language == "auto":
            language = _detect_language(text_msgs)

        sample = _sample(text_msgs, _MAX_SAMPLE)

        from app.analyzers.sentiment_llm import SentimentLLMError, score_messages
        from app.core.config import settings

        backend = _resolve_backend(settings)
        scores: list[float] | None = None
        emotions: list[str] | None = None
        model_id = ""

        if backend == "gemini":
            try:
                scores, emotions = score_messages(
                    [m.content for m in sample],
                    api_key=settings.GEMINI_API_KEY,
                    model=settings.GEMINI_SENTIMENT_MODEL,
                )
                model_id = f"gemini:{settings.GEMINI_SENTIMENT_MODEL}"
            except SentimentLLMError as exc:
                logger.warning("gemini sentiment failed, falling back to local models: %s", exc)
                backend = "local"

        if backend == "local":
            try:
                scores, emotions, model_id = _score_local(sample, language)
            except ImportError as exc:
                logger.warning("local sentiment models unavailable: %s", exc)
                return AnalysisResult(analyzer=self.name, data={"error": "sentiment_unavailable"})

        data: dict = {
            "per_person": _per_person(sample, scores),
            "evolution": _evolution(sample, scores),
            "weekly": _evolution_weekly(sample, scores),
            "emotional_drift": _emotional_drift(sample, scores),
            "sample_size": len(sample),
            "total_text_messages": len(text_msgs),
            "model": model_id,
            "language": language,  # resolved language (never "auto")
            "backend": backend,
        }
        recent = _recent_shift(sample, scores)
        if recent is not None:
            data["recent"] = recent
        if emotions is not None:
            data["emotions_per_person"] = _emotions_per_person(sample, emotions)

        return AnalysisResult(analyzer=self.name, data=data)


def _resolve_backend(settings) -> str:
    """Pick "gemini" or "local" from the settings; without an API key it is always "local"."""
    if settings.SENTIMENT_BACKEND == "local" or not settings.GEMINI_API_KEY:
        return "local"
    return "gemini"  # "auto" or "gemini", with a key


def _score_local(sample: list[ParsedMessage], language: str) -> tuple[list[float], list[str] | None, str]:
    """Score with the local models; raises ImportError when they are not installed."""
    if language == "es":
        scores = _score_messages_spanish(sample)
        emotions = _emotion_messages_spanish(sample)
        return scores, emotions, "pysentimiento-es + hybrid lexicon (sentiment + emotion)"
    return _score_messages_multilingual(sample), None, "distilbert-multilingual"


# ── Metric functions (pure once scores are computed) ─────────────────────────

def _per_person(msgs: list[ParsedMessage], scores: list[float]) -> dict:
    by_person: dict[str, list[float]] = defaultdict(list)
    for msg, score in zip(msgs, scores):
        by_person[msg.sender].append(score)

    result = {}
    for person, s in by_person.items():
        pos_count = sum(1 for x in s if x > _POSITIVE_THRESHOLD)
        neg_count = sum(1 for x in s if x < _NEGATIVE_THRESHOLD)
        positive = pos_count / len(s)
        negative = neg_count / len(s)
        neutral = 1.0 - positive - negative
        # Short, uncharged messages win the "neutral" bucket by majority and read
        # as coldness; `charged` describes only the messages that carry tone.
        charged_count = pos_count + neg_count
        if charged_count:
            charged = {
                "share": round(charged_count / len(s), 3),
                "positive": round(pos_count / charged_count, 3),
                "negative": round(neg_count / charged_count, 3),
            }
        else:
            charged = {"share": 0.0, "positive": None, "negative": None}
        avg = statistics.mean(s)
        dominant = (
            "positive" if positive > max(neutral, negative)
            else "negative" if negative > neutral
            else "neutral"
        )
        result[person] = {
            "positive": round(positive, 3),
            "neutral": round(neutral, 3),
            "negative": round(negative, 3),
            "dominant": dominant,
            "avg_score": round(avg, 3),
            "charged": charged,
        }
    return result


def _evolution(msgs: list[ParsedMessage], scores: list[float]) -> list[dict]:
    by_quarter: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for msg, score in zip(msgs, scores):
        by_quarter[_quarter(msg.timestamp)][msg.sender].append(score)

    participants = sorted({m.sender for m in msgs})
    return [
        {
            "period": q,
            **{
                p: round(statistics.mean(by_quarter[q][p]), 3)
                for p in participants
                if by_quarter[q].get(p)
            },
        }
        for q in sorted(by_quarter)
    ]


def _emotional_drift(msgs: list[ParsedMessage], scores: list[float]) -> dict:
    """
    Measures how much the two participants' emotional tones diverged.
    score: 0.0 = always in sync, 1.0 = completely opposite tones.
    """
    participants = sorted({m.sender for m in msgs})
    if len(participants) != 2:
        return {"score": 0.0, "note": "only supported for 2-person chats"}

    p1, p2 = participants
    by_quarter: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for msg, score in zip(msgs, scores):
        by_quarter[_quarter(msg.timestamp)][msg.sender].append(score)

    quarters = sorted(by_quarter)
    divergences: list[float] = []
    avgs_p1: list[tuple[str, float]] = []
    avgs_p2: list[tuple[str, float]] = []

    for q in quarters:
        s1 = by_quarter[q].get(p1)
        s2 = by_quarter[q].get(p2)
        if s1 and s2:
            a1, a2 = statistics.mean(s1), statistics.mean(s2)
            divergences.append(abs(a1 - a2))
            avgs_p1.append((q, a1))
            avgs_p2.append((q, a2))

    if not divergences:
        return {"score": 0.0}

    drift_score = round(min(statistics.mean(divergences) / 2.0, 1.0), 3)

    direction = "aligned"
    if avgs_p1 and avgs_p2:
        last1, last2 = avgs_p1[-1][1], avgs_p2[-1][1]
        if last1 > last2 + 0.1:
            direction = f"{p1}_positive_{p2}_negative"
        elif last2 > last1 + 0.1:
            direction = f"{p2}_positive_{p1}_negative"

    turning_point = None
    if len(divergences) >= 2:
        max_increase, max_idx = 0.0, 0
        for i in range(1, len(divergences)):
            increase = divergences[i] - divergences[i - 1]
            if increase > max_increase:
                max_increase, max_idx = increase, i
        if max_increase > 0.05:
            turning_point = quarters[max_idx]

    return {
        "score": drift_score,
        "direction": direction,
        "turning_point": turning_point,
    }


def _evolution_weekly(msgs: list[ParsedMessage], scores: list[float]) -> list[dict]:
    """
    One entry per ISO week with each person's mean score.

    A person's mean is included only with at least _MIN_WEEK_POINTS scored
    messages that week; fewer points make the mean too noisy to plot.
    `n` counts every scored message of the week, whoever wrote it.
    """
    by_week: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for msg, score in zip(msgs, scores):
        by_week[_week(msg.timestamp)][msg.sender].append(score)

    result = []
    for week in sorted(by_week):
        people = by_week[week]
        entry: dict = {"period": week, "period_start": _week_start(week)}
        for person in sorted(people):
            if len(people[person]) >= _MIN_WEEK_POINTS:
                entry[person] = round(statistics.mean(people[person]), 3)
        entry["n"] = sum(len(v) for v in people.values())
        result.append(entry)
    return result


def _recent_shift(msgs: list[ParsedMessage], scores: list[float]) -> dict | None:
    """
    Compares the last _RECENT_WINDOW_DAYS days (counted from the last sampled
    message) against everything before them.

    Returns None when the chat spans less than twice the window (no meaningful
    baseline) or the window has fewer than _RECENT_MIN_POINTS scored messages.
    `shift` follows the mean of the per-person deltas (recent - baseline)
    against _RECENT_SHIFT_DELTA; only people present in both periods count.
    """
    if not msgs:
        return None
    first, last = msgs[0].timestamp, msgs[-1].timestamp
    if (last - first).days < 2 * _RECENT_WINDOW_DAYS:
        return None

    start = last - timedelta(days=_RECENT_WINDOW_DAYS)
    recent: dict[str, list[float]] = defaultdict(list)
    base: dict[str, list[float]] = defaultdict(list)
    for msg, score in zip(msgs, scores):
        (recent if msg.timestamp >= start else base)[msg.sender].append(score)

    if sum(len(v) for v in recent.values()) < _RECENT_MIN_POINTS:
        return None

    def _neg(s: list[float]) -> float:
        return sum(1 for x in s if x < _NEGATIVE_THRESHOLD) / len(s)

    per_person = {}
    for person in sorted(set(recent) & set(base)):
        r_avg, b_avg = statistics.mean(recent[person]), statistics.mean(base[person])
        per_person[person] = {
            "recent_avg": round(r_avg, 3),
            "baseline_avg": round(b_avg, 3),
            "delta": round(r_avg - b_avg, 3),
            "recent_negative_share": round(_neg(recent[person]), 3),
            "baseline_negative_share": round(_neg(base[person]), 3),
            "n": len(recent[person]),
        }

    mean_delta = (
        statistics.mean(p["delta"] for p in per_person.values()) if per_person else 0.0
    )
    shift = (
        "more_negative" if mean_delta <= -_RECENT_SHIFT_DELTA
        else "more_positive" if mean_delta >= _RECENT_SHIFT_DELTA
        else "stable"
    )
    return {
        "window_days": _RECENT_WINDOW_DAYS,
        "start": start.strftime("%Y-%m-%d"),
        "per_person": per_person,
        "shift": shift,
    }


def _emotions_per_person(msgs: list[ParsedMessage], emotions: list[str]) -> dict:
    """Aggregate dominant emotion frequencies per participant (pysentimiento only)."""
    by_person: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    counts: dict[str, int] = defaultdict(int)
    for msg, emo in zip(msgs, emotions):
        by_person[msg.sender][emo] += 1
        counts[msg.sender] += 1

    result = {}
    for person, emo_counts in by_person.items():
        total = counts[person]
        shares = {emo: round(c / total, 3) for emo, c in emo_counts.items()}
        dominant = max(shares.items(), key=lambda kv: kv[1])[0]
        result[person] = {"shares": shares, "dominant": dominant, "sample_size": total}
    return result


# ── Inference ─────────────────────────────────────────────────────────────────

def _score_messages_multilingual(msgs: list[ParsedMessage]) -> list[float]:
    """Multilingual baseline: returns a score per message in [-1, +1]."""
    pipe = _get_multilingual_pipe()
    texts = [m.content[:512] for m in msgs]
    results: list[float] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch_outputs = pipe(texts[i : i + _BATCH_SIZE])
        for output in batch_outputs:
            label_scores = {item["label"]: item["score"] for item in output}
            score = label_scores.get("positive", 0.0) - label_scores.get("negative", 0.0)
            results.append(round(score, 4))

    return results


def _score_messages_spanish(msgs: list[ParsedMessage]) -> list[float]:
    """
    pysentimiento sentiment, **hybrid-augmented for couple-chat register**.

    Steps:
      1. Get raw `POS - NEG` score per message from pysentimiento (Twitter-trained).
      2. Run each message through the lexicon-fusion scorer, which combines
         the ML score with NRC/AFINN Spanish, hand-curated intimate-couple
         vocabulary, and the Emoji Sentiment Ranking — see
         `app.analyzers.lexicons.scorer.hybrid_score` for the fusion formula.

    The augmentation corrects for the documented register mismatch between
    tweet-trained sentiment models and intimate WhatsApp conversations
    (affectionate words like "amor"/"❤️"/diminutives are systematically
    under-weighted; playful drama like "me muero"/"convulsiono" is read as
    negative). See `lexicons/CITATIONS.md`.
    """
    from app.analyzers.lexicons.scorer import hybrid_score

    analyzer = _get_es_sentiment()
    texts = [m.content[:512] for m in msgs]
    scores: list[float] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch_texts = texts[i : i + _BATCH_SIZE]
        outputs = analyzer.predict(batch_texts)
        for text, out in zip(batch_texts, outputs):
            probas = out.probas
            base = probas.get("POS", 0.0) - probas.get("NEG", 0.0)
            scores.append(hybrid_score(text, base))

    return scores


def _emotion_messages_spanish(msgs: list[ParsedMessage]) -> list[str]:
    """pysentimiento emotion: returns the top-1 emotion label per message."""
    analyzer = _get_es_emotion()
    texts = [m.content[:512] for m in msgs]
    labels: list[str] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        outputs = analyzer.predict(texts[i : i + _BATCH_SIZE])
        for out in outputs:
            labels.append(out.output)

    return labels


# ── Helpers ───────────────────────────────────────────────────────────────────

def _spaced(items: list[ParsedMessage], k: int) -> list[ParsedMessage]:
    """Pick k evenly spaced items, always including the first and the last."""
    n = len(items)
    if k >= n:
        return items
    if k == 1:
        return [items[-1]]
    return [items[round(i * (n - 1) / (k - 1))] for i in range(k)]


def _sample(msgs: list[ParsedMessage], max_n: int) -> list[ParsedMessage]:
    """
    Stratified sample by ISO week, chronological, at most max_n messages.

    A plain stride (`msgs[::step][:max_n]`) overshoots max_n and the trailing
    slice then drops the end of the chat, so the latest conversation was never
    scored. Here every week keeps a floor of _MIN_PER_WEEK messages: low-volume
    weeks (typically the last ones of a relationship that fades out) would
    otherwise get too few points for a weekly mean. The remaining budget is
    split in proportion to each week's surplus over its floor. If the floors
    alone exceed max_n (multi-year chats) the floor drops to
    max_n // number_of_weeks (minimum 1). Inside a week the positions are
    evenly spaced and include its first and last message.
    """
    if len(msgs) <= max_n:
        return msgs

    weeks: dict[str, list[ParsedMessage]] = defaultdict(list)
    for m in msgs:
        weeks[_week(m.timestamp)].append(m)
    keys = sorted(weeks)

    floor = _MIN_PER_WEEK
    if sum(min(floor, len(weeks[k])) for k in keys) > max_n:
        floor = max(1, max_n // len(keys))
    alloc = {k: min(floor, len(weeks[k])) for k in keys}

    remaining = max_n - sum(alloc.values())
    surplus = {k: len(weeks[k]) - alloc[k] for k in keys}
    total_surplus = sum(surplus.values())
    if remaining > 0 and total_surplus > 0:
        shares = {k: remaining * surplus[k] / total_surplus for k in keys}
        extra = {k: min(int(shares[k]), surplus[k]) for k in keys}
        leftover = remaining - sum(extra.values())
        # Hand out rounding leftovers by largest fractional part.
        for k in sorted(keys, key=lambda k: shares[k] - int(shares[k]), reverse=True):
            if leftover <= 0:
                break
            if extra[k] < surplus[k]:
                extra[k] += 1
                leftover -= 1
        for k in keys:
            alloc[k] += extra[k]

    sampled = [m for k in keys for m in _spaced(weeks[k], alloc[k])]
    sampled.sort(key=lambda m: m.timestamp)  # stable; no-op when input is ordered
    # More weeks than max_n: one message per week still overshoots; thin evenly.
    return _spaced(sampled, max_n) if len(sampled) > max_n else sampled


def _quarter(dt: datetime) -> str:
    return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
