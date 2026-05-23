"""
Sentiment analyzer — measures emotional tone and its evolution over time.

Language routing (read from context["_meta"]["language"]):
  - "es" → pysentimiento (BETO-based, Spanish sentiment + emotion)
  - anything else / "auto" → lxyuan/distilbert multilingual baseline

Both backends produce a per-message score in [-1, +1] so the downstream
aggregations (per_person / evolution / emotional_drift) stay backend-agnostic.
The Spanish backend additionally fills an `emotions` block (joy/sadness/...).

Sampling: at most MAX_SAMPLE messages are analyzed. When the chat exceeds
this limit, messages are sampled uniformly to preserve the temporal
distribution while keeping inference time under ~30s on CPU.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime

from app.analyzers.base import AnalysisResult, BaseAnalyzer
from app.parsers.base import ParsedChat, ParsedMessage

_MULTILINGUAL_MODEL_ID = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
_MAX_SAMPLE = 2000
_BATCH_SIZE = 32
_LANG_DETECT_SAMPLE = 30  # messages used to auto-detect language

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

        if language == "es":
            scores = _score_messages_spanish(sample)
            emotions = _emotion_messages_spanish(sample)
            model_id = "pysentimiento-es + hybrid lexicon (sentiment + emotion)"
        else:
            scores = _score_messages_multilingual(sample)
            emotions = None
            model_id = "distilbert-multilingual"

        data: dict = {
            "per_person": _per_person(sample, scores),
            "evolution": _evolution(sample, scores),
            "emotional_drift": _emotional_drift(sample, scores),
            "sample_size": len(sample),
            "total_text_messages": len(text_msgs),
            "model": model_id,
            "language": language,  # resolved language (never "auto")
        }
        if emotions is not None:
            data["emotions_per_person"] = _emotions_per_person(sample, emotions)

        return AnalysisResult(analyzer=self.name, data=data)


# ── Metric functions (pure once scores are computed) ─────────────────────────

def _per_person(msgs: list[ParsedMessage], scores: list[float]) -> dict:
    by_person: dict[str, list[float]] = defaultdict(list)
    for msg, score in zip(msgs, scores):
        by_person[msg.sender].append(score)

    result = {}
    for person, s in by_person.items():
        positive = sum(1 for x in s if x > 0.2) / len(s)
        negative = sum(1 for x in s if x < -0.2) / len(s)
        neutral = 1.0 - positive - negative
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

def _sample(msgs: list[ParsedMessage], max_n: int) -> list[ParsedMessage]:
    if len(msgs) <= max_n:
        return msgs
    step = len(msgs) // max_n
    return msgs[::step][:max_n]


def _quarter(dt: datetime) -> str:
    return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
