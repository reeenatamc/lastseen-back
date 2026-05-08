"""
Sentiment analyzer — measures emotional tone and its evolution over time.

Model: lxyuan/distilbert-base-multilingual-cased-sentiments-student
  - Multilingual (Spanish + English + 100 other languages)
  - Outputs: positive / neutral / negative with confidence scores
  - Lazy-loaded on first use to avoid startup overhead

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

_MODEL_ID = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
_MAX_SAMPLE = 2000
_BATCH_SIZE = 32

_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        _pipe = pipeline(
            "sentiment-analysis",
            model=_MODEL_ID,
            top_k=None,
            truncation=True,
            max_length=128,
            device=-1,  # CPU; set to 0 for GPU
        )
    return _pipe


# ── Analyzer ──────────────────────────────────────────────────────────────────

class SentimentAnalyzer(BaseAnalyzer):
    name = "sentiment"

    def analyze(self, chat: ParsedChat) -> AnalysisResult:
        text_msgs = [m for m in chat.messages if not m.is_media and m.content.strip()]
        if len(text_msgs) < 5:
            return AnalysisResult(analyzer=self.name, data={"error": "insufficient_data"})

        sample = _sample(text_msgs, _MAX_SAMPLE)
        scores = _score_messages(sample)

        return AnalysisResult(
            analyzer=self.name,
            data={
                "per_person": _per_person(sample, scores),
                "evolution": _evolution(sample, scores),
                "emotional_drift": _emotional_drift(sample, scores),
                "sample_size": len(sample),
                "total_text_messages": len(text_msgs),
            },
        )


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

    # Normalize: max possible divergence is 2.0 (+1 vs -1)
    drift_score = round(min(statistics.mean(divergences) / 2.0, 1.0), 3)

    # Direction: who is more positive in the final quarter?
    direction = "aligned"
    if avgs_p1 and avgs_p2:
        last1, last2 = avgs_p1[-1][1], avgs_p2[-1][1]
        if last1 > last2 + 0.1:
            direction = f"{p1}_positive_{p2}_negative"
        elif last2 > last1 + 0.1:
            direction = f"{p2}_positive_{p1}_negative"

    # Turning point: quarter where divergence increased the most
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


# ── Inference ─────────────────────────────────────────────────────────────────

def _score_messages(msgs: list[ParsedMessage]) -> list[float]:
    """Run batched inference; returns a score per message in [-1, +1]."""
    pipe = _get_pipe()
    texts = [m.content[:512] for m in msgs]
    results: list[float] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch_outputs = pipe(texts[i : i + _BATCH_SIZE])
        for output in batch_outputs:
            label_scores = {item["label"]: item["score"] for item in output}
            score = label_scores.get("positive", 0.0) - label_scores.get("negative", 0.0)
            results.append(round(score, 4))

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample(msgs: list[ParsedMessage], max_n: int) -> list[ParsedMessage]:
    if len(msgs) <= max_n:
        return msgs
    step = len(msgs) // max_n
    return msgs[::step][:max_n]


def _quarter(dt: datetime) -> str:
    return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
