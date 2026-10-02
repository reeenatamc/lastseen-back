"""Paywall views of an analysis result. Pure functions, no I/O."""
from copy import deepcopy

# ── Constants ─────────────────────────────────────────────────────────────────

# Temporal sub-sections visible before unlocking: they show the chat is real
# without revealing the relationship dynamics the user is paying for.
_PREVIEW_TEMPORAL_KEYS = ("overview", "response_time", "activity_patterns", "message_length")

# Narrative field kept as-is in the preview; every other narrative field is nulled.
_PREVIEW_NARRATIVE_KEY = "resumen"

# What the preview hides, returned to the frontend to render the lock screen.
LOCKED_SECTIONS: list[str] = [
    "temporal.initiative_balance",
    "temporal.conversation_gaps",
    "temporal.response_decay",
    "temporal.delayed_replies",
    "sentiment",
    "conflict",
    "narrative",
]

# ── Builders ──────────────────────────────────────────────────────────────────

def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def build_preview(analysis: dict) -> dict:
    """Return a reduced copy of the analyzers dict. The input is never mutated."""
    preview: dict = {}

    temporal = _as_dict(analysis.get("temporal"))
    if "temporal" in analysis:
        if "error" in temporal:
            preview["temporal"] = deepcopy(temporal)
        else:
            preview["temporal"] = {
                key: deepcopy(temporal[key]) for key in _PREVIEW_TEMPORAL_KEYS if key in temporal
            }

    narrative = _as_dict(analysis.get("narrative"))
    if "narrative" in analysis:
        if "error" in narrative:
            preview["narrative"] = deepcopy(narrative)
        else:
            preview["narrative"] = {
                key: (deepcopy(value) if key == _PREVIEW_NARRATIVE_KEY else None)
                for key, value in narrative.items()
            }

    return preview


def build_teaser(analysis: dict) -> dict:
    """Facts for the lock screen. Counts and flags only: no dates, names or text.

    Only uses analyzers that run in the free preview (no sentiment).
    """
    temporal = _as_dict(analysis.get("temporal"))
    conflict = _as_dict(analysis.get("conflict"))
    decay = _as_dict(temporal.get("response_decay"))

    episodes = conflict.get("episodes")

    return {
        "conflict_episodes": len(episodes) if isinstance(episodes, list) else None,
        "turning_point_detected": bool(decay.get("turning_point")),
        "closing_phase_detected": bool(decay.get("closing_phase")),
    }
