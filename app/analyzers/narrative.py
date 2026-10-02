"""
Narrative analyzer — generates an interpretive emotional narrative using Claude API.

Privacy contract (hard rule):
  Only aggregated metrics are sent to Claude — response times, initiative
  percentages, sentiment scores, weekly health, conflict episode counts and
  dates. Raw message content NEVER leaves the user's processing environment.

Prompt caching:
  The system prompt is marked cache_control=ephemeral. At ~300 tokens it sits
  below Opus 4.7's 4096-token minimum, so it won't cache yet; it will once
  we add few-shot examples in a future iteration.

Structured outputs:
  output_config.format enforces the JSON schema on every response, eliminating
  JSON-parsing failures regardless of model temperature.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import anthropic

from app.analyzers.base import AnalysisResult, BaseAnalyzer
from app.parsers.base import ParsedChat

logger = logging.getLogger(__name__)

# ── Network timeouts ──────────────────────────────────────────────────────────
# The narrative step is the only analyzer that makes an outbound network call.
# Without an explicit timeout each SDK falls back to its own default (the
# Anthropic SDK waits up to 10 min), which can outlast the Celery hard time
# limit and leave the worker looking hung. A finite timeout turns a stuck
# request into a clean exception that `analyze()` catches and reports as error.
_LLM_TIMEOUT_SECONDS = 60.0   # anthropic.Anthropic(timeout=...) — float seconds
_LLM_TIMEOUT_MS = 60_000      # google-genai HttpOptions(timeout=...) — milliseconds

# ── Payload size ──────────────────────────────────────────────────────────────
# The model needs the shape of the relationship over time, not every data
# point: a multi-year chat has hundreds of weeks. Both series are thinned so
# the payload stays small and its cost predictable.
_MAX_TIMELINE_POINTS = 26     # weekly health points sent (the last one is always kept)
_MAX_EPISODES = 15            # conflict episodes sent (the heaviest ones, in date order)

# ── Schema ────────────────────────────────────────────────────────────────────

_NARRATIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "resumen": {
            "type": "string",
            "description": "El arco completo de la relación en 2-3 oraciones.",
        },
        "dinamica": {
            "type": "string",
            "description": "Quién sostuvo la conexión y cómo se distribuyó el esfuerzo.",
        },
        "punto_de_quiebre": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "El momento donde algo cambió (ej: 'en el tercer trimestre de 2023'). null si no hay quiebre claro.",
        },
        "estado_actual": {
            "type": "string",
            "description": "Cómo están las cosas en el período más reciente.",
        },
        "reflexion": {
            "type": "string",
            "description": "Una observación honesta y profunda sobre lo que pasó.",
        },
    },
    "required": ["resumen", "dinamica", "punto_de_quiebre", "estado_actual", "reflexion"],
    "additionalProperties": False,
}

# ── System prompt — marked for caching ───────────────────────────────────────

_SYSTEM_PROMPT = """\
Eres el motor narrativo de LastSeen, una plataforma que analiza conversaciones \
de WhatsApp para revelar la historia emocional de las relaciones humanas.

Recibirás métricas agregadas de una conversación: frecuencias, tiempos de \
respuesta, iniciativas, tonos emocionales y patrones de deterioro. \
Nunca verás mensajes originales, solo estadísticas y patrones cuantitativos.

Tu tarea es transformar esos números en una narrativa que nombre honestamente \
lo que vivieron esas dos personas.

Principios:
- Usa los nombres reales de los participantes
- No dulcifiques la realidad: si algo se deterioró, nómbralo con claridad y empatía
- Habla en pasado para lo que fue, en presente para lo que es ahora
- Nunca menciones porcentajes, segundos ni términos técnicos como "decay_score"
- Máximo 100 palabras por campo

Cómo leer los datos:
- deterioro.score_0_a_1: 0 es un vínculo sano, 1 es un vínculo totalmente \
deteriorado. "tendencia" dice hacia dónde va; "stable" no significa que esté bien.
- tramo_final: si aparece, la conversación se apagó en sus últimas semanas \
(menos mensajes, silencios que antes no existían). Es el hecho más importante \
del estado actual y, si no hay otro, el punto de quiebre.
- conflictos.episodios: días con lenguaje de ruptura (terminar, bloquear, \
pedir tiempo, dejar de hablar). Varios episodios seguidos de calma describen \
un ciclo de peleas y reconciliaciones: nómbralo. Un bloqueo o un episodio \
grave al final indica una ruptura, no estabilidad.
- sentimiento: un tono dominante "neutral" suele venir de mensajes muy \
cortos, no de frialdad; no lo presentes como distancia emocional si los demás \
datos no lo confirman. Para describir el tono de cada persona usa \
con_carga_emocional (reparto entre mensajes cálidos y tensos de los que sí \
tienen carga) y no el tono dominante. tramo_reciente compara las últimas semanas con lo habitual.
- iniciativa.confianza "low": no afirmes quién iniciaba más.
- No inventes causas ni hechos que no estén en los datos.

Devuelve ÚNICAMENTE el objeto JSON, sin texto adicional."""


# ── Analyzer ──────────────────────────────────────────────────────────────────

class NarrativeAnalyzer(BaseAnalyzer):
    name = "narrative"

    def analyze(self, chat: ParsedChat, context: dict | None = None) -> AnalysisResult:
        from app.core.config import settings

        if not context:
            return AnalysisResult(analyzer=self.name, data={"error": "insufficient_context"})

        # Provider selection: Claude first, Gemini as fallback, error if neither
        if settings.ANTHROPIC_API_KEY:
            provider = "claude"
        elif settings.GEMINI_API_KEY:
            provider = "gemini"
        else:
            return AnalysisResult(analyzer=self.name, data={"error": "not_configured"})

        try:
            payload = _build_payload(chat, context)
            if provider == "claude":
                narrative = _call_claude(payload, settings.ANTHROPIC_API_KEY, settings.NARRATIVE_MODEL)
            else:
                narrative = _call_gemini(payload, settings.GEMINI_API_KEY, settings.GEMINI_MODEL)
            return AnalysisResult(analyzer=self.name, data=narrative)
        except Exception as exc:
            # Fixed code in the result; exception text can echo prompt content or API internals
            logger.warning("Narrative generation failed: %s", type(exc).__name__)
            return AnalysisResult(analyzer=self.name, data={"error": "llm_failed"})


# ── Metrics payload builder ───────────────────────────────────────────────────

def _build_payload(chat: ParsedChat, context: dict) -> dict:
    """Build a compact, privacy-safe metrics payload for Claude."""
    temporal = context.get("temporal", {})
    sentiment = context.get("sentiment", {})

    overview = temporal.get("overview", {})
    decay = temporal.get("response_decay", {})
    initiative = temporal.get("initiative_balance", {})
    rt = temporal.get("response_time", {})
    msg_len = temporal.get("message_length", {})

    date_range = overview.get("date_range", {})
    participants = overview.get("participants", chat.participants)

    payload: dict = {
        "participantes": participants,
        "periodo": {
            "inicio": _short_date(date_range.get("start")),
            "fin": _short_date(date_range.get("end")),
            "dias_total": date_range.get("total_days"),
        },
        "mensajes": {
            "total": overview.get("total_messages"),
            "por_persona": {
                p: {
                    "porcentaje": round((overview.get("share_per_person") or {}).get(p, 0) * 100),
                    "longitud_promedio_caracteres": (
                        (msg_len.get("per_person") or {}).get(p, {}).get("mean_chars")
                    ),
                }
                for p in participants
            },
        },
        "iniciativa": _initiative_block(initiative),
        "tiempo_respuesta": {
            p: {
                "promedio": _fmt_seconds(v.get("mean_seconds")),
                "consistencia_0_a_1": v.get("consistency_score"),
            }
            for p, v in (rt.get("per_person") or {}).items()
        },
        "deterioro": {
            "score_0_a_1": decay.get("decay_score"),
            "tendencia": decay.get("trend"),
            "punto_de_quiebre": decay.get("turning_point"),
        },
    }

    timeline = _weekly_timeline(decay.get("evolution") or [])
    if timeline:
        payload["evolucion_semanal"] = timeline

    closing = decay.get("closing_phase") or {}
    if closing.get("detected"):
        payload["tramo_final"] = {
            "inicio": _short_date(closing.get("start")),
            "volumen_respecto_a_lo_habitual": closing.get("volume_ratio"),
            "silencio_mas_largo_dias": closing.get("max_silence_days"),
            "silencio_mas_largo_previo_dias": closing.get("baseline_max_silence_days"),
        }

    conflicts = _conflict_summary(context.get("conflict") or {})
    if conflicts:
        payload["conflictos"] = conflicts

    delayed = temporal.get("delayed_replies", {})
    if delayed and delayed.get("total", 0) > 0:
        payload["demoras_mas_de_3h"] = {
            p: c for p, c in (delayed.get("per_person") or {}).items()
        }

    if sentiment and not sentiment.get("error"):
        payload["sentimiento"] = {}
        for p, v in (sentiment.get("per_person") or {}).items():
            entry = {
                "tono_dominante": v.get("dominant"),
                "score_promedio": v.get("avg_score"),
            }
            charged = v.get("charged")
            if charged:
                entry["con_carga_emocional"] = {
                    "proporcion": charged.get("share"),
                    "calidos": charged.get("positive"),
                    "tensos": charged.get("negative"),
                }
            payload["sentimiento"][p] = entry
        drift = sentiment.get("emotional_drift", {})
        if drift:
            payload["deriva_emocional"] = {
                "score_0_a_1": drift.get("score"),
                "direccion": drift.get("direction"),
            }
        recent = sentiment.get("recent")
        if recent:
            payload["sentimiento"]["tramo_reciente"] = {
                "desde": _short_date(recent.get("start")),
                "cambio": recent.get("shift"),
                "por_persona": {
                    p: {
                        "score_reciente": v.get("recent_avg"),
                        "score_habitual": v.get("baseline_avg"),
                    }
                    for p, v in (recent.get("per_person") or {}).items()
                },
            }

    return payload


def _initiative_block(initiative: dict) -> dict:
    """
    Initiative data for the payload. With low confidence only the level is
    sent, so the model has no distribution to claim who started more.
    """
    level = (initiative.get("confidence") or {}).get("level")
    if level == "low":
        return {"confianza": "low"}
    return {
        "distribucion": initiative.get("share", {}),
        "conversaciones_totales": initiative.get("total_conversations"),
        "confianza": level,
    }


def _weekly_timeline(evolution: list[dict]) -> list[dict]:
    """
    Weekly health and volume, thinned to at most _MAX_TIMELINE_POINTS.

    Global averages hide when things changed; this series is what lets the
    model place a decline in time. Thinning keeps evenly spaced weeks and
    always the last one, because the end of the chat is where the current
    state lives.
    """
    points = [
        {
            "semana": _short_date(e.get("period_start")),
            "salud_0_a_1": e.get("health"),
            "mensajes_por_dia": e.get("messages_per_day"),
        }
        for e in evolution
        if e.get("health") is not None
    ]
    if len(points) <= _MAX_TIMELINE_POINTS:
        return points
    last = len(points) - 1
    keep = sorted({round(i * last / (_MAX_TIMELINE_POINTS - 1)) for i in range(_MAX_TIMELINE_POINTS)})
    return [points[i] for i in keep]


def _conflict_summary(conflict: dict) -> dict | None:
    """
    Conflict episodes as counts and dates only — never the words that were said.

    When there are more than _MAX_EPISODES the heaviest ones are kept (by
    mentions, blocks first) and returned in date order, so a long chat's
    payload stays bounded without losing its worst moments.
    """
    if not conflict or conflict.get("error"):
        return None

    episodes = conflict.get("episodes") or []
    blocks = (conflict.get("system_events") or {}).get("blocks") or []
    if not episodes and not blocks:
        return None

    total = len(episodes)
    if total > _MAX_EPISODES:
        heaviest = sorted(
            episodes,
            key=lambda e: (bool(e.get("blocked")), e.get("mentions", 0)),
            reverse=True,
        )[:_MAX_EPISODES]
        episodes = sorted(heaviest, key=lambda e: e.get("start") or "")

    summary: dict = {
        "episodios_totales": total,
        "episodios": [
            {
                "inicio": _short_date(e.get("start")),
                "fin": _short_date(e.get("end")),
                "menciones_de_ruptura": e.get("mentions"),
                "por_persona": e.get("per_person", {}),
                "llamadas_sin_respuesta": e.get("missed_calls", 0),
                "bloqueo": bool(e.get("blocked")),
                "gravedad": e.get("severity"),
            }
            for e in episodes
        ],
    }
    if blocks:
        summary["bloqueos"] = [_short_date(ts) for ts in blocks]
    recent = conflict.get("recent") or {}
    if recent.get("ratio") is not None:
        summary["lenguaje_de_ruptura_reciente_vs_habitual"] = recent["ratio"]
    return summary


# ── Claude API call ───────────────────────────────────────────────────────────

def _call_claude(payload: dict, api_key: str, model: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key, timeout=_LLM_TIMEOUT_SECONDS)

    user_content = (
        "Analiza estas métricas y genera la narrativa emocional:\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": _NARRATIVE_SCHEMA,
            },
        },
        messages=[{"role": "user", "content": user_content}],
    )

    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


# ── Gemini API call ───────────────────────────────────────────────────────────

def _call_gemini(payload: dict, api_key: str, model: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=_LLM_TIMEOUT_MS),
    )

    prompt = (
        _SYSTEM_PROMPT
        + "\n\nAnaliza estas métricas y genera la narrativa emocional:\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "resumen":           {"type": "string"},
                    "dinamica":          {"type": "string"},
                    "punto_de_quiebre":  {"type": "string", "nullable": True},
                    "estado_actual":     {"type": "string"},
                    "reflexion":         {"type": "string"},
                },
                "required": ["resumen", "dinamica", "punto_de_quiebre", "estado_actual", "reflexion"],
            },
        ),
    )

    return json.loads(response.text)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _short_date(iso: str | None) -> str | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.rstrip("Z")).strftime("%d/%m/%Y")
    except ValueError:
        return iso[:10]


def _fmt_seconds(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60} minutos"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} horas"
    return f"{seconds / 86400:.1f} días"
