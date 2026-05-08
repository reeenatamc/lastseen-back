"""
Narrative analyzer — generates an interpretive emotional narrative using Claude API.

Privacy contract (hard rule):
  Only aggregated metrics are sent to Claude — response times, initiative
  percentages, sentiment scores, dates. Raw message content NEVER leaves
  the user's processing environment.

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
from datetime import datetime

import anthropic

from app.analyzers.base import AnalysisResult, BaseAnalyzer
from app.parsers.base import ParsedChat

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

Devuelve ÚNICAMENTE el objeto JSON, sin texto adicional."""


# ── Analyzer ──────────────────────────────────────────────────────────────────

class NarrativeAnalyzer(BaseAnalyzer):
    name = "narrative"

    def analyze(self, chat: ParsedChat, context: dict | None = None) -> AnalysisResult:
        from app.core.config import settings

        if not settings.ANTHROPIC_API_KEY:
            return AnalysisResult(analyzer=self.name, data={"error": "not_configured"})

        if not context:
            return AnalysisResult(analyzer=self.name, data={"error": "insufficient_context"})

        try:
            payload = _build_payload(chat, context)
            narrative = _call_claude(payload, settings.ANTHROPIC_API_KEY, settings.NARRATIVE_MODEL)
            return AnalysisResult(analyzer=self.name, data=narrative)
        except Exception as exc:
            return AnalysisResult(analyzer=self.name, data={"error": str(exc)})


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
        "iniciativa": {
            "distribucion": initiative.get("share", {}),
            "conversaciones_totales": initiative.get("total_conversations"),
        },
        "tiempo_respuesta": {
            p: _fmt_seconds(v.get("mean_seconds"))
            for p, v in (rt.get("per_person") or {}).items()
        },
        "deterioro": {
            "score_0_a_1": decay.get("decay_score"),
            "tendencia": decay.get("trend"),
            "punto_de_quiebre": decay.get("turning_point"),
        },
    }

    if sentiment and not sentiment.get("error"):
        payload["sentimiento"] = {
            p: {
                "tono_dominante": v.get("dominant"),
                "score_promedio": v.get("avg_score"),
            }
            for p, v in (sentiment.get("per_person") or {}).items()
        }
        drift = sentiment.get("emotional_drift", {})
        if drift:
            payload["deriva_emocional"] = {
                "score_0_a_1": drift.get("score"),
                "direccion": drift.get("direction"),
            }

    return payload


# ── Claude API call ───────────────────────────────────────────────────────────

def _call_claude(payload: dict, api_key: str, model: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key)

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
