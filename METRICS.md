# LastSeen — Métricas calculadas

Resumen exhaustivo de cada métrica del pipeline, con la fórmula exacta tal como está implementada **hoy** (post-reformulación de mayo 2026), su utilidad, y los puntos que aún conviene revisar.

Fuentes: `app/analyzers/temporal.py`, `app/analyzers/sentiment.py`, `app/analyzers/narrative.py`, `app/parsers/whatsapp.py`.

> **Qué se reformuló en esta ronda** (lo demás sigue igual que el diseño original):
> - `response_decay`: mensual → **semanal ISO**; nueva fórmula de `health` (4 señales ponderadas que suman 1); `rt_score` logístico; **`neglect_score`** nuevo (reemplazó al `delay_factor` diluido); `trend` simétrico; `turning_point` por ventana deslizante devolviendo fecha.
> - `initiative_balance`: bloque con **gap adaptativo** (no 4h fijo) + flag **`confidence`**.
> - `delayed_replies`: unificada la definición de "turno" con `response_decay`.
> - Parser WhatsApp: detección de orden de fecha a nivel de archivo + normalización Unicode (antes rechazaba exports reales de iOS).
> - **Sin tocar** (sus "puntos a revisar" siguen abiertos): `response_time`/`consistency_score`, `activity_patterns`, `conversation_gaps`, `message_length`, todo `sentiment` (`per_person`/`evolution`/`emotional_drift`), `narrative`.

---

## 0. Tuning surface (todas las perillas en un solo sitio)

Bloque `# ── response_decay tuning surface ──` en `temporal.py` + constantes de bloque arriba del archivo.

| Constante | Valor | Para qué |
|---|---|---|
| `_INITIATIVE_GAP` | 4 h | **Solo fallback** del gap de bloque cuando hay <30 gaps (muestra insuficiente para inferir ritmo) |
| `_ADAPTIVE_GAP_PCT` | 0.95 | El gap de bloque = percentil 95 de las pausas del propio chat |
| `_ADAPTIVE_GAP_FLOOR` | 1 h | Nunca cortar conversación por una pausa < 1 h |
| `_ADAPTIVE_GAP_CEILING` | 6 h | Una pausa > 6 h siempre termina una conversación |
| `_ADAPTIVE_GAP_MIN_SAMPLE` | 30 gaps | Mínimo para usar el adaptativo; por debajo, fallback fijo |
| `_MAX_RESPONSE_WINDOW` | 24 h | Cap superior: más allá no es respuesta ni neglect, es silencio |
| `_DELAY_THRESHOLD` | 3 h | Piso para que un handoff cuente como espera unilateral (neglect) y como `delay_rate` (informativo) |
| `_RT_HALF_LIFE_SECONDS` | 5400 (90 min) | Tiempo de respuesta que puntea `rt_score = 0.5` |
| `_RT_STEEPNESS` | 1.2 | Pendiente de la logística de `rt_score` |
| `_SILENCE_GRACE_DAYS` | 1.0 | Silencio mutuo ≤ esto → `silence_score = 1.0` |
| `_SILENCE_DEAD_DAYS` | 3.0 | Silencio mutuo ≥ esto → `silence_score = 0.0` esa semana |
| `_NEGLECT_UNIT_HOURS` | 12.0 | Una espera unilateral de 12 h = 1.0 unidad de neglect |
| `_NEGLECT_SATURATION` | 3.0 | Unidades de neglect/semana que llevan `neglect_score` a 0 |
| `_HEALTH_W_RT` | 0.25 | Peso: velocidad *cuando está presente* |
| `_HEALTH_W_BALANCE` | 0.15 | Peso: equilibrio de iniciativa |
| `_HEALTH_W_NEGLECT` | 0.30 | Peso: abandono unilateral |
| `_HEALTH_W_SILENCE` | 0.30 | Peso: silencio mutuo (los 4 pesos suman 1.0) |
| `_TREND_DELTA` | 0.12 | Umbral **simétrico** de tendencia |
| `_TURNING_POINT_MIN_DROP` | 0.15 | Caída mínima (ventana deslizante) para marcar inflexión |
| `_TOP_GAPS` | 5 | Cuántos silencios largos expone `conversation_gaps` |
| Consistency threshold | 1 h | Respuesta < 1h cuenta como "rápida" (`response_time.consistency_score`) |
| Sentimiento ± | 0.2 | Score >0.2 positivo, <−0.2 negativo, resto neutral |
| Sentiment min msgs / sample / cap | 5 / 2000 / 512 ch / 128 tok | Gating + sampling + truncado de sentiment |
| Temporal gating | 50 msgs / 7 días | Mínimo para correr el analizador temporal |

---

# A. TEMPORAL ANALYZER

Gating: ≥ 50 mensajes y ≥ 7 días → si no, `{"error": "insufficient_data"}`.

## A.1 `overview` — Cifras de cabecera *(sin cambios)*

`_overview`. `date_range` (start/end/total_days), `total_messages`, `messages_per_person`, `share_per_person[p]=count/total`. Contexto base; alimenta `narrative`.

---

## A.2 `response_time` — Velocidad y consistencia *(sin cambios)*

`_response_time`. Para cada par cross-sender con `0 < secs ≤ 24h`: acumula por persona y por trimestre. Por persona: `mean/median/p90/std_seconds` y `consistency_score = |{s ≤ 3600}| / n`. Evolución por trimestre.

**Puntos a revisar (abiertos):**
- `consistency_score` mal nombrada: mide **velocidad** (% < 1h), no regularidad. Reemplazar por CV (`std/mean`) o IQR. *(Sigue pendiente — siguiente en la lista.)*
- Hard cap 24h descarta respuestas legítimas a las 30h.
- `_p90` indexa con redondeo hacia abajo, sin interpolar.

---

## A.3 `initiative_balance` — Quién inicia y se queda *(gap adaptativo + flag de confianza)*

**Bloque de conversación**: ahora el umbral es **adaptativo** (`_adaptive_gap`), no 4h fijo:

```
p95 = percentil 95 de TODAS las pausas entre mensajes consecutivos del chat
si hay < 30 gaps:              gap = _INITIATIVE_GAP (4h, fallback)
si no:                         gap = clamp(p95, 1h, 6h)
```

Razón: 4h fijo rompe en los dos extremos — parejas hiperactivas casi nunca pausan 4h (≈ todo un bloque → iniciativa sin sentido), y conocidos lentos siempre superan 4h (≈ cada mensaje su propio bloque). El adaptativo se ajusta al ritmo *de esta pareja*. Fuente única del percentil: `_gap_p95_seconds` (compartida con el flag de confianza).

**Bloque "sostenido"** (`_block_was_sustained`): la otra persona habló ≥1 vez **y** el último mensaje del bloque es del opener.

**Clasificación de cada transición de bloque:**

| Clase | Condición | Cuenta para |
|---|---|---|
| `initiative` | opener ≠ último-hablante anterior **Y** sostenido | opener |
| `abandoned_open` | opener ≠ último-hablante anterior **pero** no sostenido | opener |
| `late_reply` | bloque anterior abierto y cerrado por la misma persona → ahora responde el otro | responder |
| `double_text` | primer-hablante nuevo == último-hablante anterior | esa persona |

**Outputs**: `total_conversations`, `per_person`/`share` (iniciativas), `abandoned_open`/`late_reply`/`double_text` (mismo shape), `evolution` por trimestre, y **nuevo**:

```json
"confidence": {"level": "low"|"ok", "reason": "continuous_thread"|null}
```

`low/continuous_thread` cuando `_is_continuous_thread` → el p95 de pausas está **por debajo del piso de 1h**: la pareja está en hilo continuo, "conversaciones separadas por silencio" es un corte arbitrario. Los números siguen siendo reales pero el flag avisa que **no hay que sobre-interpretarlos**. Misma filosofía que `insufficient_data` / supresión de turning_point.

**Utilidad**: separar "quién escribe primero" de "quién sostiene". 4 patrones (inicia y se queda / inicia y huye / por fin vuelve / habla solo).

**Resuelto en esta ronda**: `_INITIATIVE_GAP=4h arbitrario` → ahora adaptativo + flag de confianza para el régimen donde el modelo de bloque es débil.

**Puntos a revisar (abiertos):**
- `_block_was_sustained` exige que el otro haya hablado ≥1 vez: si abrís y te ignoran del todo, cuenta como `abandoned_open` **tuyo** aunque el abandonado seas vos. Atribución ambigua.
- `late_reply` se atribuye al que vuelve, no al que dejó colgado.
- N>2 participantes: la lógica corre pero la lectura se diluye.
- **Hallazgo de fondo**: en chats hiperdensos el concepto de "bloque" es inherentemente débil (de ahí el flag). La señal real ahí no es *iniciativa* sino *abandono dentro del hilo* → la cubre `neglect_score`. No se "arregla" el modelo de bloque; se acota su confianza.

---

## A.4 `activity_patterns` — Cuándo se habla *(sin cambios)*

`_activity_patterns`: `by_hour` (0–23), `by_weekday` (0=lun), `by_month` (lista). Heatmap, descriptivo.

**Puntos a revisar (abiertos):** sin zona horaria; shape inconsistente (`by_month` lista vs dicts).

---

## A.5 `conversation_gaps` — Silencios *(sin cambios)*

`_conversation_gaps`: por par consecutivo `{start,end,hours,days}`. `top_gaps` (5 más largos) + `distribution` (buckets <1h / 1–6h / 6–24h / 1–7d / ≥7d). Anclas narrativas.

**Puntos a revisar (abiertos):** cuenta gaps esperables (sueño infla `6–24h`); no distingue silencio mutuo de abandono unilateral *(esto último ahora sí lo separa `neglect_score` en decay, pero no acá)*.

---

## A.6 `message_length` — Longitud de mensajes *(sin cambios)*

`_message_length`: solo texto no-media. Por persona `mean_chars`/`median_chars`, evolución por trimestre.

**Puntos a revisar (abiertos):** caracteres ≠ esfuerzo (emojis, URLs); media sensible a outliers (copy-paste largo).

---

## A.7 `response_decay` — La métrica núcleo *(reescrita completa)*

**Objetivo**: detectar deterioro progresivo de la reciprocidad.

### Agregación: semana ISO (antes mes)

`_week(dt)` → `"2026-W19"` (año ISO + semana, zero-padded → orden lexicográfico = cronológico). Cada semana lleva `period_start` = el lunes (`_week_start`, vía `datetime.fromisocalendar`). Gate: `len(all_weeks) < 2 → {"trend": "insufficient_data"}`. La resolución semanal deja que la métrica hable en chats reales cortos (12 días ≈ 2 semanas, no 1 mes).

### Las 4 señales por semana (todas [0,1], 1 = sano)

**`rt_score`** (`_rt_score`) — logística sobre el tiempo de respuesta promedio:
```
rt_score = 1 / (1 + (avg_rt / 5400) ** 1.2)        # None → 0.0
```
seg/min → ~1.0; 90 min → 0.5; ≥24 h → ~0. Mide "rápida *cuando está presente*".

**`balance`** = `1 − initiative_imbalance`. `imbalance = |share_p0 − 0.5| × 2` solo si la semana tiene ≥2 iniciativas y son exactamente 2 participantes; si no, `imbalance = 1.0` (sin datos = no sano, no pase libre).

**`neglect_score`** (`_neglect_score`) — **nuevo, reemplaza al `delay_factor` diluido**. Cada handoff de turno en la ventana `3h < gap ≤ 24h` aporta `min(wait_h / 12, 1.0)` **unidades** a su semana. `neglect_score = max(0, 1 − units / 3)`. Es un **conteo severizado, no una tasa** → el volumen masivo de mensajes no lo puede tapar (ese era el bug: 9 abandonos de horas diluidos en ~640 turnos daban delay_rate ~1.4%). Comparte la definición de turno con `delayed_replies` (`_iter_turn_handoffs`).

**`silence_score`** (`_silence_score`) — silencio **mutuo** más largo de la semana (gap entre cualquier par consecutivo, a resolución de timestamp):
```
silence_score = clamp(1 − max(0, max_gap_dias − 1) / 2, 0, 1)
# <1d → 1.0   |   2d → 0.5   |   ≥3d → 0.0
```
El borde final del export (sin mensaje siguiente) no genera gap → un chat que simplemente deja de exportarse no se penaliza ("salvo que sea el final").

### `health` de la semana

```
health = clamp(rt×0.25 + balance×0.15 + neglect×0.30 + silence×0.30, 0, 1)
```
Pesos suman 1, un solo espacio. Cualquier señal sin datos aporta 0 (ausencia ≠ salud). **No hay constante de "mes/semana inactivo"** — una semana muerta cae sola cerca de 0. `rt` y `silence` son ciegas al abandono unilateral por construcción (la media la lavan las ráfagas; el spam a un ausente no genera silencio mutuo) → `neglect_score` es la señal que sí lo ve.

### `trend`, `decay_score`, `turning_point`

```
third = max(1, len(health) // 3)
delta = mean(health[-third:]) − mean(health[:third])
delta < −0.12 → "deteriorating" | > +0.12 → "improving" | si no → "stable"   # SIMÉTRICO

decay_score = clamp(1 − mean(health[-third:]), 0, 1)

# turning_point: el corte i que maximiza mean(health[:i]) − mean(health[i:]),
# si supera 0.15; suprimido en el último 20%. Devuelve la FECHA (period_start),
# no el código de semana → legible para narrativa y UI.
```

`evolution[]` por semana expone: `period`, `period_start`, `avg_response_seconds`, `message_count`, `initiative_imbalance`, `delay_rate` (informativo, ya no entra a health), `neglect_units`, `silence_gap_days`, `turns`, `health`.

**Utilidad**: la métrica que da nombre al producto. `decay_score` 0→1 + `trend` + `turning_point` (fecha). Alimenta la narrativa.

**Resuelto en esta ronda:** discrepancia 40/30/30 vs código (fórmula nueva consistente, pesos suman 1); `health=0.2` mágico (eliminado); `monthly_delayed` sin cap 24h (neglect usa ventana 3–24h); trend asimétrico (ahora ±0.12); turning point solo mes-a-mes (ahora ventana deslizante, capta declives graduales); granularidad mensual inútil en chats cortos (ahora semanal); definición de turno doble (unificada).

**Puntos a revisar (abiertos / nuevos):**
- **Pocas semanas → trend frágil**: con 2 semanas `third=1`, el trend es "¿semana 2 difirió de 1 en >0.12?". Habla, pero para señal robusta querés ≥6 semanas. Inherente a chats cortos.
- **`balance` aún asume 2 personas** (`len(participants)==2`, si no `imbalance=1.0`). Falta versión N-persona (entropía / distancia a uniforme).
- **`rt_score` sigue sobre la media**: deliberado — `neglect_score` ahora cubre la cola (las desapariciones), `rt` mide solo "rápida cuando presente". Documentado, no es bug, pero tenerlo presente.
- **Sin horario**: una espera de 8h de madrugada pesa igual que 8h de día (tanto en neglect como en silence).
- Pesos `0.25/0.15/0.30/0.30` y knobs (`5400`, `12h`, `sat 3`, `±0.12`): ahora documentados con razón en la tuning surface, pero **sin calibrar contra datos reales** — son defaults razonados, no derivados.

---

## A.8 `delayed_replies` — Conteo de demoras > 3h *(turno unificado)*

`_delayed_replies` ahora consume `_iter_turn_handoffs` (turno = secuencia consecutiva del mismo sender; el gap se mide del fin de un turno al inicio del siguiente). Cuenta por responder los handoffs con `gap > threshold_hours` (default 3.0). Output: `per_person`, `share`, `total`, `threshold_hours`.

**Resuelto:** la **doble definición de "turno"** — ahora `delayed_replies` y el componente de neglect de `response_decay` usan exactamente `_iter_turn_handoffs`, así sus números son directamente comparables.

**Puntos a revisar (abiertos):**
- Sin cap superior y sin pesar por magnitud: 4h y 72h cuentan igual 1. *(En `response_decay` esto sí se resolvió vía `neglect_units` severizado; `delayed_replies` sigue siendo conteo crudo a propósito, como vista por-persona simple.)*
- `threshold_hours` hardcodeada vía default param.

---

# B. SENTIMENT ANALYZER *(sin cambios esta ronda)*

Modelo `lxyuan/distilbert-base-multilingual-cased-sentiments-student`. Sampling uniforme si >2000 msgs. Score por mensaje = `P(positive) − P(negative)` ∈ [−1,1]. Gating ≥5 msgs de texto.

## B.1 `per_person`
`positive=|{s>0.2}|/n`, `negative=|{s<−0.2}|/n`, `neutral=1−pos−neg`, `avg=mean`, `dominant=argmax`. **Abiertos:** umbral ±0.2 arbitrario; `dominant` flipa con 0.001; `avg` engañoso si alterna.

## B.2 `evolution`
`mean(scores_p_q)` por trimestre. **Abiertos:** trimestres escasos en chats cortos; no reporta `n` por trimestre.

## B.3 `emotional_drift` (solo 2 personas)
`divergence[q]=|mean(p1)−mean(p2)|`; `drift_score=min(mean(divergences)/2.0,1.0)`; dirección según último trimestre (±0.1); turning point = mayor aumento de divergencia (>0.05). **Abiertos:** hard-coded 2 personas; dirección solo último Q (ruidosa); divide por rango teórico 2.0 → score real vive en ~[0,0.3] y parece siempre bajo.

---

# C. NARRATIVE ANALYZER *(payload casi igual)*

Síntesis vía Claude (Gemini fallback) → 5 campos (`resumen`, `dinamica`, `punto_de_quiebre`, `estado_actual`, `reflexion`). **Privacy contract**: solo métricas agregadas, nunca texto ni timestamps. Schema-enforced JSON, sin reintentos.

**Mejora colateral de esta ronda**: `response_decay.turning_point` ahora es una **fecha** (`YYYY-MM-DD`) en vez de código de semana/mes → el LLM la puede frasear en lenguaje humano sin tecnicismos.

**Puntos a revisar (abiertos):** el prompt dice "nunca menciones porcentajes/segundos" pero el payload los contiene (confía en el modelo); "máx 100 palabras" no se valida; sin tercera capa si fallan Claude+Gemini. **Pendiente nuevo**: `_build_payload` no consume aún `initiative_balance.confidence` — debería suavizar el lenguaje cuando es `low`.

---

# D. Tabla maestra: input → output → utilidad

| Métrica | Input | Output clave | Para qué |
|---|---|---|---|
| `overview` | timestamps, senders | total, share | Contexto base |
| `response_time.per_person` | pares cross-sender <24h | mean/median/p90/std/consistency | Velocidad |
| `initiative_balance` | bloques (gap **adaptativo**) | iniciativa / abandoned_open / late_reply / double_text + **confidence** | Quién sostiene (con flag de fiabilidad) |
| `activity_patterns` | timestamps | hora/día/mes | Heatmap |
| `conversation_gaps` | pares ordenados | 5 silencios + buckets | Anclas |
| `message_length` | texto | mean/median chars | Cambio de esfuerzo |
| `response_decay.health[]` | rt+balance+**neglect**+silence **semanal** | score por semana | Serie de salud |
| `response_decay.decay_score` | 1−mean(health[-third:]) | 0→1 | El score de la app |
| `response_decay.trend` | Δ first vs last third (±0.12) | etiqueta | Dirección |
| `response_decay.turning_point` | ventana deslizante (>0.15) | **fecha** o null | Inflexión |
| `delayed_replies` | `_iter_turn_handoffs` >3h | conteo + share | Demoras por persona (comparable con decay) |
| `sentiment.per_person` | scores | %pos/%neg/dominant | Tono |
| `emotional_drift` | divergencia trimestral | 0→1 + dir + TP | Sincronía emocional |

---

# E. Cosas transversales

**Resuelto / mejorado en esta ronda:**
- Granularidad de la métrica núcleo: `response_decay` semanal (antes mensual, inútil en chats cortos). *(Persiste mezcla: `response_time`/`message_length`/`emotional_drift` siguen trimestrales — decidir si unificar.)*
- Pesos sin justificación: ahora centralizados y razonados en la tuning surface. *(Siguen sin calibrar empíricamente.)*
- Confianza/incertidumbre: primer paso dado — `initiative_balance.confidence` + gating `insufficient_data`. *(Falta IC en `decay_score`.)*
- Robustez de parser: detección de orden de fecha a nivel de archivo + normalización Unicode (`U+200E/202F/...`) → exports reales de iOS ya no se rechazan ni se mal-parsean.
- Cruce silencio mutuo vs abandono unilateral: `neglect_score` los separa dentro de decay.

**Abierto:**
1. **2-persona vs grupo**: `balance`/`initiative_imbalance` y `emotional_drift` asumen 2 personas. Falta familia N-persona (entropía / divergencia desde uniforme).
2. **Atribución de culpa**: `late_reply`/`abandoned_open`/`double_text`/`neglect` se reportan sueltos; la narrativa los mezcla sin un puntaje compuesto por-persona.
3. **Sentimiento ↔ temporal sin cruzar**: falta `correlation(health_semanal, sentiment_semanal)` — métrica nueva potente.
4. **`narrative._build_payload` parcial**: hay que actualizarlo a mano por métrica nueva; aún no lee `confidence` ni `neglect_units`.
5. **Sin horario / zona horaria**: ni `activity_patterns` ni `neglect`/`silence` lo consideran.
6. **Calibración**: ningún knob está derivado de un dataset; son defaults razonados.

---

# F. Orden sugerido para lo que queda

1. ~~`response_decay`~~ — **hecho** (semanal + neglect + trend simétrico + turning point por ventana).
2. ~~`initiative_balance` gap adaptativo + confianza~~ — **hecho**. Falta su generalización N-persona.
3. **`consistency_score` → CV/IQR** — renombrar o reemplazar por dispersión real (siguiente natural; cambio chico).
4. **`emotional_drift`** — renormalizar por rango efectivo (no el teórico 2.0) y extender a >2 personas.
5. **Familia N-persona** — `balance`/`imbalance`/`drift` con entropía o distancia a uniforme.
6. **Cruce `health` ↔ `sentiment`** semanal — métrica de correlación nueva.
7. **Consumir `confidence`/`neglect` en `narrative._build_payload`** — para que la historia refleje el modelo nuevo.
