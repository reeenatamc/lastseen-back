<div align="center">
<br/>

```
▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
█  ░░░░  ░  ░░░░  ░░░░░░░  ░░░░  ░░░░  █
█  ▄▄▄▄  █  ████  ████████  ▄▄▄▄  ████  █
█  ████  █  ████  ████████  ████  ████  █
█▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█

     L  A  S  T  S  E  E  N  ·  B A C K E N D
```

<br/>

### *The engine that reads what was left unsaid.*

<br/>

[![Status](https://img.shields.io/badge/–%20in%20development%20–-000000?style=for-the-badge)](.)
[![Stack](https://img.shields.io/badge/FastAPI%20·%20Celery%20·%20PostgreSQL-000000?style=for-the-badge)](.)

</div>

---

<br/>

> You upload a chat.
> We tell you when it started dying.

> The backend doesn't store your conversations.
> It reads them, extracts the patterns, and forgets.

<br/>

FastAPI + Celery pipeline that parses exported WhatsApp chats, runs three layers of analysis — **temporal**, **emotional**, **narrative** — and returns a structured JSON that the frontend turns into a story.

<br/>

---

<br/>

<div align="center">

```
 .txt file
    │
    ▼
 PARSER ─────────────────────────────────────────────────────────────
 WhatsApp · Telegram · iMessage                                      │
    │                                                                │
    ▼                                                                │
 ANALYZERS (in memory — raw messages never leave)                    │
    │                                                                │
    ├── temporal.py ──── response time · initiative balance          │
    │                    double text · response decay                │
    │                    silence map · activity patterns             │
    │                    delayed replies (>3h)                       │
    │                                                                │
    ├── sentiment.py ─── language router → hybrid stack              │
    │      │              ┌─ pysentimiento (RoBERTuito ES)           │
    │      ├─ if es ──────┤  + NRC + AFINN lexicons (7.6k words)     │
    │      │              ├─ + intimate-couple vocab (~250 entries)  │
    │      │              └─ + emoji sentiment (Kralj Novak 2015)    │
    │      │                                                          │
    │      └─ else ───── distilbert multilingual (baseline)          │
    │                                                                │
    └── narrative.py ─── Claude / Gemini  (metrics only — no text)   │
                         5-field emotional interpretation             │
    │                                                                │
    ▼                                                                │
 result JSON ───────────────────────────────────────────────────── DB
```

</div>

<br/>

---

<br/>

## API

```
POST   /api/v1/auth/register        create account
POST   /api/v1/auth/token           login → JWT
POST   /api/v1/auth/google          Google Sign-In (ID token) → JWT
GET    /api/v1/auth/me              current user

POST   /api/v1/upload/              upload .txt → queues Celery task
GET    /api/v1/upload/status/{id}   poll task (guests)

GET    /api/v1/analysis/            list analyses
GET    /api/v1/analysis/{id}        full result
GET    /api/v1/analysis/{id}/status lightweight poll (registered users)
DELETE /api/v1/analysis/{id}        delete

GET    /admin                       SQLAdmin panel
```

<br/>

---

<br/>

## What the analyzers measure

**→ Initiative balance**
Who actually starts conversations — not just who messages first, but who breaks the silence after the other person was the last to speak. Plus double text tracking: who followed up unanswered. Auto-flags `low_confidence` on continuous-thread chats so the metric is honest about when it can't speak.

**→ Response decay**
Are response times getting longer? Is reciprocity deteriorating? A score from 0 to 1 with per-ISO-week evolution and a turning point. Catches slow sustained declines, not just one-week cliffs.

**→ Response time + consistency**
Per-person mean, median, p90 and a `consistency_score` — % of replies within 1h. Low consistency reveals hot-and-cold patterns even when the average looks healthy.

**→ Delayed replies**
Times each person made the other wait more than 3 hours mid-conversation, volume-robust so a high-volume reply burst can't mask occasional ghostings.

**→ Hybrid sentiment (Spanish)**
RoBERTuito on its own under-rates affection ("amor", diminutives, ❤️) because it learned its valence from Twitter. The Spanish path fuses four signals per message:

```
 base ML score   (pysentimiento)    weight 0.38
 NRC + AFINN ES  (7.6k words)       weight 0.20   abstains if no match
 intimate vocab  (~250 entries)     weight 0.32   abstains if no match
 emoji sentiment (970 emojis)       weight 0.10   abstains if no emoji
                                    ─────────────
                                    weights renormalised on abstention
                                    × orthographic-emphasis multiplier
                                    × clamp to [-1, +1]
```

Result: avg-score uplifts of ≈ +0.10 over plain pysentimiento on real intimate chats, and a typical jump from `dominant=neutral` to `dominant=positive` for the more expressive partner. See `app/analyzers/lexicons/CITATIONS.md` for the data sources and licenses.

**→ Emotional drift**
How aligned both people's tones are over time, with a per-chat drift score and a direction label. Up to 2000 sampled messages, sampled uniformly to preserve the temporal distribution.

**→ Narrative**
Five-field interpretation generated by Claude (or Gemini as fallback) — *resumen, dinamica, punto_de_quiebre, estado_actual, reflexion*. Only aggregated metrics reach the LLM — never message content, never names sent to anyone external (the metrics ARE labeled with participant names so the narrative can address them by name; that's the only thing that crosses the wire).

<br/>

---

<br/>

## Coming in v2

**→ Dead chat detection**
The exact moment a conversation stopped being mutual and became one-sided. Not a score — a date and a reason.

**→ Group chat support**
Multi-participant analysis. Who holds the group together. Who went quiet first. The social graph of a conversation.

**→ Telegram and iMessage**
Parsers already built. Full pipeline support coming with v2.

**→ Per-domain calibrated sentiment**
The current hybrid layer is calibrated for intimate couple Spanish. v2 extends to friend chats, family, and work chats — each with its own intimate-vocab module.

**→ Message storage (opt-in)**
For users who want search, history replay, and richer analysis over time. Explicit consent required. Off by default.

**→ Aggregate insights (anonymized)**
Opt-in data layer for trend analysis — when do conversations die, what patterns precede silence, how do response times evolve in different relationship types. Never individual, always aggregate.

<br/>

---

<br/>

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI · Python 3.11 · Uvicorn |
| Queue | Celery · Redis (prefork pool, soft/hard time limits) |
| Database | PostgreSQL · SQLAlchemy 2.0 · Alembic |
| Parsers | WhatsApp (Android + iOS) · Telegram · iMessage |
| NLP base | HuggingFace Transformers · `pysentimiento` (RoBERTuito ES) · distilbert multilingual |
| NLP overlay | NRC + AFINN ES lexicons · Emoji Sentiment Ranking · curated intimate-Spanish vocab |
| Language ID | `langdetect` (seeded, deterministic) |
| LLM | Anthropic Claude (primary) · Google Gemini (fallback) — both with explicit 60 s timeouts |
| Admin | SQLAdmin |
| Auth | JWT · bcrypt · Google Sign-In (ID-token verification) |
| Deploy | Docker · Railway → AWS |

<br/>

---

<br/>

## Repo layout

```
app/
├── api/v1/routes/        auth · upload · analysis · payments
├── core/                 config · database · dependencies (DI)
├── models/               SQLAlchemy: user · analysis · message
├── parsers/              base + whatsapp/telegram/imessage
├── analyzers/
│   ├── temporal.py       pure functions over parsed messages
│   ├── sentiment.py      language router + hybrid Spanish stack
│   ├── narrative.py      Claude/Gemini call with 60 s timeout
│   └── lexicons/         hybrid sentiment overlay
│       ├── lexicon_es.csv      NRC + AFINN merged (7.6k words)
│       ├── emoji_sentiment.csv Emoji Sentiment Ranking 1.0
│       ├── intimate_es.py      curated couple-chat vocab (~250)
│       ├── loader.py           lazy cached CSV loaders
│       ├── scorer.py           4-layer late-fusion scorer
│       └── CITATIONS.md        sources + licenses (read this!)
├── workers/              pipeline.py + Celery tasks.py
└── admin/                SQLAdmin views
tests/
├── parsers/              format-specific parser tests
└── analyzers/
    ├── test_temporal.py · test_sentiment.py · test_narrative.py
    └── lexicons/test_scorer.py   25 unit tests for the hybrid stack
```

<br/>

---

<br/>

## Run locally

```bash
# 1. Generate secrets for .env
python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(48))"
python3 -c "import bcrypt; print('ADMIN_PASSWORD_HASH=' + bcrypt.hashpw(b'<your-password>', bcrypt.gensalt()).decode())"

# 2. Create .env (required keys)
#    DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/lastseen
#    REDIS_URL=redis://redis:6379/0
#    CELERY_BROKER_URL=redis://redis:6379/0
#    CELERY_RESULT_BACKEND=redis://redis:6379/0
#    SECRET_KEY=...                  (from step 1)
#    ADMIN_USERNAME=admin
#    ADMIN_PASSWORD_HASH=...         (from step 1)
#    ACCESS_TOKEN_EXPIRE_MINUTES=10080   (7 days; default 30 min is too short for the analysis loop)
#    ANTHROPIC_API_KEY=...           (optional — Gemini used as fallback)
#    GEMINI_API_KEY=...
#    GOOGLE_CLIENT_ID=...            (optional — required only for Google Sign-In)

# 3. Bring up the stack (api · worker · postgres · redis)
docker compose up -d --build --wait

# 4. Apply migrations (first time only)
docker compose exec api alembic upgrade head
```

```
API     → http://localhost:8000
Admin   → http://localhost:8000/admin
DB      → localhost:5433
Redis   → localhost:6380
```

Run the test suite (43 tests total):

```bash
docker compose exec worker python -m pytest tests/ -v
```

Common commands:

```bash
docker compose logs -f api worker          # tail api + worker logs
docker compose exec api alembic revision --autogenerate -m "msg"   # new migration
docker compose down                        # stop everything (keeps volume)
docker compose down -v                     # stop + drop DB volume
```

**First-run note:** the worker downloads ~1.4 GB of HuggingFace model weights on the first Spanish analysis (RoBERTuito sentiment + emotion). Subsequent runs reuse the in-container cache.

<br/>

---

<br/>

## Privacy, by design

- Raw messages are **never stored permanently** (the `messages` table exists for the future opt-in feature only — currently unused by the pipeline)
- Message content **never leaves the worker process** — pysentimiento, NRC, AFINN, intimate-vocab and emoji scoring all run in-memory
- Claude/Gemini receive **only aggregated metrics** — counts, percentages, response times, drift scores, participant names
- Processing is ephemeral — the result JSON is all that persists
- Time limits on every external call (Anthropic and Gemini SDKs configured with `timeout=60 s`) so a hung API can never block the worker
- Opt-in data features require explicit user consent at every step

*You share something intimate. We treat it that way.*

<br/>

---

<br/>

## Citations

The hybrid sentiment scorer stands on four published resources. **Read [`app/analyzers/lexicons/CITATIONS.md`](app/analyzers/lexicons/CITATIONS.md) before any commercial use** — the NRC lexicon is licensed for research-only and requires a separate agreement for commercial deployment.

- Kralj Novak P. et al. *Sentiment of Emojis.* PLOS ONE (2015). CC BY-SA 4.0.
- Mohammad S.M., Turney P.D. *Crowdsourcing a Word-Emotion Association Lexicon.* Comp. Intelligence (2013). Research-only.
- Nielsen F.Å. *A new ANEW.* ESWC (2011). ODbL.
- Aragón M.E. et al. *Improved emotion recognition in Spanish social media through the incorporation of lexical knowledge.* Future Generation Computer Systems (2020). — methodology reference for the hybrid lexicon-ML fusion.

<br/>

---

<br/>

## Business model

Freemium. Guests can run one analysis without an account — limited output, no history. Registered users get full analysis, saved history, and evolution tracking over multiple uploads. Premium tier (Stripe, coming v0.4) unlocks the narrative AI and deeper insights.

The anonymized aggregate layer is a separate data product — opt-in, never tied to individual users, sold as trend intelligence to researchers and product teams.

<br/>

---

<br/>

<div align="center">

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  v0.1  parsers + temporal analyzer   [ ████████ ]
  v0.2  sentiment + pipeline + API    [ ████████ ]
  v0.3  narrative + hybrid sentiment  [ ████████ ]
  v0.4  payments (Stripe)             [ ░░░░░░░░ ]
  v1.0  public launch                 [ ░░░░░░░░ ]
  v2.0  dead chat · groups · opt-in   [ ░░░░░░░░ ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

<br/>

*Currently in development.*

<br/>

---

*The data was always there.*
*LASTSEEN just knows how to read it.*

</div>
