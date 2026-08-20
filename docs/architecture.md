# AI Employee Platform — Architecture

## High-Level Pipeline

```
User Input
    │
    ▼
┌──────────────────────┐
│  Pre-Classify (regex)│  ← Greetings, math expressions, escalation patterns
│  Returns or falls to  │
│  LLM                  │
└──────────┬───────────┘
           │ (if not caught by regex)
           ▼
┌──────────────────────┐
│  Sarvam Intent       │  ← Few-shot prompt with 8 examples
│  Classifier          │     Role-based confidence scoring
│  (LLMProvider)       │     Returns: intent, confidence, suggested_tools
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Tool Resolution     │  ← INTENT_TOOL_MAP + LLM suggested_tools
│                      │     Validated against ToolRegistry
│                      │     Unknown tools silently ignored
└──────────┬───────────┘
           │
     ┌─────┴──────┐
     │ Is Escalation? │
     └─────┬──────┘
      YES  │  NO
      ▼    │    ▼
  ┌─────────┐ ┌─────────┐
  │ Escalate│ │ Execute │  ← ToolRegistry.invoke() per PlanStep
  │ Message │ │ Tools   │     tool_invoke → execute loop
  └────┬────┘ └────┬────┘
       │            │
       └─────┬──────┘
             ▼
      ┌──────────────┐
      │  Respond     │  ← LLMProvider.generate()
      │  (LLM/No LLM)│     Context: memory + docs + tool results + user input
      └──────────────┘
             │
             ▼
      Final Response (natural language)
```

## Graph Flow (LangGraph)

```
START → receive → build_context → plan → after_plan_route
                                            │
                    ┌───────────────────────┴────────────┐
                    ▼ (escalation/complaint)             ▼ (normal)
                respond (pre-set)            execute → tool_invoke → execute → ...
                    │                                    │
                    └──────────── respond (LLM) ──────────┘
                                               │
                                              END
```

## Dependency Injection Diagram

```
Settings (.env / docker-compose)
    │
    ├─→ SARVAM_API_KEY ─→ SarvamProvider ──→ LLMPlanner
    ├─→ RAG_URL         ─→ HttpRAGClient ──→ SearchDocumentsTool
    ├─→ LLM_TEMPERATURE ─→ SarvamProvider (via constructor)
    ├─→ LLM_MAX_TOKENS  ─→ SarvamProvider (via constructor)
    │
    └─→ container.py factories (@lru_cache singletons):
           get_tool_registry()    ─→ ToolRegistry (10 tools)
           get_planner()          ─→ LLMPlanner | MockPlanner
           get_context_builder()  ─→ MockContextBuilder
           get_agent_service()    ─→ AgentService(graph, tools, planner)
           get_order_service()    ─→ MockOrderService
           get_calendar_service() ─→ MockCalendarService
           get_pricing_service()  ─→ MockPricingService
           _build_rag_client()    ─→ HttpRAGClient(fallback=MockRAGClient)
           _build_llm_provider()  ─→ SarvamProvider | None
```

## Tool Registry (10 tools)

| Tool | Intent | Service | Purpose |
|------|--------|---------|---------|
| `calculator` | math, general | none | Evaluate arithmetic expressions |
| `search_documents` | sales, support, general | `RAGClient` | Knowledge base search |
| `search_pricing` | sales | `PricingService` | Plan/price search |
| `lookup_order` | support | `OrderService` | Order status lookup |
| `get_weather` | general (suggested) | none | Weather for location |
| `schedule_meeting` | LLM suggested | none | Legacy meeting scheduler |
| `calendar` | booking | `CalendarService` | Availability check |
| `schedule_demo` | booking | `CalendarService` | Demo/appointment booking |
| `send_email` | LLM suggested only | none | Send email |
| `transfer_to_human` | complaint, escalate | none | Human escalation |

## Intent → Tool Mapping

| Intent | Tools Executed | Short-Circuit? |
|--------|---------------|----------------|
| `sales` | `search_pricing` → `search_documents` | No |
| `support` | `lookup_order` → `search_documents` | No |
| `booking` | `calendar` → `schedule_demo` | No |
| `general` | `search_documents` (or LLM-suggested tool) | No |
| `math` | `calculator` | No |
| `complaint` | none → escalation message | **Yes** |
| `escalate` | none → escalation message | **Yes** |

## RAG Architecture

```
SearchDocumentsTool
    │
    └─→ RAGClient.search(query, top_k)
            │
            ├─→ HttpRAGClient ─→ POST {RAG_URL}/api/v1/documents/query
            │       │
            │       ├─ Success → [{title, snippet, score, document_id, chunk_index}]
            │       │
            │       └─ Failure → fallback_client.search()
            │                       │
            │                       └─→ MockRAGClient (5 hardcoded docs)
            │
            └─→ health_check() ── GET {RAG_URL}/health → bool
                    (called at startup, logs warning if unhealthy)

Query path inside the RAG service (/api/v1/documents/query):
    RAGService.query(query, top_k)
        │
        ├─ HybridRetriever (RRF fusion, const=60, scale=20, pool=top_k*3)
        │     ├─ vector search  → Retriever.retrieve() (embeddings)
        │     └─ lexical search → VectorStore.search_lexical() (Postgres FTS ts_rank)
        │
        └─ Ranker (keyword-overlap boost 0.08 on hybrid results)
              └─ _build_citations(limit=3, 200-char snippet)
                    → QueryResponse.citations [{document_id, title, chunk_index, content, score}]
```

## Channel Adapters & Web Chat

The gateway is the single entry point for all inbound channels. Each connector
normalizes its native payload into the canonical `shared.schemas.channels` shape
and calls the shared agent entrypoint, which the orchestrator exposes at
`POST /api/agent/run`.

```
Gateway
 ├─ POST /api/channels/web       → ChannelService (injectable transport) → orchestrator
 ├─ POST /api/channels/{channel} → dispatch by ChannelType (whatsapp, email, crm, api, sms)
 ├─ GET  /api/channels/stats     → channel traffic summary for the dashboard widget
 ├─ GET  /api/channels/events    → recent channel events (limit/channel/status/start/end filters)
 ├─ GET  /chat                   → self-contained single-file web chat (static/chat.html)
 └─ GET  /                       → 307 redirect to /chat
```

Every inbound message outcome is recorded into the `channel_events` table
(accepted / rate_limited / blocked + violation category, reason, redaction count,
request id — never the raw message text). The dashboard's "Channels & Guardrails"
widget reads it via the two GET endpoints above.

- `shared/schemas/channels.py` defines `ChannelType`, `ChannelContact`,
  `ChannelMessage`, `ChannelResponse`.
- The orchestrator `AgentRequest`/`AgentResponse`/`AgentState` carry the channel
  context (`channel`, `channel_message_id`, `tenant_id`, `contact`, `metadata`).

## Guardrails & Rate Limiting

Applied in the gateway before any request is forwarded to the orchestrator:

```
inbound message
    │
    ├─ Rate limiter (Redis fixed window, 429 when exceeded)
    │     shared/guardrails/rate_limiter.py · RedisRateLimiter
    │
    ├─ Guardrails service (400 when blocked)
    │     shared/guardrails/service.py · GuardrailsService
    │        ├─ InputSanitizer   → strips control chars, trims whitespace
    │        ├─ PIIRedactor      → masks emails/phones/credit-card/IP (longest-match merge)
    │        ├─ ContentFilter    → blocks prompt-injection + toxic phrases
    │        └─ redact_pii flag  → sanitized text replaces original when enabled
    │
    └─ forwarded with redacted body
```

Settings: `GUARDRAILS_ENABLED`, `RATE_LIMIT_ENABLED`, `RATE_LIMIT_LIMIT` (30/min),
`RATE_LIMIT_WINDOW_SECONDS` (60). Guardrails degrade gracefully to "allowed" when
Redis is unreachable.

## Shared Task Queue

Generic, Redis-backed async job queue used by background workers (e.g. the
orchestrator's memory writer):

```
shared/queue/queue.py   RedisTaskQueue
                           enqueue(task, payload) → rpush envelope {job_id, task, payload, enqueued_at}
                           poll()                 → lpop + json decode (None on empty/bad payload)
                           length()               → llen

shared/queue/worker.py  RedisTaskWorker
                           register(task_name, handler)
                           start()/stop()         → background asyncio task running _run_loop/_dispatch
                           unknown task / handler error → logged, loop continues
```

`MemoryWriterWorker` (`services/orchestrator/workers/memory_writer.py`) is a thin
wrapper: it builds a `RedisTaskQueue` + `RedisTaskWorker`, registers
`TASK_NAME = "memory_writer"` and delegates `start`/`stop`/`enqueue`. The worker
degrades to a no-op (logged) when Redis is down.

## Samvaad Channel (hosted Sarvam voice agent)

Opt-in integration with the agent you build in the Sarvam dashboard
(`indus.sarvam.ai/samvaad/build/my-agents`, e.g. `AI-Employee-33c6c05a-c14f`).
It is a **channel**, not a replacement: the dashboard gets a "Local Agent |
Samvaad Agent" toggle, and the platform services stay the tool/data backend.

### Inbound — Samvaad bridge (orchestrator)

`services/orchestrator/services/samvaad_client.py` wraps the official
`sarvam-conv-ai-sdk` (`AsyncSamvaadAgent`) headlessly (no PyAudio — audio moves
through callbacks + `send_audio`). Server messages are normalised onto a
per-session outbox: `text`, `transcript`, `audio`, `event`.

Exposed by `routers/samvaad.py`:

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/api/samvaad/status` | enabled? agent id, active sessions, reason |
| POST | `/api/samvaad/sessions` | open a session (chat/call) → session_id |
| GET  | `/api/samvaad/sessions` | list open sessions |
| GET  | `/api/samvaad/sessions/{id}` | session detail |
| POST | `/api/samvaad/sessions/{id}/text` | send a chat message |
| POST | `/api/samvaad/sessions/{id}/audio` | send base64 PCM16 audio (call) |
| POST | `/api/samvaad/sessions/{id}/close` | close a session |
| GET  | `/api/samvaad/sessions/{id}/messages` | poll normalised replies |
| WS   | `/api/samvaad/ws` | full-duplex proxy (`init`/`text`/`audio`/`poll`) |

When the SDK is missing or the config is incomplete, endpoints degrade to
`503` with a reason instead of crashing.

### Outbound — real-data webhook tools (`routers/samvaad_tools.py`)

The Samvaad agent acts on real platform data via webhook Tools / On-Start /
On-End hooks authored in the Sarvam dashboard. Base URL = this orchestrator's
public URL (or the gateway proxy path `/api/orchestrator/samvaad/tools/...`).
Auth: `X-API-Key` required, or `X-Samvaad-Secret` when `SAMVAAD_TOOL_SECRET` is
set.

| Agent hook / tool | Endpoint | Real backend |
|-------------------|----------|--------------|
| On-Start | `POST /api/samvaad/tools/on-start/context` | memory profile + session language + today's meetings → `agent_variables` |
| On-End | `POST /api/samvaad/tools/on-end/record` | writes transcript + summary fact to the memory service |
| calendar | `POST /api/samvaad/tools/calendar/availability` | `CalendarService.check_availability` (Google/ICS) |
| calendar | `POST /api/samvaad/tools/calendar/schedule` | `CalendarService.create_meeting` |
| calendar | `POST /api/samvaad/tools/calendar/update` | `CalendarService.update_meeting` / `cancel_meeting` (action) |
| email | `POST /api/samvaad/tools/email/send` | `EmailClient` (SMTP/Gmail, action) |
| email | `POST /api/samvaad/tools/email/search` | `EmailClient` (IMAP search, read-only) |
| knowledge | `POST /api/samvaad/tools/search/documents` | RAG search |
| orders | `POST /api/samvaad/tools/orders/lookup` | `OrderService` |
| pricing | `POST /api/samvaad/tools/pricing/search` | `PricingService` |
| tasks | `POST /api/samvaad/tools/tasks/manage` | `TaskService` (user_tasks table, action) |
| handoff | `POST /api/samvaad/tools/human/transfer` | `EscalationService` |

#### Wiring the Sarvam dashboard

The webhooks are entered in the Sarvam build canvas (`Tools`, plus the On-Start /
On-End hooks). Every call is a `POST` to `<PUBLIC_BASE>/api/samvaad/tools/<name>`
where `PUBLIC_BASE` is a **publicly reachable** URL for the orchestrator. An ngrok
tunnel to `localhost:8001` works for testing:

```sh
ngrok http 8001 --domain=comic-paragraph-peroxide.ngrok-free.dev
# → https://comic-paragraph-peroxide.ngrok-free.dev
# Persistent runner: scripts/run-ngrok.ps1 (auto-restarts, logs to %TEMP%\opencode\ngrok)
```

Windows Defender may flag ngrok as a false positive — add an exclusion for the
ngrok folders (scripts/ngrok-unblock.ps1 does this elevated).

Every request must carry the header `X-Samvaad-Secret: <SAMVAAD_TOOL_SECRET>`
(from `.env`) — or, as a fallback when the webhook editor cannot send custom
headers, a `?token=<SAMVAAD_TOOL_SECRET>` query parameter on the URL. Header
auth takes precedence. When `SAMVAAD_TOOL_SECRET` is unset, a non-empty
`X-API-Key` header is required instead.

| Tool name | URL (method POST) | JSON body the agent sends |
|-----------|-------------------|---------------------------|
| on_start_context | `.../tools/on-start/context` | `{"user_identifier": "...", "session_id": "..."}` |
| on_end_record | `.../tools/on-end/record` | `{"session_id":"...","user_id":"...","transcript":[{"role":"user","text":"..."}],"duration_ms":1000}` |
| check_calendar_availability | `.../tools/calendar/availability` | `{"start_at":"2026-08-18T09:00:00+05:30","duration_minutes":30,"timezone":"Asia/Kolkata"}` |
| schedule_meeting | `.../tools/calendar/schedule` | `{"session_id":"...","title":"...","start_at":"...","end_at":"...","attendees":["a@b.c"],"description":"..."}` |
| update_calendar_meeting | `.../tools/calendar/update` | `{"action":"reschedule","meeting_id":"...","new_start_at":"...","new_end_at":"..."}` or `{"action":"cancel","meeting_id":"..."}` (also matches by `session_id` + `start_at`) |
| send_email | `.../tools/email/send` | `{"to":"...","subject":"...","body":"..."}` |
| search_emails | `.../tools/email/search` | `{"query":"FROM \"boss@example.com\"","max_results":10,"with_body":false}` |
| search_documents | `.../tools/search/documents` | `{"query":"return policy","top_k":5}` |
| lookup_order | `.../tools/orders/lookup` | `{"order_id":"ORD-1234"}` |
| search_pricing | `.../tools/pricing/search` | `{"query":"enterprise plan","top_k":5}` |
| manage_task | `.../tools/tasks/manage` | `{"action":"create","session_id":"...","user_id":"...","title":"...","due_at":"..."}`, `{"action":"list","session_id":"..."}`, `{"action":"complete","task_id":"..."}`, `{"action":"update","task_id":"...","status":"in_progress"}`, `{"action":"delete","task_id":"..."}` |
| transfer_to_human | `.../tools/human/transfer` | `{"reason":"...","user_input":"...","priority":"NORMAL"}` |

The webhooks are thin and delegate to the platform's production services, so the
agent reads/writes real calendar, email, RAG, orders, pricing, memory and task
data.

##### Confirmation gate for action tools

The read-only tools (`on-start/context`, `on-end/record`,
`calendar/availability`, `email/search`, `search/documents`, `orders/lookup`,
`pricing/search`) are always available. The real-action tools
**`email/send`, `calendar/schedule`, `human/transfer`, `calendar/update`,
`tasks/manage` are blocked by default** (403) and only run after they are
explicitly listed in `SAMVAAD_TOOLS_ALLOWLIST` in `.env` — e.g.
`SAMVAAD_TOOLS_ALLOWLIST='["email/send","calendar/schedule","human/transfer","calendar/update","tasks/manage"]'`.
This is the explicit confirmation gate: production cannot use the action tools
until an operator deliberately enables them. The read-only defaults are always
available even when the allowlist is set; the allowlist only unlocks the action
tools on top of them. `GET /api/samvaad/status` reports the current
`tools: {allowed, blocked}` state.

### Config

`SAMVAAD_ENABLED`, `SAMVAAD_API_KEY`, `SAMVAAD_AGENT_ID`, `SAMVAAD_ORG_ID`,
`SAMVAAD_WORKSPACE_ID`, `SAMVAAD_APP_RUNTIME_URL`, `SAMVAAD_AGENT_VERSION`,
`SAMVAAD_SAMPLE_RATE`, `SAMVAAD_DEFAULT_LANGUAGE`, `SAMVAAD_CONNECT_TIMEOUT`,
`SAMVAAD_TOOL_SECRET`, `SAMVAAD_TOOLS_ALLOWLIST`.

Verify with `python scripts/samvaad_verify.py` (key auth + committed-version
probe; auto-falls back chat → call since the hosted agent is voice-only). The
agent must have a **committed version** — the SDK refuses to connect otherwise.

### Frontend

`services/orchestrator/static/dashboard.html` shows the channel toggle when
`/api/samvaad/status` reports enabled. The hosted Samvaad agent is a **voice
(CALL) agent**, so:

- **Text chat always uses the local agent** — the toggle never reroutes typed
  messages to Samvaad (chat interaction type returns 404 for voice agents).
- **Mic** records via the browser, transcodes webm → 16 kHz PCM16 in the page,
  sends it as `audio` over the WS proxy, and plays the returned audio chunks
  back. With the "Samvaad Agent" toggle selected the mic talks to the hosted
  voice agent; otherwise it uses the local STT pipeline.

## Configuration Guide

### Local Development

```bash
# .env file
RAG_URL=http://localhost:8004
SARVAM_API_KEY=sk_xxx
SARVAM_MODEL=sarvam-105b

# Terminal 1: Orchestrator
uvicorn orchestrator.main:app --port 8001

# Terminal 2: RAG service (optional)
uvicorn rag.main:app --port 8004

# Terminal 3: CLI
python scripts/cli.py
```

### Docker

```bash
docker compose --profile all up
# RAG_URL → http://rag:8004 (docker-compose.yml override)
# SARVAM_API_KEY → from .env file
```

### Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SARVAM_API_KEY` | `""` | Empty = MockPlanner mode |
| `SARVAM_MODEL` | `sarvam-105b` | LLM model for classify + generate |
| `RAG_URL` | `http://localhost:8004` | RAG service host (Docker: `http://rag:8004`) |
| `RAG_QUERY_PATH` | `/api/v1/documents/query` | RAG query endpoint |
| `RAG_HEALTH_PATH` | `/health` | RAG health check endpoint |
| `RAG_TIMEOUT` | `5.0` | RAG HTTP request timeout (seconds) |
| `LLM_TEMPERATURE` | `0.1` | LLM sampling temperature |
| `LLM_MAX_TOKENS` | `1024` | Max output tokens for generation |
| `LLM_CLASSIFY_MAX_TOKENS` | `512` | Max tokens for intent classification |
| `LLM_FALLBACK_INTENT` | `general` | Default intent on classification failure |
| `GUARDRAILS_ENABLED` | `true` | Apply input sanitization/redaction/PII filters |
| `RATE_LIMIT_ENABLED` | `true` | Enforce per-client rate limit on channels |
| `RATE_LIMIT_LIMIT` | `30` | Max requests per window per client |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window length |

## Service Interfaces (Protocols)

```
services/interfaces.py
    ├── OrderService       → lookup_order(order_id) → dict
    ├── CalendarService    → get_availability(query, days) → list[dict]
    │                         schedule_demo(title, ...) → dict
    └── PricingService     → search_pricing(query, top_k) → list[dict]

services/mock_services.py
    ├── MockOrderService       (deterministic order statuses)
    ├── MockCalendarService     (business-day availability slots)
    └── MockPricingService     (Free/Pro/Enterprise tiers)
```

## Shared Libraries

```
shared/llm/
    ├── base.py        → LLMProvider ABC + IntentClassification + LLMResponse
    ├── sarvam_provider.py → Sarvam AI implementation (httpx + tenacity + structlog)
    ├── schemas.py     → Pydantic models for intent classification
    └── __init__.py

shared/schemas/
    └── channels.py    → ChannelType, ChannelContact, ChannelMessage, ChannelResponse

shared/guardrails/
    ├── redactor.py    → PIIRedactor (emails, phones, credit cards, IPs)
    ├── sanitizer.py   → InputSanitizer
    ├── filter.py      → ContentFilter (injection + toxicity blocking)
    ├── rate_limiter.py → RedisRateLimiter (fixed window)
    ├── service.py     → GuardrailsService (enabled / redact_pii flags)
    └── __init__.py

shared/queue/
    ├── queue.py       → RedisTaskQueue (rpush/lpop JSON envelopes)
    ├── worker.py      → RedisTaskWorker (async dispatch loop)
    └── __init__.py

shared/utils/
    ├── exceptions.py  → AppException hierarchy
    ├── logging.py     → structlog configuration
    └── response.py    → FastAPI response helpers
```
