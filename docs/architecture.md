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
