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
            │       ├─ Success → [{title, snippet, score}]
            │       │
            │       └─ Failure → fallback_client.search()
            │                       │
            │                       └─→ MockRAGClient (5 hardcoded docs)
            │
            └─→ health_check() ── GET {RAG_URL}/health → bool
                    (called at startup, logs warning if unhealthy)
```

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

shared/utils/
    ├── exceptions.py  → AppException hierarchy
    ├── logging.py     → structlog configuration
    └── response.py    → FastAPI response helpers
```
