# AI Employee Platform

A distributed multi-agent orchestration platform with LLM-powered intent classification and natural-language response generation.

## Architecture

```
User Input → Pre-Classify (regex) → Sarvam Intent Classifier → Tool Resolution
→ Tool Execution → Context Assembly → LLM Response Generation → Final Response
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| **gateway** | 8000 | API gateway and request routing |
| **orchestrator** | 8001 | Intent classification, tool routing, response generation |
| **tool-registry** | 8002 | Tool discovery and execution |
| **memory** | 8003 | Long-term and working memory store |
| **rag** | 8004 | Retrieval-augmented generation |
| **workflow** | 8005 | Workflow definition and execution |

### Orchestrator Pipeline

```
receive → build_context → plan → after_plan_route
                                     │
              ┌──────────────────────┴──────────────┐
              ▼ (escalation/complaint)              ▼ (normal)
          respond (pre-set)              execute → tool_invoke → execute → ...
              │                                       │
              └────────── respond (LLM) ──────────────┘
                                    │
                                   END
```

### Intent → Tool Mapping

| Intent | Tools Invoked |
|--------|--------------|
| **sales** | `search_pricing` → `search_documents` |
| **support** | `lookup_order` → `search_documents` |
| **booking** | `calendar` → `schedule_demo` |
| **general** | `search_documents` (or LLM-suggested tool) |
| **math** | `calculator` |
| **complaint** | escalation short-circuit (no tools) |
| **escalate** | escalation short-circuit (no tools) |

### Tool Registry (10 tools)

`calculator`, `search_documents`, `search_pricing`, `lookup_order`, `get_weather`, `calendar`, `schedule_demo`, `schedule_meeting`, `send_email`, `transfer_to_human`

## Tech Stack

- **Python 3.12** with **FastAPI** and **Uvicorn**
- **LangGraph** for stateful agent orchestration
- **Multi-LLM support**: Ollama (local), Sarvam AI (cloud), extensible for OpenAI
- **PostgreSQL 16** via **SQLAlchemy** (async) and **asyncpg**
- **Redis 7** for caching and pub/sub
- **Alembic** for database migrations
- **OpenTelemetry** for observability
- **Structlog** for structured logging

### LLM Provider Architecture

```
                    LLMProvider (abstract)
                         │
          ┌──────────────┼──────────────┐
          │              │              │
   OllamaProvider  SarvamProvider  (OpenAI future)

    Local llama     Cloud API       Drop-in via
    qwen3:8b       sarvam-105b     same interface
```

Switch providers by changing one env var — no code changes needed.

## Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL 16 (for database-backed services)
- **Ollama** (recommended for local development) or a Sarvam AI API key

### 0. Install Ollama (recommended)

```bash
# Install Ollama
# Windows: https://ollama.com/download/windows
# macOS:   brew install ollama
# Linux:   curl -fsSL https://ollama.com/install.sh | sh

# Pull the recommended model
ollama pull qwen3:8b

# Verify it works
ollama run qwen3:8b "Hello, how are you?"
```

### 1. Clone and install

```bash
git clone <repo-url>
cd ai-employee-platform

# Create .env from example
cp .env.example .env

# Install dependencies
uv sync
```

### 2. Configure your LLM provider

Edit `.env` and choose your provider:

```bash
# For Ollama (local, free, no API key):
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:8b

# For Sarvam AI (cloud, requires API key):
LLM_PROVIDER=sarvam
SARVAM_API_KEY=sk_your_key_here

# For mock mode (offline, regex-based):
LLM_PROVIDER=
```

### 3. Run the orchestrator

```bash
uvicorn orchestrator.main:app --port 8001
```

Expected startup log:
```
{"event": "startup_configuration", "llm_provider": "OllamaProvider", "llm_model": "qwen3:8b", ...}
{"event": "llm_healthy", "model": "qwen3:8b"}
```

### 4. Interactive CLI
```

You should see:
```json
{"event": "startup_configuration", "version": "0.1.0", "rag_url": "http://localhost:8004", "llm_provider": "SarvamProvider"}
{"event": "llm_healthy", "model": "sarvam-105b"}
{"event": "rag_unhealthy", "message": "RAG service unavailable; SearchDocumentsTool will use mock fallback"}
```

### 3. Interactive CLI

```bash
python scripts/cli.py
```

```
============================================================
  AI Employee Platform - Interactive CLI
  Type 'exit' or 'quit' to stop
============================================================

You: My order ORD-7891 hasn't arrived yet
→ Executes lookup_order and search_documents tools
→ LLM synthesizes a natural-language response about order status

You: Talk to a human
→ Escalation short-circuit — responds immediately with human agent message

You: What is 5 + 5?
→ Pre-filter catches math expression → calculator tool → "5 + 5 equals 10"

You: exit
Goodbye!
```

### 4. Test an endpoint directly

```bash
# Health check
curl http://localhost:8001/health

# Run agent
curl -X POST http://localhost:8001/api/agent/run \
  -H "Content-Type: application/json" \
  -d '{"user_input": "What is 5 + 5?"}'
```

## Running Tests

```bash
# All 81 tests (~24 seconds, no API key needed)
uv run pytest services/orchestrator/tests/ -v

# Specific test file
uv run pytest services/orchestrator/tests/test_planner.py -v
uv run pytest services/orchestrator/tests/test_agent.py -v
uv run pytest services/orchestrator/tests/test_tools.py -v
uv run pytest services/orchestrator/tests/test_rag_client.py -v
```

Tests auto-force mock mode via `conftest.py` — no API key or database required.

## Docker

```bash
# All services
docker compose --profile all up -d

# Specific services
docker compose --profile orchestrator --profile rag up -d

# Infrastructure only
docker compose up -d postgres redis
```

Docker Compose overrides `RAG_URL` to `http://rag:8004` and passes `SARVAM_API_KEY` from `.env`.

## Configuration

All configuration lives in `services/orchestrator/config.py` and `.env`.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `""` | `"ollama"`, `"sarvam"`, or `""` (mock) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen3:8b` | Ollama model name |
| `OLLAMA_TIMEOUT` | `60.0` | Ollama request timeout (seconds) |
| `SARVAM_API_KEY` | `""` | Sarvam AI API key |
| `SARVAM_MODEL` | `sarvam-105b` | Sarvam model name |
| `SARVAM_BASE_URL` | `https://api.sarvam.ai` | Sarvam API endpoint |
| `SARVAM_TIMEOUT` | `30.0` | Sarvam request timeout (seconds) |
| `RAG_URL` | `http://localhost:8004` | RAG service URL |
| `RAG_TIMEOUT` | `5.0` | RAG request timeout (seconds) |
| `LLM_TEMPERATURE` | `0.1` | LLM sampling temperature |
| `LLM_MAX_TOKENS` | `1024` | Max tokens for response generation |
| `LLM_CLASSIFY_MAX_TOKENS` | `512` | Max tokens for intent classification |
| `LLM_FALLBACK_INTENT` | `general` | Default intent on classification failure |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/agent/run` | Execute an agent run |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc documentation |

### Agent Run Request

```json
{
  "user_input": "My order ORD-7891 hasn't arrived",
  "user_id": "optional-user-id",
  "session_id": "optional-session-id"
}
```

### Agent Run Response

```json
{
  "request_id": "uuid",
  "user_input": "My order ORD-7891 hasn't arrived",
  "final_response": "Your order ORD-7891 is currently in transit...",
  "steps": [
    {
      "step_index": 0,
      "tool_name": "lookup_order",
      "parameters": {"order_id": "ORD-7891"},
      "result": {"tool_name": "lookup_order", "success": true, "data": {...}}
    }
  ],
  "execution_log": [...],
  "completed_at": "2026-07-27T...",
  "duration_ms": 1234.56
}
```

## Project Structure

```
ai-employee-platform/
├── services/
│   ├── gateway/           # API gateway
│   ├── orchestrator/      # Agent orchestration (main service)
│   │   ├── graph/         # LangGraph nodes, edges, state, builder
│   │   ├── planner/       # LLMPlanner + MockPlanner + pre-classifier
│   │   ├── tools/         # BaseTool, ToolRegistry, RAGClient, 10 tools
│   │   ├── services/      # AgentService, service interfaces, mock services
│   │   ├── context/       # Context builder
│   │   └── tests/         # 81 tests across 5 test files
│   ├── tool-registry/     # Tool registration service
│   ├── memory/            # Memory service (Redis + pgvector)
│   ├── rag/               # RAG service (document ingestion + search)
│   └── workflow/          # Workflow engine
├── shared/
│   ├── llm/               # LLMProvider ABC, SarvamProvider, schemas
│   ├── auth/              # JWT + API key middleware
│   ├── events/            # Redis pub/sub event bus
│   ├── models/            # SQLAlchemy base + mixins
│   └── utils/             # Exceptions, logging, response helpers
├── scripts/
│   └── cli.py             # Interactive terminal client
├── docs/
│   └── architecture.md    # Architecture documentation
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

## Architecture Documentation

See [docs/architecture.md](docs/architecture.md) for:
- Full pipeline diagram
- Dependency injection wiring
- Tool execution flow
- RAG architecture with fallback
- Configuration guide (Docker vs Local)
