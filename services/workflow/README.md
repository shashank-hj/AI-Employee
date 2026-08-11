# AI Employee — Workflow Engine

A LangGraph-backed workflow execution service. Workflows are declared as a list of
steps (JSON) and compiled into a runnable state graph. It supports linear task chains,
parallel fan-out, data-driven branching, and human-in-the-loop approval gates with a
durable checkpointer (Postgres, with an in-memory fallback).

## Run

```bash
# local (uses DATABASE_URL default postgres://postgres:postgres@localhost:5432/ai_employee)
uvicorn workflow.main:app --port 8005

# docker
docker compose --profile workflow up -d --build workflow
```

Tables are auto-created on startup (`Base.metadata.create_all`). On boot the service
also reconciles any workflow stuck in `running` (left behind by a crash) to `failed`.

## Step DSL

A workflow is created with a `steps` array. Each step must have a unique `name`.
`next` (single name or list) routes to the next step; omitting it ends the workflow.

| type      | fields                                      | behaviour                                                    |
| --------- | ------------------------------------------- | ------------------------------------------------------------ |
| `task`    | `handler`, `params`, `requires_approval`    | Runs one handler. `requires_approval: true` pauses the graph for a human decision. |
| `fan_out` | `handlers` (list), `params`                 | Runs several handlers in parallel; results collected under the step keyed by handler. |
| `branch`  | `field`, `branches`, `default`              | No-op router. `field` is a `$input.x` / `$outputs.x` path; `branches` maps values → next step. |

`params` support references: `$input.x` (run input) and `$outputs.a.b` (prior step
output), resolved at execution time.

```json
{
  "name": "refund-flow",
  "steps": [
    { "name": "validate", "type": "task", "handler": "validate_refund",
      "params": { "order_id": "$input.order_id", "amount": "$input.amount" },
      "next": "refund" },
    { "name": "refund", "type": "task", "handler": "process_refund",
      "params": { "order_id": "$input.order_id", "amount": "$input.amount" },
      "requires_approval": true, "next": "notify" },
    { "name": "notify", "type": "task", "handler": "send_email",
      "params": { "to": "$input.email", "subject": "Refund processed" } }
  ]
}
```

## API

| Method | Path                              | Purpose                                        |
| ------ | --------------------------------- | ---------------------------------------------- |
| POST   | `/api/workflows`                  | Create a workflow (`steps` optional; defaults to an echo chain). |
| GET    | `/api/workflows`                  | List, filter by `?status=`.                    |
| GET    | `/api/workflows/{id}`             | Fetch (includes `pending_approval` while waiting). |
| POST   | `/api/workflows/{id}/run`         | Execute; `{"stream": true}` for SSE.           |
| POST   | `/api/workflows/{id}/continue`    | Resume at an approval gate (`{"payload": {"approved": true/false, "reason": "..."}}`). |
| POST   | `/api/workflows/{id}/pause`       | Pause a running workflow.                      |
| POST   | `/api/workflows/{id}/resume`      | Resume a paused workflow.                      |
| POST   | `/api/workflows/{id}/cancel`      | Cancel pending/running/paused workflow.        |
| GET    | `/api/workflows/{id}/history`     | Per-step execution history.                    |
| DELETE | `/api/workflows/{id}`             | Delete a workflow (and its checkpoint thread). |

### Human-in-the-loop

`run` on a workflow with an approval-gated step returns `interrupted: true` and leaves
the workflow `paused`. The pending decision is exposed on
`GET /api/workflows/{id}` as `pending_approval`:

```json
{
  "status": "paused",
  "current_step": "refund",
  "pending_approval": {
    "type": "human_approval",
    "workflow_step": "refund",
    "handler": "process_refund",
    "params": { "order_id": "123", "amount": 50 },
    "reason": "'refund' requires human approval"
  }
}
```

Post `{"approved": true}` to `/continue` to run the gated handler, or
`{"approved": false, "reason": "..."}` to record it as rejected (the step is skipped).
Approval state lives in the checkpointer, so it survives server restarts (Postgres).

### Streaming

`POST /run` with `{"stream": true}` returns SSE. Each frame is a JSON object on a
`data:` line: `{"type":"update","data":{...}}` per graph update, then
`{"type":"done","data":{...}}` with the final `WorkflowRunResponse`.

### Guards & timeouts

- `run` on a `paused` workflow → `409` (use `/continue`); on a `running` workflow → `409`.
- `run` on a terminal workflow (completed/failed/cancelled) resets the checkpoint thread
  and starts fresh, so re-runs are safe.
- `continue` on a non-paused workflow → `409`.
- `run` accepts `timeout_seconds` (default 300); the execution is aborted and the
  workflow marked `failed` if it exceeds it.

## Handlers

Built-in handlers in `workflow/graph/handlers.py` (`HANDLERS`):

`echo`, `calculator`, `sleep`, `validate_refund`, `process_refund`, `send_email`,
`check_inventory`, `check_pricing`.

To add a capability, add an `async def handler(params: dict) -> dict` and register it in
`HANDLERS` (and `APPROVAL_HANDLERS` if it should always require approval).

## Tests

```bash
python -m pytest services/workflow/tests -q   # 15 tests: graph + API integration
```

## Layout

```
workflow/
  graph/        # builder (steps -> StateGraph), checkpointer, handlers, state
  services/     # WorkflowService (run/stream/continue/CRUD)
  routers/      # FastAPI endpoints
  schemas/      # pydantic request/response models
  repositories/ # SQLAlchemy data access
  models/       # ORM model (workflows table)
```
