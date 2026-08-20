import time
import uuid

import structlog

from shared.llm.base import LLMProvider

from orchestrator.graph.state import AgentState
from orchestrator.graph.builder import build_orchestrator_graph
from orchestrator.graph.checkpointer import get_checkpoint_engine
from orchestrator.planner.base import BasePlanner
from orchestrator.tools.registry import ToolRegistry
from orchestrator.context.builder import ContextBuilder
from orchestrator.schemas.agent import AgentRequest, AgentResponse, ExecutionStep, ToolResult
from orchestrator.services.memory_client import MemoryClient
from orchestrator.workers.memory_writer import MemoryWriterWorker
from shared.usage.context import reset_usage_context, set_usage_context
from shared.utils.exceptions import AppException

logger = structlog.get_logger(__name__)


class AgentService:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        planner: BasePlanner,
        context_builder: ContextBuilder,
        llm_provider: LLMProvider | None = None,
        memory_writer: MemoryWriterWorker | None = None,
        memory_client: MemoryClient | None = None,
        approval_service=None,
    ) -> None:
        self._tool_registry = tool_registry
        self._planner = planner
        self._context_builder = context_builder
        self._llm_provider = llm_provider
        self._memory_writer = memory_writer
        self._memory_client = memory_client
        self._approval_service = approval_service
        self._graph = None

    async def _get_graph(self):
        """Build the compiled graph lazily so the checkpointer is attached only
        after the engine has had a chance to connect to Postgres (or fall back
        to memory)."""
        if self._graph is None:
            engine = get_checkpoint_engine()
            await engine.setup()
            self._graph = build_orchestrator_graph(
                self._tool_registry,
                self._planner,
                self._context_builder,
                self._llm_provider,
                checkpointer=engine.saver,
                approval_service=self._approval_service,
            )
        return self._graph

    async def run(self, request: AgentRequest) -> AgentResponse:
        start = time.perf_counter()
        request_id = str(uuid.uuid4())

        initial_state: AgentState = {
            "request_id": request_id,
            "user_input": request.user_input,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "channel": request.channel.value,
            "channel_message_id": request.channel_message_id,
            "tenant_id": request.tenant_id,
            "contact": request.contact.model_dump() if request.contact else None,
            "request_metadata": request.metadata,
            "memory_context": [],
            "document_context": [],
            "user_preferences": {},
            "plan": [],
            "current_step_index": 0,
            "tool_results": [],
            "execution_log": [],
            "awaiting_approval": False,
            "approval_task_id": None,
            "final_response": None,
            "error": None,
        }

        logger.info("agent_run_started", request_id=request_id, user_input=request.user_input[:100])

        token = set_usage_context(
            request_id=request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        )
        try:
            graph = await self._get_graph()
            config = {
                "configurable": {
                    "thread_id": request.session_id or request_id,
                }
            }
            final_state = await graph.ainvoke(initial_state, config=config)
        except Exception as exc:
            logger.error("agent_run_failed", request_id=request_id, error=str(exc))
            raise AppException(
                detail=f"Agent execution failed: {str(exc)}",
                status_code=500,
                error_code="AGENT_EXECUTION_ERROR",
            )
        finally:
            reset_usage_context(token)

        elapsed_ms = (time.perf_counter() - start) * 1000

        steps = self._build_execution_steps(final_state)
        response = AgentResponse(
            request_id=request_id,
            user_input=request.user_input,
            final_response=final_state.get("final_response", "No response generated."),
            steps=steps,
            execution_log=final_state.get("execution_log", []),
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            duration_ms=round(elapsed_ms, 2),
            channel=request.channel,
            channel_message_id=request.channel_message_id,
            tenant_id=request.tenant_id,
        )

        logger.info(
            "agent_run_completed",
            request_id=request_id,
            num_steps=len(steps),
            duration_ms=round(elapsed_ms, 2),
        )

        # ── M5 Memory Writer: enqueue conversation for background fact extraction ──
        if self._memory_writer is not None:
            await self._memory_writer.enqueue(
                {
                    "request_id": request_id,
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "user_input": request.user_input,
                    "final_response": response.final_response,
                    "tool_results": final_state.get("tool_results", []),
                    "execution_log": final_state.get("execution_log", []),
                    "completed_at": response.completed_at,
                }
            )

        # ── Store session messages for context retrieval ──
        if self._memory_client is not None and request.session_id:
            try:
                await self._memory_client.upsert_session(
                    request.session_id,
                    user_id=request.user_id,
                )
                await self._memory_client.add_message(
                    request.session_id, "user", request.user_input,
                    user_id=request.user_id,
                )
                await self._memory_client.add_message(
                    request.session_id, "assistant", response.final_response,
                    user_id=request.user_id,
                )
                logger.info("session_message_stored", session_id=request.session_id)
            except Exception as exc:
                logger.warning("session_message_store_failed", error=str(exc))

        return response

    def _build_execution_steps(self, state: dict) -> list[ExecutionStep]:
        plan = state.get("plan", [])
        tool_results = state.get("tool_results", [])

        steps: list[ExecutionStep] = []
        for i, step in enumerate(plan):
            result = next(
                (r for r in tool_results if r.get("step_index") == i),
                None,
            )
            tr = None
            if result:
                tr = ToolResult(
                    tool_name=result.get("tool_name", step["tool_name"]),
                    success=result.get("success", False),
                    data=result.get("data"),
                    error=result.get("error"),
                )
            steps.append(ExecutionStep(
                step_index=i,
                tool_name=step["tool_name"],
                parameters=step["parameters"],
                result=tr,
            ))
        return steps
