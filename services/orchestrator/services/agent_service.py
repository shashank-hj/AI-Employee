import time
import uuid

import structlog

from shared.llm.base import LLMProvider

from orchestrator.graph.state import AgentState
from orchestrator.graph.builder import build_orchestrator_graph
from orchestrator.planner.base import BasePlanner
from orchestrator.tools.registry import ToolRegistry
from orchestrator.context.builder import ContextBuilder
from orchestrator.schemas.agent import AgentRequest, AgentResponse, ExecutionStep, ToolResult
from shared.utils.exceptions import AppException

logger = structlog.get_logger(__name__)


class AgentService:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        planner: BasePlanner,
        context_builder: ContextBuilder,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self._graph = build_orchestrator_graph(
            tool_registry, planner, context_builder, llm_provider,
        )
        self._tool_registry = tool_registry

    async def run(self, request: AgentRequest) -> AgentResponse:
        start = time.perf_counter()
        request_id = str(uuid.uuid4())

        initial_state: AgentState = {
            "request_id": request_id,
            "user_input": request.user_input,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "memory_context": [],
            "document_context": [],
            "user_preferences": {},
            "plan": [],
            "current_step_index": 0,
            "tool_results": [],
            "execution_log": [],
            "final_response": None,
            "error": None,
        }

        logger.info("agent_run_started", request_id=request_id, user_input=request.user_input[:100])

        try:
            final_state = await self._graph.ainvoke(initial_state)
        except Exception as exc:
            logger.error("agent_run_failed", request_id=request_id, error=str(exc))
            raise AppException(
                detail=f"Agent execution failed: {str(exc)}",
                status_code=500,
                error_code="AGENT_EXECUTION_ERROR",
            )

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
        )

        logger.info(
            "agent_run_completed",
            request_id=request_id,
            num_steps=len(steps),
            duration_ms=round(elapsed_ms, 2),
        )

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
