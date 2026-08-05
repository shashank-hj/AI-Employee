import json
import uuid
from typing import Any

from shared.llm.base import LLMProvider

from orchestrator.graph.state import AgentState, PlanStep
from orchestrator.planner.base import BasePlanner
from orchestrator.tools.registry import ToolRegistry
from orchestrator.context.builder import ContextBuilder


def create_receive_node() -> Any:
    async def receive(state: AgentState) -> dict[str, Any]:
        request_id = state.get("request_id") or str(uuid.uuid4())
        return {
            "request_id": request_id,
            "current_step_index": 0,
            "tool_results": [],
            "execution_log": [
                {
                    "node": "receive",
                    "event": "request_received",
                    "user_input": state["user_input"],
                    "request_id": request_id,
                }
            ],
            "final_response": None,
            "error": None,
        }
    return receive


def create_context_node(context_builder: ContextBuilder) -> Any:
    async def build_context(state: AgentState) -> dict[str, Any]:
        ctx = await context_builder.build(
            user_input=state["user_input"],
            user_id=state.get("user_id"),
            session_id=state.get("session_id"),
        )
        return {
            "memory_context": ctx.get("memory_context", []),
            "document_context": ctx.get("document_context", []),
            "user_preferences": ctx.get("user_preferences", {}),
            "execution_log": [
                {
                    "node": "build_context",
                    "event": "context_loaded",
                    "memory_items": len(ctx.get("memory_context", [])),
                    "document_items": len(ctx.get("document_context", [])),
                }
            ],
        }
    return build_context


def create_plan_node(planner: BasePlanner) -> Any:
    async def plan(state: AgentState) -> dict[str, Any]:
        plan_steps = await planner.create_plan(state)
        result: dict[str, Any] = {
            "plan": plan_steps,
            "current_step_index": 0,
            "execution_log": [
                {
                    "node": "plan",
                    "event": "plan_created",
                    "num_steps": len(plan_steps),
                    "steps": [s["tool_name"] for s in plan_steps],
                }
            ],
        }
        if state.get("final_response") is not None:
            result["final_response"] = state["final_response"]
        return result
    return plan


def create_execute_node(tool_registry: ToolRegistry) -> Any:
    async def execute(state: AgentState) -> dict[str, Any]:
        plan = state["plan"]
        index = state["current_step_index"]

        if index >= len(plan):
            return {
                "execution_log": [{"node": "execute", "event": "plan_complete", "total_steps": len(plan)}],
            }

        current_step = plan[index]
        tool = tool_registry.get(current_step["tool_name"])
        step_log = {
            "node": "execute",
            "event": "step_started",
            "step_index": index,
            "tool_name": current_step["tool_name"],
            "parameters": current_step["parameters"],
        }

        if tool is None:
            return {
                "execution_log": [
                    {**step_log, "event": "step_failed", "error": f"Tool '{{current_step['tool_name']}}' not found in registry"},
                ],
                "current_step_index": index + 1,
                "tool_results": [{"tool_name": current_step["tool_name"], "success": False, "error": "Tool not found"}],
            }

        result = await tool.invoke(current_step["parameters"])
        return {
            "current_step_index": index + 1,
            "tool_results": [
                {
                    "tool_name": current_step["tool_name"],
                    "step_index": index,
                    **result,
                }
            ],
            "execution_log": [
                {
                    **step_log,
                    "event": "step_completed",
                    "success": result.get("success", False),
                }
            ],
        }

    return execute


def create_tool_invoke_node(tool_registry: ToolRegistry) -> Any:
    async def tool_invoke(state: AgentState) -> dict[str, Any]:
        plan = state["plan"]
        index = state["current_step_index"]

        if index >= len(plan):
            return {
                "execution_log": [{"node": "tool_invoke", "event": "no_steps_remaining"}],
            }

        current_step = plan[index]
        tool = tool_registry.get(current_step["tool_name"])

        if tool is None:
            return {
                "tool_results": [
                    {"tool_name": current_step["tool_name"], "success": False, "error": "Tool not found in registry"}
                ],
                "execution_log": [
                    {"node": "tool_invoke", "event": "tool_not_found", "tool_name": current_step["tool_name"]}
                ],
            }

        result = await tool.invoke(current_step["parameters"])
        return {
            "tool_results": [
                {
                    "tool_name": current_step["tool_name"],
                    "step_index": index,
                    **result,
                }
            ],
            "execution_log": [
                {
                    "node": "tool_invoke",
                    "event": "tool_executed",
                    "tool_name": current_step["tool_name"],
                    "success": result.get("success", False),
                }
            ],
        }

    return tool_invoke


RESPONSE_SYSTEM_PROMPT = (
    "You are a helpful AI employee assistant for an enterprise platform. "
    "You have access to tool results that provide factual information about the company. "
    "Rules: "
    "1. Use the provided tool results as your primary source to answer the user's question. "
    "2. If tool results are empty, irrelevant, or insufficient, use your own general knowledge to answer directly. "
    "You know basic facts like today's date, geography, history, math, definitions, and common knowledge. "
    "3. If a user asks about past conversation details, reference the conversation history provided in the prompt. "
    "4. Keep responses friendly, professional, and under 3 paragraphs when possible. "
    "5. If the user's request requires a human, confirm the escalation has been initiated."
)

PERSONA_PROMPTS: dict[str, str] = {
    "sales": (
        "You are a Sales AI Employee. You help prospects and customers with product information, pricing, "
        "and purchase decisions. Be enthusiastic, knowledgeable, and solution-oriented. "
        "When presenting pricing, highlight the value proposition of each tier. "
        "Offer to schedule a demo if appropriate. End with a helpful next step."
    ),
    "support": (
        "You are a Support AI Employee. You help customers with order tracking, returns, troubleshooting, "
        "and account issues. Be empathetic, patient, and thorough. If you cannot resolve the issue, "
        "offer to escalate to a human agent. Always confirm that the customer's concern is addressed."
    ),
    "booking": (
        "You are a Booking AI Employee. You help users schedule demos, appointments, and meetings. "
        "Be efficient and precise with dates, times, and availability. Confirm all booking details "
        "before finalizing. Offer alternative slots if the requested time is unavailable."
    ),
    "general": RESPONSE_SYSTEM_PROMPT,
    "complaint": (
        "You are handling a customer complaint. Be deeply empathetic, acknowledge their frustration, "
        "and confirm that their concern has been escalated to a human agent who will address it personally. "
        "Do not make promises about refunds or compensation — the human agent will handle that."
    ),
    "escalate": (
        "The user has requested to speak with a human agent. Confirm that their request "
        "has been received and that a human agent will review their conversation history "
        "and respond shortly. Be courteous and reassuring."
    ),
}


def create_respond_node(llm_provider: LLMProvider | None = None) -> Any:
    import structlog
    logger = structlog.get_logger(__name__)

    async def respond(state: AgentState) -> dict[str, Any]:
        if state.get("final_response") is not None:
            logger.info(
                "respond_escalation_short_circuit",
                request_id=state.get("request_id"),
                response_length=len(state["final_response"]),
            )
            return {
                "execution_log": [
                    {
                        "node": "respond",
                        "event": "escalation_response_used",
                        "response_length": len(state["final_response"]),
                    }
                ],
            }

        plan = state.get("plan", [])
        tool_results = state.get("tool_results", [])

        if not tool_results and not plan:
            msg = "I received your request but I'm not sure how to help with that. Could you provide more details?"
            logger.info("respond_no_results", request_id=state.get("request_id"))
            return {
                "final_response": msg,
                "execution_log": [
                    {"node": "respond", "event": "no_tool_results", "response_length": len(msg)}
                ],
            }

        tool_context = _build_tool_context(tool_results)
        prompt_parts = [f"User's question: {state['user_input']}"]

        memory_context = state.get("memory_context", [])
        if memory_context:
            memory_lines = []
            for msg in memory_context:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                memory_lines.append(f"[{role}]: {content}")
            prompt_parts.insert(1, f"Conversation history:\n" + "\n".join(memory_lines))

        prompt_parts.append(f"Tool results:\n{tool_context}")
        user_message = "\n\n".join(prompt_parts)[:3000]

        logger.info(
            "respond_assembling_context",
            request_id=state.get("request_id"),
            tool_count=len(tool_results),
            prompt_length=len(user_message),
        )

        if llm_provider is not None:
            logger.info(
                "respond_calling_llm",
                request_id=state.get("request_id"),
            )

            try:
                response = await llm_provider.generate(
                    system_prompt=_resolve_persona(state),
                    user_message=user_message,
                )
                logger.info(
                    "respond_llm_success",
                    request_id=state.get("request_id"),
                    model=response.model,
                    output_tokens=response.output_tokens,
                    duration_ms=response.duration_ms,
                )
                return {
                    "final_response": response.content,
                    "execution_log": [
                        {
                            "node": "respond",
                            "event": "llm_response_generated",
                            "model": response.model,
                            "output_tokens": response.output_tokens,
                            "duration_ms": response.duration_ms,
                        }
                    ],
                }
            except Exception as exc:
                logger.error(
                    "respond_llm_failed_fallback_natural",
                    request_id=state.get("request_id"),
                    error=str(exc),
                )

        logger.info("respond_no_llm_fallback", request_id=state.get("request_id"))
        summary = _build_natural_summary(tool_results)
        return {
            "final_response": summary,
            "execution_log": [
                {"node": "respond", "event": "no_llm_summary", "response_length": len(summary)}
            ],
        }

    return respond


def _build_tool_context(tool_results: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for result in tool_results:
        tool_name = result.get("tool_name", "unknown")
        if result.get("success"):
            data = result.get("data", {})
            try:
                parts.append(f"[{tool_name}]: {json.dumps(data, default=str)}")
            except (TypeError, ValueError):
                parts.append(f"[{tool_name}]: {str(data)}")
        else:
            error = result.get("error", result.get("data", {}).get("error", "Unknown error"))
            parts.append(f"[{tool_name} - FAILED]: {error}")
    return "\n".join(parts)


def _build_natural_summary(tool_results: list[dict[str, Any]]) -> str:
    """Fallback response when LLM is unavailable. Summarizes tool results in plain English."""
    parts: list[str] = []
    for result in tool_results:
        tool_name = result.get("tool_name", "")
        if not result.get("success"):
            continue
        data = result.get("data", {})

        if tool_name == "calculator":
            parts.append(f"{data.get('expression', '')} = {data.get('result', '')}")
        elif tool_name in ("search_documents", "search_pricing"):
            items = data.get("results", [])
            if items:
                names = [r.get("title", r.get("tier", "")) for r in items[:3]]
                parts.append(f"Found: {', '.join(names)}")
        elif tool_name == "lookup_order":
            order = data
            parts.append(
                f"Order {order.get('order_id', '')}: {order.get('status', 'unknown')}"
                f", delivery by {order.get('estimated_delivery', 'unknown')}"
            )
        elif tool_name == "get_weather":
            parts.append(
                f"Weather in {data.get('location', '')}: "
                f"{data.get('temperature', '?')}{chr(176)}{data.get('unit', 'celsius')[0].upper()}, {data.get('conditions', '')}"
            )
        elif tool_name == "calendar":
            slots = data.get("available_slots", [])
            dates = [s.get("date", "") for s in slots[:3]]
            parts.append(f"Available on: {', '.join(dates)}")
        elif tool_name in ("schedule_demo", "schedule_meeting"):
            parts.append(f"Scheduled: {data.get('title', '')} on {data.get('datetime', '')}")
        elif tool_name == "transfer_to_human":
            parts.append(data.get("message", "Escalated to human agent."))
        elif tool_name == "send_email":
            parts.append(f"Email sent to {data.get('to', 'recipient')}")

    return "\n".join(parts) if parts else "I received your request but found no results to share."


def _resolve_persona(state: AgentState) -> str:
    plan = state.get("plan", [])
    if not plan:
        if state.get("final_response"):
            return PERSONA_PROMPTS.get("escalate", RESPONSE_SYSTEM_PROMPT)
        return RESPONSE_SYSTEM_PROMPT

    tool_names = [s["tool_name"] for s in plan]
    if "search_pricing" in tool_names:
        return PERSONA_PROMPTS["sales"]
    if "lookup_order" in tool_names:
        return PERSONA_PROMPTS["support"]
    if "calendar" in tool_names or "schedule_demo" in tool_names:
        return PERSONA_PROMPTS["booking"]
    return PERSONA_PROMPTS.get("general", RESPONSE_SYSTEM_PROMPT)