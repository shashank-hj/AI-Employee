from fastapi import APIRouter, Depends

from orchestrator.container import get_approval_service, get_tool_registry
from orchestrator.services.approval_service import ApprovalService
from orchestrator.tools.registry import ToolRegistry

router = APIRouter(prefix="/api", tags=["Tools"])


@router.get("/tools")
async def list_tools(registry: ToolRegistry = Depends(get_tool_registry)):
    """List registered tools with their parameter schemas (C4 tool discovery)."""
    return {"tools": registry.get_tool_schemas()}


@router.get("/tools/approval")
async def approval_settings(approval: ApprovalService = Depends(get_approval_service)):
    """Report which tools require human-in-the-loop approval."""
    return {
        "enabled": approval.enabled,
        "approval_tools": sorted(approval._approval_tools),
    }
