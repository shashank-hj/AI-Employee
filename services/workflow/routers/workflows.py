from fastapi import APIRouter, Depends, Query
from typing import Optional
from workflow.schemas.workflows import WorkflowCreate, WorkflowResponse, WorkflowHistoryEntry

router = APIRouter(prefix="/api")


@router.post("/workflows", response_model=WorkflowResponse, status_code=201)
async def create_workflow(workflow: WorkflowCreate):
    """Create and start a new workflow."""
    pass


@router.get("/workflows", response_model=list[WorkflowResponse])
async def list_workflows(status: Optional[str] = Query(None)):
    """List workflows with optional status filter."""
    return []


@router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: str):
    """Get workflow details including state."""
    pass


@router.post("/workflows/{workflow_id}/pause", response_model=WorkflowResponse)
async def pause_workflow(workflow_id: str):
    """Pause workflow."""
    pass


@router.post("/workflows/{workflow_id}/resume", response_model=WorkflowResponse)
async def resume_workflow(workflow_id: str):
    """Resume workflow."""
    pass


@router.post("/workflows/{workflow_id}/cancel", response_model=WorkflowResponse)
async def cancel_workflow(workflow_id: str):
    """Cancel workflow."""
    pass


@router.get("/workflows/{workflow_id}/history", response_model=list[WorkflowHistoryEntry])
async def get_workflow_history(workflow_id: str):
    """Get execution history (state snapshots)."""
    return []
