from fastapi import APIRouter, Depends, Query
from typing import Optional

from workflow.container import get_workflow_service
from workflow.schemas.workflows import WorkflowCreate, WorkflowResponse, WorkflowHistoryEntry
from workflow.services.workflow_service import WorkflowService
from shared.utils.exceptions import AppException

router = APIRouter(prefix="/api")


@router.post("/workflows", response_model=WorkflowResponse, status_code=201)
async def create_workflow(
    workflow: WorkflowCreate,
    service: WorkflowService = Depends(get_workflow_service),
):
    return await service.create_workflow(workflow)


@router.get("/workflows", response_model=list[WorkflowResponse])
async def list_workflows(
    status: Optional[str] = Query(None),
    service: WorkflowService = Depends(get_workflow_service),
):
    workflows, _ = await service.list_workflows(status=status)
    return workflows


@router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    service: WorkflowService = Depends(get_workflow_service),
):
    return await service.get_workflow(workflow_id)


@router.post("/workflows/{workflow_id}/pause", response_model=WorkflowResponse)
async def pause_workflow(
    workflow_id: str,
    service: WorkflowService = Depends(get_workflow_service),
):
    try:
        return await service.pause_workflow(workflow_id)
    except ValueError as e:
        raise AppException(detail=str(e), status_code=400, error_code="INVALID_STATUS")


@router.post("/workflows/{workflow_id}/resume", response_model=WorkflowResponse)
async def resume_workflow(
    workflow_id: str,
    service: WorkflowService = Depends(get_workflow_service),
):
    try:
        return await service.resume_workflow(workflow_id)
    except ValueError as e:
        raise AppException(detail=str(e), status_code=400, error_code="INVALID_STATUS")


@router.post("/workflows/{workflow_id}/cancel", response_model=WorkflowResponse)
async def cancel_workflow(
    workflow_id: str,
    service: WorkflowService = Depends(get_workflow_service),
):
    try:
        return await service.cancel_workflow(workflow_id)
    except ValueError as e:
        raise AppException(detail=str(e), status_code=400, error_code="INVALID_STATUS")


@router.get("/workflows/{workflow_id}/history", response_model=list[WorkflowHistoryEntry])
async def get_workflow_history(
    workflow_id: str,
    service: WorkflowService = Depends(get_workflow_service),
):
    return await service.get_history(workflow_id)
