
import json

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import StreamingResponse

from shared.utils.exceptions import AppException
from workflow.container import get_workflow_service
from workflow.schemas.workflows import (
    WorkflowCreate,
    WorkflowHistoryEntry,
    WorkflowResponse,
    WorkflowResumeRequest,
    WorkflowRunRequest,
    WorkflowRunResponse,
)
from workflow.services.workflow_service import WorkflowService

router = APIRouter(prefix="/api")


@router.post("/workflows", response_model=WorkflowResponse, status_code=201)
async def create_workflow(
    workflow: WorkflowCreate,
    service: WorkflowService = Depends(get_workflow_service),
):
    return await service.create_workflow(workflow)


@router.get("/workflows", response_model=list[WorkflowResponse])
async def list_workflows(
    status: str | None = Query(None),
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


@router.post("/workflows/{workflow_id}/run")
async def run_workflow(
    workflow_id: str,
    request: WorkflowRunRequest,
    service: WorkflowService = Depends(get_workflow_service),
):
    """Execute the workflow graph. Set ``stream: true`` to receive SSE updates."""
    if request.stream:
        async def event_source():
            async for event in service.stream(
                workflow_id, request.input_data, request.timeout_seconds
            ):
                yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        return await service.run(workflow_id, request.input_data, request.timeout_seconds)
    except ValueError as e:
        raise AppException(detail=str(e), status_code=409, error_code="INVALID_STATE")
    except Exception as e:
        raise AppException(detail=str(e), status_code=500, error_code="WORKFLOW_RUN_ERROR")


@router.post("/workflows/{workflow_id}/continue", response_model=WorkflowRunResponse)
async def continue_workflow(
    workflow_id: str,
    request: WorkflowResumeRequest,
    service: WorkflowService = Depends(get_workflow_service),
):
    """Resume a workflow paused at a human-approval interrupt."""
    try:
        return await service.resume(workflow_id, request.payload)
    except ValueError as e:
        raise AppException(detail=str(e), status_code=409, error_code="INVALID_STATE")
    except Exception as e:
        raise AppException(detail=str(e), status_code=500, error_code="WORKFLOW_RESUME_ERROR")


@router.delete("/workflows/{workflow_id}", status_code=204)
async def delete_workflow(
    workflow_id: str,
    service: WorkflowService = Depends(get_workflow_service),
):
    await service.delete_workflow(workflow_id)
    return Response(status_code=204)
