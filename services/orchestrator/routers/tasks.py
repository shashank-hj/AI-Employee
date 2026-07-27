from fastapi import APIRouter
from orchestrator.schemas.tasks import TaskResponse, TaskStatus

router = APIRouter()


@router.get("/api/tasks", response_model=TaskResponse)
async def list_tasks():
    return TaskResponse(status=TaskStatus.PENDING, message="Orchestrator tasks endpoint", tasks=[])


@router.post("/api/tasks", response_model=TaskResponse)
async def create_task():
    return TaskResponse(status=TaskStatus.PENDING, message="Orchestrator tasks endpoint", tasks=[])
