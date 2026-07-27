from fastapi import APIRouter, Depends

from orchestrator.container import get_agent_service
from orchestrator.schemas.agent import AgentRequest, AgentResponse
from orchestrator.services.agent_service import AgentService

router = APIRouter(prefix="/api")


@router.post("/agent/run", response_model=AgentResponse)
async def run_agent(
    request: AgentRequest,
    service: AgentService = Depends(get_agent_service),
):
    return await service.run(request)
