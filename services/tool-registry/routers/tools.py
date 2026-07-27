from fastapi import APIRouter, Depends

from tool_registry.container import get_tool_service
from tool_registry.schemas.tools import ToolCreate, ToolListParams, ToolResponse, ToolUpdate
from tool_registry.services.tool_service import ToolService
from shared.utils.response import paginated_response, success_response

router = APIRouter(prefix="/api")


@router.post("/tools", response_model=ToolResponse, status_code=201)
async def register_tool(
    data: ToolCreate,
    service: ToolService = Depends(get_tool_service),
):
    return await service.register_tool(data)


@router.get("/tools")
async def list_tools(
    params: ToolListParams = Depends(),
    service: ToolService = Depends(get_tool_service),
):
    tools, total = await service.list_tools(params)
    return paginated_response(
        items=[t.model_dump(mode="json") for t in tools],
        total=total,
        page=params.page,
        page_size=params.page_size,
    )


@router.get("/tools/{tool_id}", response_model=ToolResponse)
async def get_tool(
    tool_id: str,
    service: ToolService = Depends(get_tool_service),
):
    return await service.get_tool(tool_id)


@router.put("/tools/{tool_id}", response_model=ToolResponse)
async def update_tool(
    tool_id: str,
    data: ToolUpdate,
    service: ToolService = Depends(get_tool_service),
):
    return await service.update_tool(tool_id, data)


@router.delete("/tools/{tool_id}", status_code=204)
async def delete_tool(
    tool_id: str,
    service: ToolService = Depends(get_tool_service),
):
    await service.delete_tool(tool_id)
