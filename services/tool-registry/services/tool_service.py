from uuid import UUID

from tool_registry.models.tool import ToolModel
from tool_registry.repositories.tool_repository import ToolRepository
from tool_registry.schemas.tools import (
    RetryPolicy,
    ToolCreate,
    ToolListParams,
    ToolResponse,
    ToolUpdate,
)
from shared.utils.exceptions import ConflictException, NotFoundException


class ToolService:
    def __init__(self, repository: ToolRepository) -> None:
        self._repo = repository

    async def register_tool(self, data: ToolCreate) -> ToolResponse:
        existing = await self._repo.get_by_name(data.name)
        if existing:
            raise ConflictException(f"Tool with name '{data.name}' already exists")

        tool = ToolModel(
            name=data.name,
            description=data.description,
            version=data.version,
            category=data.category.value,
            permissions=data.permissions,
            input_schema=data.input_schema,
            output_schema=data.output_schema,
            timeout_seconds=data.timeout_seconds,
            retry_policy=data.retry_policy.model_dump(),
            tags=data.tags,
            is_active=True,
        )
        created = await self._repo.create(tool)
        return self._model_to_response(created)

    async def get_tool(self, tool_id: str) -> ToolResponse:
        tool = await self._get_or_raise(tool_id)
        return self._model_to_response(tool)

    async def list_tools(self, params: ToolListParams) -> tuple[list[ToolResponse], int]:
        tag_list = None
        if params.tags:
            tag_list = [t.strip() for t in params.tags.split(",") if t.strip()]

        tools, total = await self._repo.list(
            category=params.category.value if params.category else None,
            is_active=params.is_active,
            tags=tag_list,
            search=params.search,
            page=params.page,
            page_size=params.page_size,
        )
        return [self._model_to_response(t) for t in tools], total

    async def update_tool(self, tool_id: str, data: ToolUpdate) -> ToolResponse:
        tool = await self._get_or_raise(tool_id)

        update_dict = data.model_dump(exclude_unset=True)

        if "name" in update_dict and update_dict["name"].lower() != tool.name.lower():
            existing = await self._repo.get_by_name(update_dict["name"])
            if existing and str(existing.id) != str(tool.id):
                raise ConflictException(f"Tool with name '{update_dict['name']}' already exists")

        if "retry_policy" in update_dict and isinstance(update_dict["retry_policy"], RetryPolicy):
            update_dict["retry_policy"] = update_dict["retry_policy"].model_dump()

        updated = await self._repo.update(tool, update_dict)
        return self._model_to_response(updated)

    async def delete_tool(self, tool_id: str) -> None:
        tool = await self._get_or_raise(tool_id)
        await self._repo.delete(tool)

    async def _get_or_raise(self, tool_id: str) -> ToolModel:
        tool = await self._repo.get_by_id(tool_id)
        if tool is None:
            raise NotFoundException(f"Tool with id '{tool_id}' not found")
        return tool

    @staticmethod
    def _model_to_response(tool: ToolModel) -> ToolResponse:
        retry = tool.retry_policy or {}
        return ToolResponse(
            id=str(tool.id),
            name=tool.name,
            description=tool.description,
            version=tool.version,
            category=tool.category,
            permissions=tool.permissions or [],
            input_schema=tool.input_schema or {},
            output_schema=tool.output_schema or {},
            timeout_seconds=tool.timeout_seconds,
            retry_policy=RetryPolicy(
                max_retries=retry.get("max_retries", 3),
                delay_seconds=retry.get("delay_seconds", 1.0),
                backoff_multiplier=retry.get("backoff_multiplier", 2.0),
            ),
            tags=tool.tags or [],
            is_active=tool.is_active,
            created_at=tool.created_at,
            updated_at=tool.updated_at,
        )
