from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]: ...

    @abstractmethod
    async def invoke(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool and return result dict with 'success' and 'data' keys."""
        ...
