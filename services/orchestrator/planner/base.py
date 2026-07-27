from abc import ABC, abstractmethod

from orchestrator.graph.state import PlanStep, AgentState


class BasePlanner(ABC):
    @abstractmethod
    async def create_plan(self, state: AgentState) -> list[PlanStep]:
        """Analyze the user input and context, return an ordered list of steps."""
        ...
