from datetime import datetime
from typing import Any, Protocol


class OrderService(Protocol):
    async def lookup_order(self, order_id: str) -> dict[str, Any]:
        """Look up an order by ID. Returns order details or an error dict."""


class CalendarServiceProtocol(Protocol):
    """Production calendar service surface used by tools and routes."""

    provider_name: str
    enabled: bool

    async def health(self) -> dict[str, Any]: ...

    async def check_availability(
        self,
        *,
        start_at: datetime,
        end_at: datetime,
        timezone: str | None = None,
        duration_minutes: int = 30,
    ) -> dict[str, Any]: ...

    async def propose_booking(
        self,
        *,
        session_id: str,
        user_id: str | None,
        draft: Any,
        duration_minutes: int = 30,
    ) -> dict[str, Any]: ...

    async def confirm_booking(
        self,
        *,
        session_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any]: ...

    async def decline_booking(self, *, session_id: str) -> bool: ...

    async def create_meeting(
        self,
        draft: Any,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]: ...

    async def get_meeting(self, meeting_id: str) -> dict[str, Any]: ...

    async def list_meetings(
        self,
        *,
        session_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        status: str | None = "scheduled",
        limit: int = 50,
    ) -> dict[str, Any]: ...

    async def update_meeting(self, meeting_id: str, draft: Any) -> dict[str, Any]: ...

    async def cancel_meeting(self, meeting_id: str) -> dict[str, Any]: ...


class PricingService(Protocol):
    async def search_pricing(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search pricing plans and tiers. Returns list of matching plan dicts."""
