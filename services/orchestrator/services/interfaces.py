from typing import Any, Protocol


class OrderService(Protocol):
    async def lookup_order(self, order_id: str) -> dict[str, Any]:
        """Look up an order by ID. Returns order details or an error dict."""


class CalendarService(Protocol):
    async def get_availability(self, query: str, days_ahead: int = 5) -> list[dict[str, Any]]:
        """Return available time slots for booking."""

    async def schedule_demo(self, title: str, attendees: list[str], date: str, time: str, duration_minutes: int = 30) -> dict[str, Any]:
        """Schedule a demo or meeting. Returns confirmation dict."""


class PricingService(Protocol):
    async def search_pricing(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search pricing plans and tiers. Returns list of matching plan dicts."""
