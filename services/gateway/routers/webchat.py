"""Web chat channel (CH5): points the user to the integrated dashboard chat.

The chat experience now lives inside the orchestrator dashboard (the Chat page),
so the gateway simply redirects root / chat URLs to the dashboard. This keeps a
single, unified UI instead of a separate chat page.
"""

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from gateway.config import settings

router = APIRouter(tags=["Web Chat"])

_DASHBOARD_URL = settings.DASHBOARD_PUBLIC_URL.rstrip("/")


@router.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url=_DASHBOARD_URL)


@router.get("/chat", include_in_schema=False, summary="Redirect to the integrated dashboard chat")
async def web_chat() -> RedirectResponse:
    return RedirectResponse(url=_DASHBOARD_URL)
