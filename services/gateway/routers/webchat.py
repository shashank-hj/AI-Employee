"""Web chat channel (CH5): serves the static chat page from the gateway."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(tags=["Web Chat"])

_CHAT_PAGE = Path(__file__).resolve().parent.parent / "static" / "chat.html"


def _read_chat_page() -> str:
    if _CHAT_PAGE.exists():
        return _CHAT_PAGE.read_text(encoding="utf-8")
    return "<!DOCTYPE html><html><body><h1>Chat unavailable</h1></body></html>"


@router.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/chat")


@router.get("/chat", response_class=HTMLResponse, summary="Serve the web chat widget")
async def web_chat() -> HTMLResponse:
    return HTMLResponse(content=_read_chat_page(), media_type="text/html")
