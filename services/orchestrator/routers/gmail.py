"""Email API routes (SMTP + IMAP)."""

import asyncio

from fastapi import APIRouter, HTTPException, Query

from orchestrator.config import settings
from orchestrator.services.gmail_client import EmailClient
from shared.utils.response import success_response

router = APIRouter(prefix="/api/email", tags=["Email"])


def _get_client() -> EmailClient:
    client = EmailClient()
    if not client.enabled:
        raise HTTPException(
            status_code=503,
            detail="Email not configured. Set EMAIL_ADDRESS and EMAIL_PASSWORD in .env",
        )
    return client


@router.get("/health", summary="Check Email integration status")
async def email_health():
    client = EmailClient()
    return {"enabled": client.enabled, "provider": "smtp+imap", "host": settings.EMAIL_SMTP_HOST}


@router.get("/profile", summary="Get email account info")
async def email_profile():
    cli = _get_client()
    try:
        profile = await asyncio.to_thread(cli.get_profile)
        return success_response(data=profile)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Email error: {str(exc)}")


@router.get("/messages", summary="List recent emails")
async def list_messages(
    max_results: int = Query(default=10, ge=1, le=50),
    query: str = Query(default=""),
):
    cli = _get_client()
    try:
        messages = await asyncio.to_thread(cli.list_messages, max_results, query)
        return success_response(data=messages)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Email error: {str(exc)}")


@router.get("/messages/{message_id}", summary="Read a specific email")
async def get_message(message_id: str):
    cli = _get_client()
    try:
        message = await asyncio.to_thread(cli.get_message, message_id)
        return success_response(data=message)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Email error: {str(exc)}")


@router.post("/send", summary="Send an email via SMTP")
async def send_email(
    to: str = Query(...),
    subject: str = Query(...),
    body: str = Query(...),
    cc: str | None = Query(default=None),
    html: bool = Query(default=False),
):
    cli = _get_client()
    try:
        if html:
            result = await asyncio.to_thread(cli.send_html_message, to, subject, body, cc)
        else:
            result = await asyncio.to_thread(cli.send_message, to, subject, body, cc)
        return success_response(data=result)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Email send error: {str(exc)}")
