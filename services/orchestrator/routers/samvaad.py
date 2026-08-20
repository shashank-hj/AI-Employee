"""Samvaad bridge endpoints.

Expose the hosted Samvaad agent as an opt-in channel: open sessions, push text
or audio into them, poll or stream the normalised replies, and check status.
"""

import asyncio
import base64
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect

from orchestrator.config import settings
from orchestrator.container import get_samvaad_session_manager
from orchestrator.routers.samvaad_tools import tools_gate_status
from orchestrator.schemas.samvaad import (
    SamvaadAudioRequest,
    SamvaadMessagesResponse,
    SamvaadSessionCost,
    SamvaadSessionCreateRequest,
    SamvaadSessionInfo,
    SamvaadSessionResponse,
    SamvaadStatusResponse,
    SamvaadTextRequest,
    SamvaadUsageResponse,
)
from orchestrator.services.samvaad_client import (
    SamvaadError,
    SamvaadSession,
    SamvaadSessionManager,
)
from orchestrator.services.samvaad_usage import get_usage_client

router = APIRouter(prefix="/api/samvaad", tags=["Samvaad"])
logger = structlog.get_logger(__name__)


def _require_enabled(manager: SamvaadSessionManager) -> None:
    if not manager.enabled:
        raise HTTPException(
            status_code=503,
            detail=manager.unavailable_reason() or "Samvaad disabled",
        )


def _get_session(manager: SamvaadSessionManager, session_id: str) -> SamvaadSession:
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Samvaad session not found")
    return session


@router.get("/status", response_model=SamvaadStatusResponse)
async def samvaad_status(
    manager: SamvaadSessionManager = Depends(get_samvaad_session_manager),
) -> SamvaadStatusResponse:
    return SamvaadStatusResponse(
        status="ok" if manager.enabled else "disabled",
        enabled=manager.enabled,
        agent_id=settings.SAMVAAD_AGENT_ID,
        reason=manager.unavailable_reason(),
        active_sessions=manager.active_sessions(),
        tools=tools_gate_status(),
    )


@router.get("/usage", response_model=SamvaadUsageResponse)
async def samvaad_usage(
    days: int = Query(default=14, ge=1, le=90),
) -> SamvaadUsageResponse:
    """Estimated Samvaad spend over the last ``days`` days.

    Pulls interaction analytics from Sarvam and estimates per-session cost
    (STT + LLM at both 105B and GLM rates). Purely advisory — the authoritative
    ledger is the Sarvam billing dashboard.
    """
    result = await (await get_usage_client()).estimate_window(days=days)
    return SamvaadUsageResponse(
        available=result["available"],
        reason=result.get("reason"),
        total_105b_rs=result["total_105b_rs"],
        total_glm_rs=result["total_glm_rs"],
        stt_rs=result["stt_rs"],
        session_count=result["session_count"],
        sessions=[SamvaadSessionCost(**s) for s in result.get("sessions", [])],
        days=result["days"],
        spend_alert_rs=settings.SAMVAAD_SPEND_ALERT_RS,
        ts=result["ts"],
    )


@router.post("/sessions", response_model=SamvaadSessionResponse, status_code=201)
async def samvaad_open_session(
    request: SamvaadSessionCreateRequest,
    manager: SamvaadSessionManager = Depends(get_samvaad_session_manager),
) -> SamvaadSessionResponse:
    _require_enabled(manager)
    try:
        session = await manager.open_session(
            user_identifier=request.user_id,
            mode=request.mode,
            language=request.language,
            agent_variables=request.agent_variables,
            session_id=request.session_id,
        )
    except SamvaadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SamvaadSessionResponse(
        session_id=session.session_id,
        interaction_id=session.interaction_id,
        mode=session.mode,
        status="connected" if session.connected else "connecting",
    )


@router.get("/sessions", response_model=list[SamvaadSessionInfo])
async def samvaad_list_sessions(
    manager: SamvaadSessionManager = Depends(get_samvaad_session_manager),
) -> list[SamvaadSessionInfo]:
    _require_enabled(manager)
    return [
        SamvaadSessionInfo(
            session_id=s.session_id,
            mode=s.mode,
            connected=s.connected,
            interaction_id=s.interaction_id,
        )
        for s in manager.list_sessions()
    ]


@router.get("/sessions/{session_id}", response_model=SamvaadSessionInfo)
async def samvaad_get_session(
    session_id: str,
    manager: SamvaadSessionManager = Depends(get_samvaad_session_manager),
) -> SamvaadSessionInfo:
    _require_enabled(manager)
    session = _get_session(manager, session_id)
    return SamvaadSessionInfo(
        session_id=session.session_id,
        mode=session.mode,
        connected=session.connected,
        interaction_id=session.interaction_id,
    )


@router.post("/sessions/{session_id}/text", response_model=SamvaadSessionInfo)
async def samvaad_send_text(
    session_id: str,
    request: SamvaadTextRequest,
    manager: SamvaadSessionManager = Depends(get_samvaad_session_manager),
) -> SamvaadSessionInfo:
    _require_enabled(manager)
    session = _get_session(manager, session_id)
    if session.mode != "chat":
        raise HTTPException(status_code=400, detail="Session is in call mode")
    try:
        await session.send_text(request.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Samvaad send error: {exc}") from exc
    return SamvaadSessionInfo(
        session_id=session.session_id,
        mode=session.mode,
        connected=session.connected,
        interaction_id=session.interaction_id,
    )


@router.post("/sessions/{session_id}/audio", response_model=SamvaadSessionInfo)
async def samvaad_send_audio(
    session_id: str,
    request: SamvaadAudioRequest,
    manager: SamvaadSessionManager = Depends(get_samvaad_session_manager),
) -> SamvaadSessionInfo:
    _require_enabled(manager)
    session = _get_session(manager, session_id)
    if session.mode != "call":
        raise HTTPException(status_code=400, detail="Session is in chat mode")
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 audio") from exc
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload")
    try:
        await session.send_audio(audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Samvaad send error: {exc}") from exc
    return SamvaadSessionInfo(
        session_id=session.session_id,
        mode=session.mode,
        connected=session.connected,
        interaction_id=session.interaction_id,
    )


@router.post("/sessions/{session_id}/close", response_model=dict[str, str])
async def samvaad_close_session(
    session_id: str,
    manager: SamvaadSessionManager = Depends(get_samvaad_session_manager),
) -> dict[str, str]:
    _require_enabled(manager)
    closed = await manager.close_session(session_id)
    if not closed:
        raise HTTPException(status_code=404, detail="Samvaad session not found")
    return {"session_id": session_id, "status": "closed"}


@router.get("/sessions/{session_id}/messages", response_model=SamvaadMessagesResponse)
async def samvaad_poll_messages(
    session_id: str,
    manager: SamvaadSessionManager = Depends(get_samvaad_session_manager),
) -> SamvaadMessagesResponse:
    _require_enabled(manager)
    session = _get_session(manager, session_id)
    return SamvaadMessagesResponse(
        session_id=session.session_id,
        interaction_id=session.interaction_id,
        messages=session.drain(),
    )


@router.websocket("/ws")
async def samvaad_ws(websocket: WebSocket) -> None:
    """Full-duplex Samvaad bridge.

    Client sends one JSON message per turn::

        {"type": "init", "user_id": str, "mode": "chat"|"call",
         "language": str?, "session_id": str?, "agent_variables": {...}?}
        {"type": "text", "data": {"text": str}}
        {"type": "audio", "data": "<base64-raw-pcm16>"}
        {"type": "poll"}

    The server streams back normalised ``text`` / ``transcript`` / ``audio`` /
    ``event`` messages as they arrive, plus ``session`` on init.
    """
    await websocket.accept()
    manager = get_samvaad_session_manager()
    session: SamvaadSession | None = None
    sender_task: asyncio.Task[Any] | None = None

    async def _sender() -> None:
        assert session is not None
        while True:
            item = await session.outbox.get()
            try:
                await websocket.send_json(item)
            except Exception:
                break

    try:
        while True:
            data = await websocket.receive_json()
            mtype = data.get("type")

            if mtype == "init":
                if not manager.enabled:
                    await websocket.send_json(
                        {"type": "error", "message": manager.unavailable_reason() or "disabled"}
                    )
                    continue
                if session is not None:
                    await session.close()
                session = await manager.open_session(
                    user_identifier=data.get("user_id", "ws-user"),
                    mode=data.get("mode", "chat"),
                    language=data.get("language"),
                    agent_variables=data.get("agent_variables"),
                    session_id=data.get("session_id"),
                )
                sender_task = asyncio.create_task(_sender())
                await websocket.send_json(
                    {
                        "type": "session",
                        "session_id": session.session_id,
                        "interaction_id": session.interaction_id,
                        "mode": session.mode,
                    }
                )
            elif session is None:
                await websocket.send_json({"type": "error", "message": "Send {type:'init'} first"})
            elif mtype == "text":
                await session.send_text((data.get("data") or {}).get("text", ""))
            elif mtype == "audio":
                try:
                    audio = base64.b64decode(data.get("data") or "")
                except Exception:
                    await websocket.send_json({"type": "error", "message": "Invalid audio base64"})
                    continue
                await session.send_audio(audio)
            elif mtype == "poll":
                if reason := session.limit_reached:
                    await websocket.send_json(
                        {
                            "type": "event",
                            "event": "limit_reached",
                            "reason": reason,
                            "turns": session.turn_count,
                        }
                    )
                    await session.close()
                for item in session.drain():
                    await websocket.send_json(item)
            else:
                await websocket.send_json({"type": "error", "message": f"Unknown type: {mtype}"})
    except WebSocketDisconnect:
        logger.info("samvaad_ws_disconnected")
    except Exception as exc:
        logger.error("samvaad_ws_error", error=str(exc))
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if sender_task is not None and not sender_task.done():
            sender_task.cancel()
        if session is not None:
            await manager.close_session(session.session_id)
