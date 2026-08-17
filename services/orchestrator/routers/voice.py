
import structlog
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from orchestrator.container import get_speech_client, get_voice_service
from orchestrator.schemas.voice import (
    VoiceStatusResponse,
    VoiceTextTurnRequest,
    VoiceTextTurnResponse,
    VoiceTurnRequest,
    VoiceTurnResponse,
)
from orchestrator.services.speech_client import SpeechClient
from orchestrator.services.voice_service import VoiceService

router = APIRouter(prefix="/api/voice", tags=["Voice"])
logger = structlog.get_logger(__name__)


@router.get("/models")
async def voice_models(speech: SpeechClient = Depends(get_speech_client)) -> dict:
    """List selectable Sarvam STT/TTS models, speakers, and languages."""
    return await speech.list_models()


@router.post("/turn", response_model=VoiceTurnResponse)
async def voice_turn(
    request: VoiceTurnRequest,
    service: VoiceService = Depends(get_voice_service),
):
    """Full voice turn: STT -> agent -> TTS. Audio in, audio + text out."""
    return await service.process_audio_turn(request)


@router.post("/turn/text", response_model=VoiceTextTurnResponse)
async def voice_text_turn(
    request: VoiceTextTurnRequest,
    service: VoiceService = Depends(get_voice_service),
):
    """Text-in voice turn: detect language -> agent -> TTS (audio out)."""
    return await service.process_text_turn(request)


@router.get("/status", response_model=VoiceStatusResponse)
async def voice_status(service: VoiceService = Depends(get_voice_service)):
    return await service.status()


@router.websocket("/ws")
async def voice_turn_websocket(websocket: WebSocket):
    """Turn-based voice WebSocket.

    Client sends one JSON message per turn:
    ``{"audio_base64": str, "language_code": str?, "user_id": str?, "session_id": str?}``
    The server replies with the full :class:`VoiceTurnResponse` JSON.
    """
    await websocket.accept()
    service: VoiceService = get_voice_service()

    try:
        while True:
            data = await websocket.receive_json()
            try:
                request = VoiceTurnRequest.model_validate(data)
            except Exception as exc:
                await websocket.send_json({"error": f"Invalid voice turn payload: {exc}"})
                continue

            response = await service.process_audio_turn(request)
            await websocket.send_json(response.model_dump())
    except WebSocketDisconnect:
        logger.info("voice_ws_disconnected")
    except Exception as exc:
        logger.error("voice_ws_error", error=str(exc))
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass
        await websocket.close()
