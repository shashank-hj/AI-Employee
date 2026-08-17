"""Channel service (CH1): normalizes inbound messages and forwards to the orchestrator."""

from typing import Any

import httpx
import structlog

from gateway.config import settings
from shared.schemas.channels import ChannelMessage, ChannelResponse

logger = structlog.get_logger(__name__)


class ChannelService:
    """Routes a canonical :class:`ChannelMessage` to the orchestrator agent entrypoint."""

    def __init__(
        self,
        orchestrator_url: str | None = None,
        agent_run_path: str = "/api/agent/run",
        timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._orchestrator_url = (orchestrator_url or settings.ORCHESTRATOR_URL).rstrip("/")
        self._agent_run_path = agent_run_path
        self._timeout = timeout if timeout is not None else settings.CHANNEL_TIMEOUT_SECONDS
        self._transport = transport

    def _to_agent_payload(self, message: ChannelMessage) -> dict[str, Any]:
        contact = message.sender.model_dump(exclude_none=True) if message.sender else None
        return {
            "user_input": message.text,
            "user_id": message.canonical_user_id,
            "session_id": message.session_id,
            "channel": message.channel.value,
            "channel_message_id": message.message_id,
            "tenant_id": message.tenant_id,
            "contact": contact,
            "metadata": message.metadata,
        }

    async def process(self, message: ChannelMessage) -> ChannelResponse:
        payload = self._to_agent_payload(message)
        logger.info(
            "channel_message_forwarding",
            channel=message.channel.value,
            message_id=message.message_id,
        )
        try:
            client = httpx.AsyncClient(timeout=self._timeout, transport=self._transport)
            async with client:
                response = await client.post(
                    f"{self._orchestrator_url}{self._agent_run_path}",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "channel_message_orchestrator_error",
                channel=message.channel.value,
                status_code=exc.response.status_code,
                detail=exc.response.text[:500],
            )
            raise
        except httpx.HTTPError as exc:
            logger.error(
                "channel_message_orchestrator_unreachable",
                channel=message.channel.value,
                error=str(exc),
            )
            raise

        logger.info(
            "channel_message_forwarded",
            channel=message.channel.value,
            request_id=data.get("request_id"),
        )
        return ChannelResponse(
            message_id=data.get("channel_message_id") or message.message_id,
            channel=message.channel,
            reply_to=message.message_id,
            final_response=data.get("final_response", ""),
            request_id=data.get("request_id"),
            duration_ms=data.get("duration_ms"),
            metadata=data.get("metadata"),
        )
