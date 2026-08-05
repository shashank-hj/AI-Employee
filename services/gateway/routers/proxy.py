from fastapi import APIRouter, Request, HTTPException
import httpx
from gateway.config import settings

router = APIRouter()

SERVICE_MAP = {
    "orchestrator": settings.ORCHESTRATOR_URL,
    "tools": settings.TOOL_REGISTRY_URL,
    "memory": settings.MEMORY_URL,
    "rag": settings.RAG_URL,
    "workflow": settings.WORKFLOW_URL,
    "speech": settings.SPEECH_URL,
}


@router.api_route("/api/{service}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_request(service: str, path: str, request: Request):
    if service not in SERVICE_MAP:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service}")
    target_url = f"{SERVICE_MAP[service]}/api/{path}"
    async with httpx.AsyncClient() as client:
        resp = await client.request(
            method=request.method, url=target_url,
            headers={k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")},
            content=await request.body(), timeout=30.0,
        )
    from fastapi.responses import Response
    return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))
