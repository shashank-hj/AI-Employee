import jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from shared.auth.schemas import AuthConfig
from shared.auth.models import User


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, config: AuthConfig | None = None):
        super().__init__(app)
        self.config = config or AuthConfig()

    async def dispatch(self, request: Request, call_next):
        user = None
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[len("Bearer "):]
            try:
                payload = jwt.decode(
                    token,
                    self.config.jwt_secret,
                    algorithms=[self.config.jwt_algorithm],
                )
                user = User(
                    id=payload.get("sub", ""),
                    email=payload.get("email", ""),
                    roles=payload.get("roles", []),
                    tenant_id=payload.get("tenant_id"),
                )
            except jwt.PyJWTError:
                pass

        request.state.user = user
        response = await call_next(request)
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, valid_keys: list[str] | None = None):
        super().__init__(app)
        self.valid_keys = valid_keys or []

    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("X-API-Key")
        if api_key and api_key in self.valid_keys:
            request.state.service_identity = {"service_name": "api_key_client"}
        else:
            request.state.service_identity = None

        response = await call_next(request)
        return response


def register_middleware(app, auth_config: AuthConfig | None = None, valid_api_keys: list[str] | None = None):
    app.add_middleware(AuthMiddleware, config=auth_config or AuthConfig())
    app.add_middleware(APIKeyMiddleware, valid_keys=valid_api_keys or [])


register_auth_middleware = register_middleware
