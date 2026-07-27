from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader
from shared.auth.models import User

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(request: Request) -> User:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_role(*roles: str):
    async def role_checker(request: Request, user: User = Depends(get_current_user)) -> User:
        if not any(role in user.roles for role in roles):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user

    return role_checker


async def get_api_key(api_key: str = Depends(api_key_header)) -> str:
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    return api_key
