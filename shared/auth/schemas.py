from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional


class TokenPayload(BaseModel):
    sub: str
    exp: int
    iat: int
    roles: list[str] = []


class AuthConfig(BaseSettings):
    jwt_secret: str = Field(default="change-me-in-production")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiry_minutes: int = Field(default=60)

    model_config = {"env_prefix": "AUTH_"}
