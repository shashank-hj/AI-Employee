
from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings


class TokenPayload(BaseModel):
    sub: str
    exp: int
    iat: int
    roles: list[str] = []


class AuthConfig(BaseSettings):
    jwt_secret: str = Field(
        default="change-me-in-production",
        # Fall back to the platform-wide SECRET_KEY so the JWT signing key is
        # not silently left at the public default when operators only set
        # SECRET_KEY (as docker-compose does). AUTH_JWT_SECRET wins when both
        # are present.
        validation_alias=AliasChoices("AUTH_JWT_SECRET", "SECRET_KEY"),
    )
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiry_minutes: int = Field(default=60)

    model_config = {"env_prefix": "AUTH_"}
