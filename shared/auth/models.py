from pydantic import BaseModel, EmailStr
from typing import Optional


class User(BaseModel):
    id: str
    email: EmailStr
    roles: list[str] = []
    tenant_id: Optional[str] = None


class ServiceIdentity(BaseModel):
    service_name: str
    permissions: list[str] = []
