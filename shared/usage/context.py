"""Context variables used to attribute usage records to a request/session/user.

The LLM provider hook fires inside ``_chat_completion`` where the request id is
not in scope. Services set this context before invoking the graph / worker so the
recorder can fill in attribution fields without threading ids through every layer.
"""

import contextvars
from contextvars import Token

USAGE_CONTEXT: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "usage_context", default={}
)


def set_usage_context(**kwargs) -> Token:
    """Set one or more attribution fields (request_id, session_id, user_id, operation)."""
    return USAGE_CONTEXT.set({**USAGE_CONTEXT.get(), **{k: v for k, v in kwargs.items() if v is not None}})


def reset_usage_context(token: Token) -> None:
    USAGE_CONTEXT.reset(token)


def get_usage_context() -> dict:
    return USAGE_CONTEXT.get()
