from dataclasses import dataclass, field


@dataclass
class UsageRecord:
    """A single billable call against an external API provider.

    ``category`` is one of: llm | speech | embedding.
    ``unit`` is the billable quantity: tokens | characters | audio_seconds.
    ``cost_inr`` is filled in by the recorder using the pricing table; callers
    may pre-set it to override pricing.
    """

    service: str = ""
    category: str = "llm"
    operation: str = "generate"
    model: str = ""
    unit: str = "tokens"
    input_units: int = 0
    output_units: int = 0
    total_units: int | None = None
    cost_inr: float | None = None
    request_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    status: str = "success"
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)
