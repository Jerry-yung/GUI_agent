"""Extract VLM token usage from LangChain responses."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class VlmCallStats:
    input_tokens: int = 0
    output_tokens: int = 0
    vlm_elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def add(self, other: VlmCallStats) -> None:
        self.input_tokens += int(other.input_tokens)
        self.output_tokens += int(other.output_tokens)
        self.vlm_elapsed_ms = round(self.vlm_elapsed_ms + float(other.vlm_elapsed_ms), 2)


def extract_vlm_token_usage(resp: Any) -> tuple[int, int]:
    um = getattr(resp, "usage_metadata", None)
    if um is not None:
        if isinstance(um, dict):
            input_t = um.get("input_tokens", um.get("prompt_tokens", 0))
            output_t = um.get("output_tokens", um.get("completion_tokens", 0))
        else:
            input_t = getattr(um, "input_tokens", None) or getattr(um, "prompt_tokens", 0)
            output_t = getattr(um, "output_tokens", None) or getattr(um, "completion_tokens", 0)
        return int(input_t or 0), int(output_t or 0)

    meta = getattr(resp, "response_metadata", None) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    if usage:
        input_t = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        output_t = usage.get("completion_tokens", usage.get("output_tokens", 0))
        return int(input_t or 0), int(output_t or 0)
    return 0, 0
