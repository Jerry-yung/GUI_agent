"""从 LangChain VLM 响应中提取 input/output token 用量。"""

from __future__ import annotations


def _coerce_token_pair(data: dict) -> dict[str, int] | None:
    if not isinstance(data, dict):
        return None
    inp = data.get("input_tokens")
    if inp is None:
        inp = data.get("prompt_tokens")
    out = data.get("output_tokens")
    if out is None:
        out = data.get("completion_tokens")
    if inp is None or out is None:
        return None
    try:
        return {"input": int(inp), "output": int(out)}
    except (TypeError, ValueError):
        return None


def extract_vlm_token_usage(response) -> dict[str, int] | None:
    """
    解析 VLM AIMessage 的 token 用量。

    Returns:
        {"input": int, "output": int} 或 None（API 未返回时）
    """
    if response is None:
        return None

    usage = getattr(response, "usage_metadata", None)
    tokens = _coerce_token_pair(usage)
    if tokens is not None:
        return tokens

    meta = getattr(response, "response_metadata", None)
    if isinstance(meta, dict):
        for key in ("token_usage", "usage"):
            tokens = _coerce_token_pair(meta.get(key))
            if tokens is not None:
                return tokens
        tokens = _coerce_token_pair(meta)
        if tokens is not None:
            return tokens

    return None


def invoke_vlm(model, messages) -> tuple[str, dict[str, int] | None]:
    """调用 VLM 并返回 (raw_text, vlm_tokens)。"""
    response = model.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)
    return raw, extract_vlm_token_usage(response)
