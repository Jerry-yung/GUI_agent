"""VLM invocation helper."""
from __future__ import annotations

import base64
import time

from langchain_core.messages import HumanMessage

from llm_set.llm import vlm
from utils.vlm_stats import VlmCallStats, extract_vlm_token_usage


def call_vlm(prompt: str, image_path: str) -> tuple[bool, str, VlmCallStats]:
    stats = VlmCallStats()
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        )
        t0 = time.perf_counter()
        resp = vlm.model.invoke([msg])
        stats.vlm_elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        in_t, out_t = extract_vlm_token_usage(resp)
        stats.input_tokens = in_t
        stats.output_tokens = out_t

        content = resp.content
        if isinstance(content, list):
            text_parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            content = "".join(text_parts)
        return True, str(content), stats
    except Exception as exc:  # noqa: BLE001
        return False, str(exc), stats
