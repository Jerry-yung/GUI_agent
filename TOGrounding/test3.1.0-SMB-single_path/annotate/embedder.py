"""qwen3-vl-embedding with text+patch multimodal fusion."""
from __future__ import annotations

import base64
import hashlib
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from annotate.node_serializer import node_embedding_text
from annotate.patch_utils import crop_node_patch

EMBEDDING_DIM = 2560
DEFAULT_EMBED_WORKERS = max(1, int(os.getenv("SMB_EMBED_WORKERS", "16")))


def has_text_semantics(node: dict) -> bool:
    return bool(node_embedding_text(node))


def count_dual_empty_nodes(nodes: list[dict]) -> int:
    return sum(1 for n in nodes if not has_text_semantics(n))


def image_to_base64(pil_img: Image.Image, fmt: str = "png") -> str:
    buffer = io.BytesIO()
    pil_img.save(buffer, format=fmt.upper())
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt};base64,{b64}"


def _vec_path(step_dir: Path, index: int) -> Path:
    return step_dir / f"{index + 1:03d}.npy"


def _load_vec(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    vec = np.load(path).astype(np.float32)
    if vec.shape != (EMBEDDING_DIM,):
        path.unlink(missing_ok=True)
        return None
    return vec


def _save_vec(path: Path, vec: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vec.astype(np.float32))


def _embed_multimodal_to_path(
    text: str, patch: Image.Image, path: Path, *, retry: int = 3
) -> tuple[np.ndarray, bool]:
    cached = _load_vec(path)
    if cached is not None:
        return cached, True
    from llm_set.llm import vlm_embedding

    b64 = image_to_base64(patch)
    last_err: Exception | None = None
    for attempt in range(retry):
        try:
            vec = np.asarray(vlm_embedding.embed_multimodal(text, b64), dtype=np.float32)
            _save_vec(path, vec)
            return vec, False
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"vlm_embedding multimodal failed: {last_err}") from last_err


def node_multimodal_text(node: dict) -> str:
    text = node_embedding_text(node)
    if text:
        return text
    for key in ("label", "ager_label", "text"):
        value = (node.get(key) or "").strip()
        if value:
            return value
    sman_type = node.get("sman_type") or node.get("kind") or "node"
    area = node.get("sman_area_idx") or node.get("id") or ""
    return f"{sman_type} {area}".strip()


def node_multimodal_patch(
    node: dict,
    screenshot: Image.Image | None,
) -> Image.Image | None:
    if screenshot is None:
        return None
    return crop_node_patch(screenshot, node.get("bounds") or [])


@dataclass
class EmbeddingBuildStats:
    total: int = 0
    dual_empty: int = 0
    multimodal_path: int = 0
    patch_only_path: int = 0
    text_only_path: int = 0
    skipped: int = 0
    cache_hits: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)

    def bump_skip(self, reason: str) -> None:
        self.skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1


@dataclass
class _NodeEmbedResult:
    index: int
    vector: np.ndarray
    cache_hit: bool = False
    multimodal: bool = False
    skip_reason: str | None = None


def _embed_node_multimodal(
    index: int,
    node: dict,
    screenshot: Image.Image | None,
    step_dir: Path,
) -> _NodeEmbedResult:
    path = _vec_path(step_dir, index)
    cached = _load_vec(path)
    if cached is not None:
        return _NodeEmbedResult(index=index, vector=cached, cache_hit=True)

    patch = node_multimodal_patch(node, screenshot)
    if patch is None:
        return _NodeEmbedResult(
            index=index,
            vector=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            skip_reason="missing_patch_or_screenshot",
        )

    text = node_multimodal_text(node)
    try:
        vec, _ = _embed_multimodal_to_path(text, patch, path)
    except RuntimeError:
        return _NodeEmbedResult(
            index=index,
            vector=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            skip_reason="multimodal_api_failed",
        )
    return _NodeEmbedResult(index=index, vector=vec, multimodal=True)


def _apply_node_result(stats: EmbeddingBuildStats, result: _NodeEmbedResult) -> None:
    if result.cache_hit:
        stats.cache_hits += 1
        return
    if result.multimodal:
        stats.multimodal_path += 1
        return
    if result.skip_reason:
        stats.bump_skip(result.skip_reason)


def build_node_embeddings(
    nodes: list[dict],
    step_dir: Path,
    screenshot: Image.Image | None = None,
    *,
    max_workers: int | None = None,
) -> np.ndarray:
    stats = EmbeddingBuildStats(total=len(nodes), dual_empty=count_dual_empty_nodes(nodes))
    out = np.zeros((len(nodes), EMBEDDING_DIM), dtype=np.float32)
    if not nodes:
        return out

    workers = DEFAULT_EMBED_WORKERS if max_workers is None else max(1, int(max_workers))
    pending: list[tuple[int, dict]] = []

    for idx, node in enumerate(nodes):
        path = _vec_path(step_dir, idx)
        cached = _load_vec(path)
        if cached is not None:
            out[idx] = cached
            stats.cache_hits += 1
        else:
            pending.append((idx, node))

    if not pending:
        return out

    if workers <= 1 or len(pending) == 1:
        for idx, node in pending:
            result = _embed_node_multimodal(idx, node, screenshot, step_dir)
            out[idx] = result.vector
            _apply_node_result(stats, result)
        return out

    pool_workers = min(workers, len(pending))
    with ThreadPoolExecutor(max_workers=pool_workers) as executor:
        futures = [
            executor.submit(_embed_node_multimodal, idx, node, screenshot, step_dir)
            for idx, node in pending
        ]
        for future in as_completed(futures):
            result = future.result()
            out[result.index] = result.vector
            _apply_node_result(stats, result)

    return out


def embed_instruction(
    query_text: str,
    step_dir: Path,
    *,
    retry: int = 3,
    reuse_cache: bool = True,
) -> np.ndarray:
    if not query_text.strip():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    digest = hashlib.md5(query_text.encode("utf-8")).hexdigest()[:16]
    path = step_dir / f"_instruction_{digest}.npy"
    if reuse_cache:
        cached = _load_vec(path)
        if cached is not None:
            return cached
    from llm_set.llm import vlm_embedding

    last_err: Exception | None = None
    for attempt in range(retry):
        try:
            vec = np.asarray(vlm_embedding.embed_text(query_text), dtype=np.float32)
            if reuse_cache:
                _save_vec(path, vec)
            return vec
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"vlm_embedding query failed: {last_err}") from last_err
