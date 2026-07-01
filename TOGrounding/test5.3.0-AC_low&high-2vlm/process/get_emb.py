#!/usr/bin/env python3
"""
get_emb.py

读取 steps/{episode_id}/{step_idx}/{stem}_nodes.json，
按 bounds 从 {stem}_screenshot.png crop patch，
为每个节点生成 embedding：
  优先 multimodal(text+patch) → patch only → text only → zero fallback

保存到 AC_data/embeddings/nodes_emb/{episode_id}/{step_idx}/{stem}_{node_id}.npy
若已存在（含迁移前旧路径）则跳过。

TEST_START / TEST_END 表示 episode 切片下标（非 step 数）。
"""

from __future__ import annotations

import base64
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm_set.llm import vlm_embedding
from process.paths import (
    iter_episode_ids,
    node_emb_legacy_path,
    node_emb_path,
    step_paths,
    stems_in_episode,
)

EMBED_DIR = PROJECT_ROOT / "AC_data" / "embeddings"

TEST_START = 50
TEST_END = 100  # None 表示到最后一个 episode

EMBEDDING_DIM = 2560


def _crop_patch(screenshot: Image.Image | None, bounds: list[int]) -> Image.Image | None:
    if screenshot is None or len(bounds) != 4:
        return None
    left, top, right, bottom = bounds
    width, height = screenshot.size
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    if right - left < 1 or bottom - top < 1:
        return None
    patch = screenshot.crop((left, top, right, bottom))
    w, h = patch.size
    if w >= 32 and h >= 32:
        return patch
    canvas = Image.new("RGB", (max(w, 32), max(h, 32)), (255, 255, 255))
    canvas.paste(patch, (0, 0))
    return canvas


def _image_to_base64(pil_img: Image.Image, fmt: str = "png") -> str:
    buffer = io.BytesIO()
    pil_img.save(buffer, format=fmt.upper())
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt};base64,{b64}"


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


def _resolve_vec_path(stem: str, node_id: int) -> tuple[Path, np.ndarray | None]:
    """优先新路径；若仅旧路径存在则迁移后复用。"""
    new_path = node_emb_path(stem, node_id)
    cached = _load_vec(new_path)
    if cached is not None:
        return new_path, cached

    legacy_path = node_emb_legacy_path(stem, node_id)
    cached = _load_vec(legacy_path)
    if cached is not None:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_path), str(new_path))
        return new_path, cached

    return new_path, None


def embed_node(
    stem: str,
    node_id: int,
    text: str,
    bounds: list[int],
    screenshot: Image.Image | None,
) -> np.ndarray:
    path, cached = _resolve_vec_path(stem, node_id)
    if cached is not None:
        return cached

    patch = _crop_patch(screenshot, bounds)

    if text and patch is not None:
        try:
            b64 = _image_to_base64(patch)
            vec = np.asarray(vlm_embedding.embed_multimodal(text, b64), dtype=np.float32)
            _save_vec(path, vec)
            return vec
        except Exception:
            pass

    if patch is not None:
        try:
            b64 = _image_to_base64(patch)
            vec = np.asarray(vlm_embedding.embed_image(b64), dtype=np.float32)
            _save_vec(path, vec)
            return vec
        except Exception:
            pass

    if text:
        try:
            vec = np.asarray(vlm_embedding.embed_text(text), dtype=np.float32)
            _save_vec(path, vec)
            return vec
        except Exception:
            pass

    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    _save_vec(path, vec)
    return vec


def _gt_action_type(paths: dict[str, Path]) -> str:
    gt_path = paths["gt"]
    if gt_path.is_file():
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        action_type = gt.get("action_type")
        if action_type:
            return str(action_type)
    meta_path = paths["meta"]
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        action_type = meta.get("action_type")
        if action_type:
            return str(action_type)
    return "unknown"


def process_stem(stem: str, index: int, total: int) -> None:
    paths = step_paths(stem)
    nodes_path = paths["nodes"]
    screenshot_path = paths["screenshot"]
    action_type = _gt_action_type(paths)

    if not nodes_path.is_file():
        print(f"[{index}/{total}] {stem} action={action_type}: 缺少 nodes 文件，跳过")
        return

    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    if not nodes:
        print(f"[{index}/{total}] {stem} action={action_type}: nodes 为空，跳过 embedding")
        return

    screenshot = None
    if screenshot_path.is_file():
        screenshot = Image.open(screenshot_path).convert("RGB")
    else:
        print(
            f"[{index}/{total}] {stem} action={action_type}: "
            f"缺少 screenshot，将使用 zero/text fallback"
        )

    success_count = 0
    zero_count = 0
    skip_count = 0

    for node in nodes:
        node_id = node.get("node_id")
        bounds = node.get("bounds", [0, 0, 0, 0])
        text = node.get("text_for_emb", "")

        _, cached = _resolve_vec_path(stem, node_id)
        if cached is not None:
            skip_count += 1
            continue

        vec = embed_node(stem, node_id, text, bounds, screenshot)
        if np.allclose(vec, 0):
            zero_count += 1
        else:
            success_count += 1

    print(
        f"[{index}/{total}] {stem} action={action_type}: "
        f"节点={len(nodes)} | 新生成={success_count} 零向量={zero_count} 跳过={skip_count}"
    )


def main() -> None:
    episode_ids = iter_episode_ids()
    start = TEST_START if TEST_START is not None else 0
    end = TEST_END if TEST_END is not None else len(episode_ids)
    selected_episodes = episode_ids[start:end]

    stems: list[str] = []
    for episode_id in selected_episodes:
        stems.extend(stems_in_episode(episode_id))

    print("=" * 60)
    print("get_emb.py — 节点 embedding 生成")
    print("=" * 60)
    print(f"episode 总数: {len(episode_ids)}")
    print(f"本次 episode 范围: [{start}, {end}) → {len(selected_episodes)} 个")
    print(f"对应 step 数: {len(stems)}")
    if selected_episodes:
        print(f"  episode 示例: {selected_episodes[0]} ... {selected_episodes[-1]}")
    if stems:
        print(f"  step 示例: {stems[0]} ... {stems[-1]}")
    print("=" * 60)

    if not stems:
        print("没有样本，退出。")
        sys.exit(1)

    for i, stem in enumerate(stems, 1):
        process_stem(stem, i, len(stems))

    print(f"\n{'=' * 60}")
    print("处理完成")
    print(f"  embeddings: {EMBED_DIR / 'nodes_emb'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
