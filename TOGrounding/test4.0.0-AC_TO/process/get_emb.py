#!/usr/bin/env python3
"""
get_emb.py

读取 AC_data/nodes/*.json，按 bounds 从截图 crop patch，
为每个节点生成 embedding：
  优先 multimodal(text+patch) → patch only → text only → zero fallback
保存到 AC_data/embeddings/nodes_emb/{stem}/{stem}_{node_id}.npy
若已存在则跳过。
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm_set.llm import vlm_embedding

BASE_DIR = PROJECT_ROOT / "AC_data"
NODES_DIR = BASE_DIR / "nodes"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
EMBED_DIR = BASE_DIR / "embeddings"

TEST_START = 300
TEST_END = 500 # 0-500 done

EMBEDDING_DIM = 2560


# ============================================================
# 工具函数
# ============================================================
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


def _vec_path(cache_dir: Path, stem: str, node_id: int) -> Path:
    return cache_dir / "nodes_emb" / stem / f"{stem}_{node_id}.npy"


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


# ============================================================
# Embedding 生成
# ============================================================
def embed_node(
    stem: str,
    node_id: int,
    text: str,
    bounds: list[int],
    screenshot: Image.Image | None,
) -> np.ndarray:
    """为单个节点生成 embedding，带缓存机制。"""
    path = _vec_path(EMBED_DIR, stem, node_id)
    cached = _load_vec(path)
    if cached is not None:
        return cached

    patch = _crop_patch(screenshot, bounds)

    # 1. multimodal (text + patch)
    if text and patch is not None:
        try:
            b64 = _image_to_base64(patch)
            vec = np.asarray(vlm_embedding.embed_multimodal(text, b64), dtype=np.float32)
            _save_vec(path, vec)
            return vec
        except Exception:
            pass

    # 2. patch only
    if patch is not None:
        try:
            b64 = _image_to_base64(patch)
            vec = np.asarray(vlm_embedding.embed_image(b64), dtype=np.float32)
            _save_vec(path, vec)
            return vec
        except Exception:
            pass

    # 3. text only
    if text:
        try:
            vec = np.asarray(vlm_embedding.embed_text(text), dtype=np.float32)
            _save_vec(path, vec)
            return vec
        except Exception:
            pass

    # 4. zero fallback
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    _save_vec(path, vec)
    return vec


def process_stem(stem: str, index: int, total: int) -> None:
    nodes_path = NODES_DIR / f"{stem}.json"
    screenshot_path = SCREENSHOTS_DIR / f"{stem}.png"

    if not nodes_path.is_file():
        print(f"[{index}/{total}] {stem}: 缺少 nodes 文件，跳过")
        return

    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    if not nodes:
        print(f"[{index}/{total}] {stem}: nodes 为空，跳过 embedding")
        return

    screenshot = None
    if screenshot_path.is_file():
        screenshot = Image.open(screenshot_path).convert("RGB")
    else:
        print(f"[{index}/{total}] {stem}: 缺少 screenshot，将使用 zero/text fallback")

    success_count = 0
    zero_count = 0
    skip_count = 0

    for node in nodes:
        node_id = node.get("node_id")
        bounds = node.get("bounds", [0, 0, 0, 0])
        text = node.get("text_for_emb", "")

        path = _vec_path(EMBED_DIR, stem, node_id)
        if path.is_file():
            skip_count += 1
            continue

        vec = embed_node(stem, node_id, text, bounds, screenshot)
        if np.allclose(vec, 0):
            zero_count += 1
        else:
            success_count += 1

    print(
        f"[{index}/{total}] {stem}: "
        f"节点={len(nodes)} | 新生成={success_count} 零向量={zero_count} 跳过={skip_count}"
    )


# ============================================================
# Main
# ============================================================
def main() -> None:
    all_stems = sorted(p.stem for p in NODES_DIR.glob("*.json"))
    start = TEST_START if TEST_START is not None else 0
    end = TEST_END if TEST_END is not None else len(all_stems)
    stems = all_stems[start:end]

    print("=" * 60)
    print("get_emb.py — 节点 embedding 生成")
    print("=" * 60)
    print(f"nodes 总数: {len(all_stems)}")
    print(f"本次范围: [{start}, {end}) → {len(stems)} 个")
    if stems:
        print(f"  示例: {stems[0]} ... {stems[-1]}")
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
