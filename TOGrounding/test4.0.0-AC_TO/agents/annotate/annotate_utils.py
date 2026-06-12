"""在截图上绘制带 node_id 标签的候选框（参考 test1.6.1 annotate_utils）。"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BOX_COLOR = (60, 130, 255)
FILL_ALPHA = 45
BORDER_ALPHA = 220
BORDER_WIDTH = 3
TEXT_COLOR = (255, 255, 255, 255)
BG_COLOR = (0, 0, 0, 180)
PAD = 4
DEFAULT_SCREEN = (1080, 2400)


def _load_font():
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for fc in font_candidates:
        if os.path.exists(fc):
            try:
                return ImageFont.truetype(fc, size=32)
            except Exception:
                pass
    return ImageFont.load_default()


def load_base_image(
    screenshot_path: Path | None,
    screen_size: tuple[int, int] = DEFAULT_SCREEN,
) -> Image.Image:
    if screenshot_path and screenshot_path.is_file():
        return Image.open(screenshot_path).convert("RGB")
    return Image.new("RGB", screen_size, (255, 255, 255))


def _node_area(node: dict) -> int:
    x1, y1, x2, y2 = node["bounds"]
    return max(0, x2 - x1) * max(0, y2 - y1)


def _contains(outer_bounds: list, inner_bounds: list) -> bool:
    ox1, oy1, ox2, oy2 = outer_bounds
    ix1, iy1, ix2, iy2 = inner_bounds
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2


def _container_node_ids(nodes: list[dict]) -> set[int]:
    """包含其他候选节点的容器：只画边框，不画半透明填充，避免遮挡内部小框。"""
    containers: set[int] = set()
    for outer in nodes:
        outer_id = int(outer.get("node_id", outer.get("ager_id", 0)))
        outer_area = _node_area(outer)
        for inner in nodes:
            inner_id = int(inner.get("node_id", inner.get("ager_id", 0)))
            if inner_id == outer_id:
                continue
            if _contains(outer["bounds"], inner["bounds"]) and outer_area > _node_area(inner):
                containers.add(outer_id)
                break
    return containers


def _draw_label(overlay_draw: ImageDraw.ImageDraw, font, node_id: int, x1: int, y1: int) -> None:
    text = f"#{node_id}"
    try:
        tbbox = overlay_draw.textbbox((0, 0), text, font=font)
        tw = tbbox[2] - tbbox[0]
        th = tbbox[3] - tbbox[1]
    except AttributeError:
        tw, th = overlay_draw.textsize(text, font=font)

    bg_x1 = x1 + BORDER_WIDTH + 1
    bg_y1 = y1 + BORDER_WIDTH + 1
    bg_x2 = bg_x1 + tw + PAD * 2
    bg_y2 = bg_y1 + th + PAD * 2
    overlay_draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=BG_COLOR)
    overlay_draw.text((bg_x1 + PAD, bg_y1 + PAD), text, font=font, fill=TEXT_COLOR)


def annotate_image(base_img: Image.Image, nodes: list[dict], out_path: Path) -> None:
    """在 base_img 上绘制节点框与 #node_id 标签。

    绘制顺序（避免大框遮挡小框）：
      1. 填充：非容器节点，大→小
      2. 边框：全部节点，小→大
      3. 标签：全部节点，小→大
    """
    img = base_img.copy()
    font = _load_font()
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    rgb = BOX_COLOR

    container_ids = _container_node_ids(nodes)
    by_area_desc = sorted(nodes, key=_node_area, reverse=True)
    by_area_asc = sorted(nodes, key=_node_area)

    for node in by_area_desc:
        node_id = int(node.get("node_id", node.get("ager_id", 0)))
        if node_id in container_ids:
            continue
        x1, y1, x2, y2 = node["bounds"]
        overlay_draw.rectangle([x1, y1, x2, y2], fill=(*rgb, FILL_ALPHA))

    for node in by_area_asc:
        node_id = int(node.get("node_id", node.get("ager_id", 0)))
        x1, y1, x2, y2 = node["bounds"]
        overlay_draw.rectangle(
            [x1, y1, x2, y2], outline=(*rgb, BORDER_ALPHA), width=BORDER_WIDTH
        )
        _draw_label(overlay_draw, font, node_id, x1, y1)

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
