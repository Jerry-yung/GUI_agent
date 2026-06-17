"""Draw labeled bounding boxes (cN / sN tags)."""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from annotate.node_filter import filter_nodes_for_annotation

TYPE_COLORS = {
    "clickable": (255, 60, 60),
    "scroll": (60, 130, 255),
    "other": (160, 160, 160),
}
FILL_ALPHA = 45
BORDER_ALPHA = 220
CLICK_BORDER_WIDTH = 3
SCROLL_BORDER_WIDTH = 2
SCROLL_DASH_LEN = 10
SCROLL_GAP_LEN = 6
LABEL_GAP = 3
LABEL_FONT_SIZE = 24
LABEL_PAD_X = 5
LABEL_PAD_Y = 3
LABEL_TEXT_COLOR = (0, 0, 0)
LABEL_BG_ALPHA = 255
OUTSIDE_NUDGE_STEPS = 4


def _load_font(size: int = LABEL_FONT_SIZE):
    for fc in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(fc):
            try:
                return ImageFont.truetype(fc, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, label: str, font) -> tuple[int, int]:
    try:
        tb = draw.textbbox((0, 0), label, font=font)
        return tb[2] - tb[0], tb[3] - tb[1]
    except AttributeError:
        return draw.textsize(label, font=font)


def _label_dims(text_w: int, text_h: int) -> tuple[int, int]:
    return text_w + LABEL_PAD_X * 2, text_h + LABEL_PAD_Y * 2


def _box_area(bounds: list[int]) -> int:
    return max(0, bounds[2] - bounds[0]) * max(0, bounds[3] - bounds[1])


def _node_rgb(node: dict) -> tuple[int, int, int]:
    kind = node.get("kind", "clickable")
    return TYPE_COLORS.get(kind, TYPE_COLORS["other"])


def _is_scroll(node: dict) -> bool:
    return node.get("kind") == "scroll"


def _rects_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _rect_in_image(
    rect: tuple[int, int, int, int], img_size: tuple[int, int]
) -> bool:
    img_w, img_h = img_size
    return rect[0] >= 0 and rect[1] >= 0 and rect[2] <= img_w and rect[3] <= img_h


def _draw_box_fill(draw: ImageDraw.ImageDraw, node: dict) -> None:
    if _is_scroll(node):
        return
    x1, y1, x2, y2 = node["bounds"]
    rgb = _node_rgb(node)
    draw.rectangle([x1, y1, x2, y2], fill=(*rgb, FILL_ALPHA))


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill: tuple[int, int, int, int],
    width: int,
    dash_len: int,
    gap_len: int,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    pos = 0.0
    while pos < length:
        seg_end = min(pos + dash_len, length)
        sx = int(round(x1 + dx * pos))
        sy = int(round(y1 + dy * pos))
        ex = int(round(x1 + dx * seg_end))
        ey = int(round(y1 + dy * seg_end))
        draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)
        pos += dash_len + gap_len


def _draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    *,
    width: int,
    dash_len: int,
    gap_len: int,
) -> None:
    x1, y1, x2, y2 = bbox
    sides = (
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    )
    for start, end in sides:
        _draw_dashed_line(draw, start, end, outline, width, dash_len, gap_len)


def _draw_box_border(draw: ImageDraw.ImageDraw, node: dict) -> None:
    x1, y1, x2, y2 = node["bounds"]
    rgb = _node_rgb(node)
    outline = (*rgb, BORDER_ALPHA)
    if _is_scroll(node):
        _draw_dashed_rectangle(
            draw,
            (x1, y1, x2, y2),
            outline,
            width=SCROLL_BORDER_WIDTH,
            dash_len=SCROLL_DASH_LEN,
            gap_len=SCROLL_GAP_LEN,
        )
    else:
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=outline,
            width=CLICK_BORDER_WIDTH,
        )


def _label_fits_inside(
    x: int,
    y: int,
    lw: int,
    lh: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> bool:
    rect = (x, y, x + lw, y + lh)
    return (
        rect[0] >= x1 + LABEL_GAP
        and rect[1] >= y1 + LABEL_GAP
        and rect[2] <= x2 - LABEL_GAP
        and rect[3] <= y2 - LABEL_GAP
    )


def _label_candidate_groups(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    lw: int,
    lh: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    margin = CLICK_BORDER_WIDTH + LABEL_GAP
    inside = [
        (x1 + margin, y1 + margin),
        (x2 - lw - margin, y1 + margin),
        (x1 + margin, y2 - lh - margin),
        (x2 - lw - margin, y2 - lh - margin),
    ]
    outside = [
        (x1 + margin, y1 - lh - margin),
        (x1 - lw - margin, y1 + margin),
        (x2 + margin, y1 + margin),
        (x1 + margin, y2 + margin),
    ]
    return inside, outside


def _expand_with_nudges(
    positions: list[tuple[int, int]], lh: int
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    nudge = max(8, lh // 2)
    shifts = [(0, 0)]
    for step in range(1, OUTSIDE_NUDGE_STEPS + 1):
        d = step * nudge
        shifts.extend([(0, -d), (0, d), (-d, 0), (d, 0)])
    for bx, by in positions:
        for dx, dy in shifts:
            pos = (bx + dx, by + dy)
            if pos in seen:
                continue
            seen.add(pos)
            candidates.append(pos)
    return candidates


def _find_label_position(
    node: dict,
    lw: int,
    lh: int,
    *,
    img_size: tuple[int, int],
    occupied: list[tuple[int, int, int, int]],
) -> tuple[int, int]:
    x1, y1, x2, y2 = node["bounds"]
    force_outside = bool(node.get("force_outside_label"))
    inside_base, outside_base = _label_candidate_groups(x1, y1, x2, y2, lw, lh)

    phases: list[tuple[list[tuple[int, int]], bool]] = []
    if not force_outside:
        phases.append((_expand_with_nudges(inside_base, lh), True))
    phases.append((_expand_with_nudges(outside_base, lh), False))

    for positions, must_fit_inside in phases:
        for x, y in positions:
            rect = (x, y, x + lw, y + lh)
            if not _rect_in_image(rect, img_size):
                continue
            if must_fit_inside and not _label_fits_inside(
                x, y, lw, lh, x1, y1, x2, y2
            ):
                continue
            if any(_rects_intersect(rect, occ) for occ in occupied):
                continue
            return x, y

    fallback_x = max(0, min(x1, img_size[0] - lw))
    fallback_y = max(0, y1 - lh - LABEL_GAP)
    if fallback_y + lh > img_size[1]:
        fallback_y = max(0, min(y2 + LABEL_GAP, img_size[1] - lh))
    return fallback_x, fallback_y


def _draw_label_text(
    draw: ImageDraw.ImageDraw,
    label: str,
    x: int,
    y: int,
    font,
) -> tuple[int, int, int, int]:
    tw, th = _text_size(draw, label, font)
    lw, lh = _label_dims(tw, th)
    draw.rectangle([x, y, x + lw, y + lh], fill=(255, 255, 255, LABEL_BG_ALPHA))
    draw.text(
        (x + LABEL_PAD_X, y + LABEL_PAD_Y),
        label,
        font=font,
        fill=(*LABEL_TEXT_COLOR, 255),
    )
    return x, y, x + lw, y + lh


def default_label_dims() -> tuple[int, int]:
    """Approximate label pad size for pre-render filtering (P4)."""
    img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(img)
    tw, th = _text_size(draw, "c99", _load_font())
    return _label_dims(tw, th)


def load_base_image(screenshot_path: Path | str | None) -> Image.Image:
    if screenshot_path and Path(screenshot_path).is_file():
        return Image.open(screenshot_path).convert("RGB")
    return Image.new("RGB", (1080, 2400), (255, 255, 255))


def annotate_nodes(
    base_img: Image.Image,
    nodes: list[dict],
    out_path: Path,
    *,
    resize_to: tuple[int, int] | None = None,
    apply_node_filter: bool = True,
) -> None:
    img = base_img.copy()
    font = _load_font()
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    render_nodes = list(nodes)
    if apply_node_filter:
        lw, lh = default_label_dims()
        render_nodes = filter_nodes_for_annotation(
            render_nodes, label_w=lw, label_h=lh
        )

    by_area_desc = sorted(render_nodes, key=lambda n: _box_area(n["bounds"]), reverse=True)
    by_area_asc = sorted(render_nodes, key=lambda n: _box_area(n["bounds"]))

    for node in by_area_desc:
        _draw_box_fill(draw, node)
    for node in by_area_asc:
        _draw_box_border(draw, node)

    label_order = sorted(
        render_nodes,
        key=lambda n: (_box_area(n["bounds"]), n.get("sman_area_idx", 0)),
    )
    placed_labels: list[tuple[int, int, int, int]] = []

    for node in label_order:
        label = str(node.get("label") or "")
        if not label:
            continue
        tw, th = _text_size(draw, label, font)
        lw, lh = _label_dims(tw, th)
        x, y = _find_label_position(
            node, lw, lh, img_size=img.size, occupied=placed_labels
        )
        rect = _draw_label_text(draw, label, x, y, font)
        placed_labels.append(rect)

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    if resize_to is not None and resize_to != img.size:
        img = img.resize(resize_to, Image.Resampling.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
