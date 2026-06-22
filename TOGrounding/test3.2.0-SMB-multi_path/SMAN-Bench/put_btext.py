"""Minimal putBText helper (from pyshine.convenience) without optional audio/keras deps."""
from __future__ import annotations

import cv2
import numpy as np


def put_btext(
    img,
    text,
    text_offset_x=20,
    text_offset_y=20,
    vspace=10,
    hspace=10,
    font_scale=1.0,
    background_RGB=(228, 225, 222),
    text_RGB=(1, 1, 1),
    font=cv2.FONT_HERSHEY_DUPLEX,
    thickness=2,
    alpha=0.6,
    gamma=0,
):
    r, g, b = background_RGB[0], background_RGB[1], background_RGB[2]
    text_r, text_g, text_b = text_RGB[0], text_RGB[1], text_RGB[2]
    text_width, text_height = cv2.getTextSize(
        text, font, fontScale=font_scale, thickness=thickness
    )[0]
    x, y, w, h = text_offset_x, text_offset_y, text_width, text_height
    crop = img[y - vspace : y + h + vspace, x - hspace : x + w + hspace]
    white_rect = np.ones(crop.shape, dtype=np.uint8)
    b_ch, g_ch, r_ch = cv2.split(white_rect)
    rect_changed = cv2.merge((b * b_ch, g * g_ch, r * r_ch))
    res = cv2.addWeighted(crop, alpha, rect_changed, 1 - alpha, gamma)
    img[y - vspace : y + vspace + h, x - hspace : x + w + hspace] = res
    cv2.putText(
        img,
        text,
        (x, y + h),
        font,
        fontScale=font_scale,
        color=(text_b, text_g, text_r),
        thickness=thickness,
    )
    return img
