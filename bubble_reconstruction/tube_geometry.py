from __future__ import annotations

import cv2
import numpy as np

def point_in_tube(xc: float, yc: float, tube: dict, margin_px: float = 0.0) -> bool:
    y_top = tube["a_top"] * xc + tube["b_top"]
    y_bot = tube["a_bot"] * xc + tube["b_bot"]
    y_min = min(y_top, y_bot)
    y_max = max(y_top, y_bot)
    return (y_min + margin_px) <= yc <= (y_max - margin_px)


def draw_tubes(rgb, tubes):
    out = rgb.copy()
    h, w, _ = out.shape

    for i, t in enumerate(tubes):
        x0, x1 = 0, w - 1
        y0_top = int(round(t["a_top"] * x0 + t["b_top"]))
        y1_top = int(round(t["a_top"] * x1 + t["b_top"]))
        y0_bot = int(round(t["a_bot"] * x0 + t["b_bot"]))
        y1_bot = int(round(t["a_bot"] * x1 + t["b_bot"]))

        cv2.line(out, (x0, y0_top), (x1, y1_top), (255, 255, 0), 2)
        cv2.line(out, (x0, y0_bot), (x1, y1_bot), (255, 255, 0), 2)

        y_mid = int((y0_top + y0_bot) / 2)
        cv2.putText(out, f"tube {i+1}", (10, max(20, y_mid)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return out


def overlay_mask(rgb, mask, color=(255, 0, 0), alpha=0.4):
    out = rgb.copy()
    colored = np.zeros_like(out)
    colored[:, :] = color
    m = mask.astype(bool)
    out[m] = (out[m] * (1 - alpha) + colored[m] * alpha).astype(np.uint8)
    return out
