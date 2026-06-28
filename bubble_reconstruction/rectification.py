from __future__ import annotations

import cv2
import numpy as np

def _tube_quad_points(tube: dict, x_left: float, x_right: float, margin_px: float = 0.0):
    a_top, b_top = float(tube["a_top"]), float(tube["b_top"])
    a_bot, b_bot = float(tube["a_bot"]), float(tube["b_bot"])

    y_tl = a_top * x_left + b_top + margin_px
    y_tr = a_top * x_right + b_top + margin_px
    y_bl = a_bot * x_left + b_bot - margin_px
    y_br = a_bot * x_right + b_bot - margin_px

    if y_bl < y_tl:
        y_tl, y_bl = y_bl, y_tl
    if y_br < y_tr:
        y_tr, y_br = y_br, y_tr

    return np.array([
        [x_left,  y_tl],
        [x_right, y_tr],
        [x_right, y_br],
        [x_left,  y_bl],
    ], dtype=np.float32)


def crop_and_rectify_tube(img, tube: dict, x_left: int, x_right: int,
                          inner_margin_px: float = 0.0,
                          out_width: int = None,
                          interp=cv2.INTER_LINEAR):
    x_left_f = float(x_left)
    x_right_f = float(x_right)

    src = _tube_quad_points(tube, x_left_f, x_right_f, margin_px=inner_margin_px)

    a_top, b_top = float(tube["a_top"]), float(tube["b_top"])
    a_bot, b_bot = float(tube["a_bot"]), float(tube["b_bot"])

    thick_left = abs((a_bot * x_left_f + b_bot) - (a_top * x_left_f + b_top)) - 2.0 * inner_margin_px
    thick_right = abs((a_bot * x_right_f + b_bot) - (a_top * x_right_f + b_top)) - 2.0 * inner_margin_px
    out_h = int(round(max(1.0, 0.5 * (thick_left + thick_right))))

    if out_width is None:
        out_w = int(round(max(2.0, x_right_f - x_left_f)))
    else:
        out_w = int(out_width)

    dst = np.array([
        [0,         0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0,         out_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)

    border = cv2.BORDER_CONSTANT
    border_val = 0 if img.ndim == 2 else (0, 0, 0)

    rect = cv2.warpPerspective(img, M, (out_w, out_h),
                               flags=interp, borderMode=border, borderValue=border_val)
    return rect, src


def resize_to_match_height(img, target_h, keep_aspect=True, is_mask=False):
    h, w = img.shape[:2]
    if h == target_h:
        return img

    scale = target_h / float(h)
    if keep_aspect:
        new_w = max(1, int(round(w * scale)))
        new_h = target_h
    else:
        new_w = w
        new_h = target_h

    if is_mask:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA

    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def center_crop_or_pad_width(img, target_w, pad_value=0):
    h, w = img.shape[:2]
    if w == target_w:
        return img

    if w > target_w:
        x0 = (w - target_w) // 2
        return img[:, x0:x0 + target_w].copy()

    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left

    if img.ndim == 2:
        return cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right,
                                  borderType=cv2.BORDER_CONSTANT, value=pad_value)
    else:
        return cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right,
                                  borderType=cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))


def rectify_and_align_pair(mask_top_orig: np.ndarray, tube_top: dict,
                           mask_side_orig: np.ndarray, tube_side: dict,
                           x_left: int, x_right: int, inner_margin_px: float,
                           keep_aspect: bool):
    rect_top, quad_top = crop_and_rectify_tube(
        mask_top_orig, tube_top, x_left=x_left, x_right=x_right,
        inner_margin_px=inner_margin_px, interp=cv2.INTER_NEAREST
    )
    rect_side, quad_side = crop_and_rectify_tube(
        mask_side_orig, tube_side, x_left=x_left, x_right=x_right,
        inner_margin_px=inner_margin_px, interp=cv2.INTER_NEAREST
    )

    H_target = max(rect_top.shape[0], rect_side.shape[0])
    rect_top = resize_to_match_height(rect_top, H_target, keep_aspect=keep_aspect, is_mask=True)
    rect_side = resize_to_match_height(rect_side, H_target, keep_aspect=keep_aspect, is_mask=True)

    W_target = min(rect_top.shape[1], rect_side.shape[1])
    rect_top = center_crop_or_pad_width(rect_top, W_target, pad_value=0)
    rect_side = center_crop_or_pad_width(rect_side, W_target, pad_value=0)

    return rect_top, rect_side, quad_top, quad_side
