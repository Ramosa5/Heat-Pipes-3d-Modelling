from __future__ import annotations

import os
from typing import Any, Iterable

import cv2
import numpy as np

from .tube_geometry import draw_tubes, overlay_mask


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _bbox_from_ann(ann: dict[str, Any]) -> tuple[int, int, int, int] | None:
    bbox = ann.get("bbox")
    if bbox is None or len(bbox) != 4:
        return None
    x, y, w, h = map(float, bbox)
    return int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))


def _bbox_center(ann: dict[str, Any]) -> tuple[float, float] | None:
    bbox = ann.get("bbox")
    if bbox is None or len(bbox) != 4:
        return None
    x, y, w, h = map(float, bbox)
    return x + 0.5 * w, y + 0.5 * h


def _tube_midline_y(tube: dict[str, float], x: float) -> float:
    y_top = float(tube["a_top"]) * float(x) + float(tube["b_top"])
    y_bot = float(tube["a_bot"]) * float(x) + float(tube["b_bot"])
    return 0.5 * (y_top + y_bot)


def _text_size(lines: list[str], font_scale: float, thickness: int) -> tuple[int, int, int]:
    widths: list[int] = []
    heights: list[int] = []
    baselines: list[int] = []
    for line in lines:
        (w, h), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        widths.append(w)
        heights.append(h)
        baselines.append(baseline)
    return max(widths or [0]), sum(heights) + max(0, len(lines) - 1) * 5, max(baselines or [0])


def _draw_text_block(img: np.ndarray,
                     x: int,
                     y: int,
                     lines: Iterable[str],
                     color: tuple[int, int, int] = (0, 255, 255),
                     font_scale: float = 0.42,
                     thickness: int = 1) -> None:
    lines = [str(line) for line in lines if str(line)]
    if not lines:
        return

    h_img, w_img = img.shape[:2]
    text_w, text_h, baseline = _text_size(lines, font_scale, thickness)
    pad = 4
    line_h = max(12, int(round(text_h / max(1, len(lines))))) + 5

    x = int(np.clip(x, 0, max(0, w_img - text_w - 2 * pad - 1)))
    y = int(np.clip(y, text_h + 2 * pad + baseline, max(text_h + 2 * pad + baseline, h_img - 1)))

    x0 = max(0, x - pad)
    y0 = max(0, y - text_h - pad - baseline)
    x1 = min(w_img - 1, x + text_w + pad)
    y1 = min(h_img - 1, y + pad)

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)

    ty = y0 + pad + 12
    for line in lines:
        cv2.putText(img, line, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        ty += line_h


def _draw_annotation_boxes(img: np.ndarray,
                           tube_anns: dict[int, list[dict[str, Any]]],
                           color: tuple[int, int, int] = (0, 255, 0)) -> None:
    """Draw original COCO bubble annotation boxes for reference."""
    for anns in tube_anns.values():
        for ann in anns:
            bbox = _bbox_from_ann(ann)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)


def _draw_mask_contours(img: np.ndarray,
                        masks_by_tube: dict[int, np.ndarray],
                        color: tuple[int, int, int] = (0, 160, 255)) -> None:
    """Draw original mask contours so labels are visually attached to bubble annotations."""
    for mask in masks_by_tube.values():
        if mask is None or mask.size == 0:
            continue
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, 1)


def _sort_rows_by_z(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(rows, key=lambda r: (_safe_float(r.get("centroid_z_mm")), _safe_int(r.get("detection_index"))))


def _sort_ecc_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(rows, key=lambda r: _safe_int(r.get("bubble_index")))


def _pair_rows_with_eccentricity(tracking_rows: list[dict[str, object]],
                                 eccentricity_rows: list[dict[str, object]],
                                 tube_pair: str) -> list[tuple[dict[str, object] | None, dict[str, object] | None, int]]:
    """
    Pair tracking rows and eccentricity rows for one tube pair.

    Tracking rows have centroids and persistent IDs. Eccentricity rows have e*, e_x, e_y.
    Both are sorted along the longitudinal direction/bubble index so the text block can
    combine them on the frame even though the two calculations are separate pipeline stages.
    """
    tr = _sort_rows_by_z([r for r in tracking_rows if r.get("tube_pair") == tube_pair])
    er = _sort_ecc_rows([r for r in eccentricity_rows if r.get("tube_pair") == tube_pair])
    n = max(len(tr), len(er))
    paired: list[tuple[dict[str, object] | None, dict[str, object] | None, int]] = []
    for i in range(n):
        paired.append((tr[i] if i < len(tr) else None, er[i] if i < len(er) else None, i + 1))
    return paired


def _label_anchor_from_tracking(row: dict[str, object],
                                tube: dict[str, float],
                                rect_shape: tuple[int, int] | tuple[int, int, int],
                                diameter_mm: float,
                                frame_width: int) -> tuple[int, int]:
    """Project reconstructed Z centroid back to an approximate x-position on the original frame."""
    rect_h = max(1, int(rect_shape[0]))
    rect_w = max(2, int(rect_shape[1]))
    mm_per_pixel = float(diameter_mm) / float(rect_h)
    rect_x = _safe_float(row.get("centroid_z_mm")) / max(mm_per_pixel, 1e-9)
    x = int(round(rect_x * float(frame_width - 1) / float(rect_w - 1)))
    x = int(np.clip(x, 0, frame_width - 1))
    y = int(round(_tube_midline_y(tube, x)))
    return x, y


def _label_anchor_from_annotations(tube_annotations: list[dict[str, Any]],
                                   fallback_index: int,
                                   tube: dict[str, float],
                                   frame_width: int) -> tuple[int, int]:
    """Use original COCO bbox centers when tracking centroids are unavailable."""
    anns = sorted(tube_annotations, key=lambda ann: (_bbox_center(ann) or (frame_width, 0))[0])
    if 0 <= fallback_index - 1 < len(anns):
        center = _bbox_center(anns[fallback_index - 1])
        if center is not None:
            return int(round(center[0])), int(round(center[1]))
    x = int(round((fallback_index / max(1, len(anns) + 1)) * (frame_width - 1)))
    y = int(round(_tube_midline_y(tube, x)))
    return x, y


def _format_parameter_lines(track_row: dict[str, object] | None,
                            ecc_row: dict[str, object] | None,
                            fallback_index: int) -> list[str]:
    lines: list[str] = []
    if track_row is not None:
        lines.append(
            f"ID {int(track_row['track_id'])}  det {int(track_row['detection_index'])}"
        )
        lines.append(
            f"z={_safe_float(track_row.get('centroid_z_mm')):.1f} mm  V={_safe_int(track_row.get('volume_voxels'))}"
        )
    else:
        idx = _safe_int(ecc_row.get("bubble_index"), fallback_index) if ecc_row is not None else fallback_index
        lines.append(f"B{idx}")

    if ecc_row is not None:
        lines.append(
            f"e*={_safe_float(ecc_row.get('e_star')):.3f}  "
            f"ex={_safe_float(ecc_row.get('e_x_mm')):.1f}  ey={_safe_float(ecc_row.get('e_y_mm')):.1f}"
        )
    return lines


def build_parameter_overlay_frame(gray_or_bgr: np.ndarray,
                                  tubes: list[dict[str, float]],
                                  tube_anns: dict[int, list[dict[str, Any]]],
                                  masks_by_tube: dict[int, np.ndarray],
                                  rect_shapes: dict[str, tuple[int, ...]],
                                  tracking_rows: list[dict[str, object]],
                                  eccentricity_rows: list[dict[str, object]],
                                  diameter_mm: float,
                                  frame_no: int,
                                  file_name: str,
                                  show_mask_overlay: bool = True) -> np.ndarray:
    """
    Return an original video frame with per-bubble parameters drawn directly on it.

    The overlay includes:
    - tube reference lines,
    - original COCO annotation boxes and mask contours,
    - persistent tracking ID when tracking is enabled,
    - eccentricity parameters when eccentricity is enabled.
    """
    if gray_or_bgr.ndim == 2:
        out = cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2BGR)
    else:
        out = gray_or_bgr.copy()

    if show_mask_overlay:
        merged_mask = np.zeros(out.shape[:2], dtype=np.uint8)
        for mask in masks_by_tube.values():
            if mask is not None:
                merged_mask = cv2.bitwise_or(merged_mask, mask.astype(np.uint8))
        out = overlay_mask(out, merged_mask, color=(255, 0, 0), alpha=0.22)

    out = draw_tubes(out, tubes)
    _draw_annotation_boxes(out, tube_anns)
    _draw_mask_contours(out, masks_by_tube)

    h_img, w_img = out.shape[:2]
    cv2.putText(
        out,
        f"Frame {frame_no}: {file_name}",
        (8, max(18, min(h_img - 8, 18))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    pair_to_tube_index = {"tube12": 0, "tube34": 2}
    pair_to_top_annotations = {"tube12": tube_anns.get(0, []), "tube34": tube_anns.get(2, [])}
    vertical_offsets = {"tube12": -10, "tube34": -10}

    for tube_pair in ("tube12", "tube34"):
        paired = _pair_rows_with_eccentricity(tracking_rows, eccentricity_rows, tube_pair)
        if not paired:
            continue

        tube_idx = pair_to_tube_index[tube_pair]
        tube = tubes[tube_idx]
        rect_shape = rect_shapes.get(tube_pair, (h_img, w_img))
        color = (0, 255, 255) if tube_pair == "tube12" else (255, 255, 0)

        for track_row, ecc_row, fallback_index in paired:
            if track_row is not None:
                x, y = _label_anchor_from_tracking(track_row, tube, rect_shape, diameter_mm, w_img)
            else:
                x, y = _label_anchor_from_annotations(
                    pair_to_top_annotations[tube_pair], fallback_index, tube, w_img
                )
            y = int(np.clip(y + vertical_offsets[tube_pair], 18, h_img - 5))
            lines = _format_parameter_lines(track_row, ecc_row, fallback_index)

            cv2.circle(out, (x, y), 4, color, -1)
            cv2.line(out, (x, y), (min(w_img - 1, x + 8), max(0, y - 8)), color, 1)
            _draw_text_block(out, min(w_img - 1, x + 10), max(18, y - 8), lines, color=color)

    return out


def save_parameter_overlay_frame(gray_or_bgr: np.ndarray,
                                 out_dir: str,
                                 file_stem: str,
                                 **kwargs: Any) -> str:
    """Build and save the 2D per-bubble parameter overlay frame."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{file_stem}_parameters.png")
    frame = build_parameter_overlay_frame(gray_or_bgr, **kwargs)
    print(f"[SAVE] Zapisywana jest klatka z parametrami: {out_path}")
    cv2.imwrite(out_path, frame)
    return out_path
