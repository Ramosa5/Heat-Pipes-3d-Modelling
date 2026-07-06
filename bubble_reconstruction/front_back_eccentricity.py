from __future__ import annotations

from typing import Any

import numpy as np

from .volume import volume_to_points_mm


FRONT_BACK_ECCENTRICITY_FIELDS = [
    "frame_no",
    "file_name",
    "tube_pair",
    "bubble_index",
    "detection_index",
    "diameter_mm",
    "window_length_mm",
    "tip_percentile",
    "edge_margin_mm",
    "min_tip_points",
    "front_eccentricity",
    "back_eccentricity",
    "front_shift_x_mm",
    "front_shift_y_mm",
    "back_shift_x_mm",
    "back_shift_y_mm",
    "front_clipped",
    "back_clipped",
    "point_count",
]


def _none_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def calculate_front_back_eccentricity(
    points_mm: np.ndarray | None,
    diameter_mm: float,
    window_length_mm: float,
    tip_percentile: float = 99.0,
    edge_margin_mm: float = 0.1,
    min_tip_points: int = 20,
) -> dict[str, object]:
    """
    Calculate front/tip and back/tail eccentricity for one reconstructed bubble.

    This is the pipeline-ready version of the uploaded validation function. The
    point cloud is expected in millimetres with X/Y centred on the pipe axis and
    Z aligned with the pipe/window length.
    """
    if points_mm is None or len(points_mm) == 0:
        return {
            "front_eccentricity": None,
            "back_eccentricity": None,
            "front_shift": (None, None),
            "back_shift": (None, None),
            "front_clipped": False,
            "back_clipped": False,
            "point_count": 0,
        }

    if diameter_mm <= 0:
        raise ValueError("diameter_mm must be positive.")
    if window_length_mm <= 0:
        raise ValueError("window_length_mm must be positive.")
    if not (0.0 < tip_percentile < 100.0):
        raise ValueError("tip_percentile must lie between 0 and 100.")
    if min_tip_points < 1:
        raise ValueError("min_tip_points must be at least 1.")

    pts = np.asarray(points_mm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_mm must have shape (N, 3), got {pts.shape}")

    z_coords = pts[:, 2]
    z_max = float(np.max(z_coords))
    z_min = float(np.min(z_coords))

    front_clipped = z_max >= (float(window_length_mm) - float(edge_margin_mm))
    back_clipped = z_min <= float(edge_margin_mm)

    sorted_idx = np.argsort(z_coords)
    min_points = min(int(min_tip_points), len(pts))

    front_eccentricity: float | None
    back_eccentricity: float | None
    front_shift: tuple[float | None, float | None]
    back_shift: tuple[float | None, float | None]

    if not front_clipped:
        z_thresh_front = np.percentile(z_coords, float(tip_percentile))
        front_pts = pts[z_coords >= z_thresh_front]
        if len(front_pts) < min_points:
            front_pts = pts[sorted_idx[-min_points:]]
        if len(front_pts) > 0:
            front_x = float(np.median(front_pts[:, 0]))
            front_y = float(np.median(front_pts[:, 1]))
            front_eccentricity = float((2.0 / diameter_mm) * np.sqrt(front_x ** 2 + front_y ** 2))
            front_shift = (front_x, front_y)
        else:
            front_eccentricity = None
            front_shift = (None, None)
    else:
        front_eccentricity = None
        front_shift = (None, None)

    if not back_clipped:
        tail_percentile = 100.0 - float(tip_percentile)
        z_thresh_back = np.percentile(z_coords, tail_percentile)
        back_pts = pts[z_coords <= z_thresh_back]
        if len(back_pts) < min_points:
            back_pts = pts[sorted_idx[:min_points]]
        if len(back_pts) > 0:
            back_x = float(np.median(back_pts[:, 0]))
            back_y = float(np.median(back_pts[:, 1]))
            back_eccentricity = float((2.0 / diameter_mm) * np.sqrt(back_x ** 2 + back_y ** 2))
            back_shift = (back_x, back_y)
        else:
            back_eccentricity = None
            back_shift = (None, None)
    else:
        back_eccentricity = None
        back_shift = (None, None)

    return {
        "front_eccentricity": front_eccentricity,
        "back_eccentricity": back_eccentricity,
        "front_shift": front_shift,
        "back_shift": back_shift,
        "front_clipped": bool(front_clipped),
        "back_clipped": bool(back_clipped),
        "point_count": int(len(pts)),
    }


def calculate_front_back_eccentricity_for_bubbles(
    bubbles: list[dict[str, Any]],
    frame_no: int,
    file_name: str,
    tube_pair: str,
    diameter_mm: float,
    center_radial_xy: bool = True,
    tip_percentile: float = 99.0,
    edge_margin_mm: float = 0.1,
    min_tip_points: int = 20,
    max_points: int = 500_000,
) -> list[dict[str, object]]:
    """Return CSV-ready front/back eccentricity rows for each reconstructed bubble."""
    rows: list[dict[str, object]] = []

    for bubble_index, bubble in enumerate(bubbles, start=1):
        volume = bubble.get("volume")
        voxel_mm = float(bubble.get("voxel_mm", 1.0))
        detection_index = int(bubble.get("detection_index", bubble_index))

        points = volume_to_points_mm(
            volume,
            voxel_mm,
            center_radial_xy=center_radial_xy,
            max_points=max_points,
        )
        window_length_mm = 0.0
        if volume is not None:
            window_length_mm = float(volume.shape[2]) * float(voxel_mm)

        result = calculate_front_back_eccentricity(
            points_mm=points,
            diameter_mm=diameter_mm,
            window_length_mm=window_length_mm,
            tip_percentile=tip_percentile,
            edge_margin_mm=edge_margin_mm,
            min_tip_points=min_tip_points,
        )

        front_shift = result["front_shift"]
        back_shift = result["back_shift"]
        front_x, front_y = front_shift  # type: ignore[misc]
        back_x, back_y = back_shift  # type: ignore[misc]

        rows.append({
            "frame_no": int(frame_no),
            "file_name": file_name,
            "tube_pair": tube_pair,
            "bubble_index": int(bubble_index),
            "detection_index": int(detection_index),
            "diameter_mm": float(diameter_mm),
            "window_length_mm": float(window_length_mm),
            "tip_percentile": float(tip_percentile),
            "edge_margin_mm": float(edge_margin_mm),
            "min_tip_points": int(min_tip_points),
            "front_eccentricity": _none_float(result["front_eccentricity"]),
            "back_eccentricity": _none_float(result["back_eccentricity"]),
            "front_shift_x_mm": _none_float(front_x),
            "front_shift_y_mm": _none_float(front_y),
            "back_shift_x_mm": _none_float(back_x),
            "back_shift_y_mm": _none_float(back_y),
            "front_clipped": bool(result["front_clipped"]),
            "back_clipped": bool(result["back_clipped"]),
            "point_count": int(result["point_count"]),
        })

    return rows


def front_back_eccentricity_label_map(rows: list[dict[str, object]], tube_pair: str) -> dict[int, dict[str, object]]:
    """Return rows keyed by detection_index for 3D label merging."""
    out: dict[int, dict[str, object]] = {}
    for row in rows:
        if str(row.get("tube_pair", "")) != tube_pair:
            continue
        out[int(row.get("detection_index", row.get("bubble_index", 0)))] = row
    return out
