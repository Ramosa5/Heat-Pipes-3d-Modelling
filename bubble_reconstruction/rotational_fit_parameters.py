from __future__ import annotations

from typing import Any

import numpy as np

from .fit_score import rotational_fit_score
from .volume import volume_to_surface_points_mm


ROTATIONAL_FIT_PARAMETER_FIELDS = [
    "frame_no",
    "file_name",
    "tube_pair",
    "bubble_index",
    "detection_index",
    "component_iou",
    "filled_voxels",
    "surface_points",
    "rotational_fit_score",
    "rotational_fit_mean_error_mm",
    "rotational_fit_radius_mm",
    "rotational_fit_n_sections",
    "rotational_fit_min_points_per_section",
    "rotational_fit_radius_statistic",
]


def calculate_rotational_fit_for_bubbles(
    bubbles: list[dict[str, Any]],
    frame_no: int,
    file_name: str,
    tube_pair: str,
    center_radial_xy: bool = True,
    max_points: int = 500_000,
    n_sections: int = 50,
    min_points_per_section: int = 5,
    radius_statistic: str = "median",
) -> list[dict[str, object]]:
    """
    Calculate rotational-fit parameters for each reconstructed bubble.

    This is the pipeline version of the uploaded rotational-fit validation script:
    it does not rerun the old standalone analysis, but applies the same
    rotational_fit_score() parameter logic to every individual reconstructed
    bubble produced by the current modular pipeline.
    """
    rows: list[dict[str, object]] = []

    for bubble_index, bubble in enumerate(bubbles, start=1):
        volume = bubble.get("volume")
        voxel_mm = float(bubble.get("voxel_mm", 1.0))
        detection_index = int(bubble.get("detection_index", bubble_index))

        surface_points = volume_to_surface_points_mm(
            volume,
            voxel_mm,
            center_radial_xy=center_radial_xy,
            max_points=max_points,
        )

        if surface_points is None or len(surface_points) < 10:
            score, mean_error, radius = 0.0, None, None
            n_surface_points = 0
        else:
            score, mean_error, radius = rotational_fit_score(
                surface_points,
                n_sections=int(n_sections),
                min_points_per_section=int(min_points_per_section),
                radius_statistic=str(radius_statistic),
            )
            n_surface_points = int(len(surface_points))

        rows.append({
            "frame_no": int(frame_no),
            "file_name": file_name,
            "tube_pair": tube_pair,
            "bubble_index": int(bubble_index),
            "detection_index": int(detection_index),
            "component_iou": float(bubble.get("component_iou", 0.0)),
            "filled_voxels": int(bubble.get("volume_voxels", 0)),
            "surface_points": int(n_surface_points),
            "rotational_fit_score": float(score),
            "rotational_fit_mean_error_mm": None if mean_error is None else float(mean_error),
            "rotational_fit_radius_mm": None if radius is None else float(radius),
            "rotational_fit_n_sections": int(n_sections),
            "rotational_fit_min_points_per_section": int(min_points_per_section),
            "rotational_fit_radius_statistic": str(radius_statistic),
        })

    return rows


def rotational_fit_label_map(rows: list[dict[str, object]], tube_pair: str) -> dict[int, dict[str, object]]:
    """Return rows keyed by detection_index for 3D label merging."""
    out: dict[int, dict[str, object]] = {}
    for row in rows:
        if str(row.get("tube_pair", "")) != tube_pair:
            continue
        out[int(row.get("detection_index", row.get("bubble_index", 0)))] = row
    return out
