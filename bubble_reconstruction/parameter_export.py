from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable


ECCENTRICITY_FIELDS = [
    "frame_no",
    "file_name",
    "tube_pair",
    "bubble_index",
    "bubble_count",
    "e_star",
    "e_x_mm",
    "e_y_mm",
    "tip_percentile",
    "diameter_mm",
]

TRACKING_FIELDS = [
    "frame_no",
    "file_name",
    "tube_pair",
    "track_id",
    "detection_index",
    "match_status",
    "match_distance_mm",
    "track_age_frames",
    "missed_frames",
    "centroid_x_mm",
    "centroid_y_mm",
    "centroid_z_mm",
    "z_min_mm",
    "z_max_mm",
    "volume_voxels",
    "point_count",
]


MASK_PARAMETER_FIELDS = [
    "frame_no",
    "file_name",
    "tube_pair",
    "view",
    "diameter_mm",
    "mm_per_pixel",
    "columns_total",
    "columns_nonzero",
    "avg_height_mm",
    "max_height_mm",
    "avg_alpha_filled",
    "avg_alpha_empty",
    "avg_s_filled_mm",
    "avg_s_interface_mm",
]


def append_dict_rows(path: str | os.PathLike[str], rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    rows = list(rows)
    if not rows:
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0

    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"[SAVE] Appended {len(rows)} parameter rows: {out_path}")


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
