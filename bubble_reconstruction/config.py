from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def default_tubes() -> list[dict[str, float]]:
    """Default tube definitions: tube1=TOP, tube2=SIDE, tube3=TOP, tube4=SIDE."""
    return [
        {"a_top": 0.007, "b_top": 50,  "a_bot": 0.007, "b_bot": 100},
        {"a_top": 0.019, "b_top": 105, "a_bot": 0.019, "b_bot": 155},
        {"a_top": 0.005, "b_top": 322, "a_bot": 0.005, "b_bot": 380},
        {"a_top": 0.005, "b_top": 408, "a_bot": 0.005, "b_bot": 470},
    ]


@dataclass
class ReconstructionConfig:
    """
    Central configuration object for the reconstruction pipeline.

    main.py should only create/update this object and call run_pipeline(config).
    The reconstruction logic lives in the package modules.
    """

    dataset_dir: str = "bubble.coco/train"
    coco_file: str = "_annotations.coco.json"
    start_frame: int = 100
    n_frames: int = 10

    save_masks: bool = False
    save_point_clouds: bool = False
    show_preview: bool = True

    tubes: list[dict[str, float]] = field(default_factory=default_tubes)

    margin_px: float = 2.0
    inner_margin_px: float = 2.0
    keep_aspect: bool = True

    diameter_mm: float = 20.0
    voxel_mm: Optional[float] = None
    smooth_sigma_z: float = 2.0
    min_radius_vox: float = 0.8

    min_area_cc: int = 80
    iou_thr: float = 0.15

    masks_dir: str = "masks"
    point_clouds_dir: str = "point_clouds"

    center_radial_xy: bool = True
    preview_max_points: int = 200_000
    preview_point_size: float = 3.0
    preview_pause_s: float = 0.25
    preview_zoom_out: float = 1.8
    show_pipe: bool = True
    pipe_opacity: float = 0.5
    center_view_on_origin: bool = True
    show_origin_marker: bool = True
