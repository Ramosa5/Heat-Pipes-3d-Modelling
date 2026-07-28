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

    # Optional 2D frame output with per-bubble parameters drawn on the original image.
    # Enabled from CLI with --annotate-frame-parameters.
    annotate_frame_parameters: bool = False
    annotated_frames_dir: str = "annotated_frames"

    # Optional frame-by-frame bubble tracking.
    # Enabled from CLI with --tracking.
    tracking: bool = False
    tracking_csv: str = "tracking_parameters.csv"
    tracking_max_distance_mm: float = 12.0
    tracking_max_missing_frames: int = 0
    tracking_debug: bool = False
    tracking_labels: bool = False
    # Optional rich per-bubble parameter labels directly in the 3D PyVista preview.
    # Enabled from CLI with --preview-parameter-labels.
    preview_parameter_labels: bool = False

    # Optional frame-by-frame eccentricity/parameter calculation.
    # Enabled from CLI with --eccentricity.
    eccentricity: bool = False
    parameters_dir: str = "parameters"
    eccentricity_csv: str = "eccentricity_parameters.csv"
    mask_parameters_csv: str = "mask_parameters.csv"
    tip_percentile: float = 99.0
    eccentricity_debug: bool = False
    # Optional PyVista visualization of eccentricity tip regions.
    # Enabled from CLI with --eccentricity-visualize.
    eccentricity_visualize: bool = False
    eccentricity_visualize_every: int = 1
    eccentricity_visualize_max_points: int = 200_000

    # Optional per-bubble rotational-fit parameters.
    # Enabled from CLI with --rotational-fit-parameters.
    rotational_fit_parameters: bool = False
    rotational_fit_csv: str = "rotational_fit_parameters.csv"
    rotational_fit_n_sections: int = 50
    rotational_fit_min_points_per_section: int = 5
    rotational_fit_radius_statistic: str = "median"
    rotational_fit_max_surface_points: int = 500_000

    # Optional front/back eccentricity parameter from the uploaded validation code.
    # Enabled from CLI with --front-back-eccentricity.
    front_back_eccentricity: bool = False
    front_back_eccentricity_csv: str = "front_back_eccentricity_parameters.csv"
    front_back_edge_margin_mm: float = 0.1
    front_back_min_tip_points: int = 20
    front_back_max_points: int = 500_000

    center_radial_xy: bool = True
    preview_max_points: int = 200_000
    preview_point_size: float = 3.0
    preview_pause_s: float = 0.25
    preview_zoom_out: float = 1.8
    # Additional camera distance multiplier for the PyVista preview.
    # Higher values make the pipes/bubbles start smaller, like scrolling backward in the viewer.
    preview_distance_scale: float = 1.0
    # Preview window/display options.
    # --preview-fullscreen tries to start the PyVista/VTK window in fullscreen.
    # --preview-compact reduces visual clutter: smaller labels, smaller points and more zoom-out.
    preview_fullscreen: bool = False
    preview_compact: bool = False
    preview_window_width: int = 1500
    preview_window_height: int = 720
    show_pipe: bool = True
    pipe_opacity: float = 0.5
    center_view_on_origin: bool = True
    show_origin_marker: bool = True

    # End-of-processing summary window: top = original video frame, middle = 3D pipes,
    # bottom = e(t) and eta(t) time-series for all tracked bubbles.
    summary_visualization: bool = False
    summary_pause_s: float = 0.20
    summary_pipe_max_points: int = 120_000
