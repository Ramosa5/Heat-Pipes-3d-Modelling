from __future__ import annotations

import os
from typing import Any

import cv2

from .coco_utils import build_bubble_mask_from_anns, load_coco
from .config import ReconstructionConfig
from .export_io import safe_stem, save_mask, save_point_cloud_from_volume
from .frame_annotation import save_parameter_overlay_frame
from .fit_score import rotational_fit_score
from .bubble_physics import summarize_rectified_mask_parameters
from .eccentricity import eccentricity_records_for_points
from .parameter_export import (
    ECCENTRICITY_FIELDS,
    MASK_PARAMETER_FIELDS,
    TRACKING_FIELDS,
    append_dict_rows,
)
from .pipe import build_pipe_cylinder_mesh_from_rectified_shape
from .reconstruction import reconstruct_pair_no_stick_with_bubbles
from .rectification import rectify_and_align_pair
from .tube_geometry import point_in_tube
from .tracking import (
    BubbleTracker,
    detections_from_reconstructed_bubbles,
    tracking_label_records,
)
from .volume import volume_to_points_mm, volume_to_surface_points_mm


def find_category_id(categories: list[dict[str, Any]], category_name: str) -> int:
    """Return COCO category id for a given category name."""
    for category in categories:
        if category.get("name") == category_name:
            return int(category["id"])
    raise RuntimeError(f"Nie znaleziono kategorii '{category_name}'")


def select_images(images: list[dict[str, Any]], start_frame: int, n_frames: int) -> tuple[list[dict[str, Any]], int, int, int]:
    """Sort COCO image entries by file_name and select a 1-based frame range."""
    images_sorted = sorted(images, key=lambda d: d.get("file_name", ""))
    start_idx = max(0, int(start_frame) - 1)  # 1-based -> 0-based
    end_idx = min(len(images_sorted), start_idx + max(1, int(n_frames)))
    images_sel = images_sorted[start_idx:end_idx]
    return images_sel, start_idx, end_idx, len(images_sorted)


def split_annotations_by_tube(bubble_anns: list[dict[str, Any]],
                              tubes: list[dict[str, float]],
                              margin_px: float) -> dict[int, list[dict[str, Any]]]:
    """Assign each bubble annotation to the first tube containing the bbox centre."""
    tube_anns: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: [], 3: []}

    for ann in bubble_anns:
        bbox = ann.get("bbox", None)
        if bbox is None:
            continue

        x, y, w_box, h_box = map(float, bbox)
        xc = x + 0.5 * w_box
        yc = y + 0.5 * h_box

        for i_tube, tube in enumerate(tubes):
            if point_in_tube(xc, yc, tube, margin_px=margin_px):
                tube_anns[i_tube].append(ann)
                break

    return tube_anns


def print_rotational_fit(label: str, score: float, mean_error: float | None, radius: float | None) -> None:
    print(
        f"Rotational fit {label}: "
        f"score={score:.3f}, "
        f"mean_error={mean_error if mean_error is not None else 0:.3f} mm, "
        f"R={radius if radius is not None else 0:.3f} mm"
    )


def reconstruct_and_score_pair(rect_top,
                               rect_side,
                               config: ReconstructionConfig,
                               label: str):
    """Run per-bubble reconstruction and rotational fit score for one TOP/SIDE pair."""
    volume, voxel_mm, bubbles = reconstruct_pair_no_stick_with_bubbles(
        rect_top,
        rect_side,
        diameter_mm=config.diameter_mm,
        voxel_mm=config.voxel_mm,
        smooth_sigma_z=config.smooth_sigma_z,
        min_radius_vox=config.min_radius_vox,
        min_area_cc=config.min_area_cc,
        iou_thr=config.iou_thr,
    )

    surface_points = volume_to_surface_points_mm(volume, voxel_mm)
    score, mean_error, radius = rotational_fit_score(
        surface_points,
        n_sections=50,
        min_points_per_section=5,
    )
    print_rotational_fit(label, score, mean_error, radius)

    return volume, voxel_mm, len(bubbles), bubbles, score, mean_error, radius




def calculate_and_save_frame_parameters(file_name: str,
                                        frame_no: int,
                                        rect_top_12,
                                        rect_side_12,
                                        rect_top_34,
                                        rect_side_34,
                                        vol_12,
                                        voxel_mm_12: float,
                                        n_pairs_12: int,
                                        vol_34,
                                        voxel_mm_34: float,
                                        n_pairs_34: int,
                                        config: ReconstructionConfig) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Calculate eccentricity and mask-derived parameters for one processed frame."""
    if not config.eccentricity:
        return [], []

    eccentricity_rows: list[dict[str, object]] = []

    pts_12 = volume_to_points_mm(
        vol_12,
        voxel_mm_12,
        center_radial_xy=config.center_radial_xy,
        max_points=500_000,
    )
    eccentricity_rows.extend(
        eccentricity_records_for_points(
            points=pts_12,
            bubble_count=n_pairs_12,
            frame_no=frame_no,
            file_name=file_name,
            tube_pair="tube12",
            diameter_mm=config.diameter_mm,
            tip_percentile=config.tip_percentile,
            pipe_center=(0.0, 0.0),
            debug=config.eccentricity_debug,
        )
    )

    pts_34 = volume_to_points_mm(
        vol_34,
        voxel_mm_34,
        center_radial_xy=config.center_radial_xy,
        max_points=500_000,
    )
    eccentricity_rows.extend(
        eccentricity_records_for_points(
            points=pts_34,
            bubble_count=n_pairs_34,
            frame_no=frame_no,
            file_name=file_name,
            tube_pair="tube34",
            diameter_mm=config.diameter_mm,
            tip_percentile=config.tip_percentile,
            pipe_center=(0.0, 0.0),
            debug=config.eccentricity_debug,
        )
    )

    if config.eccentricity_visualize:
        every = max(1, int(config.eccentricity_visualize_every))
        if (frame_no - 1) % every == 0:
            from .visualization import visualize_bubble_tip

            visualize_bubble_tip(
                pts_12,
                bubble_count=n_pairs_12,
                tip_percentile=config.tip_percentile,
                title=f"Eccentricity visualization — frame {frame_no}, tube12",
                max_points=config.eccentricity_visualize_max_points,
            )
            visualize_bubble_tip(
                pts_34,
                bubble_count=n_pairs_34,
                tip_percentile=config.tip_percentile,
                title=f"Eccentricity visualization — frame {frame_no}, tube34",
                max_points=config.eccentricity_visualize_max_points,
            )

    if eccentricity_rows:
        append_dict_rows(
            os.path.join(config.parameters_dir, config.eccentricity_csv),
            eccentricity_rows,
            ECCENTRICITY_FIELDS,
        )
        for row in eccentricity_rows:
            print(
                f"[ECC] frame={row['frame_no']} {row['tube_pair']} "
                f"bubble={row['bubble_index']}/{row['bubble_count']} "
                f"e*={float(row['e_star']):.4f} "
                f"e_x={float(row['e_x_mm']):.3f} mm "
                f"e_y={float(row['e_y_mm']):.3f} mm"
            )
    else:
        print(f"[ECC] frame={frame_no}: no reconstructed bubble points, no eccentricity rows saved")

    mask_parameter_rows = [
        summarize_rectified_mask_parameters(rect_top_12, config.diameter_mm, "top_yz", "tube12", frame_no, file_name),
        summarize_rectified_mask_parameters(rect_side_12, config.diameter_mm, "side_xz", "tube12", frame_no, file_name),
        summarize_rectified_mask_parameters(rect_top_34, config.diameter_mm, "top_yz", "tube34", frame_no, file_name),
        summarize_rectified_mask_parameters(rect_side_34, config.diameter_mm, "side_xz", "tube34", frame_no, file_name),
    ]
    append_dict_rows(
        os.path.join(config.parameters_dir, config.mask_parameters_csv),
        mask_parameter_rows,
        MASK_PARAMETER_FIELDS,
    )

    return eccentricity_rows, mask_parameter_rows

def build_preview_parameter_labels(tracking_rows: list[dict[str, object]],
                                   eccentricity_rows: list[dict[str, object]],
                                   tube_pair: str) -> list[dict[str, object]]:
    """Build multi-line per-bubble labels for the 3D PyVista preview."""
    track_rows = [row for row in tracking_rows if str(row.get("tube_pair", "")) == tube_pair]
    ecc_rows = [row for row in eccentricity_rows if str(row.get("tube_pair", "")) == tube_pair]

    if not track_rows:
        return []

    track_rows = sorted(
        track_rows,
        key=lambda row: (float(row.get("centroid_z_mm", 0.0)), float(row.get("centroid_x_mm", 0.0)), float(row.get("centroid_y_mm", 0.0))),
    )
    ecc_rows = sorted(ecc_rows, key=lambda row: int(row.get("bubble_index", 0)))

    labels: list[dict[str, object]] = []
    for i, track_row in enumerate(track_rows):
        ecc_row = ecc_rows[i] if i < len(ecc_rows) else None
        lines = [
            f"ID {int(track_row['track_id'])}",
            f"det {int(track_row['detection_index'])}",
            f"z {float(track_row['centroid_z_mm']):.2f} mm",
            f"V {int(track_row['volume_voxels'])} vox",
        ]
        if ecc_row is not None:
            lines.extend([
                f"e* {float(ecc_row['e_star']):.3f}",
                f"e_x {float(ecc_row['e_x_mm']):.2f} mm",
                f"e_y {float(ecc_row['e_y_mm']):.2f} mm",
            ])

        labels.append({
            "position": (
                float(track_row["centroid_x_mm"]),
                float(track_row["centroid_y_mm"]),
                float(track_row["centroid_z_mm"]),
            ),
            "label": "\n".join(lines),
        })

    return labels


def update_tracking_for_frame(file_name: str,
                              frame_no: int,
                              bubbles_12: list[dict[str, Any]],
                              bubbles_34: list[dict[str, Any]],
                              tracker: BubbleTracker | None,
                              config: ReconstructionConfig) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Assign persistent track IDs to reconstructed bubbles from this frame."""
    if not config.tracking or tracker is None:
        return [], [], []

    tracking_rows: list[dict[str, object]] = []
    label_rows_12: list[dict[str, object]] = []
    label_rows_34: list[dict[str, object]] = []

    for tube_pair, bubbles in (("tube12", bubbles_12), ("tube34", bubbles_34)):
        detections = detections_from_reconstructed_bubbles(
            bubbles=bubbles,
            frame_no=frame_no,
            file_name=file_name,
            tube_pair=tube_pair,
            center_radial_xy=config.center_radial_xy,
        )
        rows, closed_ids = tracker.update(
            detections=detections,
            frame_no=frame_no,
            file_name=file_name,
            tube_pair=tube_pair,
        )
        tracking_rows.extend(rows)

        if tube_pair == "tube12":
            label_rows_12 = tracking_label_records(rows)
        else:
            label_rows_34 = tracking_label_records(rows)

        for row in rows:
            print(
                f"[TRACK] frame={row['frame_no']} {row['tube_pair']} "
                f"track={row['track_id']} det={row['detection_index']} "
                f"{row['match_status']} z={float(row['centroid_z_mm']):.2f} mm"
            )
        for closed_id in closed_ids:
            print(f"[TRACK] frame={frame_no} {tube_pair} closed track={closed_id}")

    if tracking_rows:
        append_dict_rows(
            os.path.join(config.parameters_dir, config.tracking_csv),
            tracking_rows,
            TRACKING_FIELDS,
        )
    else:
        print(f"[TRACK] frame={frame_no}: no detections, no tracking rows saved")

    return tracking_rows, label_rows_12, label_rows_34


def process_frame(img_info: dict[str, Any],
                  annotations: list[dict[str, Any]],
                  bubble_cat_id: int,
                  global_frame_no: int,
                  local_frame_no: int,
                  total_selected: int,
                  config: ReconstructionConfig,
                  tracker: BubbleTracker | None = None) -> dict[str, Any] | None:
    """Process one image frame and return data needed for preview/export."""
    img_id = img_info["id"]
    img_path = os.path.join(config.dataset_dir, img_info["file_name"])

    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"[WARN] Nie mogę wczytać: {img_path} — pomijam")
        return None

    h, w = gray.shape
    x_left = 0
    x_right = w - 1

    bubble_anns = [
        ann for ann in annotations
        if ann["image_id"] == img_id and ann["category_id"] == bubble_cat_id
    ]
    tube_anns = split_annotations_by_tube(bubble_anns, config.tubes, config.margin_px)

    mask1_orig = build_bubble_mask_from_anns(tube_anns[0], h, w)
    mask2_orig = build_bubble_mask_from_anns(tube_anns[1], h, w)
    rect_top_12, rect_side_12, _, _ = rectify_and_align_pair(
        mask1_orig,
        config.tubes[0],
        mask2_orig,
        config.tubes[1],
        x_left=x_left,
        x_right=x_right,
        inner_margin_px=config.inner_margin_px,
        keep_aspect=config.keep_aspect,
    )

    mask3_orig = build_bubble_mask_from_anns(tube_anns[2], h, w)
    mask4_orig = build_bubble_mask_from_anns(tube_anns[3], h, w)
    rect_top_34, rect_side_34, _, _ = rectify_and_align_pair(
        mask3_orig,
        config.tubes[2],
        mask4_orig,
        config.tubes[3],
        x_left=x_left,
        x_right=x_right,
        inner_margin_px=config.inner_margin_px,
        keep_aspect=config.keep_aspect,
    )

    file_stem = safe_stem(img_info["file_name"])

    if config.save_masks:
        # Zapis masek po każdym przetworzonym zdjęciu
        save_mask(mask1_orig, config.masks_dir, file_stem, "tube1_orig")
        save_mask(mask2_orig, config.masks_dir, file_stem, "tube2_orig")
        save_mask(mask3_orig, config.masks_dir, file_stem, "tube3_orig")
        save_mask(mask4_orig, config.masks_dir, file_stem, "tube4_orig")
        save_mask(rect_top_12, config.masks_dir, file_stem, "tube12_top_rect")
        save_mask(rect_side_12, config.masks_dir, file_stem, "tube12_side_rect")
        save_mask(rect_top_34, config.masks_dir, file_stem, "tube34_top_rect")
        save_mask(rect_side_34, config.masks_dir, file_stem, "tube34_side_rect")

    vol_12, voxel_mm_12, n_pairs_12, bubbles_12, *_ = reconstruct_and_score_pair(
        rect_top_12,
        rect_side_12,
        config,
        label="tube12",
    )

    vol_34, voxel_mm_34, n_pairs_34, bubbles_34, *_ = reconstruct_and_score_pair(
        rect_top_34,
        rect_side_34,
        config,
        label="tube34",
    )

    tracking_rows, track_labels_12, track_labels_34 = update_tracking_for_frame(
        file_name=img_info["file_name"],
        frame_no=global_frame_no,
        bubbles_12=bubbles_12,
        bubbles_34=bubbles_34,
        tracker=tracker,
        config=config,
    )

    eccentricity_rows, mask_parameter_rows = calculate_and_save_frame_parameters(
        file_name=img_info["file_name"],
        frame_no=global_frame_no,
        rect_top_12=rect_top_12,
        rect_side_12=rect_side_12,
        rect_top_34=rect_top_34,
        rect_side_34=rect_side_34,
        vol_12=vol_12,
        voxel_mm_12=voxel_mm_12,
        n_pairs_12=n_pairs_12,
        vol_34=vol_34,
        voxel_mm_34=voxel_mm_34,
        n_pairs_34=n_pairs_34,
        config=config,
    )

    parameter_labels_12 = build_preview_parameter_labels(tracking_rows, eccentricity_rows, "tube12") if config.preview_parameter_labels else []
    parameter_labels_34 = build_preview_parameter_labels(tracking_rows, eccentricity_rows, "tube34") if config.preview_parameter_labels else []

    if config.annotate_frame_parameters:
        masks_by_tube = {
            0: mask1_orig,
            1: mask2_orig,
            2: mask3_orig,
            3: mask4_orig,
        }
        save_parameter_overlay_frame(
            gray,
            out_dir=config.annotated_frames_dir,
            file_stem=file_stem,
            tubes=config.tubes,
            tube_anns=tube_anns,
            masks_by_tube=masks_by_tube,
            rect_shapes={
                "tube12": rect_top_12.shape,
                "tube34": rect_top_34.shape,
            },
            tracking_rows=tracking_rows,
            eccentricity_rows=eccentricity_rows,
            diameter_mm=config.diameter_mm,
            frame_no=global_frame_no,
            file_name=img_info["file_name"],
        )

    pipe_mesh_12 = None
    pipe_mesh_34 = None
    if config.show_preview:
        # Coordinate system after conversion:
        #   X,Y = radial circular cross-section, Z = long pipe axis.
        # Origin is at the centre of the circular cylinder face at z=0.
        pipe_mesh_12 = build_pipe_cylinder_mesh_from_rectified_shape(
            rect_top_12.shape,
            diameter_mm=config.diameter_mm,
            voxel_mm=voxel_mm_12,
            resolution=96,
            open_ends=True,
        )
        pipe_mesh_34 = build_pipe_cylinder_mesh_from_rectified_shape(
            rect_top_34.shape,
            diameter_mm=config.diameter_mm,
            voxel_mm=voxel_mm_34,
            resolution=96,
            open_ends=True,
        )

    if config.save_point_clouds:
        # Zapis chmur punktów po stworzeniu każdej chmury punktów
        save_point_cloud_from_volume(vol_12, voxel_mm_12, config.point_clouds_dir, file_stem, "tube12")
        save_point_cloud_from_volume(vol_34, voxel_mm_34, config.point_clouds_dir, file_stem, "tube34")

    print(
        f"[frame {global_frame_no} ({local_frame_no}/{total_selected})] {img_info['file_name']} | "
        f"pairs12={n_pairs_12} filled12={int(vol_12.sum())} | "
        f"pairs34={n_pairs_34} filled34={int(vol_34.sum())}"
    )

    return {
        "title": f"Frame {global_frame_no}: {img_info['file_name']}",
        "vol_12": vol_12,
        "vox_12": voxel_mm_12,
        "pipe_mesh_12": pipe_mesh_12,
        "track_labels_12": track_labels_12,
        "parameter_labels_12": parameter_labels_12,
        "vol_34": vol_34,
        "vox_34": voxel_mm_34,
        "pipe_mesh_34": pipe_mesh_34,
        "track_labels_34": track_labels_34,
        "parameter_labels_34": parameter_labels_34,
    }


def run_pipeline(config: ReconstructionConfig) -> list[dict[str, Any]]:
    """Orchestrate the full reconstruction workflow."""
    coco = load_coco(os.path.join(config.dataset_dir, config.coco_file))
    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    bubble_cat_id = find_category_id(categories, "bubble")
    images_sel, start_idx, end_idx, total_images = select_images(
        images,
        config.start_frame,
        config.n_frames,
    )

    print(f"Start frame (1-based) = {config.start_frame}  -> idx={start_idx}")
    print(f"Biorę klatki: [{start_idx}:{end_idx}] z {total_images} total")
    if not images_sel:
        raise RuntimeError("Zakres start_frame/n_frames poza listą obrazów")

    frames_data: list[dict[str, Any]] = []
    tracker = BubbleTracker(
        max_distance_mm=config.tracking_max_distance_mm,
        max_missing_frames=config.tracking_max_missing_frames,
        debug=config.tracking_debug,
    ) if config.tracking else None

    for local_i, img_info in enumerate(images_sel, start=0):
        global_i = start_idx + local_i + 1  # 1-based do logu
        frame_data = process_frame(
            img_info=img_info,
            annotations=annotations,
            bubble_cat_id=bubble_cat_id,
            global_frame_no=global_i,
            local_frame_no=local_i + 1,
            total_selected=len(images_sel),
            config=config,
            tracker=tracker,
        )
        if frame_data is not None:
            frames_data.append(frame_data)

    if not frames_data:
        raise RuntimeError("Brak poprawnie przetworzonych klatek (nie wczytano obrazów?)")

    if config.show_preview:
        from .visualization import pv_live_animate_keep_last

        # Live preview: animates + KEEPS LAST FRAME until you close the window
        pv_live_animate_keep_last(
            frames_data,
            center_radial_xy=config.center_radial_xy,
            max_points=config.preview_max_points,
            point_size=config.preview_point_size,
            pause_s=config.preview_pause_s,
            zoom_out=config.preview_zoom_out,
            show_pipe=config.show_pipe,
            pipe_opacity=config.pipe_opacity,
            center_view_on_origin=config.center_view_on_origin,
            show_origin_marker=config.show_origin_marker,
            show_tracking_labels=config.tracking_labels,
            show_parameter_labels=config.preview_parameter_labels,
        )

    return frames_data
