from __future__ import annotations

import os
from typing import Any

import cv2

from .coco_utils import build_bubble_mask_from_anns, load_coco
from .config import ReconstructionConfig
from .export_io import safe_stem, save_mask, save_point_cloud_from_volume
from .fit_score import rotational_fit_score
from .pipe import build_pipe_cylinder_mesh_from_rectified_shape
from .reconstruction import reconstruct_pair_no_stick
from .rectification import rectify_and_align_pair
from .tube_geometry import point_in_tube
from .volume import volume_to_surface_points_mm


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
    volume, voxel_mm, n_pairs = reconstruct_pair_no_stick(
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

    return volume, voxel_mm, n_pairs, score, mean_error, radius


def process_frame(img_info: dict[str, Any],
                  annotations: list[dict[str, Any]],
                  bubble_cat_id: int,
                  global_frame_no: int,
                  local_frame_no: int,
                  total_selected: int,
                  config: ReconstructionConfig) -> dict[str, Any] | None:
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

    vol_12, voxel_mm_12, n_pairs_12, *_ = reconstruct_and_score_pair(
        rect_top_12,
        rect_side_12,
        config,
        label="tube12",
    )

    vol_34, voxel_mm_34, n_pairs_34, *_ = reconstruct_and_score_pair(
        rect_top_34,
        rect_side_34,
        config,
        label="tube34",
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
        "vol_34": vol_34,
        "vox_34": voxel_mm_34,
        "pipe_mesh_34": pipe_mesh_34,
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
        )

    return frames_data
