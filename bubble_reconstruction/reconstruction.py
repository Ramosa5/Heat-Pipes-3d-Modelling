from __future__ import annotations

import numpy as np

from .components import connected_components, match_components_by_longitudinal_overlap
from .volume import build_volume_elliptic_from_two_masks


def _merge_volume(volume_out: np.ndarray | None, new_volume: np.ndarray) -> np.ndarray:
    """OR two boolean volumes, padding if rare rounding differences create shape mismatch."""
    if volume_out is None:
        return new_volume.copy()

    if new_volume.shape == volume_out.shape:
        return volume_out | new_volume

    x_rad = max(volume_out.shape[0], new_volume.shape[0])
    y_rad = max(volume_out.shape[1], new_volume.shape[1])
    z_len = max(volume_out.shape[2], new_volume.shape[2])

    merged_old = np.zeros((x_rad, y_rad, z_len), dtype=bool)
    merged_new = np.zeros((x_rad, y_rad, z_len), dtype=bool)
    merged_old[:volume_out.shape[0], :volume_out.shape[1], :volume_out.shape[2]] = volume_out
    merged_new[:new_volume.shape[0], :new_volume.shape[1], :new_volume.shape[2]] = new_volume
    return merged_old | merged_new


def reconstruct_pair_no_stick_with_bubbles(rect_top: np.ndarray,
                                           rect_side: np.ndarray,
                                           diameter_mm: float,
                                           voxel_mm: float,
                                           smooth_sigma_z: float,
                                           min_radius_vox: float,
                                           min_area_cc: int = 80,
                                           iou_thr: float = 0.15):
    """
    Reconstruct a TOP/SIDE pair and keep individual bubble volumes.

    This is the tracking-friendly variant. It preserves the original behaviour
    by also returning a merged volume, but additionally returns a list of
    per-bubble dictionaries. Each dictionary contains the individual volume and
    the component-matching metadata needed later by the tracker.
    """
    top_comps = connected_components(rect_top, min_area=min_area_cc)
    side_comps = connected_components(rect_side, min_area=min_area_cc)
    pairs = match_components_by_longitudinal_overlap(top_comps, side_comps, iou_thr=iou_thr)

    vol_out: np.ndarray | None = None
    voxel_out: float | None = None
    bubbles: list[dict[str, object]] = []

    for detection_index, (mt, ms, iou) in enumerate(pairs, start=1):
        volume, _, voxel_used = build_volume_elliptic_from_two_masks(
            mt,
            ms,
            diameter_mm=diameter_mm,
            voxel_mm=voxel_mm,
            smooth_sigma_z=smooth_sigma_z,
            min_radius_vox=min_radius_vox,
        )
        vol_out = _merge_volume(vol_out, volume)
        voxel_out = voxel_used
        bubbles.append({
            "detection_index": detection_index,
            "volume": volume,
            "voxel_mm": voxel_used,
            "component_iou": float(iou),
            "volume_voxels": int(volume.sum()),
        })

    if vol_out is None:
        vol_out = np.zeros((1, 1, 1), dtype=bool)
        voxel_out = (diameter_mm / float(rect_top.shape[0])) if rect_top.shape[0] > 0 else 1.0

    return vol_out, float(voxel_out), bubbles


def reconstruct_pair_no_stick(rect_top: np.ndarray,
                              rect_side: np.ndarray,
                              diameter_mm: float,
                              voxel_mm: float,
                              smooth_sigma_z: float,
                              min_radius_vox: float,
                              min_area_cc: int = 80,
                              iou_thr: float = 0.15):
    """Backward-compatible wrapper returning the merged volume and pair count."""
    volume, voxel_out, bubbles = reconstruct_pair_no_stick_with_bubbles(
        rect_top=rect_top,
        rect_side=rect_side,
        diameter_mm=diameter_mm,
        voxel_mm=voxel_mm,
        smooth_sigma_z=smooth_sigma_z,
        min_radius_vox=min_radius_vox,
        min_area_cc=min_area_cc,
        iou_thr=iou_thr,
    )
    return volume, voxel_out, len(bubbles)
