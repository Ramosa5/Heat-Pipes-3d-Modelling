from __future__ import annotations

import numpy as np

from .components import connected_components, match_components_by_longitudinal_overlap
from .volume import build_volume_elliptic_from_two_masks

def reconstruct_pair_no_stick(rect_top: np.ndarray,
                              rect_side: np.ndarray,
                              diameter_mm: float,
                              voxel_mm: float,
                              smooth_sigma_z: float,
                              min_radius_vox: float,
                              min_area_cc: int = 80,
                              iou_thr: float = 0.15):
    top_comps = connected_components(rect_top, min_area=min_area_cc)
    side_comps = connected_components(rect_side, min_area=min_area_cc)

    pairs = match_components_by_longitudinal_overlap(top_comps, side_comps, iou_thr=iou_thr)

    vol_out = None
    voxel_out = None

    for mt, ms, iou in pairs:
        v, _, vox = build_volume_elliptic_from_two_masks(
            mt, ms,
            diameter_mm=diameter_mm,
            voxel_mm=voxel_mm,
            smooth_sigma_z=smooth_sigma_z,
            min_radius_vox=min_radius_vox
        )
        if vol_out is None:
            vol_out = v
            voxel_out = vox
        else:
            # ensure same shape (should be, because same H,W -> same vol dims)
            if v.shape == vol_out.shape:
                vol_out |= v
            else:
                # rare: shape mismatch due to rounding (very unlikely). pad to max.
                X_rad = max(vol_out.shape[0], v.shape[0])
                Y_rad = max(vol_out.shape[1], v.shape[1])
                Z_len = max(vol_out.shape[2], v.shape[2])
                vv = np.zeros((X_rad, Y_rad, Z_len), dtype=bool)
                oo = np.zeros((X_rad, Y_rad, Z_len), dtype=bool)
                oo[:vol_out.shape[0], :vol_out.shape[1], :vol_out.shape[2]] = vol_out
                vv[:v.shape[0], :v.shape[1], :v.shape[2]] = v
                vol_out = oo | vv
                voxel_out = vox  # close enough

    if vol_out is None:
        # empty result
        vol_out = np.zeros((1, 1, 1), dtype=bool)
        voxel_out = (diameter_mm / float(rect_top.shape[0])) if rect_top.shape[0] > 0 else 1.0

    return vol_out, voxel_out, len(pairs)
