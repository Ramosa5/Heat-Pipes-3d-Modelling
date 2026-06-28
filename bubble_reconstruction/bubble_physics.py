from __future__ import annotations

import numpy as np


def calc_theta(h, D):
    """
    Calculate the internal angle theta for a local filled height h in a pipe of diameter D.

    h can be a scalar or a NumPy array. Units of h and D must be consistent.
    """
    h_ratio = np.asarray(h, dtype=np.float64) / float(D)
    val = np.clip(1.0 - 2.0 * h_ratio, -1.0, 1.0)
    return 2.0 * np.arccos(val)


def calc_fractions(h, D):
    """Calculate filled and remaining cross-section fractions from local height h."""
    theta = calc_theta(h, D)
    alpha_filled = (theta - np.sin(theta)) / (2.0 * np.pi)
    alpha_empty = 1.0 - alpha_filled
    return alpha_filled, alpha_empty


def calc_perimeters(h, D):
    """Calculate wetted/perimeter-like arc length and interface chord length."""
    theta = calc_theta(h, D)
    s_filled = (theta * float(D)) / 2.0
    s_interface = float(D) * np.sin(theta / 2.0)
    return s_filled, s_interface


def compute_liquid_height(mask: np.ndarray) -> int:
    ys = np.where(mask > 0)[0]
    if len(ys) == 0:
        return 0
    return int(ys.max() - ys.min())


def compute_local_heights(mask: np.ndarray) -> np.ndarray:
    """Return local white-region height for every column of a binary mask."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    heights = np.zeros(mask.shape[1], dtype=np.float64)
    for x in range(mask.shape[1]):
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) > 0:
            heights[x] = ys.max() - ys.min() + 1
    return heights


def summarize_rectified_mask_parameters(
    mask: np.ndarray,
    diameter_mm: float,
    view_name: str,
    tube_pair: str,
    frame_no: int,
    file_name: str,
) -> dict[str, object]:
    """
    Compute frame-level mask parameters using the formulas from 20260301_bubble_physics.py.

    The rectified mask height is treated as the pipe diameter, so mm_per_pixel = D / H.
    The white region is the reconstructed bubble/silhouette region in that view.
    """
    if diameter_mm <= 0:
        raise ValueError("diameter_mm must be positive")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    H_px = max(1, int(mask.shape[0]))
    mm_per_pixel = float(diameter_mm) / float(H_px)

    heights_px = compute_local_heights(mask)
    heights_mm = np.clip(heights_px * mm_per_pixel, 0.0, float(diameter_mm))

    alpha_filled, alpha_empty = calc_fractions(heights_mm, diameter_mm)
    s_filled, s_interface = calc_perimeters(heights_mm, diameter_mm)

    nonzero = heights_mm > 0
    return {
        "frame_no": frame_no,
        "file_name": file_name,
        "tube_pair": tube_pair,
        "view": view_name,
        "diameter_mm": float(diameter_mm),
        "mm_per_pixel": mm_per_pixel,
        "columns_total": int(mask.shape[1]),
        "columns_nonzero": int(nonzero.sum()),
        "avg_height_mm": float(np.mean(heights_mm)) if heights_mm.size else 0.0,
        "max_height_mm": float(np.max(heights_mm)) if heights_mm.size else 0.0,
        "avg_alpha_filled": float(np.mean(alpha_filled)) if np.size(alpha_filled) else 0.0,
        "avg_alpha_empty": float(np.mean(alpha_empty)) if np.size(alpha_empty) else 0.0,
        "avg_s_filled_mm": float(np.mean(s_filled)) if np.size(s_filled) else 0.0,
        "avg_s_interface_mm": float(np.mean(s_interface)) if np.size(s_interface) else 0.0,
    }
