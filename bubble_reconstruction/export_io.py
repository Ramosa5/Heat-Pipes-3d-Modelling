from __future__ import annotations

import os

import cv2
import numpy as np

from .volume import volume_to_points_mm


def safe_stem(file_name: str) -> str:
    return os.path.splitext(os.path.basename(file_name))[0]


def save_mask(mask: np.ndarray, out_dir: str, file_stem: str, suffix: str):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{file_stem}_{suffix}.png")

    # TUTAJ DZIEJE SIĘ ZAPIS MASKI DO FOLDERU masks
    print(f"[SAVE] Zapisywana jest maska: {out_path}")
    cv2.imwrite(out_path, mask)


def save_point_cloud_from_volume(volume_bool: np.ndarray,
                                 voxel_mm: float,
                                 out_dir: str,
                                 file_stem: str,
                                 suffix: str,
                                 center_radial_xy: bool = True,
                                 max_points: int = 500_000):
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("PyVista is required for PLY export. Install it with: pip install pyvista") from exc

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{file_stem}_{suffix}.ply")

    pts = volume_to_points_mm(
        volume_bool,
        voxel_mm,
        center_radial_xy=center_radial_xy,
        max_points=max_points
    )

    if pts is None or pts.size == 0:
        pts = np.empty((0, 3), dtype=np.float32)
    else:
        pts = pts.astype(np.float32, copy=False)

    # TUTAJ DZIEJE SIĘ ZAPIS CHMURY PUNKTÓW DO FOLDERU point_clouds
    print(f"[SAVE] Zapisywana jest chmura punktów: {out_path}")
    pv.PolyData(pts).save(out_path)
