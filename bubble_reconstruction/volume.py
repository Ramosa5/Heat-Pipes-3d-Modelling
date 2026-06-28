from __future__ import annotations

import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

def _segments_from_valid(valid: np.ndarray, min_len: int = 3):
    segs = []
    n = valid.size
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        if (j - i) >= min_len:
            segs.append((i, j))
        i = j
    return segs


def _fill_missing_1d_within_segments(arr: np.ndarray, valid: np.ndarray):
    out = arr.astype(np.float32).copy()
    x = np.arange(arr.size)

    segs = _segments_from_valid(valid, min_len=2)
    for l, r in segs:
        v = valid[l:r]
        if v.sum() == 0:
            continue
        xx = x[l:r]
        yy = out[l:r]
        yy[~v] = np.interp(xx[~v], xx[v], yy[v])
        out[l:r] = yy
    return out


def _profile_from_mask_columns(mask_2d: np.ndarray, p_lo=5, p_hi=95):
    """
    mask_2d: uint8 [H,W], rows=vertical axis
    returns: center[W], half[W], valid[W]
    robust: uses percentiles instead of min/max (less tails)
    """
    H, W = mask_2d.shape
    center = np.zeros(W, dtype=np.float32)
    half = np.zeros(W, dtype=np.float32)
    valid = np.zeros(W, dtype=bool)

    for x in range(W):
        ys = np.where(mask_2d[:, x] > 0)[0]
        if ys.size == 0:
            center[x] = (H - 1) / 2.0
            half[x] = 0.0
            valid[x] = False
            continue

        # robust bounds
        y0 = int(np.percentile(ys, p_lo))
        y1 = int(np.percentile(ys, p_hi))
        if y1 < y0:
            y0, y1 = y1, y0

        center[x] = 0.5 * (y0 + y1)
        half[x] = 0.5 * (y1 - y0)
        valid[x] = True

    return center, half, valid


def build_volume_elliptic_from_two_masks(mask_top_yz: np.ndarray,
                                        mask_side_xz: np.ndarray,
                                        diameter_mm: float,
                                        voxel_mm: float = None,
                                        smooth_sigma_z: float = 2.0,
                                        min_radius_vox: float = 0.8):
    """
    Builds a 3D bubble volume using the SAME coordinate convention as the view/export.

    Required convention used everywhere after rectification:
    - X: first radial direction of the circular pipe cross-section,
    - Y: second radial direction of the circular pipe cross-section,
    - Z: long pipe/cylinder axis.

    Returned volume layout is therefore:
        volume[x_radial_index, y_radial_index, z_length_index]

    The 2D masks are rectified tube silhouettes:
    - mask_side_xz columns follow the pipe length, rows describe X radial direction,
    - mask_top_yz  columns follow the pipe length, rows describe Y radial direction.
    """
    if mask_top_yz.ndim != 2 or mask_side_xz.ndim != 2:
        raise ValueError("maski muszą być 2D")
    H, W = mask_top_yz.shape
    if mask_side_xz.shape != (H, W):
        raise ValueError("mask_top i mask_side muszą mieć ten sam rozmiar po pipeline")

    mm_per_px = diameter_mm / float(H)
    if voxel_mm is None:
        voxel_mm = mm_per_px

    R_vox = int(round((diameter_mm / 2.0) / voxel_mm))
    radial_size = 2 * R_vox + 1
    z_length_size = max(1, int(round((W * mm_per_px) / voxel_mm)))
    z_map = np.linspace(0, z_length_size - 1, W).round().astype(int)

    cx = R_vox
    cy = R_vox

    # TOP view -> Y profile (rows are radial Y, columns are length/Z)
    yc_px, yhalf_px, yvalid = _profile_from_mask_columns(mask_top_yz)
    # SIDE view -> X profile (rows are radial X, columns are length/Z)
    xc_px, xhalf_px, xvalid = _profile_from_mask_columns(mask_side_xz)

    valid_both = xvalid & yvalid

    # Fill only INSIDE valid longitudinal segments (no bridging between bubbles)
    xc_px    = _fill_missing_1d_within_segments(xc_px, valid_both)
    yc_px    = _fill_missing_1d_within_segments(yc_px, valid_both)
    xhalf_px = _fill_missing_1d_within_segments(xhalf_px, valid_both)
    yhalf_px = _fill_missing_1d_within_segments(yhalf_px, valid_both)

    # Critical: outside valid -> radii 0 => no fill => no centre strip
    xhalf_px[~valid_both] = 0.0
    yhalf_px[~valid_both] = 0.0

    px_center = (H - 1) / 2.0
    x0_mm = (xc_px - px_center) * mm_per_px
    y0_mm = (yc_px - px_center) * mm_per_px

    x0_vox = np.round(x0_mm / voxel_mm).astype(np.int32) + cx
    y0_vox = np.round(y0_mm / voxel_mm).astype(np.int32) + cy

    a_vox = np.maximum(0.0, (xhalf_px * mm_per_px) / voxel_mm).astype(np.float32)  # X semi-axis
    b_vox = np.maximum(0.0, (yhalf_px * mm_per_px) / voxel_mm).astype(np.float32)  # Y semi-axis

    # Smooth along the pipe length, i.e. along the Z direction.
    if smooth_sigma_z and smooth_sigma_z > 0:
        if HAS_SCIPY:
            a_vox = gaussian_filter1d(a_vox, sigma=smooth_sigma_z).astype(np.float32)
            b_vox = gaussian_filter1d(b_vox, sigma=smooth_sigma_z).astype(np.float32)
            x0_vox = np.round(gaussian_filter1d(x0_vox.astype(np.float32), sigma=smooth_sigma_z)).astype(np.int32)
            y0_vox = np.round(gaussian_filter1d(y0_vox.astype(np.float32), sigma=smooth_sigma_z)).astype(np.int32)
        else:
            k = int(max(3, 2 * round(smooth_sigma_z) + 1))
            k = k if (k % 2 == 1) else k + 1
            ker = np.ones(k, dtype=np.float32) / k
            a_vox = np.convolve(a_vox, ker, mode="same").astype(np.float32)
            b_vox = np.convolve(b_vox, ker, mode="same").astype(np.float32)

    # Apply min radius only where we have non-zero after smoothing.
    a_vox = np.where(a_vox > 0.0, np.maximum(a_vox, min_radius_vox), 0.0).astype(np.float32)
    b_vox = np.where(b_vox > 0.0, np.maximum(b_vox, min_radius_vox), 0.0).astype(np.float32)

    # SAME LAYOUT AS VIEW/EXPORT: [X_radial, Y_radial, Z_length]
    vol = np.zeros((radial_size, radial_size, z_length_size), dtype=bool)

    xx, yy = np.meshgrid(np.arange(radial_size), np.arange(radial_size), indexing="ij")
    circular_pipe_cross_section = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (R_vox ** 2)

    for z_px in range(W):
        # if no object here -> skip
        if a_vox[z_px] <= 0.0 or b_vox[z_px] <= 0.0:
            continue

        zv = int(z_map[z_px])

        x0 = int(np.clip(x0_vox[z_px], 0, radial_size - 1))
        y0 = int(np.clip(y0_vox[z_px], 0, radial_size - 1))

        a = float(a_vox[z_px])
        b = float(b_vox[z_px])

        ell = (((xx - x0) / a) ** 2 + ((yy - y0) / b) ** 2) <= 1.0
        vol[:, :, zv] = circular_pipe_cross_section & ell

    return vol, mm_per_px, voxel_mm


def volume_to_points_mm(volume_bool: np.ndarray, voxel_mm: float, center_radial_xy: bool = True,
                        max_points: int = 500_000):
    """
    Converts a reconstructed volume to points in millimetres.

    SAME COORDINATE SYSTEM AS PROCESSING AND VIEW:
    - origin: centre of the circular cylinder face at the beginning of the pipe,
    - X axis: first radial direction of the circular pipe cross-section,
    - Y axis: second radial direction of the circular pipe cross-section,
    - Z axis: long cylinder / pipe length.

    Input volume layout is already:
        volume[x_radial_index, y_radial_index, z_length_index]

    Therefore no axis remapping is performed here.
    """
    pts = np.argwhere(volume_bool)
    if pts.size == 0:
        return None

    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]

    X_rad, Y_rad, Z_len = volume_bool.shape
    cx = (X_rad - 1) / 2.0
    cy = (Y_rad - 1) / 2.0

    x_radial = pts[:, 0].astype(np.float32)
    y_radial = pts[:, 1].astype(np.float32)
    z_length = pts[:, 2].astype(np.float32) * voxel_mm

    if center_radial_xy:
        x_radial = (x_radial - cx) * voxel_mm
        y_radial = (y_radial - cy) * voxel_mm
    else:
        x_radial = x_radial * voxel_mm
        y_radial = y_radial * voxel_mm

    return np.c_[x_radial, y_radial, z_length]


def volume_to_surface_points_mm(volume_bool: np.ndarray,
                                voxel_mm: float,
                                center_radial_xy: bool = True,
                                max_points: int = 500_000):

    if volume_bool is None or volume_bool.sum() == 0:
        return None

    try:
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(volume_bool)
    except Exception:
        # Fallback without SciPy: simple erosion using OpenCV slice by slice
        eroded = volume_bool.copy()

        # Remove boundary voxels by checking six direct neighbours
        inner = np.zeros_like(volume_bool, dtype=bool)
        inner[1:-1, 1:-1, 1:-1] = (
            volume_bool[1:-1, 1:-1, 1:-1] &
            volume_bool[:-2, 1:-1, 1:-1] &
            volume_bool[2:, 1:-1, 1:-1] &
            volume_bool[1:-1, :-2, 1:-1] &
            volume_bool[1:-1, 2:, 1:-1] &
            volume_bool[1:-1, 1:-1, :-2] &
            volume_bool[1:-1, 1:-1, 2:]
        )
        eroded = inner

    surface = volume_bool & ~eroded

    return volume_to_points_mm(
        surface,
        voxel_mm,
        center_radial_xy=center_radial_xy,
        max_points=max_points
    )
