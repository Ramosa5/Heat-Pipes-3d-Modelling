from __future__ import annotations

import numpy as np


EccentricityResult = tuple[float, float, float]


def calculate_bubble_eccentricity(
    points: np.ndarray,
    bubble_count: int = 1,
    pipe_center: tuple[float, float] = (0.0, 0.0),
    diameter: float = 25.0,
    tip_percentile: float = 99.0,
    debug: bool = False,
) -> list[EccentricityResult]:
    """
    Calculate dimensionless tip eccentricity for one or more bubbles.

    Coordinate system used by this project:
    - X: first radial direction of the pipe cross-section,
    - Y: second radial direction of the pipe cross-section,
    - Z: pipe axis / flow direction.

    The implementation is based on the uploaded 20260531_parameters.py file,
    but moved into a clean importable module.
    """
    if points is None or len(points) == 0:
        raise ValueError("The input point cloud is missing.")
    if bubble_count < 1:
        raise ValueError(f"bubble_count must be at least 1, got {bubble_count}")
    if not (0 < tip_percentile < 100):
        raise ValueError(f"tip_percentile must be between 0 and 100, got {tip_percentile}")
    if diameter <= 0:
        raise ValueError(f"diameter must be positive, got {diameter}")

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")

    x_idx, y_idx, z_idx = 0, 1, 2

    flow_coords = pts[:, z_idx]
    sort_idx = np.argsort(flow_coords)
    sorted_flow_coords = flow_coords[sort_idx]

    if bubble_count > 1 and len(sorted_flow_coords) > bubble_count:
        gaps = np.diff(sorted_flow_coords)
        if len(gaps) >= bubble_count - 1:
            gap_idxs = np.argsort(gaps)[-(bubble_count - 1):]
            split_idxs = np.sort(gap_idxs) + 1
            bubble_point_indices = np.split(sort_idx, split_idxs)
        else:
            bubble_point_indices = [sort_idx]
    else:
        bubble_point_indices = [sort_idx]

    results: list[EccentricityResult] = []

    for i, idxs in enumerate(bubble_point_indices):
        bubble_points = pts[idxs]
        if len(bubble_points) == 0:
            results.append((0.0, 0.0, 0.0))
            continue

        threshold = np.percentile(bubble_points[:, z_idx], tip_percentile)
        tip_points = bubble_points[bubble_points[:, z_idx] >= threshold]

        if len(tip_points) == 0:
            print(f"[Bubble {i + 1}] Warning: No points found within the specified tip slice.")
            results.append((0.0, 0.0, 0.0))
            continue
        if len(tip_points) < 10:
            print(
                f"[Bubble {i + 1}] Warning: tip_percentile={tip_percentile} "
                f"selected only {len(tip_points)} tip points."
            )

        e_x = float(np.median(tip_points[:, x_idx]) - pipe_center[0])
        e_y = float(np.median(tip_points[:, y_idx]) - pipe_center[1])
        e_star = float((2.0 / diameter) * np.sqrt(e_x ** 2 + e_y ** 2))

        if debug:
            tip_fraction = 100.0 * len(tip_points) / len(bubble_points)
            print(f"\n--- Bubble {i + 1} ---")
            print(f"Point count in the tip: {len(tip_points)} / {len(bubble_points)} ({tip_fraction:.2f}%)")
            print(f"Max axial: {np.max(bubble_points[:, z_idx])}")
            print(f"AXIS [X, Y, Z] Min: {bubble_points.min(axis=0)}")
            print(f"AXIS [X, Y, Z] Max: {bubble_points.max(axis=0)}")
            print(f"Median offset 1 (e_x): {e_x:.4f}")
            print(f"Median offset 2 (e_y): {e_y:.4f}")
            print(f"Eccentricity (e*): {e_star:.4f}")

        results.append((e_star, e_x, e_y))

    return results


def eccentricity_records_for_points(
    points: np.ndarray | None,
    bubble_count: int,
    frame_no: int,
    file_name: str,
    tube_pair: str,
    diameter_mm: float,
    tip_percentile: float,
    pipe_center: tuple[float, float] = (0.0, 0.0),
    debug: bool = False,
) -> list[dict[str, object]]:
    """Return CSV-ready eccentricity rows for one frame and one tube pair."""
    if points is None or len(points) == 0 or bubble_count < 1:
        return []

    results = calculate_bubble_eccentricity(
        points=points,
        bubble_count=bubble_count,
        pipe_center=pipe_center,
        diameter=diameter_mm,
        tip_percentile=tip_percentile,
        debug=debug,
    )

    rows: list[dict[str, object]] = []
    for bubble_idx, (e_star, e_x, e_y) in enumerate(results, start=1):
        rows.append({
            "frame_no": frame_no,
            "file_name": file_name,
            "tube_pair": tube_pair,
            "bubble_index": bubble_idx,
            "bubble_count": bubble_count,
            "e_star": e_star,
            "e_x_mm": e_x,
            "e_y_mm": e_y,
            "tip_percentile": tip_percentile,
            "diameter_mm": diameter_mm,
        })
    return rows
