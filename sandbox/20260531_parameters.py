import json
import os
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
import pyvista as pv

def calculate_bubble_eccentricity(
    points: np.ndarray, 
    bubble_count: int = 1,
    pipe_center: tuple = (0.0, 0.0),         
    diameter: float = 25,               # mm
    tip_percentile: float = 99.0,
    debug: bool = False
    ) -> list[tuple[float, float, float]]:
    """
    Calculates the dimensionless eccentricity of a bubble tip from a 3D point cloud.
    
    Coordinate System: 
        X (# 0, in the center of heat pipe crosssection), 
        Y (# 1, in the center of heat pipe crosssection, opposite to gravity), 
        Z (# 2, in the center of heat pipe crosssection, along the pipe).
    
    Args:
        points (np.ndarray): A 2D array of shape (N, 3) representing the point cloud coordinates (e.g., in .ply format).
        bubble_count (int): The number of bubbles being processed or tracked.
        pipe_center (tuple): The coordinates of the pipe's central axis in the transverse plane.
        tip_percentile (float): Percentile threshold used to define the bubble tip.
        diameter (float): The internal diameter of the pipe in the same units as the point cloud.
        debug (bool): If True, enables debug mode for extended logging.

    Returns:
        List of tuple[float, float, float], one for each bubble:
            - e_star (float): The dimensionless eccentricity.
            - e_x (float): The median offset of the tip along the first transverse axis.
            - e_y (float): The median offset of the tip along the second transverse axis.
    """
    
    # 0. Validate inputs
    if points is None or len(points) == 0:
        raise ValueError("The input point cloud is missing.")
    if bubble_count < 1:
        raise ValueError(f"bubble_count must be at least 1, got {bubble_count}")
    if not (0 < tip_percentile < 100):
        raise ValueError(f"tip_percentile must be between 0 and 100, got {tip_percentile}")
    if diameter <= 0:
        raise ValueError(f"diameter must be positive, got {diameter}")
    
    # 0. Define coordinate indices    
    x_idx, y_idx, z_idx = 0, 1, 2
    
    # 1. Sort all points along the flow (Z) axis
    flow_coords = points[:, z_idx]  # Coordinate from flow axis (index 2 - Z) of each bubble 
    sort_idx = np.argsort(flow_coords) 
    sorted_flow_coords = flow_coords[sort_idx]  # Sorted in ascending order based on coordinate from flow axis
    
    # 2. Group points into distinct bubbles by finding the largest gaps along the flow axis
    if bubble_count > 1:
        gaps = np.diff(sorted_flow_coords)
        
        gaps_idxs = np.argsort(gaps)[-(bubble_count - 1):] # Find the indices of the (bubble_count - 1) largest gaps
        
        split_idxs = np.sort(gaps_idxs) + 1  # Shift by 1 for np.split and sort them to split in order
        
        bubble_point_indices = np.split(sort_idx, split_idxs)
    else:
        bubble_point_indices = [sort_idx]

    results = []

    # 3. Process each isolated bubble
    for i, idxs in enumerate(bubble_point_indices):
        
        # 3.1. Extract points belonging to a bubble
        bubble_points = points[idxs]
        
        # 3.2. Find the percentile threshold and select all points that fall within the defined slice near the tip
        threshold = np.percentile(bubble_points[:, z_idx], tip_percentile)
        tip_points = bubble_points[bubble_points[:, z_idx] >= threshold]
        
        if len(tip_points) == 0:
            print(f"[Bubble {i+1}] Warning: No points found within the specified tip slice.")
            results.append((0.0, 0.0, 0.0))
            continue
        if len(tip_points) < 10:
            print(f"[Bubble {i+1}] Warning: tip_percentile={tip_percentile} selected only {len(tip_points)} tip points.")
            
        # 3.3. Calculate median transverse coordinates (offset relative to the pipe's center)
        e_x = float(np.median(tip_points[:, x_idx]) - pipe_center[x_idx])
        e_y = float(np.median(tip_points[:, y_idx]) - pipe_center[y_idx])

        # 3.4. Calculate dimensionless eccentricity e*
        e_star = float((2.0 / diameter) * np.sqrt(e_x**2 + e_y**2))
        
        if debug:
            print(f"\n--- Bubble {i+1} ---")
            tip_fraction = 100 * len(tip_points) / len(bubble_points)
            print(f"Point count in the tip: {len(tip_points)} / {len(bubble_points)} ({tip_fraction:.2f}%)")
            print(f"Max axial: {np.max(bubble_points[:, z_idx])}")
            print(f"AXIS [X, Y, Z] Min: {bubble_points.min(axis=0)}")
            print(f"AXIS [X, Y, Z] Max: {bubble_points.max(axis=0)}")
            print(f"Median offset 1 (e_x): {e_x:.4f}")
            print(f"Median offset 2 (e_y): {e_y:.4f}")
            print(f"Eccentricity (e*): {e_star:.4f}")

        results.append((e_star, e_x, e_y))

    return results


def visualize_bubble_tip(
    points: np.ndarray,
    slice_thickness: float = 1
):
    """
    Visualize:
    - full point cloud
    - detected tip slice
    - median point of tip slice
    """
    
    x_idx, y_idx, z_idx = 0, 1, 2

    max_axial_coord = np.max(points[:, z_idx])

    tip_mask = points[:, z_idx] >= (max_axial_coord - slice_thickness)
    tip_points = points[tip_mask]

    transverse_axes = [i for i in range(3) if i != z_idx]

    median_point = tip_points.mean(axis=0)
    median_point[transverse_axes[0]] = np.median(
        tip_points[:, transverse_axes[0]]
    )
    median_point[transverse_axes[1]] = np.median(
        tip_points[:, transverse_axes[1]]
    )
    median_point[z_idx] = max_axial_coord

    plotter = pv.Plotter()

    # full cloud
    plotter.add_points(
        points,
        color="lightgray",
        point_size=2,
        opacity=0.15,
    )

    # tip slice
    plotter.add_points(
        tip_points,
        color="red",
        point_size=6,
    )

    # median eccentricity point
    plotter.add_mesh(
        pv.Sphere(radius=1.0, center=median_point),
        color="blue"
    )

    plotter.add_axes()
    plotter.show()

def generate_ideal_bubble(
    radius_xy: float = 5.0,
    radius_axial: float = 15.0,
    center: tuple = (0.0, 0.0, 0.0),
    z_idx: int = 0,
    n_points: int = 20000
) -> np.ndarray:
    """
    Generate ideal rotational ellipsoid point cloud.

    radius_xy   -> radius in pipe cross-section
    radius_axial -> radius along flow direction
    center      -> bubble center
    z_idx   -> 0=X, 1=Y, 2=Z
    """

    pts = []

    while len(pts) < n_points:

        p = np.random.uniform(
            low=[-radius_axial, -radius_xy, -radius_xy],
            high=[radius_axial, radius_xy, radius_xy],
        )

        # ellipsoid equation
        result = ((p[0] / radius_axial) ** 2 + (p[1] / radius_xy) ** 2 + (p[2] / radius_xy) ** 2)

        if result <= 1:
            pts.append(p)

    pts = np.array(pts)
    
    axial = pts[:, 0].copy()
    rad1 = pts[:, 1].copy()
    rad2 = pts[:, 2].copy()

    if z_idx == 0:
        pts = np.column_stack([axial, rad1, rad2])

    elif z_idx == 1:
        pts = np.column_stack([rad1, axial, rad2])

    elif z_idx == 2:
        pts = np.column_stack([rad1, rad2, axial])

    pts += np.array(center)

    return pts

# ==========================================
# TEST SCRIPT — ideal rotational bubble
# ==========================================

print("\nIDEAL BUBBLE\n-----------------")

diameter = 20
z_idx = 0

expected_ex = 2.0
expected_ey = -3.0

points = generate_ideal_bubble(
    radius_xy=5,
    radius_axial=12,
    center=(0.0, expected_ex, expected_ey),
    z_idx=z_idx,
    n_points=30000)

e_star, e_x, e_y = calculate_bubble_eccentricity(
    points,
    pipe_center=(0.0, 0.0),
    diameter=diameter)

print("\nEXPECTED")
print(f"e_x = {expected_ex:.3f}")
print(f"e_y = {expected_ey:.3f}")

print("\nMEASURED")
print(f"e*  = {e_star:.3f}")
print(f"e_x = {e_x:.3f}")
print(f"e_y = {e_y:.3f}")

visualize_bubble_tip(points, slice_thickness=2)

# ==========================================
# TEST SCRIPT
# ==========================================

print("\nACTUAL BUBBLE\n------------------")


ply_filename = r"C:\Users\Mateusz\Desktop\CODE\Bubble\sandbox\3Dcloud_tube34.ply" # Update this to your exact filename if needed
points = pv.read(ply_filename).points

e_star, e_x, e_y = calculate_bubble_eccentricity(points)

print(f"e*={e_star:.3f}, " f"e_x={e_x:.3f}, " f"e_y={e_y:.3f}")

visualize_bubble_tip(points, slice_thickness=1)