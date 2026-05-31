# TO DO: AXIS, SINGLE BUBBLE OR SEPARATE BUBBLES, IDEAL SPHERES

import json
import os
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
import pyvista as pv

def calculate_bubble_eccentricity(
    points: np.ndarray, 
    pipe_center: tuple = (0.0, 0.0),    # HOW TO ESTIMATE?
    slice_thickness: float = 2,                  
    diameter: float = 20,               # TBC -> adjust with the main code
    flow_axis: int = 0,                 # X,Y,Z (CHECK IF MAIN CODE IS RIGHT)
    debug: bool = True
    ) -> tuple[float, float, float]:
    """
    Calculates the dimensionless eccentricity of a bubble tip from a 3D point cloud.
    Axis: X (1st, in the center of heat pipe crosssection), Y (2nd, in the center of heat pipe crosssection, opposite to gravity), Z (3rd, in the center of heat pipe crosssection, along the pipe)
    
    Args:
        points (np.ndarray): A 2D array of shape (N, 3) representing the point cloud.
        diameter (float): The diameter of the pipe (D).
        pipe_center (tuple): The (x, y) coordinates of the pipe's central axis.
        slice_thickness (float): The depth of the slice near the tip to include in the median calculation; in voxels.
        flow_axis (int): The index of the axis corresponding to the longitudinal flow 
                         (default corresponds to the Z-axis).
                         
    Returns:
        dimensionless eccentricity and the median transverse offsets.
    """
    if points is None or len(points) == 0:
        raise ValueError("The input point cloud is empty.")

    # 1. Identify the bubble tip along the flow axis
    max_axial_coord = np.max(points[:, flow_axis])
    
    # 2. Select all points that fall within the defined slice near the tip
    tip_mask = points[:, flow_axis] >= (max_axial_coord - slice_thickness)
    tip_points = points[tip_mask]
    
    if len(tip_points) == 0:
        raise ValueError("No points found within the specified tip slice.")

    # Determine which columns represent the transverse axes (x and y)
    transverse_axes = [i for i in range(3) if i != flow_axis]
    x_idx, y_idx = transverse_axes[0], transverse_axes[1]

    # 3. Calculate median transverse coordinates (offset relative to the pipe's center)
    e_x = float(np.median(tip_points[:, x_idx]) - pipe_center[0])
    e_y = float(np.median(tip_points[:, y_idx]) - pipe_center[1])

    # 4. Calculate dimensionless eccentricity e*
    e_star = float((2.0 / diameter) * np.sqrt(e_x**2 + e_y**2))
    
    if debug:
        print(f"Max axial: {max_axial_coord}")
        print(f"Tip points: {len(tip_points)}")
        print("\nAXIS [X, Y, Z]:")
        print(points.min(axis=0))
        print(points.max(axis=0))
        print(f"Median x: {e_x}")
        print(f"Median y: {e_y}")
        print(f"Eccentricity: {e_star}")

    return e_star, e_x, e_y


def visualize_bubble_tip(
    points: np.ndarray,
    slice_thickness: float = 2,
    flow_axis: int = 0,
):
    """
    Visualize:
    - full point cloud
    - detected tip slice
    - median point of tip slice
    """

    max_axial_coord = np.max(points[:, flow_axis])

    tip_mask = points[:, flow_axis] >= (max_axial_coord - slice_thickness)
    tip_points = points[tip_mask]

    transverse_axes = [i for i in range(3) if i != flow_axis]

    median_point = tip_points.mean(axis=0)
    median_point[transverse_axes[0]] = np.median(
        tip_points[:, transverse_axes[0]]
    )
    median_point[transverse_axes[1]] = np.median(
        tip_points[:, transverse_axes[1]]
    )
    median_point[flow_axis] = max_axial_coord

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
    
# ==========================================
# TEST SCRIPT
# ==========================================

ply_filename = r"C:\Users\Mateusz\Desktop\CODE\Bubble\sandbox\3Dcloud_tube34.ply" # Update this to your exact filename if needed
points = pv.read(ply_filename).points

e_star, e_x, e_y = calculate_bubble_eccentricity(points)

print(
    f"e*={e_star:.3f}, "
    f"e_x={e_x:.3f}, "
    f"e_y={e_y:.3f}"
)

visualize_bubble_tip(
    points,
    slice_thickness=2,
    flow_axis=0
)