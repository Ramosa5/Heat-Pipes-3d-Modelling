
import json
import os
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
import pyvista as pv

def calculate_bubble_eccentricity(
    points: np.ndarray, 
    diameter: float, 
    pipe_center: tuple, 
    slice_thickness: float, 
    flow_axis: int = 2
    ) -> tuple[float, float, float]:
    """
    Calculates the dimensionless eccentricity of a bubble tip from a 3D point cloud.
    
    Args:
        points (np.ndarray): A 2D array of shape (N, 3) representing the point cloud.
        diameter (float): The diameter of the pipe (D).
        pipe_center (tuple): The (x, y) coordinates of the pipe's central axis.
        slice_thickness (float): The depth of the slice near the tip to include in the median calculation.
        flow_axis (int): The index of the axis corresponding to the longitudinal flow 
                         (default is 2, which corresponds to the Z-axis).
                         
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

    return e_star, e_x, e_y



# ==========================================
# TEST SCRIPT
# ==========================================
if __name__ == "__main__":
    # 1. Load the PLY file using PyVista
    ply_filename = r"C:\Users\Mateusz\Desktop\CODE\Bubble\sandbox\3Dcloud_tube34.ply" # Update this to your exact filename if needed
    
    try:
        mesh = pv.read(ply_filename)
        points = mesh.points # This extracts the (N, 3) numpy array directly!
        print(f"Successfully loaded '{ply_filename}'")
        print(f"Point cloud shape: {points.shape}")
    except Exception as e:
        print(f"Failed to load the PLY file: {e}")
        exit()

    # 2. Define parameters
    DIAMETER_MM = 20.0
    
    # Because your pipeline uses center_yz=True, the Y and Z axes are centered around 0.
    PIPE_CENTER = (0.0, 0.0) 
    
    # Define how deep of a slice you want from the tip (in mm)
    SLICE_THICKNESS_MM = 2.0 
    
    # In your volume_to_points_mm function, X is the un-centered flow axis (index 0)
    FLOW_AXIS = 0 

    # 3. Test the function
    try:
        e_star, e_x, e_y = calculate_bubble_eccentricity(
            points=points,
            diameter=DIAMETER_MM,
            pipe_center=PIPE_CENTER,
            slice_thickness=SLICE_THICKNESS_MM,
            flow_axis=FLOW_AXIS
        )
        print("\n--- Results ---")
        print(f"e* (Dimensionless Eccentricity): {e_star:.4f}")
        print(f"e_x (Offset): {e_x:.4f} mm")
        print(f"e_y (Offset): {e_y:.4f} mm")
    except Exception as e:
        print(f"\nError calculating eccentricity: {e}")