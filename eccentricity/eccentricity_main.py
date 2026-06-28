import numpy as np

def calculate_bubble_eccentricity(
    points_mm: np.ndarray, 
    diameter_mm: float, 
    window_length_mm: float, 
    tip_percentile: float = 99.0,
    edge_margin_mm: float = 0.1,
    min_tip_points: int = 20
) -> dict:
    """
    Calculates the dimensionless eccentricity of the bubble front (tip) and back (tail).
    
    Parameters:
    - points_mm: np.ndarray of shape (N, 3), representing the bubble point cloud in mm.
        Must be centered radially (origin at the pipe centreline).
    - diameter_mm: float, the pipe diameter (D) in mm.
    - window_length_mm: float, the total length of the recorded Z-axis space in mm.
    - tip_percentile: float, the percentile used to define the tip.
    - edge_margin_mm: float, the buffer distance from the window edge to consider "clipped".
    
    Returns:
    - dict containing the dimensionless eccentricities and transverse shifts.
    """
    
    if points_mm is None or len(points_mm) == 0:
        return {
            "front_eccentricity": None, "back_eccentricity": None,
            "front_shift": (None, None), "back_shift": (None, None),
            "front_clipped": False, "back_clipped": False
        }
        
    if diameter_mm <= 0:
        raise ValueError("diameter_mm must be positive.")

    if not (0.0 < tip_percentile < 100.0):
        raise ValueError("tip_percentile must lie between 0 and 100.")

    if min_tip_points < 1:
        raise ValueError("min_tip_points must be at least 1.")

    # Z is the longitudinal axis (index 2); X and Y are transverse (indices 0, 1)
    z_coords = points_mm[:, 2]
    
    z_max = np.max(z_coords)
    z_min = np.min(z_coords)
    
    # Check if the bubble touches the camera boundaries
    is_front_clipped = z_max >= (window_length_mm - edge_margin_mm)
    is_back_clipped = z_min <= edge_margin_mm
    
    sorted_idx = np.argsort(z_coords)
    min_points = min(min_tip_points, len(points_mm))
    
    # --- 1. Bubble Front (Tip) Characterisation ---
    if not is_front_clipped:
        z_thresh_front = np.percentile(z_coords, tip_percentile)
        front_mask = z_coords >= z_thresh_front
        front_pts = points_mm[front_mask]

        if len(front_pts) > 0:
            
            if len(front_pts) < min_points:
                front_pts = points_mm[sorted_idx[-min_points:]]

            e_x_front = float(np.median(front_pts[:, 0]))
            e_y_front = float(np.median(front_pts[:, 1]))
            e_star_front = float((2.0 / diameter_mm) * np.sqrt(e_x_front**2 + e_y_front**2))
            front_shift = (e_x_front, e_y_front)

        else:
            e_star_front, front_shift = None, (None, None)

    else:
        e_star_front, front_shift = None, (None, None)

    # --- 2. Bubble Back (Tail) Characterisation ---
    if not is_back_clipped:
        tail_percentile = 100.0 - tip_percentile
        z_thresh_back = np.percentile(z_coords, tail_percentile)
        back_mask = z_coords <= z_thresh_back
        back_pts = points_mm[back_mask]
        
        if len(back_pts) > 0:
            
            if len(back_pts) < min_points:
                back_pts = points_mm[sorted_idx[:min_points]]
                
            e_x_back = float(np.median(back_pts[:, 0]))
            e_y_back = float(np.median(back_pts[:, 1]))
            e_star_back = float((2.0 / diameter_mm) * np.sqrt(e_x_back**2 + e_y_back**2))
            back_shift = (e_x_back, e_y_back)
        
        else:
            e_star_back, back_shift = None, (None, None)
    
    else:
        # Bubble is cut off by the window; data is physically meaningless
        e_star_back, back_shift = None, (None, None)

    return {
        "front_eccentricity": e_star_front,
        "back_eccentricity": e_star_back,
        "front_shift": front_shift,
        "back_shift": back_shift,
        "front_clipped": bool(is_front_clipped),
        "back_clipped": bool(is_back_clipped)
    }
    
    
def calculate_all_bubble_eccentricities(
    bubble_clouds: list[np.ndarray],
    diameter_mm: float,
    window_length_mm: float,
    tip_percentile: float = 99.0,
    edge_margin_mm: float = 0.1,
    min_tip_points: int = 20
) -> list[dict]:
    """
    Calculates eccentricity metrics for multiple bubbles, assuming input:
    bubble_clouds = [bubble1, bubble2, bubble3]

    Parameters:
    - bubble_clouds: list of point clouds, where each element is an (N, 3)
      NumPy array representing a single bubble.

    Returns:
    - List of dictionaries returned by calculate_bubble_eccentricity().
    """

    return [
        calculate_bubble_eccentricity(
            points_mm=bubble_points,
            diameter_mm=diameter_mm,
            window_length_mm=window_length_mm,
            tip_percentile=tip_percentile,
            edge_margin_mm=edge_margin_mm,
            min_tip_points=min_tip_points)
        for bubble_points in bubble_clouds
    ]