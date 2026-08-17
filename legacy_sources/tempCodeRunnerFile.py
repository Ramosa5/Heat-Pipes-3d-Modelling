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
    
    # IN PIPELINE ADD: if number of points != 0: calculate_bubble_eccentricity(...)
    
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