import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

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
            "front_eccentricity, -": None, "back_eccentricity, -": None,
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
        "front_eccentricity, -": e_star_front,
        "back_eccentricity, -": e_star_back,
        "front_shift": front_shift,
        "back_shift": back_shift,
        "front_clipped": bool(is_front_clipped),
        "back_clipped": bool(is_back_clipped)
    }
    
def generate_synthetic_bubble(
    radius_x: float, 
    radius_y: float, 
    length_z: float, 
    offset_x: float, 
    offset_y: float, 
    center_z: float, 
    num_points: int = 5000, 
    volume_noise_sigma: float = 0.0) -> np.ndarray:
    
    points = []

    while len(points) < num_points:

        xyz = np.random.uniform(-1.0, 1.0, (num_points, 3))

        inside = xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2 <= 1.0

        xyz = xyz[inside]

        xyz[:, 0] *= radius_x
        xyz[:, 1] *= radius_y
        xyz[:, 2] *= length_z / 2.0

        points.extend(xyz.tolist())

    points = np.asarray(points[:num_points])

    if volume_noise_sigma > 0:
        points += np.random.normal(0.0, volume_noise_sigma, points.shape)

    points[:, 0] += offset_x
    points[:, 1] += offset_y
    points[:, 2] += center_z

    return points

def test_eccentricity_accuracy(
    radius_x: float = 2.0, 
    radius_y: float = 2.0, 
    length_z: float = 30.0, 
    diameter_mm: float = 20.0,
    window_length_mm: float = 120.0,
    offsets_mm: np.ndarray | None = None,
    n_trials: int = 100
) -> dict:

    if offsets_mm is None:
        max_offset = diameter_mm / 2.0 - radius_x
        offsets_mm = np.linspace(0.0, max_offset, 21)

    expected = []
    measured_front = []
    measured_back = []

    for offset in offsets_mm:

        expected_value = (2.0 * offset) / diameter_mm

        for _ in range(n_trials):

            bubble = generate_synthetic_bubble(
                radius_x=radius_x,
                radius_y=radius_y,
                length_z=length_z,
                offset_x=offset,
                offset_y=0.0,
                center_z=50.0
            )

            result = calculate_bubble_eccentricity(bubble, diameter_mm, window_length_mm)

            expected.append(expected_value)
            measured_front.append(result["front_eccentricity, -"])
            measured_back.append(result["back_eccentricity, -"])

    return {
        "expected": np.asarray(expected),
        "front": np.asarray(measured_front),
        "back": np.asarray(measured_back)
    }
    
def test_noise_robustness(radius_x: float = 2.0, 
    radius_y: float = 2.0, 
    length_z: float = 30.0, 
    diameter_mm: float = 20.0,
    window_length_mm: float = 120.0,
    offset_x: float = 3.0, 
    offset_y: float = 2.0, 
    sigma_values: np.ndarray = np.linspace(0, 0.5, 11), 
    n_trials: int = 100) -> dict:

    expected = (2.0 / diameter_mm) * np.sqrt(offset_x**2 + offset_y**2)

    mean_front = []
    std_front = []

    mean_back = []
    std_back = []

    for sigma in sigma_values:

        front = []
        back = []

        for _ in range(n_trials):

            bubble = generate_synthetic_bubble(
                radius_x=radius_x,
                radius_y=radius_y,
                length_z=length_z,
                offset_x=offset_x,
                offset_y=offset_y,
                center_z=50.0,
                volume_noise_sigma=sigma
            )

            result = calculate_bubble_eccentricity(bubble, diameter_mm, window_length_mm)

            front.append(result["front_eccentricity, -"])
            back.append(result["back_eccentricity, -"])

        mean_front.append(np.mean(front))
        std_front.append(np.std(front))

        mean_back.append(np.mean(back))
        std_back.append(np.std(back))

    return {
        "sigma": sigma_values,
        "expected": expected,
        "front_mean": np.asarray(mean_front),
        "front_std": np.asarray(std_front),
        "back_mean": np.asarray(mean_back),
        "back_std": np.asarray(std_back)
    }
    
def plot_accuracy(results: dict):

    plt.figure(figsize=(6, 6))

    plt.plot(
        [0, 1],
        [0, 1],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Ideal",
        zorder=1
    )
    
    plt.scatter(results["expected"], results["front"], s=5, alpha=0.35, label="Front")
    plt.scatter(results["expected"], results["back"], s=5, alpha=0.65, label="Back")

    #lim = [0, max(results["expected"]) * 1.05]
    #lim = [0.0, 1.05]
    
    ax = plt.gca()

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")

    # Major ticks every 0.1
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    # Optional: minor ticks every 0.05
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    ax.grid(True, which="major", linewidth=0.8)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.3)

    ax.set_xlabel("Expected eccentricity, -")
    ax.set_ylabel("Measured eccentricity, -")
    
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False)

    ax.legend()

    plt.tight_layout()
    plt.show()
    
def plot_noise(results: dict):

    plt.figure(figsize=(6, 6))

    ax = plt.gca()
    ax.set_box_aspect(1)
    
    ax.plot(results["sigma"], results["front_mean"], linewidth=2, label="Front eccentricity (mean ±1σ)")
    ax.fill_between(results["sigma"], results["front_mean"] - results["front_std"], results["front_mean"] + results["front_std"], alpha=0.20)

    ax.plot(results["sigma"], results["back_mean"], linewidth=2, label="Back eccentricity (mean ±1σ)")
    ax.fill_between(results["sigma"], results["back_mean"] - results["back_std"], results["back_mean"] + results["back_std"], alpha=0.20)

    ax.axhline(results["expected"], color="black", linestyle="--", linewidth=1.2, label="True value")

    ax.set_xlabel("Noise standard deviation, mm")
    ax.set_ylabel("Bubble eccentricity, -")

    #ax.set_xlim(results["sigma"][0], results["sigma"][-1])

    step = 0.1

    y = np.concatenate([
        results["front_mean"],
        results["back_mean"],
        [results["expected"]]
    ])

    ymin = np.floor(np.min(y) / step) * step
    ymax = np.ceil(np.max(y) / step) * step

    ax.set_ylim(ymin, ymax)

    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))

    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))

    ax.grid(True, which="major", linewidth=0.8)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.3)

    ax.tick_params(direction="in", top=True, right=True)
    
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()
    
    
def generate_curved_bubble(
    diameter_mm: float = 20.0,
    window_length_mm: float = 100.0,
    length_z: float = 40.0,
    radius_mm: float = 2.0,
    centre_z: float = 50.0,
    tip_shift=(2.5, 1.0),
    tail_shift=(-1.5, -2.0),
    curvature_x: float = 1.2,
    curvature_y: float = 0.8,
    radius_variation: float = 0.10,
    rotation_deg: float = 15.0,
    n_sections: int = 500,
    points_per_section: int = 40
):
    """
    Generates a smooth, realistic synthetic bubble whose centreline varies
    continuously along its length.
    
    It demonstrates applicability to a more realistic, spatially varying geometry (not actual benchmark).
    """

    z = np.linspace(-length_z / 2, length_z / 2, n_sections)

    cloud = []

    theta_rot = np.deg2rad(rotation_deg)
    R = np.array([[np.cos(theta_rot), -np.sin(theta_rot)], [np.sin(theta_rot),  np.cos(theta_rot)]])
    
    centreline_x = []
    centreline_y = []

    for zi in z:

        # Smooth centreline
        cx = curvature_x * np.sin(2 * np.pi * zi / length_z)
        cy = curvature_y * np.cos(1.5 * np.pi * zi / length_z)

        # Additional front displacement
        w_front = np.exp(-((zi - length_z / 2) / 3.0) ** 2)
        cx += tip_shift[0] * w_front
        cy += tip_shift[1] * w_front

        # Additional rear displacement
        w_back = np.exp(-((zi + length_z / 2) / 3.0) ** 2)
        cx += tail_shift[0] * w_back
        cy += tail_shift[1] * w_back

        # Slowly varying radius

        r = radius_mm * (1 + radius_variation * np.sin(3 * np.pi * zi / length_z))
        
        pipe_radius = diameter_mm / 2.0

        max_offset = pipe_radius - r

        offset = np.hypot(cx, cy)

        if offset > max_offset:
            scale = max_offset / offset
            cx *= scale
            cy *= scale

        # Store the actual centreline used to generate the cloud
        centreline_x.extend([cx] * points_per_section)
        centreline_y.extend([cy] * points_per_section)
        
        phi = np.random.uniform(0, 2 * np.pi, points_per_section)
        rho = np.sqrt(np.random.rand(points_per_section)) * r

        xy = np.vstack((rho * np.cos(phi), rho * np.sin(phi)))

        xy = R @ xy

        x = xy[0] + cx
        y = xy[1] + cy

        section = np.column_stack((x, y, np.full(points_per_section, zi)))

        cloud.append(section)

    bubble = np.vstack(cloud)

    bubble[:, 2] += centre_z
    
    centreline_x = np.asarray(centreline_x)
    centreline_y = np.asarray(centreline_y)

    result = calculate_bubble_eccentricity(
        bubble,
        diameter_mm,
        window_length_mm
    )

    # ----- theoretical values computed over the same region -----

    tip_percentile = 99.0

    z_coords = bubble[:, 2]

    z_front = np.percentile(z_coords, tip_percentile)
    z_back = np.percentile(z_coords, 100.0 - tip_percentile)

    front_mask = z_coords >= z_front
    back_mask = z_coords <= z_back

    fx = np.median(centreline_x[front_mask])
    fy = np.median(centreline_y[front_mask])

    bx = np.median(centreline_x[back_mask])
    by = np.median(centreline_y[back_mask])

    expected_front = (2.0 / diameter_mm) * np.sqrt(fx**2 + fy**2)
    expected_back = (2.0 / diameter_mm) * np.sqrt(bx**2 + by**2)

    return {
    "bubble": bubble,
    "expected_front": expected_front,
    "expected_back": expected_back,
    "measured_front": result["front_eccentricity, -"],
    "measured_back": result["back_eccentricity, -"],
    "result": result
    }
    
def test_curved_bubbles(
    n_bubbles: int = 100,
    diameter_mm: float = 20.0,
    window_length_mm: float = 100.0,
    random_seed: int = 42
) -> dict:

    np.random.seed(random_seed)

    expected_front = []
    expected_back = []

    measured_front = []
    measured_back = []

    for _ in range(n_bubbles):

        bubble = generate_curved_bubble(
            diameter_mm=diameter_mm,
            window_length_mm=window_length_mm,

            length_z=np.random.uniform(20.0, 60.0),
            radius_mm=np.random.uniform(1.0, 4.0),

            tip_shift=(np.random.uniform(-5.0, 5.0), np.random.uniform(-5.0, 5.0)),
            tail_shift=(np.random.uniform(-5.0, 5.0), np.random.uniform(-5.0, 5.0)),

            curvature_x=np.random.uniform(-5.0, 5.0),
            curvature_y=np.random.uniform(-5.0, 5.0),

            radius_variation=np.random.uniform(0.0, 0.40),

            rotation_deg=np.random.uniform(0.0, 180.0),

            n_sections=500,
            points_per_section=40
        )

        expected_front.append(bubble["expected_front"])
        expected_back.append(bubble["expected_back"])

        measured_front.append(bubble["measured_front"])
        measured_back.append(bubble["measured_back"])

    return {
        "expected_front": np.asarray(expected_front),
        "expected_back": np.asarray(expected_back),
        "measured_front": np.asarray(measured_front),
        "measured_back": np.asarray(measured_back)
    }
    
def plot_curved_validation(results: dict):

    plt.figure(figsize=(6,6))

    ax = plt.gca()

    ax.plot(
        [0,1],
        [0,1],
        "k--",
        linewidth=1.2,
        label="Ideal"
    )

    ax.scatter(
        results["expected_front"],
        results["measured_front"],
        s=35,
        alpha=0.7,
        label="Front"
    )

    ax.scatter(
        results["expected_back"],
        results["measured_back"],
        s=35,
        alpha=0.7,
        label="Back"
    )

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    ax.set_aspect("equal")

    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.3)

    ax.tick_params(direction="in", top=True, right=True)

    ax.set_xlabel("Reference eccentricity, -")
    ax.set_ylabel("Measured eccentricity, -")

    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    
    print("\n================ Evaluation ==================")

    DIAMETER_MM = 20.0
    WINDOW_LENGTH = 100.0
    RADIUS_X = 2
    RADIUS_Y = 2
    LENGTH_Z=30.0
    
    np.random.seed(42)

    accuracy = test_eccentricity_accuracy(radius_x=RADIUS_X, radius_y=RADIUS_Y, length_z=LENGTH_Z, diameter_mm=DIAMETER_MM, window_length_mm=WINDOW_LENGTH)
     
    plot_accuracy(accuracy)

    noise = test_noise_robustness(radius_x=RADIUS_X, radius_y=RADIUS_Y, length_z=LENGTH_Z, diameter_mm=DIAMETER_MM, window_length_mm=WINDOW_LENGTH)   
     
    plot_noise(noise)
    
    front_rmse = np.sqrt(np.mean((accuracy["front"] - accuracy["expected"])**2))
    back_rmse = np.sqrt(np.mean((accuracy["back"] - accuracy["expected"])**2))

    

    front_bias = np.mean(accuracy["front"] - accuracy["expected"])
    back_bias = np.mean(accuracy["back"] - accuracy["expected"])
    
    front_mae = np.mean(np.abs(accuracy["front"] - accuracy["expected"]))
    back_mae = np.mean(np.abs(accuracy["back"] - accuracy["expected"]))
    
    ss_res_front = np.sum((accuracy["front"] - accuracy["expected"])**2)
    ss_tot = np.sum((accuracy["expected"] - np.mean(accuracy["expected"]))**2)
    r2_front = 1.0 - ss_res_front / ss_tot

    ss_res_back = np.sum((accuracy["back"] - accuracy["expected"])**2)
    r2_back = 1.0 - ss_res_back / ss_tot

    front_error = np.abs(accuracy["front"] - accuracy["expected"])
    back_error = np.abs(accuracy["back"] - accuracy["expected"])
    
    print("\n========== Nominal Bubble Applicability ==========")
    print(f"Front RMSE: {front_rmse:.5f}")
    print(f"Back RMSE:  {back_rmse:.5f}")
    print(f"Front bias: {front_bias:.5f}")
    print(f"Back bias:  {back_bias:.5f}")
    print(f"Front MAE: {front_mae:.5f}")
    print(f"Back MAE:  {back_mae:.5f}")
    print(f"Front R²: {r2_front:.5f}")
    print(f"Back  R²: {r2_back:.5f}")
    print(f"Front 95th percentile error: {np.percentile(front_error,95):.5f}")
    print(f"Back 95th percentile error:  {np.percentile(back_error,95):.5f}")
    print("================================================")
    
    curved = test_curved_bubbles(n_bubbles=100)

    plot_curved_validation(curved)

    front_rmse = np.sqrt(np.mean((curved["measured_front"] - curved["expected_front"])**2))
    back_rmse = np.sqrt(np.mean((curved["measured_back"] - curved["expected_back"])**2))

    front_mae = np.mean(np.abs(curved["measured_front"] - curved["expected_front"]))
    back_mae = np.mean(np.abs(curved["measured_back"] - curved["expected_back"]))

    front_bias = np.mean(curved["measured_front"] - curved["expected_front"])
    back_bias = np.mean(curved["measured_back"] - curved["expected_back"])

    ss_tot_front = np.sum((curved["expected_front"] - np.mean(curved["expected_front"]))**2)
    ss_tot_back = np.sum((curved["expected_back"] - np.mean(curved["expected_back"]))**2)

    front_r2 = 1 - np.sum((curved["measured_front"] - curved["expected_front"])**2) / ss_tot_front
    back_r2 = 1 - np.sum((curved["measured_back"] - curved["expected_back"])**2) / ss_tot_back
    
    front_error = np.abs(curved["measured_front"] - curved["expected_front"])
    back_error = np.abs(curved["measured_back"] - curved["expected_back"])
    
    print("\n========== Curved Bubble Applicability ==========")
    print(f"Front RMSE : {front_rmse:.5f}")
    print(f"Back RMSE  : {back_rmse:.5f}")
    print(f"Front MAE  : {front_mae:.5f}")
    print(f"Back MAE   : {back_mae:.5f}")
    print(f"Front bias : {front_bias:.5f}")
    print(f"Back bias  : {back_bias:.5f}")
    print(f"Front R²   : {front_r2:.5f}")
    print(f"Back R²    : {back_r2:.5f}")
    print(f"Front 95th percentile error: {np.percentile(front_error,95):.5f}")
    print(f"Back 95th percentile error: {np.percentile(back_error,95):.5f}")
    print("================================================")