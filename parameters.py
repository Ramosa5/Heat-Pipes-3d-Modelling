import numpy as np

def median_eccentricity_2D(
    points_mm: np.ndarray,
    diameter_mm: float,
    n_tip_points: int = 100,
    axial_axis: int = 0,
    transverse_axes: tuple[int, int] = (1, 2)):
    """
    Compute median dimensionless eccentricity of the bubble tip.

    Parameters
    ----------
    points_mm : np.ndarray
        Array of shape (N, 3) with 3D bubble points in millimetres.
        The point cloud should be centered on the pipe axis in the transverse directions.
    diameter_mm : float
        Pipe inner diameter in mm.
    n_tip_points : int
        Number of points used to define the bubble tip.
    axial_axis : int
        Index of the axial coordinate (default: 0 -> x).
    transverse_axes : tuple[int, int]
        Indices of the two transverse coordinates.

    Returns
    -------
    ecc_tilde : float
        Dimensionless median eccentricity.
    e_tilde_1 : float
        Median transverse offset in the first transverse direction [mm].
    e_tilde_2 : float
        Median transverse offset in the second transverse direction [mm].
    tip_points : np.ndarray
        Subset of points used for tip estimation.
    """
    pts = np.asarray(points_mm, dtype=np.float32)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_mm must have shape (N, 3)")
    if pts.shape[0] == 0:
        raise ValueError("points_mm is empty")
    if diameter_mm <= 0:
        raise ValueError("diameter_mm must be positive")

    n_tip_points = int(max(1, min(n_tip_points, pts.shape[0])))

    # Select the N points with the largest axial coordinate
    axial = pts[:, axial_axis]
    tip_idx = np.argpartition(axial, -n_tip_points)[-n_tip_points:]
    tip_points = pts[tip_idx]

    # Median transverse position of the tip
    e_tilde_1 = float(np.median(tip_points[:, transverse_axes[0]]))
    e_tilde_2 = float(np.median(tip_points[:, transverse_axes[1]]))

    # Normalize by pipe radius D/2
    ecc_tilde =  float((2.0 / float(diameter_mm)) * np.sqrt(e_tilde_1**2 + e_tilde_2**2))

    return ecc_tilde, e_tilde_1, e_tilde_2, tip_points