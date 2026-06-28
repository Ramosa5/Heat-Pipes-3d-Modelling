from __future__ import annotations

import numpy as np

def rotational_fit_score(points: np.ndarray,
                         n_sections: int = 50,
                         min_points_per_section: int = 5,
                         radius_statistic: str = "median"):
    """
    Computes the rotational fit coefficient of a bubble.

    points: surface point cloud of a single bubble, shape (N, 3)
    n_sections: number of cross-sections along the main axis
    min_points_per_section: minimum number of points required in one section

    return:
        score      - the closer to 1, the more rotationally symmetric the shape is
        mean_error - mean radial deviation [mm]
        R          - reference mean radius [mm]
    """
    if points is None or len(points) < 10:
        return 0.0, None, None

    pts = np.asarray(points, dtype=np.float64)

    # 1. Move the bubble to its own local centre
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    # 2. Estimate the main axis using PCA
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # The eigenvector with the largest eigenvalue is the main axis
    main_axis = eigvecs[:, np.argmax(eigvals)]
    main_axis = main_axis / np.linalg.norm(main_axis)

    # 3. Project points onto the main axis
    # This gives the local position of each point along the bubble axis
    z_local = centered @ main_axis

    # 4. Compute radial distance of each point from the main axis
    parallel = np.outer(z_local, main_axis)
    radial_vec = centered - parallel
    radii = np.linalg.norm(radial_vec, axis=1)

    # 5. Divide the bubble into cross-sections along the main axis
    z_min, z_max = z_local.min(), z_local.max()

    if np.isclose(z_min, z_max):
        return 0.0, None, None

    bins = np.linspace(z_min, z_max, n_sections + 1)

    section_ids = np.digitize(z_local, bins) - 1
    section_ids = np.clip(section_ids, 0, n_sections - 1)

    # Radius profile of the ideal rotational solid
    r_profile = np.full(n_sections, np.nan)

    for k in range(n_sections):
        mask = section_ids == k

        if np.sum(mask) >= min_points_per_section:
            if radius_statistic == "percentile":
                r_profile[k] = np.percentile(radii[mask], 95)
            else:
                r_profile[k] = np.median(radii[mask])

    # If too few sections contain data, the result is unreliable
    valid_sections = ~np.isnan(r_profile)

    if np.sum(valid_sections) < 3:
        return 0.0, None, None

    # 6. Fill missing cross-sections by interpolation
    xs = np.arange(n_sections)

    r_profile[~valid_sections] = np.interp(
        xs[~valid_sections],
        xs[valid_sections],
        r_profile[valid_sections]
    )

    # 7. Assign ideal radius to every point based on its section
    r_ideal_for_points = r_profile[section_ids]

    # 8. Compute radial deviation from the ideal rotational profile
    errors = np.abs(radii - r_ideal_for_points)

    mean_error = np.mean(errors)

    # Reference mean radius
    R = np.mean(r_profile)

    if R <= 1e-9:
        return 0.0, float(mean_error), float(R)

    # Normalized rotational fit score
    score = 1.0 - mean_error / R

    # Clamp score to range 0-1
    score = float(np.clip(score, 0.0, 1.0))

    return score, float(mean_error), float(R)


def validate_existing_rotational_fit_score():
    """
    Simple validation of the existing rotational_fit_score() function
    using synthetic point clouds with known geometry.
    """

    # ---------- 1. Ideal rotational shape: ellipsoid of revolution ----------
    length_mm = 30.0
    radius_mm = 7.0

    n_z = 80
    n_theta = 120

    z_values = np.linspace(-length_mm / 2.0, length_mm / 2.0, n_z)
    theta_values = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    ideal_points = []

    for z in z_values:
        z_norm = z / (length_mm / 2.0)
        r = radius_mm * np.sqrt(max(0.0, 1.0 - z_norm ** 2))

        for theta in theta_values:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            ideal_points.append([x, y, z])

    ideal_points = np.asarray(ideal_points, dtype=np.float64)

    ideal_score, ideal_error, ideal_R = rotational_fit_score(
        ideal_points,
        n_sections=50,
        min_points_per_section=5,
        radius_statistic="median"
    )

    # ---------- 2. Same shape, but shifted in space ----------
    shifted_points = ideal_points + np.array([100.0, -50.0, 200.0])

    shifted_score, shifted_error, shifted_R = rotational_fit_score(
        shifted_points,
        n_sections=50,
        min_points_per_section=5,
        radius_statistic="median"
    )

    # ---------- 3. Asymmetric shape ----------
    asymmetric_points = []

    asymmetry = 0.35

    for z in z_values:
        z_norm = z / (length_mm / 2.0)
        base_r = radius_mm * np.sqrt(max(0.0, 1.0 - z_norm ** 2))

        for theta in theta_values:
            # Radius depends on angle, so this is no longer rotationally symmetric
            r = base_r * (1.0 + asymmetry * np.cos(2.0 * theta))

            x = r * np.cos(theta)
            y = r * np.sin(theta)
            asymmetric_points.append([x, y, z])

    asymmetric_points = np.asarray(asymmetric_points, dtype=np.float64)

    asym_score, asym_error, asym_R = rotational_fit_score(
        asymmetric_points,
        n_sections=50,
        min_points_per_section=5,
        radius_statistic="median"
    )

    print("\n=== Validation of rotational_fit_score() ===")
    print(f"Ideal rotational shape: score={ideal_score:.4f}, error={ideal_error:.4f} mm, R={ideal_R:.4f} mm")
    print(f"Shifted same shape:      score={shifted_score:.4f}, error={shifted_error:.4f} mm, R={shifted_R:.4f} mm")
    print(f"Asymmetric shape:        score={asym_score:.4f}, error={asym_error:.4f} mm, R={asym_R:.4f} mm")

    print("\nExpected:")
    print("- ideal score should be close to 1")
    print("- shifted score should be almost the same as ideal score")
    print("- asymmetric score should be lower than ideal score")
