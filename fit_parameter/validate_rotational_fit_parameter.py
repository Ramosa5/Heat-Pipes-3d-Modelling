import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATH SETUP
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT FUNCTIONS FROM MAIN PROJECT FILE
# ============================================================
from main_old import (
    load_coco,
    point_in_tube,
    build_bubble_mask_from_anns,
    rectify_and_align_pair,
    connected_components,
    match_components_by_longitudinal_overlap,
    build_volume_elliptic_from_two_masks,
    volume_to_surface_points_mm,
    rotational_fit_score,
)


# ============================================================
# CONFIG
# ============================================================
OUTPUT_DIR = SCRIPT_DIR / "rotational_fit_parameter_validation"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_DIR = PROJECT_ROOT / "bubble.coco" / "train"
COCO_FILE = "_annotations.coco.json"

RUN_SYNTHETIC_ANALYSIS = True
RUN_REAL_ANALYSIS = True

# Real dataset range
START_FRAME = 100
N_FRAMES = 40

# Synthetic validation
N_REALISATIONS = 100
ASYMMETRY_MIN = 0.0
ASYMMETRY_MAX = 0.70
ASYMMETRY_STEP = 0.01

NOISE_MIN_MM = 0.0
NOISE_MAX_MM = 1.00
NOISE_STEP_MM = 0.05

# Rotational fit parameters
N_SECTIONS = 50
MIN_POINTS_PER_SECTION = 5
RADIUS_STATISTIC = "median"

# Reconstruction parameters
DIAMETER_MM = 20.0
VOXEL_MM = None
SMOOTH_SIGMA_Z = 2.0
MIN_RADIUS_VOX = 0.8
MIN_AREA_CC = 80
IOU_THR = 0.15

# Tube definitions
TUBES = [
    {"a_top": 0.007, "b_top": 50,  "a_bot": 0.007, "b_bot": 100},
    {"a_top": 0.019, "b_top": 105, "a_bot": 0.019, "b_bot": 155},
    {"a_top": 0.005, "b_top": 322, "a_bot": 0.005, "b_bot": 380},
    {"a_top": 0.005, "b_top": 408, "a_bot": 0.005, "b_bot": 470},
]

MARGIN_PX = 2.0
INNER_MARGIN_PX = 2.0
KEEP_ASPECT = True


# ============================================================
# GENERAL HELPERS
# ============================================================
def clear_output_directory():
    """
    Removes old output files so outdated plots do not stay in the folder.
    """
    patterns = ["*.png", "*.csv", "*.json"]

    for pattern in patterns:
        for path in OUTPUT_DIR.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass


def to_python_value(value):
    """
    Converts NumPy values to plain Python values for JSON export.
    """
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    return value


def save_csv(rows, path):
    """
    Saves a list of dictionaries to CSV.
    """
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_json(synthetic_rows, real_rows):
    """
    Saves a compact summary of the validation.
    """
    real_scores = [
        row["score"] for row in real_rows
        if row.get("score") is not None
    ]

    summary = {
        "configuration": {
            "dataset_dir": str(DATASET_DIR),
            "coco_file": COCO_FILE,
            "start_frame": int(START_FRAME),
            "n_frames": int(N_FRAMES),
            "n_realisations": int(N_REALISATIONS),
            "asymmetry_min": float(ASYMMETRY_MIN),
            "asymmetry_max": float(ASYMMETRY_MAX),
            "asymmetry_step": float(ASYMMETRY_STEP),
            "noise_min_mm": float(NOISE_MIN_MM),
            "noise_max_mm": float(NOISE_MAX_MM),
            "noise_step_mm": float(NOISE_STEP_MM),
            "diameter_mm": float(DIAMETER_MM),
            "n_sections": int(N_SECTIONS),
            "min_points_per_section": int(MIN_POINTS_PER_SECTION),
            "radius_statistic": RADIUS_STATISTIC,
        },
        "synthetic": {
            "n_rows": int(len(synthetic_rows)),
        },
        "real": {
            "n_bubbles": int(len(real_scores)),
            "mean_score": float(np.mean(real_scores)) if real_scores else None,
            "std_score": float(np.std(real_scores)) if real_scores else None,
            "min_score": float(np.min(real_scores)) if real_scores else None,
            "max_score": float(np.max(real_scores)) if real_scores else None,
        },
    }

    clean_summary = json.loads(
        json.dumps(summary, default=to_python_value)
    )

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as file:
        json.dump(clean_summary, file, indent=4)


# ============================================================
# SYNTHETIC BUBBLE GENERATION
# ============================================================
def rotation_matrix_xyz(rx_deg, ry_deg, rz_deg):
    """
    Builds a 3D rotation matrix from Euler angles in degrees.
    """
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    rx_mat = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)],
    ])

    ry_mat = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])

    rz_mat = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1],
    ])

    return rz_mat @ ry_mat @ rx_mat


def generate_synthetic_bubble_surface(
    length_mm=30.0,
    radius_mm=7.0,
    asymmetry=0.0,
    noise_std_mm=0.0,
    n_z=80,
    n_theta=120,
    translation=(0.0, 0.0, 0.0),
    rotation_deg=(0.0, 0.0, 0.0),
    seed=None,
):
    """
    Generates a synthetic bubble surface.

    The base shape is an ellipsoid of revolution.
    Asymmetry is introduced as an angular radius perturbation.
    """
    rng = np.random.default_rng(seed)

    z_values = np.linspace(-length_mm / 2.0, length_mm / 2.0, n_z)
    theta_values = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    points = []

    for z in z_values:
        axial_term = 1.0 - (z / (length_mm / 2.0)) ** 2
        base_radius = radius_mm * np.sqrt(max(0.0, axial_term))

        for theta in theta_values:
            angular_factor = 1.0 + asymmetry * np.cos(2.0 * theta)
            local_radius = base_radius * angular_factor

            x = local_radius * np.cos(theta)
            y = local_radius * np.sin(theta)

            points.append([x, y, z])

    points = np.asarray(points, dtype=float)

    if noise_std_mm > 0.0:
        points += rng.normal(
            loc=0.0,
            scale=noise_std_mm,
            size=points.shape
        )

    rot_mat = rotation_matrix_xyz(*rotation_deg)
    points = points @ rot_mat.T

    translation = np.asarray(translation, dtype=float)
    points = points + translation

    return points


def calculate_rotational_fit_for_points(points):
    """
    Calculates rotational fit score for a point cloud.
    """
    score, mean_error, ref_radius = rotational_fit_score(
        points,
        n_sections=N_SECTIONS,
        min_points_per_section=MIN_POINTS_PER_SECTION,
        radius_statistic=RADIUS_STATISTIC,
    )

    return score, mean_error, ref_radius


def analyse_synthetic_bubbles():
    """
    Runs synthetic validation:
    1. Asymmetry sweep every 0.01.
    2. Noise sweep every 0.05 mm with 100 independent realisations.
    3. Rigid transformation invariance test.
    """
    rows = []

    length_mm = 30.0
    radius_mm = 7.0
    n_z = 80
    n_theta = 120

    asymmetry_values = np.round(
        np.arange(ASYMMETRY_MIN, ASYMMETRY_MAX + 0.5 * ASYMMETRY_STEP, ASYMMETRY_STEP),
        2
    )

    noise_values = np.round(
        np.arange(NOISE_MIN_MM, NOISE_MAX_MM + 0.5 * NOISE_STEP_MM, NOISE_STEP_MM),
        2
    )

    # ------------------------------------------------------------
    # Test 1: asymmetry sweep
    # ------------------------------------------------------------
    for asymmetry in asymmetry_values:
        points = generate_synthetic_bubble_surface(
            length_mm=length_mm,
            radius_mm=radius_mm,
            asymmetry=float(asymmetry),
            noise_std_mm=0.0,
            n_z=n_z,
            n_theta=n_theta,
            seed=123,
        )

        score, mean_error, ref_radius = calculate_rotational_fit_for_points(points)

        rows.append({
            "source": "synthetic",
            "test_type": "asymmetry_sweep",
            "case_name": "asymmetry_sweep",
            "realisation": 0,
            "asymmetry": float(asymmetry),
            "noise_std_mm": 0.0,
            "translation_x_mm": 0.0,
            "translation_y_mm": 0.0,
            "translation_z_mm": 0.0,
            "rotation_x_deg": 0.0,
            "rotation_y_deg": 0.0,
            "rotation_z_deg": 0.0,
            "score": score,
            "mean_error_mm": mean_error,
            "R_mm": ref_radius,
        })

    # ------------------------------------------------------------
    # Test 2: noise sweep with independent realisations
    # ------------------------------------------------------------
    for noise_std in noise_values:
        for realisation in range(N_REALISATIONS):
            seed = 100000 + int(round(noise_std * 1000.0)) * 1000 + realisation

            points = generate_synthetic_bubble_surface(
                length_mm=length_mm,
                radius_mm=radius_mm,
                asymmetry=0.0,
                noise_std_mm=float(noise_std),
                n_z=n_z,
                n_theta=n_theta,
                seed=seed,
            )

            score, mean_error, ref_radius = calculate_rotational_fit_for_points(points)

            rows.append({
                "source": "synthetic",
                "test_type": "noise_sweep",
                "case_name": "noise_sweep",
                "realisation": int(realisation),
                "asymmetry": 0.0,
                "noise_std_mm": float(noise_std),
                "translation_x_mm": 0.0,
                "translation_y_mm": 0.0,
                "translation_z_mm": 0.0,
                "rotation_x_deg": 0.0,
                "rotation_y_deg": 0.0,
                "rotation_z_deg": 0.0,
                "score": score,
                "mean_error_mm": mean_error,
                "R_mm": ref_radius,
            })

    # ------------------------------------------------------------
    # Test 3: rigid transformation invariance
    # ------------------------------------------------------------
    invariance_cases = [
        {
            "case_name": "ideal_original",
            "translation": (0.0, 0.0, 0.0),
            "rotation_deg": (0.0, 0.0, 0.0),
        },
        {
            "case_name": "ideal_shifted",
            "translation": (5.0, -3.0, 8.0),
            "rotation_deg": (0.0, 0.0, 0.0),
        },
        {
            "case_name": "ideal_rotated",
            "translation": (0.0, 0.0, 0.0),
            "rotation_deg": (20.0, 35.0, 15.0),
        },
        {
            "case_name": "ideal_shifted_and_rotated",
            "translation": (5.0, -3.0, 8.0),
            "rotation_deg": (20.0, 35.0, 15.0),
        },
    ]

    for case in invariance_cases:
        points = generate_synthetic_bubble_surface(
            length_mm=length_mm,
            radius_mm=radius_mm,
            asymmetry=0.0,
            noise_std_mm=0.0,
            n_z=n_z,
            n_theta=n_theta,
            translation=case["translation"],
            rotation_deg=case["rotation_deg"],
            seed=123,
        )

        score, mean_error, ref_radius = calculate_rotational_fit_for_points(points)

        rows.append({
            "source": "synthetic",
            "test_type": "invariance_test",
            "case_name": case["case_name"],
            "realisation": 0,
            "asymmetry": 0.0,
            "noise_std_mm": 0.0,
            "translation_x_mm": float(case["translation"][0]),
            "translation_y_mm": float(case["translation"][1]),
            "translation_z_mm": float(case["translation"][2]),
            "rotation_x_deg": float(case["rotation_deg"][0]),
            "rotation_y_deg": float(case["rotation_deg"][1]),
            "rotation_z_deg": float(case["rotation_deg"][2]),
            "score": score,
            "mean_error_mm": mean_error,
            "R_mm": ref_radius,
        })

    return rows


# ============================================================
# REAL DATASET ANALYSIS
# ============================================================
def reconstruct_pair_individual_bubbles(
    rect_top,
    rect_side,
    diameter_mm,
    voxel_mm,
    smooth_sigma_z,
    min_radius_vox,
    min_area_cc,
    iou_thr,
):
    """
    Reconstructs every matched top-side component as a separate bubble.
    """
    top_components = connected_components(rect_top, min_area=min_area_cc)
    side_components = connected_components(rect_side, min_area=min_area_cc)

    pairs = match_components_by_longitudinal_overlap(
        top_components,
        side_components,
        iou_thr=iou_thr,
    )

    reconstructed = []

    for bubble_index, (top_mask, side_mask, pair_iou) in enumerate(pairs):
        volume, _, current_voxel_mm = build_volume_elliptic_from_two_masks(
            top_mask,
            side_mask,
            diameter_mm=diameter_mm,
            voxel_mm=voxel_mm,
            smooth_sigma_z=smooth_sigma_z,
            min_radius_vox=min_radius_vox,
        )

        reconstructed.append({
            "bubble_index": int(bubble_index),
            "pair_iou": float(pair_iou),
            "volume": volume,
            "voxel_mm": current_voxel_mm,
            "filled_voxels": int(volume.sum()),
        })

    return reconstructed


def find_bubble_category_id(categories):
    """
    Finds category id for category named 'bubble'.
    """
    for category in categories:
        if category.get("name") == "bubble":
            return category.get("id")

    raise RuntimeError("Nie znaleziono kategorii 'bubble' w COCO.")


def analyse_real_dataset(dataset_dir, coco_file, start_frame, n_frames):
    """
    Calculates the rotational fit coefficient for real reconstructed bubbles.
    """
    dataset_dir = Path(dataset_dir)
    coco_path = dataset_dir / coco_file

    print(f"[INFO] COCO path: {coco_path}")
    print(f"[INFO] COCO exists: {coco_path.exists()}")

    coco = load_coco(str(coco_path))

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    bubble_category_id = find_bubble_category_id(categories)

    images_sorted = sorted(images, key=lambda item: item.get("file_name", ""))

    start_idx = max(0, int(start_frame) - 1)
    end_idx = min(len(images_sorted), start_idx + max(1, int(n_frames)))
    selected_images = images_sorted[start_idx:end_idx]

    if not selected_images:
        raise RuntimeError("Zakres START_FRAME / N_FRAMES jest poza listą obrazów.")

    rows = []

    for local_index, image_info in enumerate(selected_images):
        global_frame_number = start_idx + local_index + 1
        image_id = image_info["id"]
        image_path = dataset_dir / image_info["file_name"]

        print(f"[REAL] frame={global_frame_number}, file={image_info['file_name']}")

        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if gray is None:
            print(f"[WARN] Nie mogę wczytać obrazu: {image_path}")
            continue

        image_h, image_w = gray.shape
        x_left = 0
        x_right = image_w - 1

        bubble_annotations = [
            ann for ann in annotations
            if ann["image_id"] == image_id and ann["category_id"] == bubble_category_id
        ]

        tube_annotations = {
            0: [],
            1: [],
            2: [],
            3: [],
        }

        for ann in bubble_annotations:
            bbox = ann.get("bbox", None)

            if bbox is None:
                continue

            x, y, w_box, h_box = map(float, bbox)
            x_center = x + 0.5 * w_box
            y_center = y + 0.5 * h_box

            for tube_index, tube in enumerate(TUBES):
                if point_in_tube(x_center, y_center, tube, margin_px=MARGIN_PX):
                    tube_annotations[tube_index].append(ann)
                    break

        mask1_orig = build_bubble_mask_from_anns(tube_annotations[0], image_h, image_w)
        mask2_orig = build_bubble_mask_from_anns(tube_annotations[1], image_h, image_w)

        rect_top_12, rect_side_12, _, _ = rectify_and_align_pair(
            mask1_orig,
            TUBES[0],
            mask2_orig,
            TUBES[1],
            x_left=x_left,
            x_right=x_right,
            inner_margin_px=INNER_MARGIN_PX,
            keep_aspect=KEEP_ASPECT,
        )

        mask3_orig = build_bubble_mask_from_anns(tube_annotations[2], image_h, image_w)
        mask4_orig = build_bubble_mask_from_anns(tube_annotations[3], image_h, image_w)

        rect_top_34, rect_side_34, _, _ = rectify_and_align_pair(
            mask3_orig,
            TUBES[2],
            mask4_orig,
            TUBES[3],
            x_left=x_left,
            x_right=x_right,
            inner_margin_px=INNER_MARGIN_PX,
            keep_aspect=KEEP_ASPECT,
        )

        pair_results = [
            (
                "tube12",
                reconstruct_pair_individual_bubbles(
                    rect_top_12,
                    rect_side_12,
                    diameter_mm=DIAMETER_MM,
                    voxel_mm=VOXEL_MM,
                    smooth_sigma_z=SMOOTH_SIGMA_Z,
                    min_radius_vox=MIN_RADIUS_VOX,
                    min_area_cc=MIN_AREA_CC,
                    iou_thr=IOU_THR,
                )
            ),
            (
                "tube34",
                reconstruct_pair_individual_bubbles(
                    rect_top_34,
                    rect_side_34,
                    diameter_mm=DIAMETER_MM,
                    voxel_mm=VOXEL_MM,
                    smooth_sigma_z=SMOOTH_SIGMA_Z,
                    min_radius_vox=MIN_RADIUS_VOX,
                    min_area_cc=MIN_AREA_CC,
                    iou_thr=IOU_THR,
                )
            ),
        ]

        for tube_pair_name, bubbles in pair_results:
            for bubble in bubbles:
                volume = bubble["volume"]
                voxel_mm = bubble["voxel_mm"]

                surface_points = volume_to_surface_points_mm(
                    volume,
                    voxel_mm,
                    center_radial_xy=True,
                    max_points=500_000,
                )

                if surface_points is None or len(surface_points) < 10:
                    continue

                score, mean_error, ref_radius = calculate_rotational_fit_for_points(surface_points)

                rows.append({
                    "source": "real",
                    "test_type": "experimental",
                    "case_name": "real_bubble",
                    "frame_number": int(global_frame_number),
                    "image_file": image_info["file_name"],
                    "tube_pair": tube_pair_name,
                    "bubble_index": int(bubble["bubble_index"]),
                    "pair_iou": float(bubble["pair_iou"]),
                    "filled_voxels": int(bubble["filled_voxels"]),
                    "surface_points": int(len(surface_points)),
                    "score": score,
                    "mean_error_mm": mean_error,
                    "R_mm": ref_radius,
                })

    return rows


# ============================================================
# PLOTTING
# ============================================================
def apply_paper_plot_style(
    ax,
    xlim=None,
    ylim=None,
    xlabel="",
    ylabel="",
    aspect_equal=False,
):
    """
    Applies a simple publication-like plot style.
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.minorticks_on()

    ax.grid(True, which="major", linewidth=0.7, alpha=0.75)
    ax.grid(True, which="minor", linewidth=0.35, alpha=0.35)

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
    )

    if aspect_equal:
        ax.set_aspect("equal", adjustable="box")


def plot_synthetic_score_vs_asymmetry(synthetic_rows):
    rows = [
        row for row in synthetic_rows
        if row["test_type"] == "asymmetry_sweep"
    ]

    if not rows:
        return

    x = [row["asymmetry"] for row in rows]
    y = [row["score"] for row in rows]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        x,
        y,
        s=18,
        label="Rotational fit",
    )

    apply_paper_plot_style(
        ax,
        xlim=(0.0, 0.75),
        ylim=(0.5, 1.0),
        xlabel="Asymmetry coefficient, -",
        ylabel="Rotational fit score, -",
    )

    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "synthetic_score_vs_asymmetry.png", dpi=300)
    plt.close(fig)


def plot_synthetic_error_vs_asymmetry(synthetic_rows):
    rows = [
        row for row in synthetic_rows
        if row["test_type"] == "asymmetry_sweep"
    ]

    if not rows:
        return

    x = [row["asymmetry"] for row in rows]
    y = [row["mean_error_mm"] for row in rows]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        x,
        y,
        s=18,
        label="Mean radial error",
    )

    apply_paper_plot_style(
        ax,
        xlim=(0.0, 0.75),
        ylim=(0.0, max(y) * 1.1),
        xlabel="Asymmetry coefficient, -",
        ylabel="Mean radial error, mm",
    )

    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "synthetic_error_vs_asymmetry.png", dpi=300)
    plt.close(fig)


def plot_synthetic_score_vs_noise(synthetic_rows):
    rows = [
        row for row in synthetic_rows
        if row["test_type"] == "noise_sweep"
    ]

    if not rows:
        return

    rng = np.random.default_rng(12345)

    x_raw = np.array([row["noise_std_mm"] for row in rows], dtype=float)
    y = np.array([row["score"] for row in rows], dtype=float)

    jitter = rng.normal(
        loc=0.0,
        scale=NOISE_STEP_MM * 0.06,
        size=x_raw.shape,
    )

    x = np.clip(x_raw + jitter, NOISE_MIN_MM, NOISE_MAX_MM)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        x,
        y,
        s=8,
        alpha=0.35,
        label="Independent realisations",
    )

    apply_paper_plot_style(
        ax,
        xlim=(0.0, 1.05),
        ylim=(0.75, 1.0),
        xlabel="Noise standard deviation, mm",
        ylabel="Rotational fit score, -",
    )

    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "synthetic_score_vs_noise.png", dpi=300)
    plt.close(fig)


def plot_real_score_distribution(real_rows):
    scores = [
        row["score"] for row in real_rows
        if row.get("score") is not None
    ]

    if not scores:
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.hist(
        scores,
        bins=15,
        edgecolor="black",
        linewidth=0.7,
    )

    apply_paper_plot_style(
        ax,
        xlim=(0.5, 1.0),
        ylim=None,
        xlabel="Rotational fit score, -",
        ylabel="Number of bubbles",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "real_score_distribution.png", dpi=300)
    plt.close(fig)


def create_plots(synthetic_rows, real_rows):
    if synthetic_rows:
        plot_synthetic_score_vs_asymmetry(synthetic_rows)
        plot_synthetic_error_vs_asymmetry(synthetic_rows)
        plot_synthetic_score_vs_noise(synthetic_rows)

    if real_rows:
        plot_real_score_distribution(real_rows)


# ============================================================
# PRINT SUMMARY
# ============================================================
def print_synthetic_summary(synthetic_rows):
    print("\n=== Synthetic asymmetry sweep ===")

    asymmetry_rows = [
        row for row in synthetic_rows
        if row["test_type"] == "asymmetry_sweep"
    ]

    if asymmetry_rows:
        selected_asymmetries = [0.00, 0.10, 0.20, 0.35, 0.50, 0.70]

        for asymmetry in selected_asymmetries:
            matching = [
                row for row in asymmetry_rows
                if abs(row["asymmetry"] - asymmetry) < 1e-9
            ]

            if not matching:
                continue

            row = matching[0]

            print(
                f"asymmetry={row['asymmetry']:.2f} | "
                f"score={row['score']:.4f} | "
                f"error={row['mean_error_mm']:.4f} mm | "
                f"R={row['R_mm']:.4f} mm"
            )

    print("\n=== Synthetic noise sweep ===")

    noise_rows = [
        row for row in synthetic_rows
        if row["test_type"] == "noise_sweep"
    ]

    if noise_rows:
        noise_levels = sorted(set(row["noise_std_mm"] for row in noise_rows))

        for noise in noise_levels:
            scores = np.array([
                row["score"] for row in noise_rows
                if row["noise_std_mm"] == noise
            ], dtype=float)

            errors = np.array([
                row["mean_error_mm"] for row in noise_rows
                if row["noise_std_mm"] == noise
            ], dtype=float)

            print(
                f"noise={noise:.2f} mm | "
                f"score mean={np.mean(scores):.4f} | "
                f"score std={np.std(scores):.4f} | "
                f"error mean={np.mean(errors):.4f} mm"
            )

    print("\n=== Invariance test ===")

    invariance_rows = [
        row for row in synthetic_rows
        if row["test_type"] == "invariance_test"
    ]

    for row in invariance_rows:
        print(
            f"{row['case_name']:28s} | "
            f"score={row['score']:.4f} | "
            f"error={row['mean_error_mm']:.4f} mm | "
            f"R={row['R_mm']:.4f} mm"
        )


def print_real_summary(real_rows):
    print("\n=== Real bubbles ===")

    scores = [
        row["score"] for row in real_rows
        if row.get("score") is not None
    ]

    if not scores:
        print("No real bubble scores calculated.")
        return

    print(f"count = {len(scores)}")
    print(f"mean score = {np.mean(scores):.4f}")
    print(f"std score  = {np.std(scores):.4f}")
    print(f"min score  = {np.min(scores):.4f}")
    print(f"max score  = {np.max(scores):.4f}")


def print_generated_files():
    print(f"\nSaved output directory: {OUTPUT_DIR}")
    print("\nGenerated files:")

    for path in sorted(OUTPUT_DIR.iterdir()):
        print(path)


# ============================================================
# MAIN RUNNER
# ============================================================
def run_analysis():
    clear_output_directory()

    synthetic_rows = []
    real_rows = []

    if RUN_SYNTHETIC_ANALYSIS:
        synthetic_rows = analyse_synthetic_bubbles()

    if RUN_REAL_ANALYSIS:
        real_rows = analyse_real_dataset(
            dataset_dir=DATASET_DIR,
            coco_file=COCO_FILE,
            start_frame=START_FRAME,
            n_frames=N_FRAMES,
        )

    if synthetic_rows:
        save_csv(
            synthetic_rows,
            OUTPUT_DIR / "synthetic_rotational_fit_results.csv",
        )

    if real_rows:
        save_csv(
            real_rows,
            OUTPUT_DIR / "real_rotational_fit_results.csv",
        )

    if synthetic_rows or real_rows:
        save_csv(
            synthetic_rows + real_rows,
            OUTPUT_DIR / "all_rotational_fit_results.csv",
        )

    save_summary_json(synthetic_rows, real_rows)
    create_plots(synthetic_rows, real_rows)

    if synthetic_rows:
        print_synthetic_summary(synthetic_rows)

    if real_rows:
        print_real_summary(real_rows)

    print_generated_files()

    try:
        os.startfile(OUTPUT_DIR)
    except Exception:
        pass


if __name__ == "__main__":
    run_analysis()