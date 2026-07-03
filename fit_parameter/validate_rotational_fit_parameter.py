import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_DIR = SCRIPT_DIR / "rotational_fit_parameter_validation"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_DIR = PROJECT_ROOT / "bubble.coco" / "train"
COCO_FILE = "_annotations.coco.json"

START_FRAME = 100
N_FRAMES = 30

DIAMETER_MM = 20.0
VOXEL_MM = None
SMOOTH_SIGMA_Z = 2.0
MIN_RADIUS_VOX = 0.8

MIN_AREA_CC = 80
IOU_THR = 0.15

MARGIN_PX = 2.0
INNER_MARGIN_PX = 2.0
KEEP_ASPECT = True

N_SECTIONS = 50
MIN_POINTS_PER_SECTION = 5
RADIUS_STATISTIC = "median"

# Same tube definitions as in main_old.py
TUBES = [
    {"a_top": 0.007, "b_top": 50,  "a_bot": 0.007, "b_bot": 100},
    {"a_top": 0.019, "b_top": 105, "a_bot": 0.019, "b_bot": 155},
    {"a_top": 0.005, "b_top": 322, "a_bot": 0.005, "b_bot": 380},
    {"a_top": 0.005, "b_top": 408, "a_bot": 0.005, "b_bot": 470},
]


# ============================================================
# SYNTHETIC BUBBLES
# ============================================================

def rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float):
    """
    Creates a 3D rotation matrix from XYZ Euler angles in degrees.
    """
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    rx_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    ry_matrix = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    rz_matrix = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    return rz_matrix @ ry_matrix @ rx_matrix


def generate_synthetic_bubble_surface(length_mm: float = 30.0,
                                      radius_mm: float = 7.0,
                                      asymmetry: float = 0.0,
                                      noise_std_mm: float = 0.0,
                                      n_z: int = 80,
                                      n_theta: int = 120,
                                      translation=(0.0, 0.0, 0.0),
                                      rotation_deg=(0.0, 0.0, 0.0),
                                      seed: int = 42):
    """
    Generates a synthetic bubble surface.

    asymmetry = 0.0:
        ideal ellipsoid of revolution

    asymmetry > 0.0:
        radius depends on angle, so the object becomes less rotationally symmetric
    """
    rng = np.random.default_rng(seed)

    half_length = length_mm / 2.0
    z_values = np.linspace(-half_length, half_length, n_z)
    theta_values = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    points = []

    for z in z_values:
        z_norm = z / half_length

        # Ellipsoid-like radius profile along the main axis
        base_radius = radius_mm * np.sqrt(max(0.0, 1.0 - z_norm ** 2))

        for theta in theta_values:
            # Angle-dependent deformation
            radius = base_radius * (1.0 + asymmetry * np.cos(2.0 * theta))

            x = radius * np.cos(theta)
            y = radius * np.sin(theta)

            points.append([x, y, z])

    points = np.asarray(points, dtype=np.float64)

    if noise_std_mm > 0.0:
        points += rng.normal(0.0, noise_std_mm, size=points.shape)

    rot = rotation_matrix_xyz(*rotation_deg)
    points = points @ rot.T

    points += np.asarray(translation, dtype=np.float64)

    return points


def analyse_synthetic_bubbles():
    """
    Generates synthetic bubbles with controlled geometry and computes rotational fit score.
    """
    parameters = {
        "length_mm": 30.0,
        "radius_mm": 7.0,
        "n_z": 80,
        "n_theta": 120,
        "n_sections": N_SECTIONS,
        "min_points_per_section": MIN_POINTS_PER_SECTION,
        "radius_statistic": RADIUS_STATISTIC,
        "asymmetry_values": [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70],
        "noise_values_mm": [0.0, 0.1, 0.25, 0.5, 1.0]
    }

    with open(OUTPUT_DIR / "synthetic_generation_parameters.json", "w", encoding="utf-8") as f:
        json.dump(parameters, f, indent=4)

    rows = []

    # ------------------------------------------------------------
    # Test 1: asymmetry sweep
    # ------------------------------------------------------------
    for asymmetry in parameters["asymmetry_values"]:
        points = generate_synthetic_bubble_surface(
            length_mm=parameters["length_mm"],
            radius_mm=parameters["radius_mm"],
            asymmetry=asymmetry,
            noise_std_mm=0.0,
            n_z=parameters["n_z"],
            n_theta=parameters["n_theta"]
        )

        score, mean_error, ref_radius = rotational_fit_score(
            points,
            n_sections=N_SECTIONS,
            min_points_per_section=MIN_POINTS_PER_SECTION,
            radius_statistic=RADIUS_STATISTIC
        )

        rows.append({
            "source": "synthetic",
            "test_type": "asymmetry_sweep",
            "asymmetry": asymmetry,
            "noise_std_mm": 0.0,
            "translation_x": 0.0,
            "translation_y": 0.0,
            "translation_z": 0.0,
            "rotation_x_deg": 0.0,
            "rotation_y_deg": 0.0,
            "rotation_z_deg": 0.0,
            "score": score,
            "mean_error_mm": mean_error,
            "R_mm": ref_radius
        })

    # ------------------------------------------------------------
    # Test 2: noise sweep
    # ------------------------------------------------------------
    for noise_std in parameters["noise_values_mm"]:
        points = generate_synthetic_bubble_surface(
            length_mm=parameters["length_mm"],
            radius_mm=parameters["radius_mm"],
            asymmetry=0.0,
            noise_std_mm=noise_std,
            n_z=parameters["n_z"],
            n_theta=parameters["n_theta"]
        )

        score, mean_error, ref_radius = rotational_fit_score(
            points,
            n_sections=N_SECTIONS,
            min_points_per_section=MIN_POINTS_PER_SECTION,
            radius_statistic=RADIUS_STATISTIC
        )

        rows.append({
            "source": "synthetic",
            "test_type": "noise_sweep",
            "asymmetry": 0.0,
            "noise_std_mm": noise_std,
            "translation_x": 0.0,
            "translation_y": 0.0,
            "translation_z": 0.0,
            "rotation_x_deg": 0.0,
            "rotation_y_deg": 0.0,
            "rotation_z_deg": 0.0,
            "score": score,
            "mean_error_mm": mean_error,
            "R_mm": ref_radius
        })

    # ------------------------------------------------------------
    # Test 3: translation and rotation invariance
    # ------------------------------------------------------------
    invariance_cases = [
        ("ideal_original", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("ideal_shifted", (100.0, -50.0, 200.0), (0.0, 0.0, 0.0)),
        ("ideal_rotated", (0.0, 0.0, 0.0), (25.0, -15.0, 40.0)),
        ("ideal_shifted_and_rotated", (100.0, -50.0, 200.0), (25.0, -15.0, 40.0)),
    ]

    for test_name, translation, rotation_deg in invariance_cases:
        points = generate_synthetic_bubble_surface(
            length_mm=parameters["length_mm"],
            radius_mm=parameters["radius_mm"],
            asymmetry=0.0,
            noise_std_mm=0.0,
            n_z=parameters["n_z"],
            n_theta=parameters["n_theta"],
            translation=translation,
            rotation_deg=rotation_deg
        )

        score, mean_error, ref_radius = rotational_fit_score(
            points,
            n_sections=N_SECTIONS,
            min_points_per_section=MIN_POINTS_PER_SECTION,
            radius_statistic=RADIUS_STATISTIC
        )

        rows.append({
            "source": "synthetic",
            "test_type": test_name,
            "asymmetry": 0.0,
            "noise_std_mm": 0.0,
            "translation_x": translation[0],
            "translation_y": translation[1],
            "translation_z": translation[2],
            "rotation_x_deg": rotation_deg[0],
            "rotation_y_deg": rotation_deg[1],
            "rotation_z_deg": rotation_deg[2],
            "score": score,
            "mean_error_mm": mean_error,
            "R_mm": ref_radius
        })

    return rows


# ============================================================
# REAL BUBBLES
# ============================================================

def reconstruct_pair_individual_bubbles(rect_top: np.ndarray,
                                        rect_side: np.ndarray,
                                        diameter_mm: float,
                                        voxel_mm: float,
                                        smooth_sigma_z: float,
                                        min_radius_vox: float,
                                        min_area_cc: int = 80,
                                        iou_thr: float = 0.15):
    """
    Reconstructs individual bubbles instead of merging them into one volume.

    This is important for validation because rotational fit should be computed
    separately for each bubble.
    """
    top_components = connected_components(rect_top, min_area=min_area_cc)
    side_components = connected_components(rect_side, min_area=min_area_cc)

    matched_pairs = match_components_by_longitudinal_overlap(
        top_components,
        side_components,
        iou_thr=iou_thr
    )

    results = []

    for bubble_index, (top_mask, side_mask, iou_value) in enumerate(matched_pairs, start=1):
        volume_bool, _, used_voxel_mm = build_volume_elliptic_from_two_masks(
            top_mask,
            side_mask,
            diameter_mm=diameter_mm,
            voxel_mm=voxel_mm,
            smooth_sigma_z=smooth_sigma_z,
            min_radius_vox=min_radius_vox
        )

        results.append({
            "bubble_index": bubble_index,
            "pair_iou": float(iou_value),
            "volume": volume_bool,
            "voxel_mm": used_voxel_mm,
            "filled_voxels": int(volume_bool.sum())
        })

    return results


def analyse_real_dataset(dataset_dir=DATASET_DIR,
                         coco_file=COCO_FILE,
                         start_frame=START_FRAME,
                         n_frames=N_FRAMES):
    """
    Computes rotational fit score for real reconstructed bubbles.
    """
    coco = load_coco(os.path.join(dataset_dir, coco_file))

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    bubble_cat_id = None
    for category in categories:
        if category["name"] == "bubble":
            bubble_cat_id = category["id"]
            break

    if bubble_cat_id is None:
        raise RuntimeError("Category 'bubble' was not found in COCO annotations.")

    images_sorted = sorted(images, key=lambda d: d.get("file_name", ""))

    start_idx = max(0, int(start_frame) - 1)
    end_idx = min(len(images_sorted), start_idx + max(1, int(n_frames)))
    selected_images = images_sorted[start_idx:end_idx]

    if not selected_images:
        raise RuntimeError("Selected frame range is empty.")

    real_rows = []

    for local_index, img_info in enumerate(selected_images, start=0):
        global_frame = start_idx + local_index + 1

        img_id = img_info["id"]
        img_path = os.path.join(dataset_dir, img_info["file_name"])

        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if gray is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue

        img_h, img_w = gray.shape

        x_left = 0
        x_right = img_w - 1

        bubble_annotations = [
            ann for ann in annotations
            if ann["image_id"] == img_id and ann["category_id"] == bubble_cat_id
        ]

        tube_annotations = {0: [], 1: [], 2: [], 3: []}

        for ann in bubble_annotations:
            bbox = ann.get("bbox", None)

            if bbox is None:
                continue

            x, y, w_box, h_box = map(float, bbox)
            xc = x + 0.5 * w_box
            yc = y + 0.5 * h_box

            for tube_index, tube in enumerate(TUBES):
                if point_in_tube(xc, yc, tube, margin_px=MARGIN_PX):
                    tube_annotations[tube_index].append(ann)
                    break

        # ------------------------------------------------------------
        # tube1 + tube2
        # ------------------------------------------------------------
        mask1_orig = build_bubble_mask_from_anns(tube_annotations[0], img_h, img_w)
        mask2_orig = build_bubble_mask_from_anns(tube_annotations[1], img_h, img_w)

        rect_top_12, rect_side_12, _, _ = rectify_and_align_pair(
            mask1_orig,
            TUBES[0],
            mask2_orig,
            TUBES[1],
            x_left=x_left,
            x_right=x_right,
            inner_margin_px=INNER_MARGIN_PX,
            keep_aspect=KEEP_ASPECT
        )

        # ------------------------------------------------------------
        # tube3 + tube4
        # ------------------------------------------------------------
        mask3_orig = build_bubble_mask_from_anns(tube_annotations[2], img_h, img_w)
        mask4_orig = build_bubble_mask_from_anns(tube_annotations[3], img_h, img_w)

        rect_top_34, rect_side_34, _, _ = rectify_and_align_pair(
            mask3_orig,
            TUBES[2],
            mask4_orig,
            TUBES[3],
            x_left=x_left,
            x_right=x_right,
            inner_margin_px=INNER_MARGIN_PX,
            keep_aspect=KEEP_ASPECT
        )

        tube_pairs = [
            ("tube12", rect_top_12, rect_side_12),
            ("tube34", rect_top_34, rect_side_34)
        ]

        for tube_pair_name, rect_top, rect_side in tube_pairs:
            individual_bubbles = reconstruct_pair_individual_bubbles(
                rect_top,
                rect_side,
                diameter_mm=DIAMETER_MM,
                voxel_mm=VOXEL_MM,
                smooth_sigma_z=SMOOTH_SIGMA_Z,
                min_radius_vox=MIN_RADIUS_VOX,
                min_area_cc=MIN_AREA_CC,
                iou_thr=IOU_THR
            )

            for bubble_data in individual_bubbles:
                surface_points = volume_to_surface_points_mm(
                    bubble_data["volume"],
                    bubble_data["voxel_mm"],
                    center_radial_xy=True
                )

                score, mean_error, ref_radius = rotational_fit_score(
                    surface_points,
                    n_sections=N_SECTIONS,
                    min_points_per_section=MIN_POINTS_PER_SECTION,
                    radius_statistic=RADIUS_STATISTIC
                )

                real_rows.append({
                    "source": "real",
                    "frame": global_frame,
                    "file_name": img_info["file_name"],
                    "tube_pair": tube_pair_name,
                    "bubble_index": bubble_data["bubble_index"],
                    "pair_iou": bubble_data["pair_iou"],
                    "filled_voxels": bubble_data["filled_voxels"],
                    "score": score,
                    "mean_error_mm": mean_error,
                    "R_mm": ref_radius
                })

        print(f"[REAL] frame={global_frame}, file={img_info['file_name']}")

    return real_rows


# ============================================================
# SAVING
# ============================================================

def save_csv(rows, path):
    if not rows:
        return

    keys = sorted(set().union(*(row.keys() for row in rows)))

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_json(synthetic_rows, real_rows):
    real_scores = [
        row["score"] for row in real_rows
        if row.get("score") is not None
    ]

    synthetic_scores = [
        row["score"] for row in synthetic_rows
        if row.get("score") is not None
    ]

    summary = {
        "real": {
            "count": len(real_scores),
            "mean_score": float(np.mean(real_scores)) if real_scores else None,
            "std_score": float(np.std(real_scores)) if real_scores else None,
            "min_score": float(np.min(real_scores)) if real_scores else None,
            "max_score": float(np.max(real_scores)) if real_scores else None,
        },
        "synthetic": {
            "count": len(synthetic_scores),
            "mean_score": float(np.mean(synthetic_scores)) if synthetic_scores else None,
            "std_score": float(np.std(synthetic_scores)) if synthetic_scores else None,
            "min_score": float(np.min(synthetic_scores)) if synthetic_scores else None,
            "max_score": float(np.max(synthetic_scores)) if synthetic_scores else None,
        },
        "configuration": {
            "dataset_dir": str(DATASET_DIR),
            "coco_file": COCO_FILE,
            "start_frame": int(START_FRAME),
            "n_frames": int(N_FRAMES),
            "diameter_mm": float(DIAMETER_MM),
            "n_sections": int(N_SECTIONS),
            "min_points_per_section": int(MIN_POINTS_PER_SECTION),
            "radius_statistic": RADIUS_STATISTIC,
        }
    }

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)


# ============================================================
# PLOTTING
# ============================================================
def apply_paper_plot_style(ax,
                           xlim=None,
                           ylim=None,
                           xlabel="",
                           ylabel="",
                           aspect_equal=False):
    """
    Applies publication-like plot formatting.
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

    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

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

    ax.plot(
        x,
        y,
        marker="o",
        linewidth=1.4,
        markersize=4,
        label="Rotational fit"
    )

    apply_paper_plot_style(
        ax,
        xlim=(0.0, max(x) + 0.05),
        ylim=(0.5, 1.0),
        xlabel="Asymmetry coefficient, -",
        ylabel="Rotational fit score, -"
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

    ax.plot(
        x,
        y,
        marker="o",
        linewidth=1.4,
        markersize=4,
        label="Mean radial error"
    )

    apply_paper_plot_style(
        ax,
        xlim=(0.0, max(x) + 0.05),
        ylim=(0.0, max(y) * 1.1),
        xlabel="Asymmetry coefficient, -",
        ylabel="Mean radial error, mm"
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

    x = [row["noise_std_mm"] for row in rows]
    y = [row["score"] for row in rows]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(
        x,
        y,
        marker="o",
        linewidth=1.4,
        markersize=4,
        label="Rotational fit"
    )

    apply_paper_plot_style(
        ax,
        xlim=(0.0, max(x) + 0.05),
        ylim=(0.8, 1.0),
        xlabel="Noise standard deviation, mm",
        ylabel="Rotational fit score, -"
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
        bins=12,
        edgecolor="black",
        linewidth=0.7
    )

    apply_paper_plot_style(
        ax,
        xlim=(0.5, 1.0),
        ylim=None,
        xlabel="Rotational fit score, -",
        ylabel="Number of bubbles"
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "real_score_distribution.png", dpi=300)
    plt.close(fig)


def create_plots(synthetic_rows, real_rows):
    plot_synthetic_score_vs_asymmetry(synthetic_rows)
    plot_synthetic_error_vs_asymmetry(synthetic_rows)
    plot_synthetic_score_vs_noise(synthetic_rows)
    plot_real_score_distribution(real_rows)


# ============================================================
# PRINTING
# ============================================================

def print_summary(synthetic_rows, real_rows):
    print("\n=== Synthetic cases ===")
    for row in synthetic_rows:
        print(
            f"{row['test_type']:28s} | "
            f"asymmetry={row['asymmetry']:.2f} | "
            f"noise={row['noise_std_mm']:.2f} mm | "
            f"score={row['score']:.4f} | "
            f"error={row['mean_error_mm']:.4f} mm | "
            f"R={row['R_mm']:.4f} mm"
        )

    real_scores = [
        row["score"] for row in real_rows
        if row.get("score") is not None
    ]

    print("\n=== Real bubbles ===")
    print(f"count = {len(real_scores)}")

    if real_scores:
        print(f"mean score = {np.mean(real_scores):.4f}")
        print(f"std score  = {np.std(real_scores):.4f}")
        print(f"min score  = {np.min(real_scores):.4f}")
        print(f"max score  = {np.max(real_scores):.4f}")

    print(f"\nSaved output directory: {OUTPUT_DIR}")


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_analysis():
    synthetic_rows = analyse_synthetic_bubbles()

    real_rows = analyse_real_dataset(
        dataset_dir=DATASET_DIR,
        coco_file=COCO_FILE,
        start_frame=START_FRAME,
        n_frames=N_FRAMES
    )

    save_csv(synthetic_rows, OUTPUT_DIR / "synthetic_rotational_fit_results.csv")
    save_csv(real_rows, OUTPUT_DIR / "real_rotational_fit_results.csv")
    save_csv(synthetic_rows + real_rows, OUTPUT_DIR / "all_rotational_fit_results.csv")

    save_summary_json(synthetic_rows, real_rows)

    create_plots(synthetic_rows, real_rows)

    print_summary(synthetic_rows, real_rows)


if __name__ == "__main__":
    run_analysis()