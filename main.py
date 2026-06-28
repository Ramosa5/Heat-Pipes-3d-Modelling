from __future__ import annotations

import argparse

from bubble_reconstruction.config import ReconstructionConfig
from bubble_reconstruction.fit_score import validate_existing_rotational_fit_score
from bubble_reconstruction.processing import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bubble 3D reconstruction runner. main.py only orchestrates the modular pipeline."
    )
    parser.add_argument("--dataset-dir", default="bubble.coco/train")
    parser.add_argument("--coco-file", default="_annotations.coco.json")
    parser.add_argument("--start-frame", type=int, default=100)
    parser.add_argument("--n-frames", type=int, default=10)

    parser.add_argument("--save-masks", action="store_true", help="Save original and rectified masks.")
    parser.add_argument("--save-point-clouds", action="store_true", help="Save reconstructed point clouds as PLY.")
    parser.add_argument("--no-preview", action="store_true", help="Disable the PyVista live preview window.")

    parser.add_argument("--diameter-mm", type=float, default=20.0)
    parser.add_argument("--voxel-mm", type=float, default=None)
    parser.add_argument("--smooth-sigma-z", type=float, default=2.0)
    parser.add_argument("--min-radius-vox", type=float, default=0.8)
    parser.add_argument("--min-area-cc", type=int, default=80)
    parser.add_argument("--iou-thr", type=float, default=0.15)

    parser.add_argument("--masks-dir", default="masks")
    parser.add_argument("--point-clouds-dir", default="point_clouds")

    parser.add_argument(
        "--validate-fit-score",
        action="store_true",
        help="Run synthetic validation of rotational_fit_score() and exit.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ReconstructionConfig:
    return ReconstructionConfig(
        dataset_dir=args.dataset_dir,
        coco_file=args.coco_file,
        start_frame=args.start_frame,
        n_frames=args.n_frames,
        save_masks=args.save_masks,
        save_point_clouds=args.save_point_clouds,
        show_preview=not args.no_preview,
        diameter_mm=args.diameter_mm,
        voxel_mm=args.voxel_mm,
        smooth_sigma_z=args.smooth_sigma_z,
        min_radius_vox=args.min_radius_vox,
        min_area_cc=args.min_area_cc,
        iou_thr=args.iou_thr,
        masks_dir=args.masks_dir,
        point_clouds_dir=args.point_clouds_dir,
    )


def main() -> None:
    args = parse_args()

    if args.validate_fit_score:
        validate_existing_rotational_fit_score()
        return

    config = build_config(args)
    run_pipeline(config)


if __name__ == "__main__":
    main()
