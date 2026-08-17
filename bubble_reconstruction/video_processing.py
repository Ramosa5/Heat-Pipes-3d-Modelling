from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import ReconstructionConfig
from .prediction import MaskDetection, draw_detections, load_maskrcnn_model, predict_frame_masks
from .processing import process_masks_frame
from .tracking import BubbleTracker
from .tube_geometry import point_in_tube


def detections_to_tube_masks(detections: list[MaskDetection],
                             frame_shape: tuple[int, int],
                             tubes: list[dict[str, float]],
                             margin_px: float = 2.0) -> tuple[dict[int, np.ndarray], dict[int, list[dict[str, Any]]]]:
    """Assign predicted masks to tube1..tube4 by bbox centre.

    Returns:
        masks_by_tube: {0..3: uint8 mask 0/255}
        tube_anns: COCO-like lightweight annotations for optional overlay/debug.
    """
    h, w = frame_shape[:2]
    masks_by_tube: dict[int, np.ndarray] = {i: np.zeros((h, w), dtype=np.uint8) for i in range(4)}
    tube_anns: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: [], 3: []}

    for det_idx, det in enumerate(detections, start=1):
        xc, yc = det.center
        assigned_tube: int | None = None
        for tube_idx, tube in enumerate(tubes):
            if point_in_tube(xc, yc, tube, margin_px=margin_px):
                assigned_tube = tube_idx
                break

        if assigned_tube is None:
            continue

        mask = det.mask
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        masks_by_tube[assigned_tube] = cv2.bitwise_or(masks_by_tube[assigned_tube], (mask > 0).astype(np.uint8) * 255)

        x, y, bw, bh = det.bbox_xywh
        tube_anns[assigned_tube].append({
            "id": int(det_idx),
            "image_id": None,
            "category_id": 1,
            "bbox": [float(x), float(y), float(bw), float(bh)],
            "score": float(det.score),
            "source": "maskrcnn_video",
        })

    return masks_by_tube, tube_anns


def run_video_pipeline(config: ReconstructionConfig,
                       video_path: str,
                       model_path: str,
                       score_threshold: float = 0.3,
                       mask_threshold: float = 0.3,
                       step: int = 1,
                       save_detection_video: str | None = None,
                       show_detection_window: bool = False) -> list[dict[str, Any]]:
    """Run the 3D reconstruction pipeline using Mask R-CNN predictions from a video."""
    model, device = load_maskrcnn_model(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Keep main.py semantics: start_frame is 1-based, like COCO image selection.
    start_frame_1based = max(1, int(config.start_frame))
    start_idx = max(0, min(start_frame_1based - 1, max(0, total_frames - 1)))
    n_frames = max(1, int(config.n_frames))
    end_idx_exclusive = min(total_frames, start_idx + n_frames)
    step = max(1, int(step))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    print(f"[VIDEO] Video: {video_path}")
    print(f"[VIDEO] Frames: {total_frames}")
    print(f"[VIDEO] FPS: {fps:.3f}")
    print(f"[VIDEO] Resolution: {width}x{height}")
    print(f"[VIDEO] Start frame (1-based) = {start_frame_1based} -> idx={start_idx}")
    print(f"[VIDEO] Processing frames: [{start_idx}:{end_idx_exclusive}] step={step}")

    writer = None
    if save_detection_video:
        out_path = Path(save_detection_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps / step if fps > 0 else 25.0, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create detection video: {save_detection_video}")

    tracker = BubbleTracker(
        max_distance_mm=config.tracking_max_distance_mm,
        max_missing_frames=config.tracking_max_missing_frames,
        debug=config.tracking_debug,
    ) if config.tracking else None

    frames_data: list[dict[str, Any]] = []
    local_processed = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if frame_idx >= end_idx_exclusive:
            break
        if (frame_idx - start_idx) % step != 0:
            continue

        frame_no = frame_idx + 1
        file_name = f"video_frame_{frame_no:06d}.png"
        local_processed += 1

        detections = predict_frame_masks(
            model=model,
            frame_bgr=frame_bgr,
            device=device,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
        )

        masks_by_tube, tube_anns = detections_to_tube_masks(
            detections=detections,
            frame_shape=frame_bgr.shape[:2],
            tubes=config.tubes,
            margin_px=config.margin_px,
        )

        print(f"[VIDEO] frame={frame_no} detections={len(detections)} assigned={sum(len(v) for v in tube_anns.values())}")

        if writer is not None or show_detection_window:
            vis = draw_detections(frame_bgr, detections)
            cv2.rectangle(vis, (0, 0), (min(width, 620), 76), (0, 0, 0), -1)
            cv2.putText(vis, f"Predicted bubbles: {len(detections)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"Frame: {frame_no}/{total_frames}", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            if writer is not None:
                writer.write(vis)
            if show_detection_window:
                cv2.imshow("Mask R-CNN predictions used for 3D reconstruction", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

        frame_data = process_masks_frame(
            frame_image=frame_bgr,
            masks_by_tube=masks_by_tube,
            tube_anns=tube_anns,
            file_name=file_name,
            global_frame_no=frame_no,
            local_frame_no=local_processed,
            total_selected=max(1, len(range(start_idx, end_idx_exclusive, step))),
            config=config,
            tracker=tracker,
        )
        if frame_data is not None:
            frames_data.append(frame_data)

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[VIDEO] Saved detection debug video: {save_detection_video}")
    if show_detection_window:
        cv2.destroyWindow("Mask R-CNN predictions used for 3D reconstruction")

    if not frames_data:
        raise RuntimeError("No video frames were reconstructed. Check --video, --start-frame, --n-frames and prediction thresholds.")

    if config.show_preview:
        from .visualization import pv_live_animate_keep_last

        pv_live_animate_keep_last(
            frames_data,
            center_radial_xy=config.center_radial_xy,
            max_points=config.preview_max_points,
            point_size=config.preview_point_size,
            pause_s=config.preview_pause_s,
            zoom_out=config.preview_zoom_out,
            show_pipe=config.show_pipe,
            pipe_opacity=config.pipe_opacity,
            center_view_on_origin=config.center_view_on_origin,
            show_origin_marker=config.show_origin_marker,
            show_tracking_labels=config.tracking_labels,
            show_parameter_labels=config.preview_parameter_labels,
            fullscreen=config.preview_fullscreen,
            compact=config.preview_compact,
            window_size=(config.preview_window_width, config.preview_window_height),
            distance_scale=config.preview_distance_scale,
        )

    if config.summary_visualization:
        from .summary_visualization import show_summary_visualization

        show_summary_visualization(frames_data, config)

    return frames_data
