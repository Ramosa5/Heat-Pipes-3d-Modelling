import argparse
import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F

DEFAULT_MODEL_PATH = "new_best_maskrcnn_bubble.pth"
DEFAULT_VIDEO_PATH = "C001H002S0001.avi"
DEFAULT_SCORE_THRESHOLD = 0.3
DEFAULT_MASK_THRESHOLD = 0.3


def get_model(num_classes=2):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features_box, num_classes
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def predict_frame(model, frame_bgr, device, score_threshold, mask_threshold):
    """Run Mask R-CNN on one OpenCV BGR frame and return visualization + count."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    x = F.to_tensor(image).to(device)

    with torch.no_grad():
        pred = model([x])[0]

    boxes = pred["boxes"].detach().cpu().numpy()
    scores = pred["scores"].detach().cpu().numpy()
    masks = pred["masks"].detach().cpu().numpy()

    output = frame_bgr.copy()
    bubble_count = 0

    for box, score, mask in zip(boxes, scores, masks):
        if score < score_threshold:
            continue

        bubble_count += 1
        x1, y1, x2, y2 = box.astype(int)

        # Bounding box + confidence
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            output,
            f"{score:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Mask overlay
        binary_mask = mask[0] > mask_threshold
        if np.any(binary_mask):
            overlay = output.copy()
            overlay[binary_mask] = (0, 255, 0)
            output = cv2.addWeighted(overlay, 0.30, output, 0.70, 0)

    return output, bubble_count


def main():
    parser = argparse.ArgumentParser(
        description="Mask R-CNN bubble detection on an entire video."
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Path to input AVI/video")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to .pth model")
    parser.add_argument("--score", type=float, default=DEFAULT_SCORE_THRESHOLD,
                        help="Detection score threshold")
    parser.add_argument("--mask", type=float, default=DEFAULT_MASK_THRESHOLD,
                        help="Mask threshold")
    parser.add_argument("--save", default=None,
                        help="Optional path for output video, e.g. bubbles_detected.avi")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Frame number to start from (0-based)")
    parser.add_argument("--step", type=int, default=1,
                        help="Process every Nth frame (1 = every frame)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = get_model(num_classes=2)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = max(0, min(args.start_frame, max(0, total_frames - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"Video: {args.video}")
    print(f"Frames: {total_frames}")
    print(f"FPS: {fps:.3f}")
    print(f"Resolution: {width}x{height}")
    print("Controls: SPACE = pause/resume, A/D = previous/next frame while paused, Q/ESC = quit")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_fps = fps / max(1, args.step) if fps > 0 else 25.0
        writer = cv2.VideoWriter(args.save, fourcc, out_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {args.save}")

    paused = False
    current_frame_idx = start_frame
    last_raw_frame = None
    last_visualized = None
    last_count = 0

    while True:
        if not paused or last_raw_frame is None:
            ok, frame = cap.read()
            if not ok:
                print("End of video.")
                break

            current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            # Skip frames when --step > 1
            if args.step > 1 and ((current_frame_idx - start_frame) % args.step != 0):
                continue

            last_raw_frame = frame.copy()
            visualized, bubble_count = predict_frame(
                model,
                frame,
                device,
                args.score,
                args.mask,
            )
            last_visualized = visualized
            last_count = bubble_count

            if writer is not None:
                writer.write(visualized)
        else:
            visualized = last_visualized.copy()
            bubble_count = last_count

        # Header with current position and bubble count
        cv2.rectangle(visualized, (0, 0), (min(width, 560), 76), (0, 0, 0), -1)
        cv2.putText(
            visualized,
            f"Bubbles: {bubble_count}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            visualized,
            f"Frame: {current_frame_idx + 1}/{total_frames}",
            (15, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Mask R-CNN - Bubble detection", visualized)

        # If running, ~1 ms; if paused, wait for keyboard input.
        key = cv2.waitKey(0 if paused else 1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif paused and key in (ord("d"), 83):  # D/right arrow
            target = min(total_frames - 1, current_frame_idx + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            if ok:
                current_frame_idx = target
                last_raw_frame = frame.copy()
                last_visualized, last_count = predict_frame(
                    model, frame, device, args.score, args.mask
                )
        elif paused and key in (ord("a"), 81):  # A/left arrow
            target = max(0, current_frame_idx - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            if ok:
                current_frame_idx = target
                last_raw_frame = frame.copy()
                last_visualized, last_count = predict_frame(
                    model, frame, device, args.score, args.mask
                )

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved output video to: {args.save}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
