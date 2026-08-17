from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass
class MaskDetection:
    """One Mask R-CNN bubble prediction prepared for the 3D pipeline."""

    box_xyxy: tuple[float, float, float, float]
    score: float
    mask: np.ndarray

    @property
    def bbox_xywh(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.box_xyxy
        return float(x1), float(y1), float(x2 - x1), float(y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box_xyxy
        return 0.5 * float(x1 + x2), 0.5 * float(y1 + y2)


def get_model(num_classes: int = 2):
    """Create the same Mask R-CNN architecture used by predict_video.py."""
    import torchvision

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features_box,
        num_classes,
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes,
    )

    return model


def load_maskrcnn_model(model_path: str, device: str | None = None, num_classes: int = 2):
    """Load a trained bubble Mask R-CNN model and return (model, torch_device)."""
    import torch

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[VIDEO] Device: {torch_device}")

    model = get_model(num_classes=num_classes)
    state = torch.load(model_path, map_location=torch_device)
    model.load_state_dict(state)
    model.to(torch_device)
    model.eval()
    print(f"[VIDEO] Loaded Mask R-CNN model: {model_path}")
    return model, torch_device


def predict_frame_masks(model,
                        frame_bgr: np.ndarray,
                        device,
                        score_threshold: float = 0.3,
                        mask_threshold: float = 0.3) -> list[MaskDetection]:
    """Run Mask R-CNN on one BGR frame and return raw binary masks.

    This function intentionally returns the masks before drawing overlays. These
    masks are the replacement for COCO annotation masks in the 3D reconstruction
    pipeline.
    """
    import torch
    from torchvision.transforms import functional as F

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    x = F.to_tensor(image).to(device)

    with torch.no_grad():
        pred: dict[str, Any] = model([x])[0]

    boxes = pred["boxes"].detach().cpu().numpy()
    scores = pred["scores"].detach().cpu().numpy()
    masks = pred["masks"].detach().cpu().numpy()

    detections: list[MaskDetection] = []
    for box, score, mask in zip(boxes, scores, masks):
        if float(score) < float(score_threshold):
            continue
        binary_mask = (mask[0] > float(mask_threshold)).astype(np.uint8) * 255
        if not np.any(binary_mask):
            continue
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        detections.append(MaskDetection(box_xyxy=(x1, y1, x2, y2), score=float(score), mask=binary_mask))

    return detections


def draw_detections(frame_bgr: np.ndarray,
                    detections: list[MaskDetection],
                    color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Optional 2D debug visualization of predicted masks and scores."""
    output = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.box_xyxy]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            output,
            f"{det.score:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        binary_mask = det.mask > 0
        overlay = output.copy()
        overlay[binary_mask] = color
        output = cv2.addWeighted(overlay, 0.30, output, 0.70, 0)
    return output
