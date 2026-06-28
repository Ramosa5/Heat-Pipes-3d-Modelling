from __future__ import annotations

import json

import cv2
import numpy as np

def load_coco(path):
    with open(path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    return coco


def ann_to_mask(ann, img_h, img_w):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    seg = ann.get("segmentation", None)

    if isinstance(seg, list) and len(seg) > 0:
        for poly in seg:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
        return mask

    bbox = ann.get("bbox", None)
    if bbox is None or len(bbox) != 4:
        return mask

    x, y, w, h = map(float, bbox)
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(img_w, int(round(x + w)))
    y1 = min(img_h, int(round(y + h)))
    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    return mask


def build_bubble_mask_from_anns(anns, img_h, img_w):
    m = np.zeros((img_h, img_w), dtype=np.uint8)
    for ann in anns:
        m = cv2.bitwise_or(m, ann_to_mask(ann, img_h, img_w))
    return m
