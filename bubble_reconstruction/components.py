from __future__ import annotations

import cv2
import numpy as np

def connected_components(mask_2d: np.ndarray, min_area: int = 80):
    m = (mask_2d > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        comps.append(comp)
    return comps


def longitudinal_span(mask_2d: np.ndarray):
    xs = np.where(mask_2d.max(axis=0) > 0)[0]
    if xs.size == 0:
        return None
    return int(xs.min()), int(xs.max())


def span_iou(a, b):
    if a is None or b is None:
        return 0.0
    ax0, ax1 = a
    bx0, bx1 = b
    inter = max(0, min(ax1, bx1) - max(ax0, bx0) + 1)
    union = (ax1 - ax0 + 1) + (bx1 - bx0 + 1) - inter
    return inter / union if union > 0 else 0.0


def match_components_by_longitudinal_overlap(top_comps, side_comps, iou_thr=0.15):
    top_spans = [longitudinal_span(c) for c in top_comps]
    side_spans = [longitudinal_span(c) for c in side_comps]

    pairs = []
    used_side = set()
    for i, ts in enumerate(top_spans):
        best_j = -1
        best_iou = 0.0
        for j, ss in enumerate(side_spans):
            if j in used_side:
                continue
            iou = span_iou(ts, ss)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            used_side.add(best_j)
            pairs.append((top_comps[i], side_comps[best_j], best_iou))
    return pairs
