from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


TRACKING_FIELDS = [
    "frame_no",
    "file_name",
    "tube_pair",
    "track_id",
    "detection_index",
    "match_status",
    "match_distance_mm",
    "track_age_frames",
    "missed_frames",
    "centroid_x_mm",
    "centroid_y_mm",
    "centroid_z_mm",
    "z_min_mm",
    "z_max_mm",
    "volume_voxels",
    "point_count",
]


@dataclass
class BubbleDetection:
    """One reconstructed bubble candidate in one frame and one tube pair."""

    frame_no: int
    file_name: str
    tube_pair: str
    detection_index: int
    centroid: np.ndarray
    z_min_mm: float
    z_max_mm: float
    volume_voxels: int
    point_count: int


@dataclass
class BubbleTrack:
    """State of a tracked bubble across frames."""

    track_id: int
    tube_pair: str
    last_centroid: np.ndarray
    last_z_min_mm: float
    last_z_max_mm: float
    last_frame_no: int
    age_frames: int = 1
    missed_frames: int = 0
    active: bool = True


def _volume_points_mm_no_sampling(volume_bool: np.ndarray,
                                  voxel_mm: float,
                                  center_radial_xy: bool = True) -> np.ndarray | None:
    """Convert all occupied voxels to metric coordinates without random sampling."""
    if volume_bool is None or volume_bool.sum() == 0:
        return None

    pts = np.argwhere(volume_bool)
    if pts.size == 0:
        return None

    x_rad, y_rad, z_len = volume_bool.shape
    cx = (x_rad - 1) / 2.0
    cy = (y_rad - 1) / 2.0

    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    z = pts[:, 2].astype(np.float64) * float(voxel_mm)

    if center_radial_xy:
        x = (x - cx) * float(voxel_mm)
        y = (y - cy) * float(voxel_mm)
    else:
        x = x * float(voxel_mm)
        y = y * float(voxel_mm)

    return np.column_stack([x, y, z])


def detections_from_reconstructed_bubbles(bubbles: list[dict[str, Any]],
                                          frame_no: int,
                                          file_name: str,
                                          tube_pair: str,
                                          center_radial_xy: bool = True) -> list[BubbleDetection]:
    """Build tracking detections from individual per-bubble reconstruction outputs."""
    detections: list[BubbleDetection] = []

    for fallback_index, bubble in enumerate(bubbles, start=1):
        volume = bubble.get("volume")
        voxel_mm = float(bubble.get("voxel_mm", 1.0))
        pts = _volume_points_mm_no_sampling(volume, voxel_mm, center_radial_xy=center_radial_xy)
        if pts is None or len(pts) == 0:
            continue

        centroid = pts.mean(axis=0).astype(np.float64)
        detection_index = int(bubble.get("detection_index", fallback_index))
        detections.append(
            BubbleDetection(
                frame_no=int(frame_no),
                file_name=file_name,
                tube_pair=tube_pair,
                detection_index=detection_index,
                centroid=centroid,
                z_min_mm=float(np.min(pts[:, 2])),
                z_max_mm=float(np.max(pts[:, 2])),
                volume_voxels=int(np.count_nonzero(volume)),
                point_count=int(len(pts)),
            )
        )

    # Stable order makes CSV rows and labels deterministic.
    detections.sort(key=lambda d: (d.centroid[2], d.centroid[0], d.centroid[1]))
    return detections


class BubbleTracker:
    """
    Simple online tracker for bubbles moving through one or more tube pairs.

    Matching rule:
    - for each frame and tube pair, compare new detections to active tracks from the same tube pair,
    - assign detections greedily by nearest 3D centroid distance,
    - accept a match only when the distance is not larger than max_distance_mm,
    - unmatched detections start new tracks,
    - unmatched tracks are closed once missed_frames > max_missing_frames.

    With max_missing_frames=0, a bubble stops being tracked immediately in the first frame
    where it is no longer detected.
    """

    def __init__(self,
                 max_distance_mm: float = 12.0,
                 max_missing_frames: int = 0,
                 debug: bool = False) -> None:
        self.max_distance_mm = float(max_distance_mm)
        self.max_missing_frames = int(max_missing_frames)
        self.debug = bool(debug)
        self._next_track_id = 1
        self.tracks: dict[int, BubbleTrack] = {}

    def update(self,
               detections: list[BubbleDetection],
               frame_no: int,
               file_name: str,
               tube_pair: str) -> tuple[list[dict[str, object]], list[int]]:
        """Update tracks for one tube pair in one frame and return CSV-ready assignment rows."""
        active_tracks = [
            track for track in self.tracks.values()
            if track.active and track.tube_pair == tube_pair
        ]

        pairs: list[tuple[float, int, int]] = []
        for track_idx, track in enumerate(active_tracks):
            for det_idx, det in enumerate(detections):
                dist = float(np.linalg.norm(det.centroid - track.last_centroid))
                if dist <= self.max_distance_mm:
                    pairs.append((dist, track_idx, det_idx))
        pairs.sort(key=lambda item: item[0])

        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        det_to_track: dict[int, tuple[BubbleTrack, float, str]] = {}

        for dist, track_idx, det_idx in pairs:
            track = active_tracks[track_idx]
            if track.track_id in assigned_tracks or det_idx in assigned_detections:
                continue
            assigned_tracks.add(track.track_id)
            assigned_detections.add(det_idx)
            det_to_track[det_idx] = (track, dist, "matched")

        closed_track_ids: list[int] = []
        for track in active_tracks:
            if track.track_id in assigned_tracks:
                continue
            track.missed_frames += 1
            if track.missed_frames > self.max_missing_frames:
                track.active = False
                closed_track_ids.append(track.track_id)

        for det_idx, det in enumerate(detections):
            if det_idx in det_to_track:
                continue
            new_track = BubbleTrack(
                track_id=self._next_track_id,
                tube_pair=tube_pair,
                last_centroid=det.centroid.copy(),
                last_z_min_mm=det.z_min_mm,
                last_z_max_mm=det.z_max_mm,
                last_frame_no=int(frame_no),
                age_frames=1,
                missed_frames=0,
                active=True,
            )
            self.tracks[new_track.track_id] = new_track
            self._next_track_id += 1
            det_to_track[det_idx] = (new_track, 0.0, "new")

        rows: list[dict[str, object]] = []
        for det_idx, det in enumerate(detections):
            track, dist, status = det_to_track[det_idx]
            if status == "matched":
                track.last_centroid = det.centroid.copy()
                track.last_z_min_mm = det.z_min_mm
                track.last_z_max_mm = det.z_max_mm
                track.last_frame_no = int(frame_no)
                track.age_frames += 1
                track.missed_frames = 0

            rows.append({
                "frame_no": int(frame_no),
                "file_name": file_name,
                "tube_pair": tube_pair,
                "track_id": int(track.track_id),
                "detection_index": int(det.detection_index),
                "match_status": status,
                "match_distance_mm": float(dist),
                "track_age_frames": int(track.age_frames),
                "missed_frames": int(track.missed_frames),
                "centroid_x_mm": float(det.centroid[0]),
                "centroid_y_mm": float(det.centroid[1]),
                "centroid_z_mm": float(det.centroid[2]),
                "z_min_mm": float(det.z_min_mm),
                "z_max_mm": float(det.z_max_mm),
                "volume_voxels": int(det.volume_voxels),
                "point_count": int(det.point_count),
            })

        if self.debug:
            for row in rows:
                print(
                    f"[TRACK] frame={row['frame_no']} {row['tube_pair']} "
                    f"track={row['track_id']} det={row['detection_index']} "
                    f"status={row['match_status']} dist={float(row['match_distance_mm']):.2f} mm"
                )
            for closed_id in closed_track_ids:
                print(f"[TRACK] frame={frame_no} {tube_pair} closed track={closed_id}")

        return rows, closed_track_ids


def tracking_label_records(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Convert tracking CSV rows into compact records for PyVista point labels."""
    labels: list[dict[str, object]] = []
    for row in rows:
        labels.append({
            "position": (
                float(row["centroid_x_mm"]),
                float(row["centroid_y_mm"]),
                float(row["centroid_z_mm"]),
            ),
            "label": f"ID {int(row['track_id'])}",
        })
    return labels
