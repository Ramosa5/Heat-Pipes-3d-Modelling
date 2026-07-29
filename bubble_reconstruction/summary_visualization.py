from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import os
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

try:
    import pyvista as pv
except Exception:
    pv = None


def _fig_to_rgb(fig) -> np.ndarray:
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = buf[:, :, :3].copy()
    plt.close(fig)
    return rgb


def _resize_to_width(img: np.ndarray, width: int, max_height: int | None = None) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return img
    scale = float(width) / float(w)
    if max_height is not None and h * scale > max_height:
        scale = float(max_height) / float(h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _pad_to_size(img: np.ndarray, width: int, height: int, value: int = 255) -> np.ndarray:
    out = np.full((height, width, 3), value, dtype=np.uint8)
    h, w = img.shape[:2]
    h_copy = min(h, height)
    w_copy = min(w, width)
    y0 = max(0, (height - h_copy) // 2)
    x0 = max(0, (width - w_copy) // 2)
    out[y0:y0 + h_copy, x0:x0 + w_copy] = img[:h_copy, :w_copy]
    return out


def _grayscale_to_rgb(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr.astype(np.uint8, copy=False)


def _series_display_label(series_key: str) -> str:
    key = str(series_key)
    if "-ID" in key:
        tube, track = key.split("-ID", 1)
        return f"ID {track} ({tube})"
    return key


def build_summary_series_color_map(tracking_rows: list[dict[str, Any]]) -> dict[str, tuple[int, int, int]]:
    """Return stable BGR colors for tracked bubble IDs.

    The color depends only on the series key, not on how many bubbles are visible in
    the current frame. Therefore the top-right frame labels, pipe labels, plots and
    legend use the same color for a given ID across the whole animation.
    """
    series_keys = sorted({
        f"{str(r.get('tube_pair', ''))}-ID{int(r.get('track_id', 0))}"
        for r in (tracking_rows or [])
        if r.get('tube_pair') is not None
    })
    palette = [
        (31, 119, 180),   # blue, RGB
        (255, 127, 14),   # orange
        (44, 160, 44),    # green
        (214, 39, 40),    # red
        (148, 103, 189),  # purple
        (140, 86, 75),    # brown
        (227, 119, 194),  # pink
        (127, 127, 127),  # gray
        (188, 189, 34),   # olive
        (23, 190, 207),   # cyan
        (174, 199, 232),
        (255, 187, 120),
        (152, 223, 138),
        (255, 152, 150),
        (197, 176, 213),
        (196, 156, 148),
        (247, 182, 210),
        (199, 199, 199),
        (219, 219, 141),
        (158, 218, 229),
    ]
    colors: dict[str, tuple[int, int, int]] = {}
    for key in series_keys:
        # deterministic checksum independent of Python's randomized hash()
        checksum = sum((i + 1) * ord(ch) for i, ch in enumerate(key))
        r, g, b = palette[checksum % len(palette)]
        colors[key] = (int(b), int(g), int(r))  # BGR for OpenCV
    return colors


def _series_color_rgb(series_key: str, color_map_bgr: dict[str, tuple[int, int, int]]) -> tuple[float, float, float]:
    bgr = color_map_bgr.get(series_key, (0, 0, 0))
    return (bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0)


def _estimate_origin_view_distance_from_frame(fd: dict[str, Any],
                                             min_distance_mm: float = 80.0,
                                             zoom_out: float = 1.8) -> float:
    max_abs = 0.0
    for key in ("pipe_mesh_12", "pipe_mesh_34"):
        mesh = fd.get(key, None)
        if mesh is None or getattr(mesh, "n_points", 0) == 0:
            continue
        try:
            bounds = mesh.bounds
            max_abs = max(max_abs, max(abs(float(v)) for v in bounds))
        except Exception:
            pass
    for vol_key, vox_key in (("vol_12", "vox_12"), ("vol_34", "vox_34")):
        vol = fd.get(vol_key, None)
        vox = fd.get(vox_key, None)
        if vol is None or vox is None:
            continue
        try:
            length_mm = float(vol.shape[2]) * float(vox)
            radius_like_mm = 0.5 * max(float(vol.shape[0]), float(vol.shape[1])) * float(vox)
            max_abs = max(max_abs, length_mm, radius_like_mm)
        except Exception:
            pass
    return max(float(min_distance_mm), float(max_abs) * float(zoom_out))


def _set_camera_centered_on_coordinate_origin(plotter, distance_mm: float, origin=(0.0, 0.0, 0.0)) -> None:
    ox, oy, oz = map(float, origin)
    d = float(distance_mm)
    camera_position = (ox + d, oy - d, oz + 0.65 * d)
    try:
        plotter.camera.position = camera_position
        plotter.camera.focal_point = (ox, oy, oz)
        plotter.camera.up = (0.0, 1.0, 0.0)
        plotter.camera.clipping_range = (0.001, max(12.0 * d, 1500.0))
    except Exception:
        pass
    try:
        plotter.set_focus((ox, oy, oz))
        plotter.set_position(camera_position)
        plotter.set_viewup((0.0, 1.0, 0.0))
    except Exception:
        pass


def _add_coordinate_origin_marker(plotter, radius_mm: float = 1.0, color: str = "red") -> None:
    if pv is None:
        return
    try:
        marker = pv.Sphere(radius=float(radius_mm), center=(0.0, 0.0, 0.0), theta_resolution=24, phi_resolution=12)
        plotter.add_mesh(marker, color=color, name="coordinate_origin_marker")
    except Exception:
        pass


def _safe_pts_for_summary(vol, vox, center_radial_xy: bool = True, max_points: int = 120000) -> np.ndarray:
    from .volume import volume_to_points_mm
    pts = volume_to_points_mm(vol, vox, center_radial_xy=center_radial_xy, max_points=max_points)
    if pts is None:
        return np.empty((0, 3), dtype=np.float32)
    return pts.astype(np.float32, copy=False)


def _set_horizontal_pipe_camera(plotter, frame_data: dict[str, Any], suffix: str, panel_width_px: int, panel_height_px: int, diameter_mm: float) -> None:
    """Set a side-view camera so the pipe appears horizontal and fills most of the panel width."""
    vol = frame_data.get(f"vol_{suffix}", None)
    vox = frame_data.get(f"vox_{suffix}", None)

    length_mm = None
    if vol is not None and vox is not None:
        try:
            length_mm = float(vol.shape[2]) * float(vox)
        except Exception:
            length_mm = None
    if length_mm is None:
        mesh = frame_data.get(f"pipe_mesh_{suffix}", None)
        try:
            bounds = mesh.bounds
            length_mm = max(1.0, float(bounds[5]) - float(bounds[4]))
        except Exception:
            length_mm = 100.0

    radius_mm = max(0.5 * float(diameter_mm), 1.0)
    z_center = 0.5 * float(length_mm)
    aspect = max(1.0, float(panel_width_px) / max(1.0, float(panel_height_px)))

    # Side view along X so Z becomes horizontal on screen and Y vertical.
    distance = max(5.0 * radius_mm, 60.0)
    parallel_scale = max(1.12 * radius_mm, 0.54 * float(length_mm) / aspect)

    try:
        plotter.camera.parallel_projection = True
    except Exception:
        pass
    try:
        plotter.camera.position = (-distance, 0.0, z_center)
        plotter.camera.focal_point = (0.0, 0.0, z_center)
        plotter.camera.up = (0.0, 1.0, 0.0)
        plotter.camera.parallel_scale = float(parallel_scale)
        plotter.camera.clipping_range = (0.001, max(20.0 * distance, 2000.0))
    except Exception:
        pass
    try:
        plotter.set_focus((0.0, 0.0, z_center))
        plotter.set_position((-distance, 0.0, z_center))
        plotter.set_viewup((0.0, 1.0, 0.0))
    except Exception:
        pass


def _render_pipe_pyvista(frame_data: dict[str, Any], config, color_map_bgr: dict[str, tuple[int, int, int]] | None = None, width: int = 1400, height: int = 330) -> np.ndarray | None:
    if pv is None:
        return None
    try:
        plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(width, height), border=False)
    except Exception:
        return None

    try:
        plotter.set_background("white")
    except Exception:
        pass
    try:
        plotter.enable_depth_peeling()
    except Exception:
        pass

    max_points = int(getattr(config, "summary_pipe_max_points", 120000))
    preview_point_size = 2.0 if bool(getattr(config, "preview_compact", False)) else 2.5
    pipe_opacity = 0.22 if bool(getattr(config, "preview_compact", False)) else 0.32
    show_pipe = bool(getattr(config, "show_pipe", True))
    diameter_mm = float(getattr(config, "diameter_mm", 20.0))

    for col, suffix, title in ((0, "12", "tube1&2"), (1, "34", "tube3&4")):
        plotter.subplot(0, col)
        plotter.add_text(f"Bubble 3D + pipe ({title})", font_size=9)
        mesh = frame_data.get(f"pipe_mesh_{suffix}", None)
        if show_pipe and mesh is not None and getattr(mesh, "n_cells", 0) > 0:
            plotter.add_mesh(mesh, opacity=pipe_opacity, color="lightgray", show_scalar_bar=False, smooth_shading=True)
        pts = _safe_pts_for_summary(frame_data.get(f"vol_{suffix}"), frame_data.get(f"vox_{suffix}"), center_radial_xy=bool(getattr(config, "center_radial_xy", True)), max_points=max_points)
        if len(pts):
            poly = pv.PolyData(pts)
            plotter.add_points(poly, render_points_as_spheres=True, point_size=preview_point_size, color="#3b82f6")
        _add_coordinate_origin_marker(plotter, radius_mm=0.06 * diameter_mm)
        try:
            plotter.add_axes()
        except Exception:
            pass
        _add_bubble_id_labels(plotter, frame_data, suffix, color_map_bgr=color_map_bgr)

    # Independent side-view camera per subplot, so each pipe is horizontal and centered.
    panel_width = max(1, int(width // 2))
    panel_height = max(1, int(height))
    for col, suffix in ((0, "12"), (1, "34")):
        plotter.subplot(0, col)
        _set_horizontal_pipe_camera(plotter, frame_data, suffix, panel_width, panel_height, diameter_mm=diameter_mm)

    try:
        img = plotter.screenshot(return_img=True, transparent_background=False)
    except Exception:
        try:
            plotter.close()
        except Exception:
            pass
        return None

    try:
        plotter.close()
    except Exception:
        pass

    return np.asarray(img, dtype=np.uint8)


def _volume_projection_points(vol: np.ndarray | None, voxel_mm: float | None, max_points: int = 50000):
    if vol is None or voxel_mm is None:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)
    occ = np.asarray(vol) > 0
    if not np.any(occ):
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)
    proj_xz = np.max(occ, axis=1)  # shape: X by Z
    xs, zs = np.where(proj_xz)
    if len(xs) > max_points:
        idx = np.linspace(0, len(xs) - 1, max_points).astype(int)
        xs = xs[idx]
        zs = zs[idx]
    voxel = float(voxel_mm)
    x_mm = (xs.astype(float) - 0.5 * (vol.shape[0] - 1)) * voxel
    z_mm = zs.astype(float) * voxel
    return z_mm, x_mm


def _render_pipe_projection(frame_data: dict[str, Any], config, color_map_bgr: dict[str, tuple[int, int, int]] | None = None, width: int = 1400, height: int = 330) -> np.ndarray:
    pyvista_img = _render_pipe_pyvista(frame_data, config, color_map_bgr=color_map_bgr, width=width, height=height)
    if pyvista_img is not None:
        return _pad_to_size(_resize_to_width(pyvista_img, width, max_height=height), width, height)

    diameter = float(getattr(config, "diameter_mm", 20.0))
    radius = diameter / 2.0
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.3), dpi=100)
    for ax, suffix, title in zip(axes, ("12", "34"), ("tube1&2", "tube3&4")):
        vol = frame_data.get(f"vol_{suffix}")
        vox = frame_data.get(f"vox_{suffix}")
        z_mm, x_mm = _volume_projection_points(vol, vox)
        if vol is not None and vox is not None:
            length_mm = float(vol.shape[2]) * float(vox)
        elif len(z_mm):
            length_mm = float(np.nanmax(z_mm))
        else:
            length_mm = 1.0
        ax.fill_between([0, length_mm], [-radius, -radius], [radius, radius], alpha=0.10)
        ax.plot([0, length_mm], [radius, radius], linewidth=1)
        ax.plot([0, length_mm], [-radius, -radius], linewidth=1)
        ax.plot([0, 0], [-radius, radius], linewidth=1)
        ax.plot([length_mm, length_mm], [-radius, radius], linewidth=1)
        if len(z_mm):
            ax.scatter(z_mm, x_mm, s=1, alpha=0.35)
        tube_pair = f"tube{suffix}"
        track_rows = [row for row in (frame_data.get("tracking_rows", []) or []) if str(row.get("tube_pair", "")) == tube_pair]
        for row in track_rows:
            zc = float(row.get("centroid_z_mm", 0.0))
            yc = float(row.get("centroid_y_mm", 0.0))
            series_key = f"{tube_pair}-ID{int(row.get('track_id', 0))}"
            rgb = _series_color_rgb(series_key, color_map_bgr or {}) if color_map_bgr else (0.0, 0.0, 0.0)
            ax.text(zc, yc, f"ID {int(row.get('track_id', 0))}", fontsize=7, ha="center", va="bottom", color=rgb, bbox=dict(facecolor="white", alpha=0.70, edgecolor="none", pad=1.5))
        ax.set_title(f"3D pipe projection ({title})")
        ax.set_xlabel("Z axis / pipe length [mm]")
        ax.set_ylabel("Radial X [mm]")
        ax.set_ylim(-radius * 1.35, radius * 1.35)
        ax.set_xlim(0, max(length_mm, 1.0))
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    img = _fig_to_rgb(fig)
    img = _resize_to_width(img, width, max_height=height)
    return _pad_to_size(img, width, height)


def _add_bubble_id_labels(plotter, frame_data: dict[str, Any], suffix: str, color_map_bgr: dict[str, tuple[int, int, int]] | None = None) -> None:
    if pv is None:
        return
    tube_pair = f"tube{suffix}"
    rows = [row for row in (frame_data.get("tracking_rows", []) or []) if str(row.get("tube_pair", "")) == tube_pair]
    if not rows:
        return
    try:
        rows = sorted(rows, key=lambda r: (float(r.get("centroid_z_mm", 0.0)), float(r.get("centroid_y_mm", 0.0))))
        for row in rows:
            point = [(
                float(row.get("centroid_x_mm", 0.0)),
                float(row.get("centroid_y_mm", 0.0)),
                float(row.get("centroid_z_mm", 0.0)),
            )]
            label = [f"ID {int(row.get('track_id', 0))}"]
            series_key = f"{tube_pair}-ID{int(row.get('track_id', 0))}"
            bgr = (0, 0, 0) if color_map_bgr is None else color_map_bgr.get(series_key, (0, 0, 0))
            text_color = (bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0)
            plotter.add_point_labels(
                point,
                label,
                point_size=0,
                font_size=10,
                text_color=text_color,
                shape_color="white",
                shape_opacity=0.65,
                margin=2,
                fill_shape=True,
                always_visible=True,
            )
    except Exception:
        pass




def _extract_series(frames_data: list[dict[str, Any]]):
    frame_nos = [int(fd.get("frame_no", i + 1)) for i, fd in enumerate(frames_data)]
    frame_to_idx = {frame_no: i for i, frame_no in enumerate(frame_nos)}
    track_map: dict[tuple[int, str, int], str] = {}
    e_temp = defaultdict(lambda: np.full(len(frame_nos), np.nan, dtype=float))
    eta_temp = defaultdict(lambda: np.full(len(frame_nos), np.nan, dtype=float))
    for fd in frames_data:
        frame_no = int(fd["frame_no"])
        idx = frame_to_idx[frame_no]
        for tr in fd.get("tracking_rows", []) or []:
            tube = str(tr.get("tube_pair", ""))
            det = int(tr.get("detection_index", 0))
            track_id = int(tr.get("track_id", 0))
            track_map[(frame_no, tube, det)] = f"{tube}-ID{track_id}"
        for er in fd.get("eccentricity_rows", []) or []:
            tube = str(er.get("tube_pair", ""))
            det = int(er.get("bubble_index", 0))
            label = track_map.get((frame_no, tube, det), f"{tube}-b{det}")
            try:
                e_temp[label][idx] = float(er.get("e_star", np.nan))
            except Exception:
                pass
        for rr in fd.get("rotational_fit_rows", []) or []:
            tube = str(rr.get("tube_pair", ""))
            det = int(rr.get("detection_index", 0))
            label = track_map.get((frame_no, tube, det), f"{tube}-b{det}")
            try:
                eta_temp[label][idx] = float(rr.get("rotational_fit_score", np.nan))
            except Exception:
                pass
    labels = sorted(set(e_temp.keys()) | set(eta_temp.keys()))
    return frame_nos, {label: e_temp[label] for label in labels}, {label: eta_temp[label] for label in labels}


def _set_ylim_from_series(ax, series_dict: dict[str, np.ndarray], fallback=(0.0, 1.0)):
    vals: list[float] = []
    for arr in series_dict.values():
        arr = np.asarray(arr, dtype=float)
        vals.extend(arr[np.isfinite(arr)].tolist())
    if not vals:
        ax.set_ylim(*fallback)
        return
    ymin = min(vals)
    ymax = max(vals)
    if np.isclose(ymin, ymax):
        delta = 0.1 if np.isfinite(ymin) else 1.0
        ymin -= delta
        ymax += delta
    else:
        pad = 0.08 * (ymax - ymin)
        ymin -= pad
        ymax += pad
    ax.set_ylim(ymin, ymax)


def _render_diagrams(frame_nos: list[int],
                     e_series: dict[str, np.ndarray],
                     eta_series: dict[str, np.ndarray],
                     current_frame_no: int,
                     color_map_bgr: dict[str, tuple[int, int, int]] | None = None,
                     width: int = 1400,
                     height: int = 330) -> np.ndarray:
    fig = plt.figure(figsize=(19, 4.2), dpi=100)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15])
    ax_e = fig.add_subplot(gs[0, 0])
    ax_eta = fig.add_subplot(gs[0, 1])
    ax_leg = fig.add_subplot(gs[0, 2])

    handles = []
    labels = []
    ordered_labels = sorted(set(list(e_series.keys()) + list(eta_series.keys())))

    for label in ordered_labels:
        color = _series_color_rgb(label, color_map_bgr or {})
        e_vals = e_series.get(label, None)
        eta_vals = eta_series.get(label, None)
        line_handle = None
        if e_vals is not None:
            (line_handle,) = ax_e.plot(frame_nos, e_vals, marker="o", markersize=2.5, linewidth=1.2, color=color)
        if eta_vals is not None:
            ax_eta.plot(frame_nos, eta_vals, marker="o", markersize=2.5, linewidth=1.2, color=color)
        if line_handle is None and eta_vals is not None:
            (line_handle,) = ax_eta.plot([], [], color=color)
        if line_handle is not None:
            handles.append(line_handle)
            labels.append(_series_display_label(label))

    ax_e.axvline(current_frame_no, linestyle="--", linewidth=1, color="black")
    ax_eta.axvline(current_frame_no, linestyle="--", linewidth=1, color="black")
    ax_e.set_title("Eccentricity e(t)")
    ax_e.set_xlabel("Frame number")
    ax_e.set_ylabel("e(t) = e*")
    ax_e.grid(True, alpha=0.25)
    ax_eta.set_title("Rotational-fit index eta(t)")
    ax_eta.set_xlabel("Frame number")
    ax_eta.set_ylabel("eta(t) = I_rot")
    ax_eta.grid(True, alpha=0.25)
    if frame_nos:
        x_min = min(frame_nos)
        x_max = max(frame_nos)
        pad = max(1, int(0.03 * max(1, x_max - x_min + 1)))
        ax_e.set_xlim(x_min - pad, x_max + pad)
        ax_eta.set_xlim(x_min - pad, x_max + pad)
    _set_ylim_from_series(ax_e, e_series, fallback=(0.0, 1.0))
    _set_ylim_from_series(ax_eta, eta_series, fallback=(0.0, 1.0))

    ax_leg.axis("off")
    if handles:
        legend_ncol = 2 if len(handles) > 18 else 1
        ax_leg.legend(handles, labels, loc="upper left", fontsize=7, ncol=legend_ncol, frameon=True, title="Bubble IDs", title_fontsize=8, borderaxespad=0.2, handlelength=1.6, columnspacing=0.9, labelspacing=0.35)

    fig.tight_layout()
    img = _fig_to_rgb(fig)
    img = _resize_to_width(img, width, max_height=height)
    return _pad_to_size(img, width, height)


def _compose_dashboard(frame_data: dict[str, Any],
                       pipe_img: np.ndarray,
                       diagrams_img: np.ndarray,
                       canvas_width: int = 1400,
                       top_height: int = 330,
                       middle_height: int = 330,
                       bottom_height: int = 330) -> np.ndarray:
    left_img = _grayscale_to_rgb(frame_data["frame_image"])
    right_src = frame_data.get("frame_image_with_ids", None)
    right_img = _grayscale_to_rgb(right_src) if right_src is not None else left_img.copy()

    panel_width = canvas_width // 2
    panel_height = top_height - 40

    left_img = _resize_to_width(left_img, panel_width, max_height=panel_height)
    right_img = _resize_to_width(right_img, panel_width, max_height=panel_height)
    left_img = _pad_to_size(left_img, panel_width, panel_height)
    right_img = _pad_to_size(right_img, panel_width, panel_height)

    top_row = np.hstack([left_img, right_img])

    header = np.full((40, canvas_width, 3), 245, dtype=np.uint8)
    title = f"Frame {int(frame_data.get('frame_no', 0))}: {frame_data.get('file_name', '')}"
    cv2.putText(header, title[:170], (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(header, "original frame", (150, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(header, "frame + bubble IDs", (canvas_width // 2 + 120, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)

    pipe_img = _pad_to_size(pipe_img, canvas_width, middle_height)
    diagrams_img = _pad_to_size(diagrams_img, canvas_width, bottom_height)
    return np.vstack([header, top_row, pipe_img, diagrams_img])


def _save_summary_outputs(frames: list[np.ndarray],
                          out_dir: str = "summary_visualization",
                          fps: float = 5.0) -> None:
    """Save the complete dashboard animation, not only the last visible frame.

    The previous package accidentally included an old one-frame MP4 in the project
    archive. This writer overwrites that file every run, saves a PNG sequence for
    verification, and also writes an AVI fallback because some Windows OpenCV builds
    create broken MP4 files with only the first frame.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not frames:
        print("[SUMMARY] No dashboard frames to save.")
        return

    # Remove older frame sequence files so the folder always reflects the current run.
    for old_png in out_path.glob("summary_frame_*.png"):
        try:
            old_png.unlink()
        except Exception:
            pass

    clean_frames: list[np.ndarray] = []
    h, w = frames[0].shape[:2]
    for frame in frames:
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.shape[:2] != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
        clean_frames.append(arr)

    last_png = out_path / "summary_last_frame.png"
    cv2.imwrite(str(last_png), cv2.cvtColor(clean_frames[-1], cv2.COLOR_RGB2BGR))

    # Save a PNG sequence as a guaranteed fallback and debugging aid.
    for i, frame in enumerate(clean_frames, start=1):
        frame_png = out_path / f"summary_frame_{i:04d}.png"
        cv2.imwrite(str(frame_png), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    fps = max(1.0, float(fps))

    def write_video(video_path: Path, fourcc_text: str) -> int:
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*fourcc_text), fps, (w, h))
        if not writer.isOpened():
            print(f"[SUMMARY] Could not open video writer for {video_path} using {fourcc_text}.")
            return 0
        written = 0
        for frame in clean_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            written += 1
        writer.release()
        return written

    mp4_path = out_path / "summary_visualization.mp4"
    avi_path = out_path / "summary_visualization.avi"
    written_mp4 = write_video(mp4_path, "mp4v")
    written_avi = write_video(avi_path, "MJPG")

    def read_frame_count(video_path: Path) -> int:
        try:
            cap = cv2.VideoCapture(str(video_path))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return count
        except Exception:
            return -1

    mp4_count = read_frame_count(mp4_path) if written_mp4 else 0
    avi_count = read_frame_count(avi_path) if written_avi else 0

    print(f"[SUMMARY] Saved {len(clean_frames)} dashboard PNG frames: {out_path / 'summary_frame_0001.png'} ...")
    print(f"[SUMMARY] Saved MP4: {mp4_path} | written={written_mp4}, readable_frames={mp4_count}")
    print(f"[SUMMARY] Saved AVI fallback: {avi_path} | written={written_avi}, readable_frames={avi_count}")
    print(f"[SUMMARY] Saved last frame: {last_png}")


def show_summary_visualization(frames_data: list[dict[str, Any]], config) -> None:
    if not frames_data:
        print("[SUMMARY] No frame data available for summary visualization.")
        return
    print("[SUMMARY] Building summary visualization frames...")
    frame_nos, e_series, eta_series = _extract_series(frames_data)
    all_tracking_rows = []
    for _fd in frames_data:
        all_tracking_rows.extend(_fd.get("tracking_rows", []) or [])
    color_map_bgr = build_summary_series_color_map(all_tracking_rows)
    canvas_width = 1650
    top_height = 330
    middle_height = 330
    bottom_height = 430
    dashboards: list[np.ndarray] = []
    for fd in frames_data:
        current_frame_no = int(fd.get("frame_no", 0))
        pipe_img = _render_pipe_projection(fd, config, color_map_bgr=color_map_bgr, width=canvas_width, height=middle_height)
        diagrams_img = _render_diagrams(frame_nos, e_series, eta_series, current_frame_no, color_map_bgr=color_map_bgr, width=canvas_width, height=bottom_height)
        dashboards.append(_compose_dashboard(fd, pipe_img, diagrams_img, canvas_width, top_height, middle_height, bottom_height))
    output_fps = max(1.0, 1.0 / max(0.001, float(getattr(config, "summary_pause_s", 0.2))))
    print(f"[SUMMARY] Built {len(dashboards)} dashboard frames from {len(frames_data)} processed frames.")
    _save_summary_outputs(dashboards, fps=output_fps)

    # On Linux/headless systems cv2.namedWindow may abort the Python process before
    # raising an exception. Detect that case and save files instead. On Windows this
    # branch is normally skipped and a real OpenCV window opens.
    if os.name != "nt" and os.environ.get("BUBBLE_SUMMARY_ALLOW_CV2_WINDOW", "0") != "1":
        print("[SUMMARY] Non-Windows or headless environment detected; files were saved instead of opening a window.")
        return

    print("[SUMMARY] Opening OpenCV dashboard window. Press Esc or Q to close.")
    pause_ms = max(1, int(1000.0 * float(getattr(config, "summary_pause_s", 0.2))))
    window_name = "Bubble summary visualization"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1650, 1120)
        for frame in dashboards:
            cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(pause_ms) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
        if dashboards:
            cv2.imshow(window_name, cv2.cvtColor(dashboards[-1], cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    except Exception as exc:
        print(f"[SUMMARY] OpenCV window could not be opened: {exc}")
        print("[SUMMARY] Dashboard files were already saved in the summary_visualization folder.")
