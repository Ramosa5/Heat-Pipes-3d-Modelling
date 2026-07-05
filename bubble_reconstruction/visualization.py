from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

try:
    import pyvista as pv
except ImportError as exc:
    raise ImportError("PyVista is required for live 3D preview. Install it with: pip install pyvista") from exc

from .volume import volume_to_points_mm

def show_step(img, title="", cmap=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    plt.waitforbuttonpress()
    plt.close(fig)


def estimate_origin_view_distance_from_frame(fd: dict,
                                             min_distance_mm: float = 80.0,
                                             zoom_out: float = 1.8) -> float:
    """
    Estimates a safe camera distance, but keeps the VIEW CENTRE at (0, 0, 0).

    This is intentionally different from reset_camera(), because reset_camera()
    centres the view on the object bounds. Here the required view centre is the
    coordinate-system origin: the centre of the circular cylinder face at z=0.
    """
    max_abs = 0.0

    for key in ("pipe_mesh_12", "pipe_mesh_34"):
        mesh = fd.get(key, None)
        if mesh is None or getattr(mesh, "n_points", 0) == 0:
            continue
        try:
            bounds = mesh.bounds  # xmin, xmax, ymin, ymax, zmin, zmax
            max_abs = max(max_abs, max(abs(float(v)) for v in bounds))
        except Exception:
            pass

    for vol_key, vox_key in (("vol_12", "vox_12"), ("vol_34", "vox_34")):
        vol = fd.get(vol_key, None)
        vox = fd.get(vox_key, None)
        if vol is None or vox is None:
            continue
        try:
            # volume shape is [X_radial, Y_radial, Z_length]
            length_mm = float(vol.shape[2]) * float(vox)
            radius_like_mm = 0.5 * max(float(vol.shape[0]), float(vol.shape[1])) * float(vox)
            max_abs = max(max_abs, length_mm, radius_like_mm)
        except Exception:
            pass

    return max(float(min_distance_mm), float(max_abs) * float(zoom_out))


def set_camera_centered_on_coordinate_origin(plotter,
                                             distance_mm: float,
                                             origin=(0.0, 0.0, 0.0)):
    """
    Sets camera focal point to the coordinate-system origin.

    Result:
    - centre of the view = (0, 0, 0),
    - mouse rotation/orbit is easier because it rotates around the origin,
    - the pipe still extends along +Z from the origin.
    """
    ox, oy, oz = map(float, origin)
    d = float(distance_mm)

    # View from an oblique direction, so both circular face and Z-length are visible.
    camera_position = (ox + d, oy - d, oz + 0.65 * d)

    try:
        plotter.camera.position = camera_position
        plotter.camera.focal_point = (ox, oy, oz)
        plotter.camera.up = (0.0, 1.0, 0.0)
        plotter.camera.clipping_range = (0.001, max(10.0 * d, 1000.0))
    except Exception:
        pass

    try:
        plotter.set_focus((ox, oy, oz))
    except Exception:
        pass

    try:
        plotter.set_position(camera_position)
    except Exception:
        pass

    try:
        plotter.set_viewup((0.0, 1.0, 0.0))
    except Exception:
        pass


def add_coordinate_origin_marker(plotter,
                                 radius_mm: float = 1.0,
                                 color: str = "red"):
    """Adds a small marker at (0, 0, 0), i.e. at the centre of the circular pipe face."""
    try:
        marker = pv.Sphere(radius=float(radius_mm), center=(0.0, 0.0, 0.0), theta_resolution=24, phi_resolution=12)
        plotter.add_mesh(marker, color=color, name="coordinate_origin_marker")
    except Exception:
        pass


def pv_live_animate_keep_last(frames_data,
                              center_radial_xy: bool = True,
                              max_points: int = 200_000,
                              point_size: float = 3.0,
                              pause_s: float = 0.05,
                              zoom_out: float = 1.8,
                              show_pipe: bool = True,
                              pipe_opacity: float = 0.5,
                              center_view_on_origin: bool = True,
                              show_origin_marker: bool = True,
                              show_tracking_labels: bool = False,
                              show_parameter_labels: bool = False):
    def safe_pts(vol, vox):
        pts = volume_to_points_mm(vol, vox, center_radial_xy=center_radial_xy, max_points=max_points)
        if pts is None:
            return np.empty((0, 3), dtype=np.float32)
        return pts.astype(np.float32, copy=False)

    def safe_pipe_mesh(fd, key_mesh):
        mesh = fd.get(key_mesh, None)
        if mesh is None:
            return pv.PolyData()
        return mesh

    def add_labels(plotter, fd, key, font_size: int = 14, text_color: str = "black", shape_opacity: float = 0.35):
        label_records = fd.get(key, []) or []
        if not label_records:
            return None
        try:
            points = [item["position"] for item in label_records]
            labels = [str(item["label"]) for item in label_records]
            return plotter.add_point_labels(
                points,
                labels,
                point_size=0,
                font_size=int(font_size),
                text_color=text_color,
                shape_opacity=float(shape_opacity),
                always_visible=True,
            )
        except Exception:
            return None

    p = pv.Plotter(shape=(1, 2), window_size=(1500, 720), off_screen=False, title="Bubbles + Z-axis cylindrical pipe (live)")

    # Lepsze renderowanie przezroczystości w PyVista/VTK, jeśli dana wersja to obsługuje.
    try:
        p.enable_depth_peeling()
    except Exception:
        pass

    fd0 = frames_data[0]
    poly_a = pv.PolyData(safe_pts(fd0["vol_12"], fd0["vox_12"]))
    poly_b = pv.PolyData(safe_pts(fd0["vol_34"], fd0["vox_34"]))

    pipe_mesh_a = safe_pipe_mesh(fd0, "pipe_mesh_12")
    pipe_mesh_b = safe_pipe_mesh(fd0, "pipe_mesh_34")

    p.subplot(0, 0)
    p.add_text("Bubble 3D + Z-axis pipe (tube1&2)", font_size=12)
    if show_pipe and pipe_mesh_a.n_cells > 0:
        # Półprzezroczysty cylinder rurki.
        p.add_mesh(pipe_mesh_a, opacity=pipe_opacity, color="lightgray", scalars=None, show_scalar_bar=False)
    p.add_points(poly_a, render_points_as_spheres=True, point_size=point_size)
    if show_origin_marker:
        add_coordinate_origin_marker(p, radius_mm=0.06 * 20.0)
    p.add_axes()
    p.show_grid()
    label_actor_a = None
    if show_parameter_labels:
        label_actor_a = add_labels(p, fd0, "parameter_labels_12", font_size=12, text_color="black", shape_opacity=0.45)
    elif show_tracking_labels:
        label_actor_a = add_labels(p, fd0, "track_labels_12", font_size=14, text_color="black", shape_opacity=0.35)

    p.subplot(0, 1)
    p.add_text("Bubble 3D + Z-axis pipe (tube3&4)", font_size=12)
    if show_pipe and pipe_mesh_b.n_cells > 0:
        # Półprzezroczysty cylinder rurki.
        p.add_mesh(pipe_mesh_b, opacity=pipe_opacity, color="lightgray", scalars=None, show_scalar_bar=False)
    p.add_points(poly_b, render_points_as_spheres=True, point_size=point_size)
    if show_origin_marker:
        add_coordinate_origin_marker(p, radius_mm=0.06 * 20.0)
    p.add_axes()
    p.show_grid()
    label_actor_b = None
    if show_parameter_labels:
        label_actor_b = add_labels(p, fd0, "parameter_labels_34", font_size=12, text_color="black", shape_opacity=0.45)
    elif show_tracking_labels:
        label_actor_b = add_labels(p, fd0, "track_labels_34", font_size=14, text_color="black", shape_opacity=0.35)

    p.link_views()

    # --- CAMERA MANAGEMENT ---
    # Do NOT use reset_camera() here, because it centres the view on object bounds.
    # Requirement: centre of the view must be the coordinate-system origin (0,0,0).
    if center_view_on_origin:
        view_distance_mm = estimate_origin_view_distance_from_frame(fd0, zoom_out=zoom_out)
        p.subplot(0, 0)
        set_camera_centered_on_coordinate_origin(p, view_distance_mm, origin=(0.0, 0.0, 0.0))
        p.subplot(0, 1)
        set_camera_centered_on_coordinate_origin(p, view_distance_mm, origin=(0.0, 0.0, 0.0))
    else:
        p.reset_camera()

        # twarde oddalenie (distance)
        try:
            p.camera.distance = float(p.camera.distance) * float(zoom_out)
        except Exception:
            pass

        # dodatkowe "zoom out" (mniej niż 1 oddala)
        try:
            p.camera.zoom(1.0 / float(zoom_out))
        except Exception:
            pass

    try:
        p.show(auto_close=False, interactive_update=True)
    except TypeError:
        # fallback: cannot animate live in this version -> just show first frame blocking
        p.show(auto_close=True)
        return

    for fd in frames_data:
        poly_a.points = safe_pts(fd["vol_12"], fd["vox_12"])
        poly_b.points = safe_pts(fd["vol_34"], fd["vox_34"])

        (poly_a.modified() if hasattr(poly_a, "modified") else poly_a.Modified())
        (poly_b.modified() if hasattr(poly_b, "modified") else poly_b.Modified())

        try:
            p.title = fd["title"]
        except Exception:
            pass

        if show_tracking_labels or show_parameter_labels:
            try:
                if label_actor_a is not None:
                    p.remove_actor(label_actor_a)
            except Exception:
                pass
            try:
                if label_actor_b is not None:
                    p.remove_actor(label_actor_b)
            except Exception:
                pass

            p.subplot(0, 0)
            if show_parameter_labels:
                label_actor_a = add_labels(p, fd, "parameter_labels_12", font_size=12, text_color="black", shape_opacity=0.45)
            else:
                label_actor_a = add_labels(p, fd, "track_labels_12", font_size=14, text_color="black", shape_opacity=0.35)
            p.subplot(0, 1)
            if show_parameter_labels:
                label_actor_b = add_labels(p, fd, "parameter_labels_34", font_size=12, text_color="black", shape_opacity=0.45)
            else:
                label_actor_b = add_labels(p, fd, "track_labels_34", font_size=14, text_color="black", shape_opacity=0.35)

        p.render()
        if hasattr(p, "process_events"):
            p.process_events()
        time.sleep(pause_s)

    # NOW keep last frame until window is closed:
    # This call blocks and keeps the current scene (last frame).
    p.show(auto_close=True)



def visualize_bubble_tip(points: np.ndarray | None,
                         bubble_count: int = 1,
                         tip_percentile: float = 99.0,
                         title: str = "Bubble tip eccentricity",
                         max_points: int = 200_000,
                         point_size: float = 2.0,
                         tip_point_size: float = 8.0,
                         median_radius_mm: float = 1.0) -> None:
    """
    Visualize eccentricity calculation in the same style as the original script:
    - full reconstructed point cloud: light gray,
    - selected tip region: red,
    - median eccentricity point: blue sphere.

    The bubble separation follows the same largest-Z-gaps logic as
    calculate_bubble_eccentricity(), so the visualized regions match the CSV rows.
    """
    if points is None:
        print("[ECC-VIS] No points to visualize.")
        return

    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        print("[ECC-VIS] Empty or invalid point cloud, skipping visualization.")
        return

    if len(pts) > max_points:
        idx = np.random.choice(len(pts), int(max_points), replace=False)
        pts = pts[idx]

    bubble_count = max(1, int(bubble_count))
    z_idx = 2
    sort_idx = np.argsort(pts[:, z_idx])
    sorted_z = pts[:, z_idx][sort_idx]

    if bubble_count > 1 and len(sorted_z) > bubble_count:
        gaps = np.diff(sorted_z)
        gap_idxs = np.argsort(gaps)[-(bubble_count - 1):]
        split_idxs = np.sort(gap_idxs) + 1
        bubble_point_indices = np.split(sort_idx, split_idxs)
    else:
        bubble_point_indices = [sort_idx]

    plotter = pv.Plotter(title=title)
    plotter.add_text(title, font_size=12)

    plotter.add_points(
        pts,
        color="lightgray",
        point_size=float(point_size),
        opacity=0.15,
        render_points_as_spheres=True,
    )

    for i, idxs in enumerate(bubble_point_indices, start=1):
        bubble_points = pts[idxs]
        if len(bubble_points) == 0:
            continue

        threshold = np.percentile(bubble_points[:, z_idx], tip_percentile)
        tip_points = bubble_points[bubble_points[:, z_idx] >= threshold]
        if len(tip_points) == 0:
            continue

        median_point = np.median(tip_points, axis=0)

        plotter.add_points(
            tip_points,
            color="red",
            point_size=float(tip_point_size),
            opacity=0.75,
            render_points_as_spheres=True,
        )
        plotter.add_mesh(
            pv.Sphere(radius=float(median_radius_mm), center=median_point),
            color="blue",
        )
        try:
            plotter.add_point_labels(
                [median_point],
                [f"B{i}"],
                point_size=0,
                font_size=12,
                text_color="blue",
            )
        except Exception:
            pass

    plotter.add_axes()
    plotter.show_grid()
    plotter.show()
