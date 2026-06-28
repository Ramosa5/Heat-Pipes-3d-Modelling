from __future__ import annotations


def build_pipe_cylinder_mesh_from_rectified_shape(rect_shape,
                                                  diameter_mm: float,
                                                  voxel_mm: float = None,
                                                  resolution: int = 96,
                                                  open_ends: bool = True):
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("PyVista is required for pipe mesh creation. Install it with: pip install pyvista") from exc

    H, W = rect_shape[:2]
    if H <= 0 or W <= 0:
        return pv.PolyData()

    mm_per_px = diameter_mm / float(H)
    if voxel_mm is None:
        voxel_mm = mm_per_px

    # Długość cylindra w tym samym układzie metrycznym co punkty bąbli.
    length_mm = max(float(voxel_mm), float(W) * float(mm_per_px))
    radius_mm = float(diameter_mm) / 2.0

    # Required coordinate system:
    # - origin at the centre of one circular cylinder face,
    # - long pipe dimension along Z,
    # - therefore the cylinder spans approximately z=[0, length_mm].
    center = (0.0, 0.0, length_mm / 2.0)

    # PyVista has a built-in cylinder mesh. capping=False keeps the pipe open,
    # so the end faces do not hide the bubbles inside.
    try:
        return pv.Cylinder(
            center=center,
            direction=(0.0, 0.0, 1.0),
            radius=radius_mm,
            height=length_mm,
            resolution=int(resolution),
            capping=not bool(open_ends)
        )
    except TypeError:
        # Older PyVista versions may not support the capping argument.
        return pv.Cylinder(
            center=center,
            direction=(0.0, 0.0, 1.0),
            radius=radius_mm,
            height=length_mm,
            resolution=int(resolution)
        )
