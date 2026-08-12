"""Render a scene manifest as one 3D world view: camera locations + trajectory.

A single static 3D figure showing where every camera sits in the world (drawn as
a small view-cone / frustum pointing along its optical axis), a ground plane, and
each entity's ground-truth 3D trajectory over the frames. This is the "see the
space" view -- unlike the per-camera tiles of ``record_multiview.py``, it draws
the whole scene once so camera coverage and overlap are directly visible.

Geometry follows ``multicam_sim.cameras.Camera``: ``R`` is the world->camera
rotation with rows ``[right, down, forward]`` (OpenCV, +z forward) and ``t`` is
the world->camera translation, so the camera centre is ``C = -R^T @ t`` and the
frustum edges are the back-projected image corners ``R^T @ (K^-1 @ [u, v, 1])``.
World up is +z; the ground plane is drawn at z = 0.

Works on any manifest with the standard schema (``cameras[*].{K,R,t,width,
height}`` and ``entities[*].frames[*].points[*].xyz_gt``); defaults to the
bundled MTMC golden fixture.

``--animate`` exports an animated GIF instead: ``--animate-mode orbit`` (the
default) sweeps the viewpoint right round the scene so coverage and frustum
overlap can be read from any angle, and ``--animate-mode trajectory`` holds the
viewpoint still and walks the object along its path. The static-PNG default is
untouched.

``matplotlib`` (and ``PIL`` for the GIF) is imported lazily inside the render
functions (neither is a package dependency), mirroring the lazy-import pattern in
``scripts/record_multiview.py``. GIFs are written with Pillow, as there too, so no
ffmpeg/imageio encoder is required. Run::

    uv run --with matplotlib python scripts/view_scene_3d.py
    uv run --with matplotlib python scripts/view_scene_3d.py --manifest PATH --out PATH
    uv run --with matplotlib python scripts/view_scene_3d.py --animate --out scene.gif
    uv run --with matplotlib python scripts/view_scene_3d.py --animate \
        --animate-mode trajectory --out walk.gif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MANIFEST = _ROOT / "tests" / "fixtures" / "manifest_golden" / "mtmc.json"
_DEFAULT_OUT = _ROOT / "docs" / "assets" / "scene_3d.png"

# How far in front of each camera to draw the frustum apex (world units).
_FRUSTUM_DEPTH = 1.2

#: Animation modes for ``--animate``.
ORBIT = "orbit"
TRAJECTORY = "trajectory"

#: Animation defaults. 36 frames is a 10-degree azimuth step — smooth enough to
#: read, small enough that the GIF stays reviewable in a PR. The figure is drawn
#: smaller and at a lower DPI than the static PNG so a 36-frame GIF does not
#: dwarf the repo's other assets.
ANIMATE_FRAMES = 36
ANIMATE_MS = 120
ANIMATE_DPI = 80
ANIMATE_FIGSIZE = (7.0, 5.5)
_DEFAULT_ANIMATE_OUT = _ROOT / "docs" / "assets" / "scene_3d.gif"


def _camera_centre(
    rotation: NDArray[np.float64], translation: NDArray[np.float64]
) -> NDArray[np.float64]:
    """World-space camera centre ``C = -R^T @ t``."""
    return -rotation.T @ translation


def _frustum_corners(
    intrinsics: NDArray[np.float64],
    rotation: NDArray[np.float64],
    centre: NDArray[np.float64],
    width: float,
    height: float,
    depth: float,
) -> NDArray[np.float64]:
    """The four image-corner rays back-projected to ``depth`` in world space.

    A pixel ``[u, v, 1]`` back-projects to the world ray direction
    ``R^T @ (K^-1 @ [u, v, 1])``; scaling so the forward (+z camera) component
    equals ``depth`` puts the corner a fixed distance in front of the camera.
    """
    inv_k = np.linalg.inv(intrinsics)
    corners_px = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        dtype=np.float64,
    )
    out = np.empty((4, 3), dtype=np.float64)
    for i, (u, v) in enumerate(corners_px):
        cam_ray = inv_k @ np.array([u, v, 1.0], dtype=np.float64)
        cam_ray = cam_ray / cam_ray[2] * depth  # forward (+z) component == depth
        out[i] = centre + rotation.T @ cam_ray
    return out


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def _trajectories(manifest: dict[str, Any]) -> list[tuple[str, NDArray[np.float64]]]:
    """Per-(entity, point) ground-truth 3D polylines over the frames."""
    out: list[tuple[str, NDArray[np.float64]]] = []
    for entity in manifest.get("entities", []):
        by_point: dict[str, list[list[float]]] = {}
        for frame in entity.get("frames", []):
            for name, point in frame.get("points", {}).items():
                xyz = point.get("xyz_gt")
                if xyz is not None:
                    by_point.setdefault(name, []).append(list(xyz))
        for name, coords in by_point.items():
            label = entity.get("id", "object")
            label = label if name == "center" else f"{label}:{name}"
            out.append((str(label), np.asarray(coords, dtype=np.float64)))
    return out


def _draw_scene(
    ax: Any, manifest: dict[str, Any], *, upto: int | None = None
) -> NDArray[np.float64]:
    """Draw cameras, trajectories and the ground plane onto ``ax``.

    Returns every scene point, for the caller's bounds/aspect computation.

    ``upto`` truncates each trajectory to its first ``upto`` samples — the
    animated trajectory mode, where the polyline grows frame by frame and the
    leading sample is marked. The returned points always span the *full* path
    regardless, so the view does not drift between frames. ``None`` (the default,
    and the static path) draws every sample with no marker.
    """
    all_points: list[NDArray[np.float64]] = []

    # Cameras: centre marker + a wireframe frustum to the four image corners.
    for cam in manifest.get("cameras", []):
        intrinsics = np.asarray(cam["K"], dtype=np.float64)
        rotation = np.asarray(cam["R"], dtype=np.float64)
        translation = np.asarray(cam["t"], dtype=np.float64)
        centre = _camera_centre(rotation, translation)
        corners = _frustum_corners(
            intrinsics,
            rotation,
            centre,
            float(cam["width"]),
            float(cam["height"]),
            _FRUSTUM_DEPTH,
        )
        ax.scatter(*centre, color="crimson", s=40, depthshade=False)
        ax.text(*centre, f"  cam {cam['id']}", color="crimson", fontsize=8)
        for corner in corners:
            edge = np.vstack([centre, corner])
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color="crimson", linewidth=0.8)
        loop = np.vstack([corners, corners[0]])
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="crimson", linewidth=0.8)
        all_points.append(centre[None, :])
        all_points.append(corners)

    # Trajectories: one polyline per entity point.
    for label, coords in _trajectories(manifest):
        all_points.append(coords)  # bounds always span the whole path
        shown = coords if upto is None else coords[: max(upto, 1)]
        line = ax.plot(
            shown[:, 0],
            shown[:, 1],
            shown[:, 2],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"trajectory: {label}",
        )
        if upto is not None:
            # Mark where the object currently is, so motion is legible even
            # when the trailing polyline is short.
            head = shown[-1]
            ax.scatter(
                *head,
                s=70,
                color=line[0].get_color(),
                edgecolor="black",
                linewidth=0.6,
                depthshade=False,
            )

    if not all_points:
        raise ValueError("manifest has no cameras or trajectories to draw")
    pts = np.vstack(all_points)

    # Ground plane at z = 0 spanning the scene's xy extent.
    pad = 1.0
    x_min, y_min = pts[:, 0].min() - pad, pts[:, 1].min() - pad
    x_max, y_max = pts[:, 0].max() + pad, pts[:, 1].max() + pad
    gx, gy = np.meshgrid(
        np.linspace(x_min, x_max, 2),
        np.linspace(y_min, y_max, 2),
    )
    ax.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.12, color="steelblue")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z (up)")
    ax.set_title("Scene: camera locations and object trajectory")
    # Only when something is labelled: a camera-only manifest has no trajectory
    # to list, and matplotlib warns ("No artists with labels found") rather than
    # quietly drawing an empty box.
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="upper left", fontsize=8)

    # Equal aspect over the combined bounds so directions are not skewed.
    span = pts.max(axis=0) - pts.min(axis=0)
    span = np.where(span > 0, span, 1.0)
    ax.set_box_aspect(tuple(span))
    return pts


def render(manifest: dict[str, Any], out_path: Path) -> Path:
    """Render the 3D scene view to ``out_path`` (PNG) and return the path."""
    import matplotlib

    matplotlib.use("Agg")  # headless-safe: no display needed
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    _draw_scene(ax, manifest)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _trajectory_length(manifest: dict[str, Any]) -> int:
    """Longest trajectory sample count, for pacing the trajectory animation."""
    return max((len(coords) for _, coords in _trajectories(manifest)), default=0)


def render_animation(
    manifest: dict[str, Any],
    out_path: Path,
    *,
    mode: str = ORBIT,
    frames: int = ANIMATE_FRAMES,
    frame_ms: int = ANIMATE_MS,
) -> Path:
    """Render an animated GIF of the scene to ``out_path`` and return the path.

    ``mode`` is :data:`ORBIT` (azimuth sweep right round the scene, geometry
    fixed) or :data:`TRAJECTORY` (fixed viewpoint, object walking its path).

    GIF is written with Pillow, mirroring the animation path in
    ``scripts/record_multiview.py`` — no ``imageio``/ffmpeg encoder needed, so it
    works on a bare headless box.
    """
    import io

    import matplotlib

    matplotlib.use("Agg")  # headless-safe: no display needed
    import matplotlib.pyplot as plt
    from PIL import Image

    if mode not in (ORBIT, TRAJECTORY):
        raise ValueError(f"unknown animation mode {mode!r}; expected {ORBIT} or {TRAJECTORY}")
    if frames < 1:
        raise ValueError(f"frames must be >= 1 (got {frames})")

    samples = _trajectory_length(manifest)
    if mode == TRAJECTORY and samples == 0:
        raise ValueError("manifest has no trajectory to animate; use --animate-mode orbit")

    def snapshot(fig: Any) -> Image.Image:
        buf = io.BytesIO()
        # Deliberately NOT bbox_inches="tight" here, unlike the static PNG: a
        # tight box is recomputed per frame, and as the view rotates the artists'
        # extent changes, so frames come out at different pixel sizes (354..436
        # px wide for the bundled fixture). A GIF pastes every frame onto the
        # first frame's canvas, so that shows up as jitter and clipped edges.
        # The full canvas is a fixed figsize x dpi for every frame.
        fig.savefig(buf, format="png", dpi=ANIMATE_DPI)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    images: list[Image.Image] = []
    if mode == ORBIT:
        # The geometry never changes, so draw once and only move the camera.
        fig = plt.figure(figsize=ANIMATE_FIGSIZE)
        ax = fig.add_subplot(111, projection="3d")
        _draw_scene(ax, manifest)
        elev = ax.elev
        for index in range(frames):
            ax.view_init(elev=elev, azim=index * 360.0 / frames)
            images.append(snapshot(fig))
        plt.close(fig)
    else:
        # More frames than trajectory samples would re-render figures that are
        # pixel-identical (Pillow collapses them on write anyway), so cap it and
        # step one sample per frame at the limit.
        frames = min(frames, samples)
        for index in range(frames):
            upto = max(1, round((index + 1) * samples / frames))
            fig = plt.figure(figsize=ANIMATE_FIGSIZE)
            ax = fig.add_subplot(111, projection="3d")
            _draw_scene(ax, manifest, upto=upto)
            images.append(snapshot(fig))
            plt.close(fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=frame_ms,
        loop=0,
        optimize=True,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="scene manifest JSON (default: bundled MTMC golden fixture)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "output path (default: docs/assets/scene_3d.png, "
            "or docs/assets/scene_3d.gif with --animate)"
        ),
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="export an animated GIF instead of a static PNG",
    )
    parser.add_argument(
        "--animate-mode",
        choices=(ORBIT, TRAJECTORY),
        default=ORBIT,
        help=(
            f"{ORBIT}: sweep the viewpoint right round the scene; "
            f"{TRAJECTORY}: fixed viewpoint, object walks its path "
            f"(default: {ORBIT})"
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=ANIMATE_FRAMES,
        help=f"animation frame count (default: {ANIMATE_FRAMES})",
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=ANIMATE_MS,
        help=f"per-frame duration in milliseconds (default: {ANIMATE_MS})",
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    if args.animate:
        out_path = render_animation(
            manifest,
            args.out or _DEFAULT_ANIMATE_OUT,
            mode=args.animate_mode,
            frames=args.frames,
            frame_ms=args.frame_ms,
        )
    else:
        out_path = render(manifest, args.out or _DEFAULT_OUT)
    size = out_path.stat().st_size
    print(f"wrote {out_path}  ({size} bytes)")


if __name__ == "__main__":
    main()
