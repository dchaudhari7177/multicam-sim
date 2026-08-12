"""Headless smoke tests for the 3D viewer's animated export.

Kept in its own module (rather than added to ``test_view_scene_3d.py``) so the
animation coverage is independent of the static-render tests.

``matplotlib``/``Pillow`` are not package dependencies, so every test skips
cleanly when they are absent. Nothing here needs a display: the script selects
the Agg backend itself.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "view_scene_3d.py"
_MANIFEST = _ROOT / "tests" / "fixtures" / "manifest_golden" / "mtmc.json"


def _viewer() -> Any:
    """Load the viewer script as a module (it is a script, not a package member)."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    spec = importlib.util.spec_from_file_location("view_scene_3d", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frames(path: Path) -> list[tuple[int, int]]:
    from PIL import Image, ImageSequence

    with Image.open(path) as img:
        return [frame.size for frame in ImageSequence.Iterator(img)]


def test_orbit_animation_writes_a_non_empty_gif(tmp_path: Path) -> None:
    viewer = _viewer()
    out = tmp_path / "orbit.gif"
    written = viewer.render_animation(
        viewer._load_manifest(_MANIFEST), out, mode=viewer.ORBIT, frames=4
    )

    assert written == out
    assert out.stat().st_size > 0
    assert len(_frames(out)) > 1  # genuinely animated, not a single still


def test_orbit_frames_all_share_one_canvas_size(tmp_path: Path) -> None:
    """A GIF pastes every frame onto the first frame's canvas.

    Saving frames with ``bbox_inches="tight"`` would size each one to its own
    artist extent — which changes as the view rotates — and the animation would
    jitter and clip. Pin that they are uniform.
    """
    viewer = _viewer()
    out = tmp_path / "orbit.gif"
    viewer.render_animation(viewer._load_manifest(_MANIFEST), out, mode=viewer.ORBIT, frames=6)

    assert len(set(_frames(out))) == 1


def test_trajectory_animation_writes_a_non_empty_gif(tmp_path: Path) -> None:
    viewer = _viewer()
    out = tmp_path / "walk.gif"
    viewer.render_animation(viewer._load_manifest(_MANIFEST), out, mode=viewer.TRAJECTORY, frames=4)

    assert out.stat().st_size > 0
    assert len(_frames(out)) > 1


def test_trajectory_frames_are_capped_at_the_sample_count(tmp_path: Path) -> None:
    """Asking for more frames than trajectory samples must not render duplicates."""
    viewer = _viewer()
    manifest = viewer._load_manifest(_MANIFEST)
    samples = viewer._trajectory_length(manifest)
    assert samples > 0

    out = tmp_path / "walk.gif"
    viewer.render_animation(manifest, out, mode=viewer.TRAJECTORY, frames=samples + 50)

    assert len(_frames(out)) <= samples


def test_static_png_path_is_unaffected(tmp_path: Path) -> None:
    """The default export is still a single PNG."""
    viewer = _viewer()
    out = tmp_path / "scene.png"
    written = viewer.render(viewer._load_manifest(_MANIFEST), out)

    assert written == out
    assert out.stat().st_size > 0
    assert out.read_bytes().startswith(b"\x89PNG")


def test_unknown_mode_and_bad_frame_count_are_rejected(tmp_path: Path) -> None:
    viewer = _viewer()
    manifest = viewer._load_manifest(_MANIFEST)

    with pytest.raises(ValueError, match="unknown animation mode"):
        viewer.render_animation(manifest, tmp_path / "x.gif", mode="spin")
    with pytest.raises(ValueError, match="frames must be >= 1"):
        viewer.render_animation(manifest, tmp_path / "x.gif", frames=0)


def test_trajectory_mode_rejects_a_manifest_with_no_trajectory(tmp_path: Path) -> None:
    """Orbit still works on a camera-only scene; walking a path does not."""
    viewer = _viewer()
    manifest = viewer._load_manifest(_MANIFEST)
    cameras_only = {"cameras": manifest["cameras"]}

    with pytest.raises(ValueError, match="no trajectory to animate"):
        viewer.render_animation(cameras_only, tmp_path / "x.gif", mode=viewer.TRAJECTORY)

    out = viewer.render_animation(cameras_only, tmp_path / "orbit.gif", frames=2)
    assert out.stat().st_size > 0
