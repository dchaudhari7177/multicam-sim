"""Coverage colouring in the 3D scene viewer.

The viewer must read camera coverage straight off the manifest's per-camera
``in_view`` flags rather than reprojecting, so these tests assert against the
handoff_ltr scene whose coverage pattern is documented in ``test_handoff_ltr``:
LEFT sees the parcel early and loses it, the RIGHT cameras do the reverse, and
WIDE sees it on every frame.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from multicam_sim import build_handoff_ltr_scene, build_manifest
from multicam_sim.handoff_ltr import CAM_LEFT, CAM_MID_RIGHT, CAM_RIGHT, CAM_WIDE


def _viewer() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "view_scene_3d.py"
    spec = importlib.util.spec_from_file_location("view_scene_3d", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _handoff_manifest() -> dict:
    manifest: dict = json.loads(build_manifest(build_handoff_ltr_scene()).to_json())
    return manifest


def test_frame_numbers_are_sorted_and_unique() -> None:
    frames = _viewer()._frame_numbers(_handoff_manifest())

    assert frames == sorted(set(frames))
    assert frames[0] == 0


def test_wide_camera_sees_the_parcel_on_every_frame() -> None:
    viewer, manifest = _viewer(), _handoff_manifest()

    for frame in viewer._frame_numbers(manifest):
        assert CAM_WIDE in viewer._coverage_at_frame(manifest, frame)


def test_left_and_right_cameras_swap_coverage_across_the_handoff() -> None:
    """LEFT is green early and grey late; the RIGHT cameras are the reverse."""
    viewer, manifest = _viewer(), _handoff_manifest()
    frames = viewer._frame_numbers(manifest)

    first = viewer._coverage_at_frame(manifest, frames[0])
    last = viewer._coverage_at_frame(manifest, frames[-1])

    assert CAM_LEFT in first
    assert CAM_LEFT not in last
    assert CAM_RIGHT not in first
    assert CAM_RIGHT in last


def test_coverage_matches_the_manifest_in_view_flags_exactly() -> None:
    """The viewer reports coverage from ``in_view`` alone, never reprojection."""
    viewer, manifest = _viewer(), _handoff_manifest()
    frame_number = viewer._frame_numbers(manifest)[0]

    frame = next(f for f in manifest["entities"][0]["frames"] if f["frame"] == frame_number)
    expected = {
        obs["cam"]
        for point in frame["points"].values()
        for obs in point["per_cam"]
        if obs["in_view"]
    }

    assert set(viewer._coverage_at_frame(manifest, frame_number)) == expected


def test_overlap_frame_has_two_near_cameras_on_the_parcel() -> None:
    """The handoff scene has a band where both right-hand cameras see it."""
    viewer, manifest = _viewer(), _handoff_manifest()

    overlaps = [
        frame
        for frame in viewer._frame_numbers(manifest)
        if {CAM_MID_RIGHT, CAM_RIGHT} <= set(viewer._coverage_at_frame(manifest, frame))
    ]

    assert overlaps, "expected at least one mid-right/right overlap frame"


def test_seeing_cameras_get_a_ray_to_the_point_they_see() -> None:
    viewer, manifest = _viewer(), _handoff_manifest()
    frame_number = viewer._frame_numbers(manifest)[0]

    seen = viewer._coverage_at_frame(manifest, frame_number)

    assert seen, "expected at least one camera to see the parcel on the first frame"
    for targets in seen.values():
        assert targets, "a seeing camera must carry the point it sees"
        assert all(target.shape == (3,) for target in targets)


def test_render_writes_a_png_for_a_chosen_frame(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    viewer, manifest = _viewer(), _handoff_manifest()
    out = tmp_path / "scene.png"

    written = viewer.render(manifest, out, viewer._frame_numbers(manifest)[-1])

    assert written == out
    assert out.stat().st_size > 0
