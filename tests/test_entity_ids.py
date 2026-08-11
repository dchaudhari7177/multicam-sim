"""Entity ids are unique across entities and distractors (issue #96).

``build`` merges ``_entities`` and ``_distractors`` into one spec list and keys the
per-frame lookup by id, so the two share a namespace whether or not the API says so. A
duplicate used to pass silently and split the scene: both entries reached
``scene.entities`` and the manifest, while ``frames_by_id`` kept only the last one
written. Every id-keyed consumer downstream then read the survivor, so annotations and
groups saw one entity where a consumer parsing the manifest saw two.
"""

from __future__ import annotations

import pytest

from multicam_sim import build_manifest
from multicam_sim.dsl import CameraRig, Path, SceneBuilder

_FPS = 10.0
_NUM_FRAMES = 5


def _ring() -> list:
    return CameraRig.ring(
        n=3, radius=4.0, height=1.5, look_at=(0.0, 0.0, 0.5), focal=800.0, width=640, height_px=480
    )


def _builder() -> SceneBuilder:
    return SceneBuilder(fps=_FPS, num_frames=_NUM_FRAMES).cameras(_ring())


def _static(at: tuple[float, float, float]) -> Path:
    return Path.linear(at, at)


def test_two_distractors_with_one_id_are_rejected() -> None:
    """The reproduction from the issue: the manifest kept both, frames_by_id kept the last."""
    builder = _builder().entity("obj", _static((0.0, 0.0, 0.5)))
    builder.distractor("dup", _static((1.0, 1.0, 0.0)))

    with pytest.raises(ValueError, match="duplicate entity id 'dup'"):
        builder.distractor("dup", _static((9.0, 9.0, 0.0)))


def test_two_entities_with_one_id_are_rejected() -> None:
    builder = _builder().entity("obj", _static((0.0, 0.0, 0.5)))

    with pytest.raises(ValueError, match="duplicate entity id 'obj'"):
        builder.entity("obj", _static((9.0, 9.0, 0.0)))


def test_an_entity_and_a_distractor_cannot_share_an_id() -> None:
    """They land in different lists but one namespace, which is the confusing case."""
    builder = _builder().entity("obj2", _static((0.0, 0.0, 0.5)))

    with pytest.raises(ValueError, match="duplicate entity id 'obj2'"):
        builder.distractor("obj2", _static((9.0, 9.0, 0.0)))


def test_a_distractor_id_cannot_be_reused_by_an_entity() -> None:
    """The other order, since the two lists are checked independently."""
    builder = _builder().distractor("d", _static((1.0, 1.0, 0.0)))

    with pytest.raises(ValueError, match="duplicate entity id 'd'"):
        builder.entity("d", _static((9.0, 9.0, 0.0)))


def test_the_error_names_what_already_held_the_id() -> None:
    """The caller needs to know which call to change, not just that something clashed."""
    builder = _builder().distractor("thing", _static((1.0, 1.0, 0.0)))

    with pytest.raises(ValueError) as exc:
        builder.entity("thing", _static((2.0, 2.0, 0.0)))

    message = str(exc.value)
    assert "already added as a distractor" in message
    assert "as an entity" in message or "as a entity" in message


def test_the_rejection_happens_before_the_spec_is_stored() -> None:
    """A rejected call must not leave a half-added entity behind."""
    builder = _builder().entity("obj", _static((0.0, 0.0, 0.5)))
    with pytest.raises(ValueError):
        builder.entity("obj", _static((9.0, 9.0, 0.0)))

    scene = builder.build()
    assert [e.id for e in scene.entities] == ["obj"]


def test_distinct_ids_still_build_and_agree_with_the_manifest() -> None:
    """The invariant the guard exists to protect: one entity per id, everywhere."""
    scene = (
        _builder()
        .entity("obj", _static((0.0, 0.0, 0.5)))
        .distractor("d1", _static((1.0, 1.0, 0.0)))
        .distractor("d2", _static((9.0, 9.0, 0.0)))
        .build()
    )

    entity_ids = [e.id for e in scene.entities]
    manifest_ids = [e.id for e in build_manifest(scene).entities]

    assert entity_ids == ["obj", "d1", "d2"]
    assert sorted(manifest_ids) == sorted(entity_ids)
    assert len(set(manifest_ids)) == len(manifest_ids)
