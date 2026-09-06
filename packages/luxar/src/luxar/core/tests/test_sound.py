"""Tests for the Sound node: ``add_sound``, the store it writes, and how the
hash, reader and CLI see it.

Covers what only the full stack can check — the group a write actually
produces, the clip bytes coming back, the ``hidden=`` sugar's row, the scene
bounds staying untouched, and the payload folding into ``content_hash``. The
pure validators live in ``validation/tests/test_sound_validation.py``.
"""

from __future__ import annotations

import struct
import warnings
import zlib
from pathlib import Path

import numpy as np
import pytest

from luxar import Dimension, Dimensions, LuxarZarrCompiler, Sound
from luxar._zarr_compat import open_group, read_raw_bytes
from luxar.io import LuxarScene
from luxar.io._compiler.finalize.hashing import (
    PAYLOAD_FILE_ATTRS,
    compute_content_hashes,
)

MP3 = bytes.fromhex("fffb9000") + bytes(range(256)) * 4
M4A = b"\x00\x00\x00\x18ftypM4A \x00\x00\x00\x00" + bytes(range(256)) * 4
LICENCE = dict(
    license="CC0", attribution="Test author", source_url="https://example.org/clip"
)


def _tiny_png() -> bytes:
    """A valid 1x1 RGB PNG (the overlay writer sniffs and decodes it)."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data))
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", idat)
        + chunk(b"IEND", b"")
    )


def _story_dims() -> Dimensions:
    return Dimensions(
        [
            Dimension(
                "story", unit="", categories=["Overview", "A", "B"], display=False
            ),
            Dimension("t", unit="s", range=(0, 10), display=False),
            Dimension("x", unit="um"),
            Dimension("y", unit="um"),
            Dimension("z", unit="um"),
        ]
    )


def _build(tmp_path: Path, name: str = "s", dims: Dimensions | None = None, **sounds):
    """Compile one scene with a points node plus the given ``name=kwargs`` sounds."""
    store = tmp_path / f"{name}.luxar.zarr"
    dims = dims or _story_dims()
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        pos = np.random.default_rng(0).random((20, dims.ndim)).astype(np.float32) * 10
        pos[:, 0] = 1.0
        pos[:, 1] = 5.0
        scene.add_points("pts", pos, extend_to_all=["story", "t"])
        nodes = {}
        for sound_name, kwargs in sounds.items():
            kwargs = {**LICENCE, **kwargs}
            clip = kwargs.pop("clip", MP3)
            nodes[sound_name] = scene.add_sound(sound_name, clip, **kwargs)
    return store, nodes


# =============================================================================
# The three placements
# =============================================================================


def test_non_spatial_bed_has_no_positions_and_is_a_sound_node(tmp_path) -> None:
    store, nodes = _build(
        tmp_path, bed=dict(gain=0.4, fade_in_ms=1500, fade_out_ms=1500)
    )
    node = nodes["bed"]
    assert isinstance(node, Sound)
    assert node.n_positions == 0 and node.n_elements == 0
    assert node.spatial is False
    assert node.trigger == "continuous"
    assert node.bus == "ambient"
    assert node.format == "mp3"
    assert node.audio_file == "audio.mp3"

    group = open_group(store, mode="r")["bed"]
    attrs = dict(group.attrs)
    assert attrs["type"] == "sound"
    assert attrs["has_positions"] is False
    assert attrs["loop"] is True
    assert attrs["gain"] == 0.4
    assert attrs["fade_in_ms"] == 1500.0
    assert attrs["ordering"] == "none"
    assert "positions" not in group
    assert "distance_model" not in attrs  # non-spatial: no panner knobs stored
    assert attrs["license"] == "CC0"
    assert read_raw_bytes(group, "audio.mp3") == MP3


def test_hidden_sugar_builds_one_row_and_extends_the_other_hidden_dims(
    tmp_path,
) -> None:
    store, nodes = _build(
        tmp_path,
        narr=dict(hidden={"story": 2}, trigger="once", delay_ms=800, bus="voice"),
    )
    group = open_group(store, mode="r")["narr"]
    attrs = dict(group.attrs)
    assert attrs["spatial"] is False
    assert attrs["has_positions"] is True
    assert attrs["n_positions"] == 1
    assert attrs["ndim"] == 5
    assert attrs["loop"] is False
    assert attrs["trigger"] == "once"
    assert attrs["delay_ms"] == 800.0
    # `t` is hidden and not named → extended; `story` is the binding.
    assert attrs["extend_to_all"] == ["t"]
    positions = LuxarScene(open_group(store, mode="r"), store)._decode_array(
        group, "positions"
    )
    assert positions.shape == (1, 5)
    assert positions[0, 0] == pytest.approx(2.0)
    assert np.all(positions[0, 1:] == 0)


def test_spatial_source_stores_positions_and_panner_knobs(tmp_path) -> None:
    store, nodes = _build(
        tmp_path,
        hum=dict(
            positions=[[1, 5, 3.0, 4.0, 5.0]],
            ref_distance=2,
            max_distance=30,
            rolloff=1,
            orientation=(0, 0, -1),
            extend_to_all=["t"],
            clip=M4A,
        ),
    )
    node = nodes["hum"]
    assert node.spatial is True
    assert node.n_positions == 1
    assert node.format == "aac"
    assert node.audio_file == "audio.m4a"
    group = open_group(store, mode="r")["hum"]
    attrs = dict(group.attrs)
    assert attrs["distance_model"] == "inverse"
    assert attrs["ref_distance"] == 2.0
    assert attrs["max_distance"] == 30.0
    assert attrs["orientation"] == [0.0, 0.0, -1.0]
    assert attrs["position_bounds"]["min"][2:] == [3.0, 4.0, 5.0]
    assert read_raw_bytes(group, "audio.m4a") == M4A


def test_distance_knobs_left_none_stay_absent_for_the_viewer_to_default(
    tmp_path,
) -> None:
    store, _ = _build(
        tmp_path, hum=dict(positions=[[1, 5, 1.0, 1.0, 1.0]], extend_to_all=["t"])
    )
    attrs = dict(open_group(store, mode="r")["hum"].attrs)
    assert "ref_distance" not in attrs and "max_distance" not in attrs
    assert attrs["distance_model"] == "inverse"


# =============================================================================
# Scene-level effects
# =============================================================================


def test_sound_positions_never_stretch_the_scene_bounds(tmp_path) -> None:
    far = [[1, 5, 1000.0, 1000.0, 1000.0]]
    store, _ = _build(tmp_path, hum=dict(positions=far, extend_to_all=["t"]))
    root = dict(open_group(store, mode="r").attrs)
    assert max(root["position_bounds"]["max"][2:]) < 100


def test_content_hash_covers_the_clip_bytes(tmp_path) -> None:
    assert "audio_file" in PAYLOAD_FILE_ATTRS
    store_a, _ = _build(tmp_path, "a", bed=dict(clip=MP3))
    other = bytes.fromhex("fffb9000") + bytes(reversed(range(256))) * 4
    assert len(other) == len(MP3)
    store_b, _ = _build(tmp_path, "b", bed=dict(clip=other))
    hash_a = dict(open_group(store_a, mode="r").attrs)["content_hash"]
    hash_b = dict(open_group(store_b, mode="r").attrs)["content_hash"]
    assert hash_a != hash_b
    # Recomputing over unchanged bytes reproduces the digest.
    assert compute_content_hashes(open_group(store_a, mode="r+")) == hash_a


def test_compile_with_sound_and_overlay_is_warning_free(tmp_path) -> None:
    """The plain payload keys (``audio.mp3``, an overlay's ``image.png``) used to
    make zarr's member enumeration warn during finalize, which under ``-W error``
    failed the whole compile. Pre-existing for overlays; fixed for both."""
    store = tmp_path / "quiet.luxar.zarr"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_sound("bed", MP3, **LICENCE)
            scene.add_image(_tiny_png(), (0.5, 0.9), name="logo")
        loaded = LuxarScene(open_group(store, mode="r"), store)
        assert loaded.list_sounds() == ["bed"]


def test_reader_lists_sound_nodes_with_their_summary(tmp_path) -> None:
    store, _ = _build(
        tmp_path,
        bed=dict(),
        narr=dict(hidden={"story": 1}, bus="voice", trigger="once"),
    )
    loaded = LuxarScene(open_group(store, mode="r"), store)
    assert sorted(loaded.list_sounds()) == ["bed", "narr"]
    info = {n["name"]: n for n in loaded.nodes if n["type"] == "sound"}
    assert info["narr"]["bus"] == "voice"
    assert info["narr"]["trigger"] == "once"
    assert info["narr"]["n_positions"] == 1
    assert info["bed"]["audio_file"] == "audio.mp3"
    assert loaded.list_points() == ["pts"]


def test_luxar_info_reports_the_sound_node(tmp_path) -> None:
    from typer.testing import CliRunner

    from luxar.cli.main import app

    store, _ = _build(tmp_path, bed=dict())
    result = CliRunner().invoke(app, ["info", str(store)])
    assert result.exit_code == 0, result.output
    assert "bed" in result.output
    assert "🔈" in result.output


# =============================================================================
# Refusals through the full adder
# =============================================================================


@pytest.mark.parametrize(
    "kwargs,error_pattern,test_id",
    [
        (dict(clip=b"OggS" + b"\x00" * 64), r"Ogg.*Safari", "ogg-refused"),
        (dict(license=""), r"license is required", "missing-licence"),
        (dict(attribution=""), r"attribution is required", "missing-attribution"),
        (dict(trigger="on_arrive"), r"Phase 2", "phase-2-trigger"),
        (dict(bus="music"), r"Invalid bus", "bad-bus"),
        (dict(gain=-1), r"gain must be finite", "negative-gain"),
        (dict(opacity=0.5), r"heard, not drawn", "appearance-attr"),
        (
            dict(trigger="once", positions=[[1, 5, 0, 0, 0]], hidden={"story": 1}),
            r"mutually exclusive",
            "positions-and-hidden",
        ),
        (dict(hidden={"x": 1}), r"displayed dimension", "hidden-names-displayed-dim"),
        (dict(hidden={"nope": 1}), r"unknown dimension", "hidden-unknown-dim"),
        (dict(hidden={}), r"at least one dimension", "hidden-empty"),
        (
            dict(spatial=True),
            r"spatial=True needs positions",
            "spatial-without-positions",
        ),
        (
            dict(spatial=True, hidden={"story": 1}),
            r"spatial=True needs positions",
            "spatial-with-hidden",
        ),
        (dict(positions=[[1, 5, 0]]), r"Dimension mismatch", "wrong-column-count"),
        (dict(positions=np.zeros((0, 5))), r"K >= 1", "empty-positions"),
    ],
)
def test_add_sound_refusals(tmp_path, kwargs, error_pattern, test_id) -> None:
    with pytest.raises(ValueError, match=error_pattern) as excinfo:
        _build(tmp_path, bad=kwargs)
    assert str(excinfo.value).startswith("Could not add sound 'bad':")


def test_failed_add_leaves_no_child_behind(tmp_path) -> None:
    store = tmp_path / "s.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError):
            scene.add_sound("bad", b"OggS" + b"\x00" * 64, **LICENCE)
        assert "bad" not in [c.name for c in scene.children]
        scene.add_sound("good", MP3, **LICENCE)
    assert "bad" not in open_group(store, mode="r")


def test_non_spatial_distance_knobs_warn_and_are_dropped(tmp_path) -> None:
    with pytest.warns(UserWarning, match="non-spatial"):
        store, _ = _build(tmp_path, bed=dict(ref_distance=3.0))
    assert "ref_distance" not in dict(open_group(store, mode="r")["bed"].attrs)


def test_duplicate_name_is_refused_before_any_write(tmp_path) -> None:
    store = tmp_path / "s.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_sound("bed", MP3, **LICENCE)
        with pytest.raises(ValueError, match="Duplicate child name"):
            scene.add_sound("bed", MP3, **LICENCE)
