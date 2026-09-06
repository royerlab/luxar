"""Tests for the pure sound-node validators (``luxar.validation.sound``).

Mirrors ``test_mesh_validation.py``: every rejection is a ``(factory,
error_pattern, test_id)`` triple so each gets its own named case, and each fails
against a no-op validator (the format sniffer returns a string, so a stub that
returned ``"mp3"`` would fail every refusal below).

The store-facing half — what ``add_sound`` actually writes and how the hash and
reader see it — lives in ``core/tests/test_sound.py``.
"""

from __future__ import annotations

import pytest

from luxar.validation.sound import (
    AUDIO_FORMAT_FILENAMES,
    SOUND_PASSTHROUGH_ATTRS,
    SOUND_RESERVED_ATTRS,
    VALID_SOUND_BUSES,
    VALID_SOUND_TRIGGERS,
    sniff_audio_format,
    validate_audio_input,
    validate_non_negative_finite,
    validate_sound_bus,
    validate_sound_licence,
    validate_sound_passthrough_attrs,
    validate_sound_trigger,
    validate_spatial_params,
)

# Minimal payloads: only the magic bytes matter to the sniffer.
MP3_FRAME = bytes.fromhex("fffb9000") + b"\x00" * 64
MP3_ID3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 64
M4A = b"\x00\x00\x00\x18ftypM4A \x00\x00\x00\x00" + b"\x00" * 64
ADTS = bytes.fromhex("fff15080") + b"\x00" * 64
OGG = b"OggS\x00\x02" + b"\x00" * 64
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 64
FLAC = b"fLaC\x00\x00\x00\x22" + b"\x00" * 64
WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 64


# =============================================================================
# Format sniffing
# =============================================================================


@pytest.mark.parametrize(
    "payload,expected",
    [
        (MP3_FRAME, "mp3"),
        (MP3_ID3, "mp3"),
        (M4A, "aac"),
        (ADTS, "aac"),
    ],
    ids=["mp3-frame-sync", "mp3-id3-tag", "aac-m4a-container", "aac-adts"],
)
def test_sniff_accepts_mp3_and_aac(payload: bytes, expected: str) -> None:
    assert sniff_audio_format(payload) == expected
    assert expected in AUDIO_FORMAT_FILENAMES


@pytest.mark.parametrize(
    "factory,error_pattern,test_id",
    [
        (lambda: sniff_audio_format(OGG), r"Ogg.*Safari", "ogg-refused-for-safari"),
        (lambda: sniff_audio_format(WAV), r"WAV", "wav-refused"),
        (lambda: sniff_audio_format(FLAC), r"FLAC", "flac-refused"),
        (lambda: sniff_audio_format(WEBM), r"Matroska", "webm-refused"),
        (lambda: sniff_audio_format(b"hello world"), r"not a recognised", "garbage"),
        (lambda: sniff_audio_format(b"\xff"), r"too short", "too-short"),
    ],
)
def test_sniff_refusals(factory, error_pattern, test_id) -> None:
    with pytest.raises(ValueError, match=error_pattern):
        factory()


def test_sniff_rejects_non_bytes() -> None:
    with pytest.raises(TypeError, match="must be bytes"):
        sniff_audio_format("clip.mp3")  # type: ignore[arg-type]


# =============================================================================
# Clip input (bytes or path)
# =============================================================================


def test_validate_audio_input_reads_a_path_and_agrees_with_its_suffix(tmp_path) -> None:
    f = tmp_path / "bed.mp3"
    f.write_bytes(MP3_FRAME)
    payload, fmt = validate_audio_input(f)
    assert payload == MP3_FRAME
    assert fmt == "mp3"
    payload, fmt = validate_audio_input(str(f))
    assert fmt == "mp3"


def test_validate_audio_input_accepts_bytes_directly() -> None:
    assert validate_audio_input(bytearray(M4A)) == (M4A, "aac")


@pytest.mark.parametrize(
    "factory,error_pattern,test_id",
    [
        (
            lambda p: validate_audio_input(p / "missing.mp3"),
            r"does not exist",
            "missing-path",
        ),
        (
            lambda p: validate_audio_input(_write(p / "lying.mp3", OGG)),
            r"Ogg",
            "ogg-in-mp3-clothing-refused-by-payload",
        ),
        (
            lambda p: validate_audio_input(_write(p / "lying.m4a", MP3_FRAME)),
            r"extension says aac but the payload is mp3",
            "suffix-disagrees-with-payload",
        ),
        (lambda p: validate_audio_input(b""), r"empty", "empty-bytes"),
        (lambda p: validate_audio_input(42), r"bytes or a path", "wrong-type"),  # type: ignore[arg-type]
    ],
)
def test_validate_audio_input_refusals(
    tmp_path, factory, error_pattern, test_id
) -> None:
    with pytest.raises((ValueError, TypeError), match=error_pattern):
        factory(tmp_path)


def _write(path, payload: bytes):
    path.write_bytes(payload)
    return path


# =============================================================================
# Scalar knobs
# =============================================================================


def test_trigger_bus_vocabulary() -> None:
    assert VALID_SOUND_TRIGGERS == ("continuous", "once")
    assert VALID_SOUND_BUSES == ("ambient", "voice", "effects")
    for t in VALID_SOUND_TRIGGERS:
        assert validate_sound_trigger(t) == t
    for b in VALID_SOUND_BUSES:
        assert validate_sound_bus(b) == b


@pytest.mark.parametrize(
    "factory,error_pattern,test_id",
    [
        (
            lambda: validate_sound_trigger("on_arrive"),
            r"Phase 2",
            "on-arrive-is-phase-2",
        ),
        (
            lambda: validate_sound_trigger("on_depart"),
            r"Phase 2",
            "on-depart-is-phase-2",
        ),
        (lambda: validate_sound_trigger("loop"), r"Invalid trigger", "unknown-trigger"),
        (lambda: validate_sound_bus("music"), r"Invalid bus", "unknown-bus"),
        (
            lambda: validate_non_negative_finite("gain", -0.1),
            r"gain must be finite and >= 0",
            "negative-gain",
        ),
        (
            lambda: validate_non_negative_finite("delay_ms", float("nan")),
            r"delay_ms",
            "nan-delay",
        ),
        (
            lambda: validate_non_negative_finite("gain", True),
            r"must be a number",
            "bool-is-not-a-number",
        ),
        (
            lambda: validate_non_negative_finite("gain", "1"),
            r"must be a number",
            "string-is-not-a-number",
        ),
    ],
)
def test_scalar_refusals(factory, error_pattern, test_id) -> None:
    with pytest.raises((ValueError, TypeError), match=error_pattern):
        factory()


def test_licence_fields_are_required_and_stripped() -> None:
    assert validate_sound_licence(" CC0 ", "Someone", "https://x") == (
        "CC0",
        "Someone",
        "https://x",
    )
    for missing in ({"license": ""}, {"attribution": "  "}, {"source_url": None}):
        kwargs = {"license": "CC0", "attribution": "A", "source_url": "https://x"}
        kwargs.update(missing)
        with pytest.raises(ValueError, match=next(iter(missing))):
            validate_sound_licence(**kwargs)


# =============================================================================
# Spatial (PannerNode) knobs
# =============================================================================


def test_spatial_params_return_only_what_was_given() -> None:
    assert (
        validate_spatial_params(
            ref_distance=None,
            max_distance=None,
            rolloff=None,
            cone_inner_deg=None,
            cone_outer_deg=None,
            cone_outer_gain=None,
            orientation=None,
        )
        == {}
    )
    out = validate_spatial_params(
        ref_distance=2,
        max_distance=30,
        rolloff=1,
        cone_inner_deg=90,
        cone_outer_deg=180,
        cone_outer_gain=0.2,
        orientation=(0, 0, -1),
    )
    assert out == {
        "ref_distance": 2.0,
        "max_distance": 30.0,
        "rolloff": 1.0,
        "cone_inner_deg": 90.0,
        "cone_outer_deg": 180.0,
        "cone_outer_gain": 0.2,
        "orientation": [0.0, 0.0, -1.0],
    }


def _spatial(**overrides):
    kwargs = dict(
        ref_distance=None,
        max_distance=None,
        rolloff=None,
        cone_inner_deg=None,
        cone_outer_deg=None,
        cone_outer_gain=None,
        orientation=None,
    )
    kwargs.update(overrides)
    return validate_spatial_params(**kwargs)


@pytest.mark.parametrize(
    "factory,error_pattern,test_id",
    [
        (lambda: _spatial(ref_distance=0), r"ref_distance must be > 0", "zero-ref"),
        (lambda: _spatial(max_distance=0), r"max_distance must be > 0", "zero-max"),
        (
            lambda: _spatial(ref_distance=10, max_distance=5),
            r"max_distance \(5.0\) must be >= ref_distance",
            "max-below-ref",
        ),
        (lambda: _spatial(cone_inner_deg=400), r"\[0, 360\]", "cone-over-360"),
        (lambda: _spatial(cone_outer_gain=1.5), r"\[0, 1\]", "outer-gain-over-1"),
        (lambda: _spatial(orientation=(1, 0)), r"finite 3-vector", "orientation-2d"),
        (lambda: _spatial(orientation=(0, 0, 0)), r"zero vector", "orientation-zero"),
        (
            lambda: _spatial(orientation="up"),
            r"sequence of 3 numbers",
            "orientation-str",
        ),
    ],
)
def test_spatial_refusals(factory, error_pattern, test_id) -> None:
    with pytest.raises((ValueError, TypeError), match=error_pattern):
        factory()


# =============================================================================
# **attrs gate
# =============================================================================


def test_passthrough_attrs_accept_compositing_keys_only() -> None:
    validate_sound_passthrough_attrs(
        {"layer": True, "visible": False, "transform": [1] * 16, "nd_transform": {}}
    )
    assert SOUND_PASSTHROUGH_ATTRS == {"layer", "visible", "transform", "nd_transform"}
    assert SOUND_PASSTHROUGH_ATTRS.isdisjoint(SOUND_RESERVED_ATTRS)


@pytest.mark.parametrize(
    "factory,error_pattern,test_id",
    [
        (
            lambda: validate_sound_passthrough_attrs({"trigger": "once"}),
            r"stamped by the sound writer",
            "reserved-key",
        ),
        (
            lambda: validate_sound_passthrough_attrs({"audio_file": "x.mp3"}),
            r"stamped by the sound writer",
            "reserved-payload-name",
        ),
        (
            lambda: validate_sound_passthrough_attrs({"opacity": 0.5}),
            r"unknown attribute\(s\) \['opacity'\].*heard, not drawn",
            "appearance-attr-refused",
        ),
        (
            lambda: validate_sound_passthrough_attrs(
                {"colormap": "viridis", "layer": True}
            ),
            r"\['colormap'\]",
            "colormap-refused-layer-fine",
        ),
    ],
)
def test_passthrough_refusals(factory, error_pattern, test_id) -> None:
    with pytest.raises(ValueError, match=error_pattern):
        factory()
