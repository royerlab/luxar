"""Pure validators for the ``sound`` node type (``scene.add_sound``).

A sound node is authored like a points node — an optional ``(K, ndim)``
``positions`` array decides *where* and, through the slab rule on hidden
dimensions, *when* it is audible — but its payload is an opaque audio clip
rather than element arrays. Everything here is store-free: the clip is sniffed
by its magic bytes, the scalar knobs are range-checked, and the writer-stamped
attr names are reserved so a caller cannot clobber them. The adder
(``core/group/adders/sound.py``) runs these gates *before* any zarr write.

The format policy follows ``docs/guides/specs/SOUND_SPEC.md`` §3.1/§7: MP3 and
AAC (``.m4a`` / ADTS) are accepted everywhere; Ogg/Opus is refused because
Safari cannot decode it, and WAV/FLAC are refused because the store is a
hosted artefact and an uncompressed bed is the wrong size class.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple, Union

#: Triggers Phase 1 plays. ``on_depart`` / ``on_arrive`` are spec §3.1 vocabulary
#: but need the waypoint driver's events (Phase 2), so they are named here only to
#: give the caller a precise refusal instead of a typo diagnostic.
VALID_SOUND_TRIGGERS: Tuple[str, ...] = ("continuous", "once")
PHASE2_SOUND_TRIGGERS: Tuple[str, ...] = ("on_depart", "on_arrive")

#: The three buses ``viewer_config.audio.buses`` carries a gain for.
VALID_SOUND_BUSES: Tuple[str, ...] = ("ambient", "voice", "effects")

#: ``PannerNode.distanceModel`` values, verbatim.
VALID_DISTANCE_MODELS: Tuple[str, ...] = ("inverse", "linear", "exponential")

#: Accepted container/codec → the filename the clip is stored under inside the
#: node group. The viewer reads ``attrs["audio_file"]`` and never guesses.
AUDIO_FORMAT_FILENAMES: Dict[str, str] = {"mp3": "audio.mp3", "aac": "audio.m4a"}

#: Path suffixes a caller may hand in, mapped to the format the payload must
#: then sniff as. A mismatch is refused (a ``.mp3`` file holding Ogg is a
#: mislabelled asset that would fail only on Safari, at playback time).
_AUDIO_SUFFIX_FORMATS: Dict[str, str] = {
    ".mp3": "mp3",
    ".m4a": "aac",
    ".aac": "aac",
    ".mp4": "aac",
}

#: Attr keys the sound writer stamps itself. Supplying one through ``**attrs``
#: is refused up front rather than silently overwritten.
SOUND_RESERVED_ATTRS: FrozenSet[str] = frozenset(
    {
        "type",
        "spatial",
        "trigger",
        "delay_ms",
        "gain",
        "bus",
        "loop",
        "fade_in_ms",
        "fade_out_ms",
        "distance_model",
        "ref_distance",
        "max_distance",
        "rolloff",
        "cone_inner_deg",
        "cone_outer_deg",
        "cone_outer_gain",
        "orientation",
        "attach_to",
        "format",
        "duration_ms",
        "sample_rate",
        "license",
        "attribution",
        "source_url",
        "audio_file",
        "has_positions",
        "n_positions",
        "ndim",
        "ordering",
        "position_bounds",
        "extend_to_all",
    }
)

#: The only free-form ``**attrs`` a sound node accepts: the compositing keys
#: that mean something for a node that is heard rather than drawn. Appearance
#: attrs (``opacity``, ``colormap``, ``blending_mode`` …) are refused because a
#: sound has no pixels for them to act on.
SOUND_PASSTHROUGH_ATTRS: FrozenSet[str] = frozenset(
    {"layer", "visible", "transform", "nd_transform"}
)


def sniff_audio_format(payload: bytes) -> str:
    """Return ``"mp3"`` or ``"aac"`` for ``payload``, or raise for anything else.

    Sniffs magic bytes only — never the caller's filename — so a mislabelled
    asset is caught at authoring time:

    * ``ID3`` tag or an MPEG frame sync with a non-reserved layer → ``mp3``
    * ISO-BMFF ``ftyp`` box (``.m4a`` / ``.mp4``) or an ADTS sync → ``aac``
    * ``OggS`` → refused by name (Safari cannot decode Ogg/Opus)
    * ``RIFF`` (WAV), ``fLaC``, Matroska/WebM → refused by name
    * anything else → refused with the accepted list
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"audio clip must be bytes, got {type(payload).__name__}. Pass the "
            "encoded MP3/AAC bytes or a path to the file."
        )
    head = bytes(payload[:12])
    if len(head) < 4:
        raise ValueError(
            f"audio clip is too short to be an encoded file ({len(head)} bytes)"
        )
    if head[:3] == b"ID3":
        return "mp3"
    if head[4:8] == b"ftyp":
        return "aac"
    if head[:4] == b"OggS":
        raise ValueError(
            "audio clip is Ogg (Vorbis/Opus), which Safari cannot decode. "
            "Re-encode as MP3 or AAC (.m4a) — e.g. "
            "`ffmpeg -i clip.ogg -c:a aac clip.m4a`."
        )
    if head[:4] == b"RIFF":
        raise ValueError(
            "audio clip is WAV (uncompressed). Encode as MP3 or AAC (.m4a) so "
            "the store stays hosting-sized."
        )
    if head[:4] == b"fLaC":
        raise ValueError(
            "audio clip is FLAC. Encode as MP3 or AAC (.m4a); the viewer does "
            "not accept lossless containers."
        )
    if head[:4] == b"\x1a\x45\xdf\xa3":
        raise ValueError("audio clip is Matroska/WebM. Encode as MP3 or AAC (.m4a).")
    if head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        # MPEG audio sync. Layer bits 00 are reserved in MPEG-1/2 audio and mark
        # an ADTS (raw AAC) header instead.
        layer_bits = head[1] & 0x06
        return "aac" if layer_bits == 0 else "mp3"
    raise ValueError(
        "audio clip is not a recognised MP3 or AAC payload (accepted: MP3 "
        "frames or an ID3-tagged MP3, AAC in an .m4a/.mp4 container or ADTS). "
        f"First bytes: {head[:8]!r}"
    )


def validate_audio_input(clip: Union[bytes, bytearray, str, Path]) -> Tuple[bytes, str]:
    """Resolve ``clip`` (bytes or a path) to ``(payload, format)``.

    A path is read whole. When it carries a known suffix the sniffed format must
    agree with it — the same discipline ``validate_image_input`` applies to
    overlay images — so a ``.mp3`` that is really Ogg is refused here, not on a
    visitor's Safari.
    """
    suffix_format: Optional[str] = None
    if isinstance(clip, (str, Path)):
        path = Path(clip)
        if not path.is_file():
            raise ValueError(f"audio clip path does not exist: {path}")
        suffix_format = _AUDIO_SUFFIX_FORMATS.get(path.suffix.lower())
        payload = path.read_bytes()
    elif isinstance(clip, (bytes, bytearray, memoryview)):
        payload = bytes(clip)
    else:
        raise TypeError(
            f"audio clip must be bytes or a path, got {type(clip).__name__}"
        )
    if len(payload) == 0:
        raise ValueError("audio clip is empty (0 bytes)")
    fmt = sniff_audio_format(payload)
    if suffix_format is not None and suffix_format != fmt:
        raise ValueError(
            f"audio clip extension says {suffix_format} but the payload is {fmt}; "
            "fix the file name or re-encode so the two agree"
        )
    return payload, fmt


def validate_sound_trigger(trigger: str) -> str:
    """Validate the ``trigger`` knob (Phase 1 accepts ``continuous`` / ``once``)."""
    if trigger in PHASE2_SOUND_TRIGGERS:
        raise ValueError(
            f"trigger={trigger!r} needs the waypoint driver's arrival/departure "
            "events, which land in Phase 2 of the sound layer. Use "
            "'continuous' or 'once' (fires when the node becomes audible)."
        )
    if trigger not in VALID_SOUND_TRIGGERS:
        raise ValueError(
            f"Invalid trigger {trigger!r}. Must be one of {VALID_SOUND_TRIGGERS}"
        )
    return trigger


def validate_sound_bus(bus: str) -> str:
    """Validate the ``bus`` knob."""
    if bus not in VALID_SOUND_BUSES:
        raise ValueError(f"Invalid bus {bus!r}. Must be one of {VALID_SOUND_BUSES}")
    return bus


def validate_distance_model(model: str) -> str:
    """Validate the panner ``distance_model``."""
    if model not in VALID_DISTANCE_MODELS:
        raise ValueError(
            f"Invalid distance_model {model!r}. Must be one of {VALID_DISTANCE_MODELS}"
        )
    return model


def validate_non_negative_finite(name: str, value: Any) -> float:
    """Coerce ``value`` to a finite float ``>= 0`` or raise naming ``name``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    v = float(value)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {value}")
    return v


def validate_sound_licence(
    license: Any, attribution: Any, source_url: Any
) -> Tuple[str, str, str]:
    """Require the three provenance fields every clip carries (spec §3.1).

    Same discipline as demo citations: a recorded or synthesised clip ships with
    its licence, who made it, and where it came from, so the Layers panel and a
    downstream reader can show them.
    """
    out = []
    for field_name, value in (
        ("license", license),
        ("attribution", attribution),
        ("source_url", source_url),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{field_name} is required for every sound clip and must be a "
                f"non-empty string (got {value!r}). Record the licence "
                "(e.g. 'CC0'), the author, and the URL the clip came from."
            )
        out.append(value.strip())
    return out[0], out[1], out[2]


def validate_spatial_params(
    *,
    ref_distance: Optional[float],
    max_distance: Optional[float],
    rolloff: Optional[float],
    cone_inner_deg: Optional[float],
    cone_outer_deg: Optional[float],
    cone_outer_gain: Optional[float],
    orientation: Optional[Sequence[float]],
) -> Dict[str, Any]:
    """Range-check the ``PannerNode`` knobs; return only the ones given.

    Distances default in the VIEWER from the scene scale (spec §3.1: ``scale/20``
    and ``scale``), so an absent knob stays absent on disk rather than being
    pinned to a number the author never chose.
    """
    out: Dict[str, Any] = {}
    if ref_distance is not None:
        v = validate_non_negative_finite("ref_distance", ref_distance)
        if v == 0:
            raise ValueError("ref_distance must be > 0")
        out["ref_distance"] = v
    if max_distance is not None:
        v = validate_non_negative_finite("max_distance", max_distance)
        if v == 0:
            raise ValueError("max_distance must be > 0")
        if "ref_distance" in out and v < out["ref_distance"]:
            raise ValueError(
                f"max_distance ({v}) must be >= ref_distance ({out['ref_distance']})"
            )
        out["max_distance"] = v
    if rolloff is not None:
        out["rolloff"] = validate_non_negative_finite("rolloff", rolloff)
    for key, value in (
        ("cone_inner_deg", cone_inner_deg),
        ("cone_outer_deg", cone_outer_deg),
    ):
        if value is not None:
            v = validate_non_negative_finite(key, value)
            if v > 360:
                raise ValueError(f"{key} must be in [0, 360], got {value}")
            out[key] = v
    if cone_outer_gain is not None:
        v = validate_non_negative_finite("cone_outer_gain", cone_outer_gain)
        if v > 1:
            raise ValueError(
                f"cone_outer_gain must be in [0, 1], got {cone_outer_gain}"
            )
        out["cone_outer_gain"] = v
    if orientation is not None:
        try:
            vec = [float(c) for c in orientation]
        except (TypeError, ValueError) as e:
            raise TypeError("orientation must be a sequence of 3 numbers") from e
        if len(vec) != 3 or not all(math.isfinite(c) for c in vec):
            raise ValueError(
                f"orientation must be a finite 3-vector, got {list(orientation)!r}"
            )
        if all(c == 0 for c in vec):
            raise ValueError("orientation must not be the zero vector")
        out["orientation"] = vec
    return out


def validate_sound_passthrough_attrs(attrs: Dict[str, Any]) -> None:
    """Refuse every ``**attrs`` key a sound node cannot honour.

    Two classes, two messages: a writer-stamped key (``trigger=`` passed twice,
    ``type=``, ``audio_file=`` …) and an appearance/unknown key. Both are loud
    because the silent alternative — an attr landing on disk that the viewer
    ignores — is exactly the failure mode the geometry adders' unknown-attr gate
    exists to prevent (#787).
    """
    reserved = sorted(k for k in attrs if k in SOUND_RESERVED_ATTRS)
    if reserved:
        raise ValueError(
            f"attrs {reserved} are stamped by the sound writer and cannot be "
            "supplied as extra attributes; pass them as add_sound() parameters."
        )
    unknown = sorted(k for k in attrs if k not in SOUND_PASSTHROUGH_ATTRS)
    if unknown:
        raise ValueError(
            f"unknown attribute(s) {unknown} for a sound node. A sound is heard, "
            "not drawn, so appearance attrs do not apply; the accepted extras are "
            f"{sorted(SOUND_PASSTHROUGH_ATTRS)}."
        )
