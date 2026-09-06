"""Sound node writer — an opaque audio clip plus an optional nD positions array.

The one node type that is heard rather than drawn (``SOUND_SPEC.md`` §3.2).
Its group holds:

* ``positions`` — ``(K, ndim)`` float32, ABSENT for a non-spatial clip with no
  hidden-dimension binding. Stored as a PLAIN float32 array (no quantizing
  encoder, no spatial ordering): K is a handful of rows and the viewer reads it
  raw to run the per-vertex slab kernel on exact hidden coordinates.
* ``audio.mp3`` / ``audio.m4a`` — the clip, a plain store key next to the zarr
  documents (``_zarr_compat.write_raw_bytes``), named by ``attrs["audio_file"]``
  so a reader never guesses the extension. ``finalize/hashing.py`` folds the
  bytes into ``content_hash`` through ``PAYLOAD_FILE_ATTRS`` — the same hook
  that closed #1720 for overlay images — and ``luxar optimise`` carries the
  file for the same reason.

Everything the adder validated lands verbatim as attrs; the writer only adds
what it alone knows (``has_positions`` / ``n_positions`` / ``ndim`` /
``position_bounds`` / ``format`` / ``audio_file`` / ``duration_ms`` when a tag
reader is importable). A sound node never touches the scene bounds: an
inaudible-when-invisible source at the far corner of a dataset should not push
the opening framing out.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from arbol import aprint
from numpy.typing import NDArray

from ...._zarr_compat import create_array, write_raw_bytes
from ....typing_utils.aliases import NodePath
from ....validation.sound import AUDIO_FORMAT_FILENAMES, FOA_CHANNELS
from ..bounds import compute_position_bounds
from ..context import GeometryWriteCtx
from ..node_common import prepare_transform_attrs, validate_node_path


def probe_audio_info(payload: bytes, fmt: str) -> tuple[Optional[float], Optional[int]]:
    """Best-effort ``(duration_ms, channels)`` via ``mutagen`` when it is importable.

    Either value is ``None`` (the attr is left absent) when the optional tag
    reader is not installed or cannot parse the payload. The viewer measures the
    decoded buffer anyway, so this is informational — what ``luxar info`` shows —
    except for an ambisonic clip, whose channel count the writer checks when it
    is known.
    """
    try:
        import io

        from mutagen.mp3 import MP3  # type: ignore[import-not-found]
        from mutagen.mp4 import MP4  # type: ignore[import-not-found]
    except ImportError:
        return None, None
    try:
        tag = MP3(io.BytesIO(payload)) if fmt == "mp3" else MP4(io.BytesIO(payload))
        length = getattr(tag.info, "length", None)
        channels = getattr(tag.info, "channels", None)
    except Exception:  # noqa: BLE001 - a tag reader failing must never fail a write
        return None, None
    duration_ms: Optional[float] = None
    if length is not None and np.isfinite(length) and length > 0:
        duration_ms = float(length) * 1000.0
    n_channels = int(channels) if isinstance(channels, int) and channels > 0 else None
    return duration_ms, n_channels


def probe_audio_duration_ms(payload: bytes, fmt: str) -> Optional[float]:
    """Best-effort clip duration in ms (see :func:`probe_audio_info`)."""
    return probe_audio_info(payload, fmt)[0]


def write_sound(
    ctx: GeometryWriteCtx,
    path: NodePath,
    payload: bytes,
    fmt: str,
    positions: Optional[NDArray[np.float32]],
    *,
    sound_attrs: Dict[str, Any],
    **attrs: Any,
) -> dict[str, Any]:
    """Write one sound node (see ``LuxarZarrCompiler.write_sound``).

    ``sound_attrs`` are the validated playback/spatial/licence knobs the adder
    resolved; ``**attrs`` are the compositing pass-throughs (``layer`` /
    ``visible`` / ``transform`` / ``nd_transform``). Returns the node metadata
    the caller records in the metadata cache.
    """
    path = validate_node_path(path)
    if fmt not in AUDIO_FORMAT_FILENAMES:
        raise ValueError(
            f"Unsupported audio format {fmt!r}; expected one of "
            f"{sorted(AUDIO_FORMAT_FILENAMES)}"
        )
    if not payload:
        raise ValueError("audio clip is empty (0 bytes)")
    # Pure attr processing belongs in the gate — and prepare_transform_attrs is
    # NOT idempotent (it transposes the matrix), so it runs exactly once.
    prepare_transform_attrs(attrs, ctx.store)

    n_positions = 0
    n_dims = 0
    if positions is not None:
        pos = np.asarray(positions, dtype=np.float32)
        if pos.ndim != 2 or pos.shape[0] == 0:
            raise ValueError(
                f"sound positions must have shape (K, D) with K >= 1, got {pos.shape}"
            )
        if not np.all(np.isfinite(pos)):
            raise ValueError("sound positions must be finite")
        n_positions, n_dims = int(pos.shape[0]), int(pos.shape[1])
    else:
        pos = None

    group = ctx.store.require_group(path)
    audio_file = AUDIO_FORMAT_FILENAMES[fmt]
    where = (
        f"{n_positions} position(s), {n_dims}D" if pos is not None else "non-spatial"
    )
    aprint(f"🔈 Writing sound clip ({fmt}, {len(payload):,} bytes; {where}) to {path}")

    if pos is not None:
        # Plain float32, NOT the quantizing encoder the geometry writers use:
        # K is a handful of rows (one per place the source exists), so the
        # per-channel uint16 fixed-point grid saves nothing and is DEGENERATE
        # for a single row (min == max collapses every coordinate to code 0 —
        # a narration bound to story 3 would decode as story 0 and play on the
        # overview). The viewer reads the array raw and runs the slab kernel on
        # exact hidden coordinates.
        create_array(
            group,
            "positions",
            data=pos,
            chunks=(n_positions, n_dims),
            compressor=None,
            overwrite=True,
            # The overview's hidden row is all zeros == the fill value, and zarr 3
            # skips a chunk equal to its fill value by default; the viewer then
            # 404s on `positions/c/0/0` (harmless, but an ERROR line on every
            # load). Write the chunk unconditionally.
            config={"write_empty_chunks": True},
        )

    write_raw_bytes(group, audio_file, payload)

    metadata: Dict[str, Any] = dict(sound_attrs)
    metadata["type"] = "sound"
    metadata["format"] = fmt
    metadata["audio_file"] = audio_file
    metadata["has_positions"] = pos is not None
    metadata["n_positions"] = n_positions
    metadata["ndim"] = n_dims
    duration_ms, channels = probe_audio_info(payload, fmt)
    if duration_ms is not None:
        metadata["duration_ms"] = duration_ms
    if channels is not None:
        metadata["channels"] = channels
        if sound_attrs.get("ambisonic") == "foa" and channels != FOA_CHANNELS:
            raise ValueError(
                f"ambisonic='foa' needs a {FOA_CHANNELS}-channel AmbiX clip; this "
                f"clip has {channels} channel(s)"
            )
    # No spatial index; stamped so a reader never has to distinguish "no
    # ordering" from "attr missing" (mesh does the same).
    metadata["ordering"] = "none"
    if pos is not None:
        position_bounds = compute_position_bounds(pos)
        metadata["position_bounds"] = position_bounds
        # Deliberately NOT ctx.update_scene_bounds(): a sound source must not
        # stretch the scene's framing.

    # Compositing pass-throughs first, then the writer's own truth on top so a
    # caller dict can never clobber a stamped key.
    group.attrs.update(attrs)
    for key, value in metadata.items():
        group.attrs[key] = value

    aprint(f"✅ Sound written to {path}")
    return metadata
