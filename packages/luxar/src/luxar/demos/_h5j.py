"""Decoding Janelia FlyLight H5J stacks.

H5J is an HDF5 container holding one HEVC (H.265) elementary stream per channel
in a 1-D uint8 dataset. It is how FlyLight distributes the *stitched* 63x/40x
brains — the raw ``.lsm.bz2`` tiles are 16-bit but carry no stitching geometry
at all (``Positions``/``TilePositions``/``OriginX/Y/Z`` are zeroed), so the H5J
is the only product that is both stitched and distortion-corrected.

Shared by the two FlyLight demos. Both need the same two guarantees, and both
would fail quietly without them: the reference channel must be found from the
file's own ``channel_spec`` rather than assumed, and a short decode must raise
instead of returning a thinner volume.
"""

import subprocess
from pathlib import Path

import numpy as np
from arbol import aprint

from luxar.demos import require_module

__all__ = ["decode_h5j_channel", "reference_channel_index", "signal_channel_indices"]


def reference_channel_index(h5j_path: Path) -> int:
    """Index of the reference channel, read from the file's own ``channel_spec``.

    Do not hard-code this. ``channel_spec`` is a per-sample string like
    ``sssr`` (three signal channels then the reference); other releases carry
    different counts, and assuming index 3 would silently decode a *signal*
    channel as if it were the neuropil — a plausible-looking but wrong scene.
    """
    h5py = require_module("h5py")
    with h5py.File(h5j_path, "r") as f:
        spec = f.attrs["channel_spec"].decode()
    if "r" not in spec:
        raise RuntimeError(
            f"{h5j_path.name} has channel_spec {spec!r} with no reference "
            "channel, so there is no neuropil to render."
        )
    return spec.index("r")


def signal_channel_indices(h5j_path: Path) -> list[int]:
    """Indices of the SIGNAL channels, from the file's own ``channel_spec``.

    The complement of :func:`reference_channel_index`. Derived from the spec for
    the same reason: a fit that swept in the dense nc82 reference channel would
    be dominated by it (10-19% occupancy against ~0.1% for the signal).
    """
    h5py = require_module("h5py")
    with h5py.File(h5j_path, "r") as f:
        spec = f.attrs["channel_spec"].decode()
    return [i for i, c in enumerate(spec) if c == "s"]


def decode_h5j_channel(h5j_path: Path, channel: int) -> np.ndarray:
    """Decode one channel of an H5J stack to a (Z, Y, X) uint8 volume.

    H5J stores each channel as an HEVC video stream inside a 1-D uint8 HDF5
    dataset. Frames are padded up to the codec's block size (``pad_right`` /
    ``pad_bottom``), so decoded frames must be cropped back to the stated size.
    """
    h5py = require_module("h5py")

    with h5py.File(h5j_path, "r") as f:
        grp = f["Channels"]
        w = int(grp.attrs["width"][0])
        h = int(grp.attrs["height"][0])
        n = int(grp.attrs["frames"][0])
        pad_r = int(grp.attrs["pad_right"][0])
        pad_b = int(grp.attrs["pad_bottom"][0])
        spec = f.attrs["channel_spec"].decode()
        blob = grp[f"Channel_{channel}"][:].tobytes()

    ew, eh = w + pad_r, h + pad_b
    aprint(f"channel {channel} of spec {spec!r}: {w}x{h}x{n} (encoded {ew}x{eh})")

    stream = h5j_path.with_suffix(f".ch{channel}.hevc")
    stream.write_bytes(blob)
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, path from our cache
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(stream),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-",
            ],
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()[:500]}")
        raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    finally:
        stream.unlink(missing_ok=True)

    # Refuse a short or ragged decode rather than silently returning a
    # thinner volume: a channel that spans a different physical extent from its
    # siblings makes every downstream product misregistered — a much harder
    # thing to notice than an exception here.
    expected = n * ew * eh
    if raw.size != expected:
        raise RuntimeError(
            f"H5J channel {channel} decoded to {raw.size} bytes, expected "
            f"{expected} ({n} frames of {ew}x{eh}). The stream is truncated or "
            "ffmpeg dropped frames, so this channel would be misregistered "
            "against the others."
        )

    vol = raw.reshape(n, eh, ew)[:, :h, :w]
    aprint(f"  decoded {n} frames -> {vol.shape}, mean={vol.mean():.2f}")
    return vol
