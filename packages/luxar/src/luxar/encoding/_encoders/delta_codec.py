"""Columnar per-chunk delta+zigzag pre-filter for quantized code arrays.

Luxar stores coordinate / Cholesky / amplitude / color arrays as Hilbert-ordered
uint8/uint16 quantization codes. Spatial ordering makes consecutive codes a
smooth ramp, but Blosc's byte-shuffle cannot exploit that smoothness. This
**filter** turns the ramp into small residuals — per-axis modular delta,
zigzag-mapped to unsigned, laid out column-major within the chunk — which zstd
then crushes (measured 12-16% whole-store lossless on real fits; see the 2026-07
compression-transfer campaign / SOG comparison).

It exists TWICE, once per zarr format, because the two spell a filter
differently: :class:`LuxarDelta` is the numcodecs filter a format-2 store
records, and :class:`LuxarDeltaV3` is the ``zarr.codecs`` array-to-array codec a
format-3 store records. They share one wire format (:func:`delta_encode` /
:func:`delta_decode`) and one NAME, and produce byte-identical chunks, so the
single TypeScript twin decodes either without knowing which wrote it.

Why a zarr *filter* and not an encoding transform: the viewer range-loader
decodes arbitrary sub-chunk element ranges with stateless per-element
kernels, so a sequential cumsum can only live where whole chunks are
reconstructed — inside the zarr codec pipeline (zarrita ``getChunk`` on the
TS side, numcodecs here). The stored ``encoding`` attrs are untouched; only the
array's own metadata gains an entry (``filters`` in a format-2 ``.zarray``, a
``codecs`` chain member in a format-3 ``zarr.json``), and each chunk is
self-contained (implicit 0 anchor per column per chunk).

Wire format (per chunk of ``rows × cols`` codes, per column, modular
``2**bits`` arithmetic)::

    encode:  d  = (code - prev) mod 2^bits          # prev = 0 at chunk start
             s  = d >= 2^(bits-1) ? d - 2^bits : d  # signed interpretation
             zz = (s << 1) ^ (s >> (bits-1))        # zigzag -> uint
    decode:  s    = (zz >> 1) ^ -(zz & 1)
             code = (prev + s) mod 2^bits

Layout is **columnar** (all column-0 residuals, then column-1, ...): this is
what unlocks the gain — interleaved row-major is weak on coordinates and
*negative* on Cholesky codes. The TypeScript twin lives in
``packages/luxar-viewer/src/data/codecs/luxar-delta.ts`` (registered under BOTH
``numcodecs.luxar_delta_v1`` for format 2 and the bare ``luxar_delta_v1`` for
format 3 — zarrita looks the two up in different namespaces); keep them in 1:1
sync.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray, ndarray_copy
from numcodecs.registry import register_codec
from zarr.abc.codec import ArrayArrayCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.common import JSON, parse_named_configuration
from zarr.registry import register_codec as zarr_register_codec

#: Minimum compressed-size advantage (plain / delta) the encode-time probe
#: requires before enabling the filter. Keeps the scheme monotonic — a
#: marginal or losing array is stored exactly as before.
DELTA_PROBE_MIN_GAIN = 1.02


def _wire_dtype(bits: int) -> np.dtype:
    """The unsigned dtype ``bits`` codes are stored in."""
    return np.dtype(np.uint8 if bits == 8 else np.uint16)


def delta_encode(codes: np.ndarray, cols: int, bits: int) -> np.ndarray:
    """``rows x cols`` codes -> columnar zigzag residuals.

    Returns a C-contiguous ``(cols, rows)`` array, so its flat byte order IS the
    wire order (all column-0 residuals, then column-1, ...).

    THE WIRE FORMAT LIVES HERE. Both the format-2 filter (:class:`LuxarDelta`)
    and the format-3 codec (:class:`LuxarDeltaV3`) delegate to this function so
    the bytes cannot drift between them — a store written under either format
    must decode with the single TypeScript twin, which has no way to tell which
    Python plumbing produced it.
    """
    rows = codes.size // cols
    m = 1 << bits
    half = m >> 1
    ch = codes.reshape(rows, cols).astype(np.int32)
    prev = np.vstack([np.zeros((1, cols), np.int32), ch[:-1]])
    d = (ch - prev) & (m - 1)
    s = np.where(d >= half, d - m, d)
    zz = ((s << 1) ^ (s >> 31)).astype(_wire_dtype(bits))
    return np.ascontiguousarray(zz.T)


def delta_decode(zz: np.ndarray, cols: int, bits: int) -> np.ndarray:
    """Inverse of :func:`delta_encode`: wire residuals -> ``rows x cols`` codes."""
    rows = zz.size // cols
    m = 1 << bits
    z = zz.reshape(cols, rows).astype(np.int32)
    s = (z >> 1) ^ -(z & 1)
    # Per-column running sum along rows undoes the delta (axis=1: rows).
    codes = (np.cumsum(s, axis=1) & (m - 1)).astype(_wire_dtype(bits))
    return np.ascontiguousarray(codes.T)  # back to row-major


class LuxarDelta(Codec):
    """Columnar per-chunk modular delta + zigzag for uint8/uint16 codes.

    Parameters
    ----------
    cols:
        Number of columns (channels) per row in each chunk. Chunks must hold
        whole rows (Luxar chunking never splits columns); ``encode``/``decode``
        raise if the chunk size is not a multiple of ``cols``.
    bits:
        Code width — 8 or 16. Must match the array dtype (uint8 / uint16).
    """

    codec_id = "luxar_delta_v1"

    def __init__(self, cols: int = 1, bits: int = 16):
        if bits not in (8, 16):
            raise ValueError(f"bits must be 8 or 16, got {bits}")
        if cols < 1:
            raise ValueError(f"cols must be >= 1, got {cols}")
        self.cols = int(cols)
        self.bits = int(bits)

    @property
    def _dtype(self) -> np.dtype:
        return np.dtype(np.uint8 if self.bits == 8 else np.uint16)

    def _as_codes(self, buf: Any) -> np.ndarray:
        """Flat 1D view of ``buf`` as this codec's unsigned dtype."""
        # numcodecs flattens in MEMORY order ('A'): an F-contiguous 2D input
        # would silently flatten column-major and scramble the stride-`cols`
        # transform. zarr v2 chunks are always C-contiguous (zarr normalizes
        # input order before filters — verified empirically), so this only
        # fires on direct misuse of the codec — fail loud, never corrupt.
        if isinstance(buf, np.ndarray) and buf.ndim >= 2 and not buf.flags.c_contiguous:
            raise ValueError(
                "luxar_delta_v1: multi-dimensional chunk must be C-contiguous"
            )
        arr = np.asarray(ensure_contiguous_ndarray(buf))
        if arr.dtype != self._dtype:
            arr = arr.view(self._dtype)
        if arr.size % self.cols != 0:
            raise ValueError(
                f"luxar_delta_v1: chunk of {arr.size} elements is not a "
                f"multiple of cols={self.cols} (columns must never be split)"
            )
        return arr

    def encode(self, buf: Any) -> np.ndarray:
        # Columnar layout within the chunk (the compression-critical choice).
        return delta_encode(self._as_codes(buf), self.cols, self.bits)

    def decode(self, buf: Any, out: Any = None) -> Any:
        dec = delta_decode(self._as_codes(buf), self.cols, self.bits)
        if out is not None:
            return ndarray_copy(dec, out)
        return dec

    def get_config(self) -> dict:
        return {"id": self.codec_id, "cols": self.cols, "bits": self.bits}

    def __repr__(self) -> str:
        return f"LuxarDelta(cols={self.cols}, bits={self.bits})"


register_codec(LuxarDelta)


@dataclass(frozen=True)
class LuxarDeltaV3(ArrayArrayCodec):
    """The format-3 spelling of :class:`LuxarDelta` — same bytes, new plumbing.

    Format 2 and format 3 disagree about WHERE a filter sits, not about what it
    does. A v2 numcodecs filter is handed the chunk's raw bytes, before the dtype
    is interpreted; a v3 ``ArrayArrayCodec`` is handed a typed array, after.
    The transform is identical either way, so both delegate to
    :func:`delta_encode` / :func:`delta_decode`.

    That positional difference is also what retires a subtlety the v2 filter had
    to guard: the TypeScript twin has ALWAYS been a zarrita ``array_to_array``
    codec, so under v2 it ran on the far side of the endian-converting bytes
    codec from its Python counterpart, and a big-endian store would have had the
    two sides disagree about which bytes were residuals. Under v3 both sides run
    array-to-array, in the same place, so that asymmetry is gone. The LE-only
    rail in :func:`probe_delta_filter` is kept anyway — Luxar emits native-LE
    only, and a rail that cannot fire costs nothing.

    Shape is PRESERVED rather than transposed, which is why there is no
    ``resolve_metadata`` override: the encoded array keeps the chunk's declared
    ``(rows, cols)`` shape and carries the columnar residuals in its flat order.
    Declaring ``(cols, rows)`` instead would be equally decodable by this codec
    but would change what the bytes codec is told about the chunk, and the wire
    bytes would no longer match a v2 store's.
    """

    is_fixed_size = True

    cols: int
    bits: int

    def __init__(self, *, cols: int = 1, bits: int = 16) -> None:
        if bits not in (8, 16):
            raise ValueError(f"bits must be 8 or 16, got {bits}")
        if cols < 1:
            raise ValueError(f"cols must be >= 1, got {cols}")
        object.__setattr__(self, "cols", int(cols))
        object.__setattr__(self, "bits", int(bits))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> "LuxarDeltaV3":
        """Rebuild from stored metadata (``name`` + ``configuration``)."""
        _, configuration = parse_named_configuration(data, LuxarDelta.codec_id)
        return cls(**configuration)

    def to_dict(self) -> dict[str, JSON]:
        """Serialize to the v3 codec-chain entry the store records."""
        # Deliberately the SAME name as the v2 filter's `codec_id`: one wire
        # name, one TypeScript twin, whichever format the store is in.
        return {
            "name": LuxarDelta.codec_id,
            "configuration": {"cols": self.cols, "bits": self.bits},
        }

    def _check(self, chunk_array: Any) -> np.ndarray:
        arr = np.asarray(chunk_array.as_ndarray_like())
        if arr.dtype != _wire_dtype(self.bits):
            raise ValueError(
                f"{LuxarDelta.codec_id}: configured bits={self.bits} does not "
                f"match the chunk dtype {arr.dtype}"
            )
        if arr.size % self.cols != 0:
            raise ValueError(
                f"{LuxarDelta.codec_id}: chunk of {arr.size} elements is not a "
                f"multiple of cols={self.cols} (columns must never be split)"
            )
        return arr

    async def _encode_single(
        self, chunk_array: Any, chunk_spec: ArraySpec
    ) -> Any | None:
        arr = self._check(chunk_array)
        wire = delta_encode(arr, self.cols, self.bits)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(wire.reshape(arr.shape))

    async def _decode_single(self, chunk_array: Any, chunk_spec: ArraySpec) -> Any:
        arr = self._check(chunk_array)
        codes = delta_decode(arr, self.cols, self.bits)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(
            codes.reshape(arr.shape)
        )

    def compute_encoded_size(self, input_byte_length: int, _spec: ArraySpec) -> int:
        """Byte-for-byte the same size — a permutation plus a bijection."""
        return input_byte_length


# Registered eagerly under the SAME name the v2 filter uses, mirroring
# `register_codec(LuxarDelta)` above. The `zarr.codecs` entry point in
# pyproject.toml is the lazy fallback for a process that never imports this
# module (a bare `zarr.open_group` on a delta store); this call covers every
# process that does, without depending on the package's metadata being
# reinstalled after an edit.
zarr_register_codec(LuxarDelta.codec_id, LuxarDeltaV3)


def _normalize_chunks(chunks: Any) -> Optional[tuple]:
    """``chunks`` as an int tuple, or ``None`` when its shape is unknowable.

    zarr's ``create_dataset`` accepts many chunk specs (``None``, ``True``,
    ``False``, a bare int, ``"auto"``, an int sequence). The probe can only
    reason about an explicit int sequence (or a bare int for 1D); everything
    else is treated as "unknown" — the caller then declines for 2D arrays and
    falls back to whole-array rows for 1D. ``bool`` is excluded explicitly
    (it subclasses ``int``).
    """
    if isinstance(chunks, (int, np.integer)) and not isinstance(chunks, bool):
        return (int(chunks),)
    if isinstance(chunks, (tuple, list)) and all(
        isinstance(c, (int, np.integer)) and not isinstance(c, bool) for c in chunks
    ):
        return tuple(int(c) for c in chunks)
    return None


def probe_delta_filter(
    codes: np.ndarray,
    chunks: Optional[tuple],
    compressor: Optional[Any],
) -> Optional[list]:
    """Encode-time probe: ``[LuxarDelta(...)]`` if delta wins, else ``None``.

    Compresses one representative chunk of ``codes`` both plain and
    delta-filtered with the RESOLVED ``compressor`` and enables the filter
    only when the plain size exceeds the delta size by
    :data:`DELTA_PROBE_MIN_GAIN`. Deterministic (middle chunk, no RNG) so
    identical inputs always produce identical stores.

    Returns ``None`` (no filter) whenever delta cannot be applied safely or
    profitably: empty / non-uint8/16 ``codes``, no compressor, or a 2D array
    whose ``chunks`` are unknown or split columns (the per-chunk stride-
    ``cols`` transform requires whole rows per chunk).
    """
    codes = np.asarray(codes)
    # Explicitly LITTLE-endian u8/u16 only (dtype.str, not dtype equality, so
    # the rule is platform-independent): the viewer's zarrita pipeline runs
    # its endian-converting bytes codec BEFORE array_to_array filters, while
    # Python zarr views the dtype AFTER filters — for a big-endian store the
    # two sides would disagree on the residual bytes. Luxar writers only emit
    # native-LE arrays in practice; this rail keeps a hypothetical big-endian
    # writer from producing stores the viewer mis-decodes.
    if codes.size == 0 or codes.dtype.str not in ("|u1", "<u2"):
        return None
    if compressor is None:
        return None
    chunk_tuple = _normalize_chunks(chunks)
    if codes.ndim == 1:
        cols = 1
        rows_per_chunk = chunk_tuple[0] if chunk_tuple else codes.shape[0]
    elif codes.ndim == 2:
        cols = codes.shape[1]
        # 2D requires explicit chunks that keep columns whole; Luxar's
        # intelligent chunking always does, but an auto-chunked (chunks=None/
        # True/"auto") array offers no such guarantee — decline rather than
        # risk it.
        if not chunk_tuple or len(chunk_tuple) != 2 or chunk_tuple[1] != cols:
            return None
        rows_per_chunk = chunk_tuple[0]
    else:
        return None
    if rows_per_chunk <= 0:
        return None

    n_rows = codes.shape[0]
    n_chunks = max(1, -(-n_rows // rows_per_chunk))
    mid = (n_chunks - 1) // 2
    sample = codes[mid * rows_per_chunk : (mid + 1) * rows_per_chunk]
    sample = np.ascontiguousarray(sample)

    codec = LuxarDelta(cols=cols, bits=8 * codes.dtype.itemsize)
    plain_size = len(compressor.encode(sample))
    delta_size = len(compressor.encode(codec.encode(sample)))
    if delta_size > 0 and plain_size > delta_size * DELTA_PROBE_MIN_GAIN:
        return [codec]
    return None
