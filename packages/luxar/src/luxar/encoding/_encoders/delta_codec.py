"""Columnar per-chunk delta+zigzag pre-filter for quantized code arrays.

Luxar stores coordinate / Cholesky / amplitude arrays as Hilbert-ordered
uint8/uint16 quantization codes. Spatial ordering makes consecutive codes a
smooth ramp, but Blosc's byte-shuffle cannot exploit that smoothness. This
zarr v2 **filter** turns the ramp into small residuals — per-axis modular
delta, zigzag-mapped to unsigned, laid out column-major within the chunk —
which zstd then crushes (measured ~12% whole-store lossless on real fits;
see the 2026-07 compression-transfer campaign / SOG comparison).

Why a zarr *filter* and not an encoding transform: the viewer range-loader
decodes arbitrary sub-chunk element ranges with stateless per-element
kernels, so a sequential cumsum can only live where whole chunks are
reconstructed — inside the zarr codec pipeline (zarrita ``getChunk`` on the
TS side, numcodecs here). The stored ``encoding`` attrs are untouched; only
``.zarray`` gains a ``filters`` entry, and each chunk is self-contained
(implicit 0 anchor per column per chunk).

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
``packages/luxar-viewer/src/data/codecs/luxar-delta.ts`` (registered as
``numcodecs.luxar_delta_v1``); keep the two in 1:1 sync.
"""

from typing import Any, Optional

import numpy as np
from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray, ndarray_copy
from numcodecs.registry import register_codec

#: Minimum compressed-size advantage (plain / delta) the encode-time probe
#: requires before enabling the filter. Keeps the scheme monotonic — a
#: marginal or losing array is stored exactly as before.
DELTA_PROBE_MIN_GAIN = 1.02


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
        codes = self._as_codes(buf)
        rows = codes.size // self.cols
        m = 1 << self.bits
        half = m >> 1
        ch = codes.reshape(rows, self.cols).astype(np.int32)
        prev = np.vstack([np.zeros((1, self.cols), np.int32), ch[:-1]])
        d = (ch - prev) & (m - 1)
        s = np.where(d >= half, d - m, d)
        zz = ((s << 1) ^ (s >> 31)).astype(self._dtype)
        # Columnar layout within the chunk (the compression-critical choice).
        return np.ascontiguousarray(zz.T)

    def decode(self, buf: Any, out: Any = None) -> Any:
        zz = self._as_codes(buf)
        rows = zz.size // self.cols
        m = 1 << self.bits
        z = zz.reshape(self.cols, rows).astype(np.int32)
        s = (z >> 1) ^ -(z & 1)
        # Per-column running sum along rows undoes the delta (axis=1: rows).
        codes = (np.cumsum(s, axis=1) & (m - 1)).astype(self._dtype)
        dec = np.ascontiguousarray(codes.T)  # back to row-major
        if out is not None:
            return ndarray_copy(dec, out)
        return dec

    def get_config(self) -> dict:
        return {"id": self.codec_id, "cols": self.cols, "bits": self.bits}

    def __repr__(self) -> str:
        return f"LuxarDelta(cols={self.cols}, bits={self.bits})"


register_codec(LuxarDelta)


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
    if codes.size == 0 or codes.dtype not in (np.uint8, np.uint16):
        return None
    if compressor is None:
        return None
    if codes.ndim == 1:
        cols = 1
        rows_per_chunk = int(chunks[0]) if chunks else codes.shape[0]
    elif codes.ndim == 2:
        cols = codes.shape[1]
        # 2D requires explicit chunks that keep columns whole; Luxar's
        # intelligent chunking always does, but an auto-chunked (chunks=None)
        # array offers no such guarantee — decline rather than risk it.
        if not chunks or len(chunks) != 2 or int(chunks[1]) != cols:
            return None
        rows_per_chunk = int(chunks[0])
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
