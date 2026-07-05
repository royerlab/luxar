"""Per-dtype compressor policy for Luxar zarr arrays.

Measured under deployment conditions (64 KiB chunks, Hilbert-ordered splats,
native + in-browser wasm decode — manuscript supplementary
``codec_selection``), the best general-purpose configuration is not one
compressor but a policy keyed on the ELEMENT WIDTH of the stored codes:

- multi-byte integer codes (uint16 fixed-point / quantized) → ``zstd`` level 9
  with BYTE shuffle (bit shuffle is silently neutralised by Blosc's internal
  heuristics above level 1 at 64 KiB chunks; byte shuffle at high level is
  what actually engages);
- single-byte codes (uint8) → ``zstd`` level 9, no shuffle (filters are
  no-ops or harmful for single-byte payloads);
- floats → ``zstd`` level 9, no shuffle (mantissa entropy dominates; the
  structured exponent bytes compress without help).

Decode speed is level-independent (natively and in wasm), so the high level
is purely a write-time budget. The policy applies whenever a writer asks for
the DEFAULT compressor (the :data:`WIDTH_AWARE_DEFAULT` sentinel); an
explicit compressor object passes through untouched, and ``None`` still
means "store uncompressed".
"""

from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from numcodecs import Blosc

if TYPE_CHECKING:
    from ..typing_utils.protocols import CompressorProtocol


class _WidthAwareDefault:
    """Sentinel: 'pick the compressor from the array's dtype at write time'.

    A distinct marker class (not ``None``, which means *no* compression, and
    not a concrete ``Blosc`` instance, which would freeze one configuration
    for every dtype). Writers thread this through untouched; the single
    resolution point is :func:`resolve_compressor`, called where the encoded
    dtype is finally known.
    """

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "<width-aware default compressor>"


#: The default compressor "value" threaded through all writers.
WIDTH_AWARE_DEFAULT = _WidthAwareDefault()

#: What a writer may pass as ``compressor``: a concrete compressor, the
#: width-aware sentinel, or ``None`` (store uncompressed).
CompressorLike = Union["CompressorProtocol", _WidthAwareDefault, None]

# One concrete compressor per width class (module-level: Blosc instances are
# stateless and reusable across arrays).
_MULTIBYTE_INT = Blosc(cname="zstd", clevel=9, shuffle=Blosc.SHUFFLE)
_PLAIN = Blosc(cname="zstd", clevel=9, shuffle=Blosc.NOSHUFFLE)


def resolve_compressor(compressor: Optional[Any], dtype: Any) -> Optional[Any]:
    """Resolve a writer-supplied compressor for an array of ``dtype``.

    - :data:`WIDTH_AWARE_DEFAULT` → the per-dtype policy above;
    - any explicit compressor object → passed through untouched;
    - ``None`` → ``None`` (store uncompressed).
    """
    if not isinstance(compressor, _WidthAwareDefault):
        return compressor
    dt = np.dtype(dtype)
    if dt.kind in ("i", "u") and dt.itemsize >= 2:
        return _MULTIBYTE_INT
    return _PLAIN
