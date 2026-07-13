"""Lock A4.0b's rounding invariant: the lifted quantizer rounds by default but
the CUSTOM encoder truncates.

The shared ``_quantize_normalized_clip`` helper must round (``round_values=True``)
for the semantic-type encoders and truncate toward zero (``round_values=False``)
for the CUSTOM encoder — that preserves the custom encoder's historical
byte-for-byte behaviour. Round-trip suites use tolerances that would mask a
1-LSB round↔truncate flip, so this pins it directly.
"""

from __future__ import annotations

import numpy as np
import zarr

from luxar.encoding import ArrayEncoder


def test_quantize_normalized_clip_rounds_vs_truncates() -> None:
    # normalized 0.109 * 100 levels = 10.9 → round = 11, truncate = 10.
    data = np.array([0.109], dtype=np.float64)
    rounded = ArrayEncoder._quantize_normalized_clip(
        data, 0.0, 1.0, 100, np.dtype(np.uint8), round_values=True
    )
    truncated = ArrayEncoder._quantize_normalized_clip(
        data, 0.0, 1.0, 100, np.dtype(np.uint8), round_values=False
    )
    assert int(rounded[0]) == 11
    assert int(truncated[0]) == 10


def test_custom_bounded_scalar_truncates() -> None:
    # 0.109 over bounds (0, 1) at 8 bits → 0.109 * 255 = 27.795.
    # The CUSTOM encoder must truncate (27); a rounding regression would store 28.
    # Exercise _encode_custom directly — going through encode() would hit the
    # broadcast/LUT priority ladder (a low-cardinality array never reaches the
    # dtype encoders), which is not what this test is pinning.
    g = zarr.group(store=zarr.MemoryStore())
    ArrayEncoder()._encode_custom(
        g,
        "a",
        np.array([0.109, 0.109], dtype=np.float32),
        "bounded_scalar_uint8",
        (0.0, 1.0),
    )
    stored = np.asarray(g["a"])
    assert int(stored[0]) == 27, (
        f"CUSTOM bounded_scalar must truncate 27.795 → 27, got {int(stored[0])}"
    )
    assert g["a"].attrs["encoding"]["name"] == "bounded_scalar_uint8"
