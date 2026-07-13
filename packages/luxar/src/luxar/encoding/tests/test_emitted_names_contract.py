"""Producer half of the #10 SSOT loop: every name the encoder EMITS ∈ contract.

``test_custom_dispatch_contract`` covers the CUSTOM-mode dispatch subset, and
``format-contract.test.ts`` covers the consumer half (decoder recognizes every
contract name). This closes the gap those leave: the AUTO / MEMORY quantization
families the ``ArrayEncoder`` actually writes (bounded/geolog scalar, the
per-channel log/linear/geolog families, rgb, coordinate fixed-point, the
Cholesky split) are exercised here and every emitted ``encoding.name`` is
asserted to be a member of the single-sourced contract. Delete an encoding from
``format-contract/contract.yaml`` (and regenerate) while the encoder still emits
it and this fails — which the drift gate alone would NOT catch.
"""

from __future__ import annotations

import numpy as np
import zarr

from luxar.encoding import ArrayEncoder, EncodingMode, SemanticType
from luxar.typing_utils._format_contract import ENCODING_NAMES


def _emit(data, semantic_type, mode=EncodingMode.AUTO, **kw) -> str:
    """Encode ``data`` and return the emitted ``encoding.name``."""
    g = zarr.group(store=zarr.MemoryStore())
    ArrayEncoder().encode(
        data=data, zarr_group=g, name="a", semantic_type=semantic_type, mode=mode, **kw
    )
    return str(g["a"].attrs["encoding"]["name"])


def _collect_emitted_names() -> set[str]:
    rng = np.random.default_rng(0)
    names: set[str] = set()

    # POSITIVE_SCALAR → bounded_scalar_uint8 / _uint16 / geolog_scalar_uint16
    names.add(_emit(np.linspace(0.1, 1.0, 1000, dtype=np.float32), SemanticType.POSITIVE_SCALAR))
    names.add(_emit(np.linspace(0.001, 1.0, 1000, dtype=np.float32), SemanticType.POSITIVE_SCALAR))
    names.add(_emit(np.array([1e-6, 1e-3, 1.0], dtype=np.float32), SemanticType.POSITIVE_SCALAR))

    # COORDINATE → linear_perchannel_u16 (per-axis fixed point)
    pos = (rng.random((4000, 3)) * [400, 1800, 2000] + [3, 100, 30]).astype(np.float32)
    names.add(_emit(pos, SemanticType.COORDINATE))

    # COLOR → rgb_uint8 (SDR) / geolog_perchannel_u16 (HDR AUTO) / _u8 (HDR MEMORY)
    hdr = (rng.random((2000, 3)).astype(np.float32) * 50.0 + 0.01)
    names.add(_emit(hdr, SemanticType.COLOR, EncodingMode.AUTO, color_mode="hdr"))
    names.add(_emit(hdr, SemanticType.COLOR, EncodingMode.MEMORY, color_mode="hdr"))
    names.add(_emit(rng.random((2000, 3)).astype(np.float32), SemanticType.COLOR, color_mode="sdr"))

    # Broadcast (uniform) and passthrough (empty) → "broadcasted" / "none"
    names.add(_emit(np.full(500, 2.0, dtype=np.float32), SemanticType.POSITIVE_SCALAR))
    names.add(_emit(np.zeros((0, 3), dtype=np.float32), SemanticType.COORDINATE))

    # UNIT_VECTOR → float32; INDEX → smallest uint
    uv = rng.standard_normal((200, 3)).astype(np.float32)
    uv /= np.linalg.norm(uv, axis=1, keepdims=True)
    names.add(_emit(uv, SemanticType.UNIT_VECTOR))
    names.add(_emit(np.arange(70000, dtype=np.int64), SemanticType.INDEX))

    # CHOLESKY split entry point → log_perchannel_u8 + signed_log_perchannel_u8
    g = zarr.group(store=zarr.MemoryStore())
    diag = rng.uniform(0.4, 5.0, size=(2000, 3)).astype(np.float32)
    off = (rng.standard_normal((2000, 3)) * 0.3).astype(np.float32)
    ArrayEncoder().encode_cholesky_split(g, diag, off, 3, EncodingMode.AUTO)
    names.add(str(g["cholesky_factors_diag"].attrs["encoding"]["name"]))
    names.add(str(g["cholesky_factors_offdiag"].attrs["encoding"]["name"]))

    return names


def test_all_emitted_encoding_names_are_in_contract() -> None:
    emitted = _collect_emitted_names()
    # Guard against silent coverage degradation (a recipe that stops emitting a
    # quantized name would otherwise make this pass vacuously).
    assert len(emitted) >= 8, f"emitted-name coverage regressed: only {sorted(emitted)}"
    unknown = emitted - set(ENCODING_NAMES)
    assert not unknown, (
        f"ArrayEncoder emits encoding names absent from format-contract/"
        f"contract.yaml: {sorted(unknown)}. Add them to the contract (and run "
        f"`make gen-contract`)."
    )
