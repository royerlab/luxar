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

import inspect
import re

import numpy as np
import pytest

from luxar._zarr_compat import memory_group
from luxar.encoding import ArrayEncoder, EncodingMode, SemanticType
from luxar.encoding._encoders import perchannel
from luxar.encoding._encoders.structural import StructuralEncoderMixin
from luxar.typing_utils._format_contract import ENCODING_NAMES

#: Quantized contract names no producer can emit, with the reason. Asserted
#: below to be EXACTLY the unreachable set, so this cannot quietly grow into an
#: excuse list: making one of these reachable, or adding a new dead name to the
#: contract, both fail.
UNREACHABLE: dict[str, str] = {
    "linear_perchannel_u8": (
        "COORDINATE is the only caller of _encode_linear_perchannel and passes "
        "_COORD_BITS unconditionally -- perchannel.py:635 states 'Coordinates "
        "always u16 (never u8) for both AUTO and MEMORY', because the decode "
        "contract for COORDINATE is float32 regardless of input dtype. No "
        "on-disk store carries this name."
    ),
}


def _emit(data, semantic_type, mode=EncodingMode.AUTO, **kw) -> str:
    """Encode ``data`` and return the emitted ``encoding.name``."""
    g = memory_group()
    ArrayEncoder().encode(
        data=data, zarr_group=g, name="a", semantic_type=semantic_type, mode=mode, **kw
    )
    return str(g["a"].attrs["encoding"]["name"])


def _collect_emitted_names() -> set[str]:
    rng = np.random.default_rng(0)
    names: set[str] = set()

    # POSITIVE_SCALAR → bounded_scalar_uint8 / _uint16 / geolog_scalar_uint16
    names.add(
        _emit(
            np.linspace(0.1, 1.0, 1000, dtype=np.float32), SemanticType.POSITIVE_SCALAR
        )
    )
    names.add(
        _emit(
            np.linspace(0.001, 1.0, 1000, dtype=np.float32),
            SemanticType.POSITIVE_SCALAR,
        )
    )
    names.add(
        _emit(
            np.array([1e-6, 1e-3, 1.0], dtype=np.float32), SemanticType.POSITIVE_SCALAR
        )
    )

    # COORDINATE → linear_perchannel_u16 (per-axis fixed point)
    pos = (rng.random((4000, 3)) * [400, 1800, 2000] + [3, 100, 30]).astype(np.float32)
    names.add(_emit(pos, SemanticType.COORDINATE))

    # COLOR → rgb_uint8 (SDR) / geolog_perchannel_u16 (HDR AUTO) / _u8 (HDR MEMORY)
    hdr = rng.random((2000, 3)).astype(np.float32) * 50.0 + 0.01
    names.add(_emit(hdr, SemanticType.COLOR, EncodingMode.AUTO, color_mode="hdr"))
    names.add(_emit(hdr, SemanticType.COLOR, EncodingMode.MEMORY, color_mode="hdr"))
    names.add(
        _emit(
            rng.random((2000, 3)).astype(np.float32),
            SemanticType.COLOR,
            color_mode="sdr",
        )
    )

    # Broadcast (uniform) and passthrough (empty) → "broadcasted" / "none"
    names.add(_emit(np.full(500, 2.0, dtype=np.float32), SemanticType.POSITIVE_SCALAR))
    names.add(_emit(np.zeros((0, 3), dtype=np.float32), SemanticType.COORDINATE))

    # INDEX → smallest uint
    names.add(_emit(np.arange(70000, dtype=np.int64), SemanticType.INDEX))

    # CHOLESKY split entry point → log_perchannel_u8 + signed_log_perchannel_u8
    g = memory_group()
    diag = rng.uniform(0.4, 5.0, size=(2000, 3)).astype(np.float32)
    off = (rng.standard_normal((2000, 3)) * 0.3).astype(np.float32)
    ArrayEncoder().encode_cholesky_split(g, diag, off, 3, EncodingMode.AUTO)
    names.add(str(g["cholesky_factors_diag"].attrs["encoding"]["name"]))
    names.add(str(g["cholesky_factors_offdiag"].attrs["encoding"]["name"]))

    # CHOLESKY certificate ESCALATION → log_perchannel_u16 +
    # signed_log_perchannel_u16. Reached by dynamic range, not by size: a
    # 1e-6..1e6 diagonal fails the cov_relf_p95 <= 0.05 certificate at uint8 and
    # the encoder escalates. This is the safety valve that keeps a bad
    # quantization off disk, so it is worth holding under test.
    g = memory_group()
    wide_diag = np.exp(rng.uniform(np.log(1e-6), np.log(1e6), (1000, 3))).astype(
        np.float32
    )
    wide_off = (rng.standard_normal((1000, 3)) * 1000.0).astype(np.float32)
    with pytest.warns(UserWarning, match="escalating to uint16"):
        ArrayEncoder().encode_cholesky_split(
            g, wide_diag, wide_off, 3, EncodingMode.AUTO
        )
    names.add(str(g["cholesky_factors_diag"].attrs["encoding"]["name"]))
    names.add(str(g["cholesky_factors_offdiag"].attrs["encoding"]["name"]))

    # POSITIVE_SCALAR wide range at MEMORY → geolog_scalar_uint8. Needs high
    # CARDINALITY too: a handful of distinct values wins a LUT instead.
    wide = np.exp(rng.uniform(np.log(1e-6), np.log(1e3), 5000)).astype(np.float32)
    names.add(_emit(wide, SemanticType.POSITIVE_SCALAR, EncodingMode.MEMORY))

    # Low cardinality → lut_uint8. This name appears on disk more than any other
    # quantized encoding in the built demo stores, and nothing required it to
    # stay reachable from the contract side.
    names.add(
        _emit(
            np.repeat(np.arange(10, dtype=np.float32), 400),
            SemanticType.POSITIVE_SCALAR,
        )
    )

    # 256 < distinct rows <= 65536 in ROW mode → lut_uint16. The uint16 tier
    # also has to beat storing the values raw, so the repeat count matters.
    rows = np.repeat((rng.random((900, 3)) * 255).astype(np.uint8), 400, axis=0)
    names.add(_emit(rows, SemanticType.COLOR, color_mode="sdr"))

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


def _quantized_contract_names() -> set[str]:
    """Contract names carrying a bit-width suffix -- the f-string-built ones."""
    return {n for n in ENCODING_NAMES if re.search(r"_u(?:int)?\d+$", n)}


def test_linear_perchannel_u8_remains_unreachable() -> None:
    assert perchannel._COORD_BITS == 16
    source = inspect.getsource(perchannel)
    assert source.count("self._encode_linear_perchannel(") == 1


def test_every_quantized_contract_name_is_reachable() -> None:
    """The reverse of the subset check: contract ⊆ (emitted ∪ dispatch ∪ known-dead).

    ``test_all_emitted_encoding_names_are_in_contract`` proves the encoder never
    invents a name. It cannot prove the contract has no DEAD entries, and it
    cannot notice that a whole family disappears from the producer surface.
    Individual producer tests cover many exact names; this assertion enforces
    the reverse subset across the complete contract.
    """
    quantized = _quantized_contract_names()
    emitted = _collect_emitted_names()
    dispatch = set(StructuralEncoderMixin._CUSTOM_DISPATCH)

    unexplained = quantized - emitted - dispatch - set(UNREACHABLE)
    assert not unexplained, (
        f"no producer path reaches {sorted(unexplained)}. Either exercise them "
        f"in _collect_emitted_names, or record them in UNREACHABLE with the "
        f"reason and the file:line that enforces it."
    )

    # The structural reason for the current entry is asserted separately above;
    # this also catches the collector or CUSTOM dispatch gaining a producer.
    now_reachable = sorted(set(UNREACHABLE) & (emitted | dispatch))
    assert not now_reachable, (
        f"{now_reachable} is recorded in UNREACHABLE but a producer now emits "
        f"it. Delete the entry — its stated reason no longer holds."
    )
