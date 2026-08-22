"""Couple positive-scalar round-trip slack to the real encoder."""

import numpy as np
import pytest

from luxar._zarr_compat import memory_group
from luxar.encoding import ArrayDecoder, ArrayEncoder, EncodingMode, SemanticType


@pytest.mark.parametrize(
    ("name", "data", "mode", "encoding_type", "expected_encoding"),
    [
        (
            "linear_u8",
            np.linspace(0.1, 5.0, 4001, dtype=np.float32),
            EncodingMode.AUTO,
            "linear",
            "bounded_scalar_uint8",
        ),
        (
            "linear_u16",
            np.geomspace(1e-4, 5.0, 4001, dtype=np.float32),
            EncodingMode.AUTO,
            "linear",
            "bounded_scalar_uint16",
        ),
        (
            "linear_geolog",
            np.geomspace(1e-8, 5.0, 4001, dtype=np.float32),
            EncodingMode.AUTO,
            "linear",
            "geolog_scalar_uint16",
        ),
        (
            "explicit_log_memory",
            np.geomspace(0.1, 5.0, 4001, dtype=np.float32),
            EncodingMode.MEMORY,
            "log",
            "geolog_scalar_uint8",
        ),
    ],
)
def test_positive_scalar_slack_bounds_real_upward_displacement(
    name: str,
    data: np.ndarray,
    mode: EncodingMode,
    encoding_type: str,
    expected_encoding: str,
) -> None:
    encoder = ArrayEncoder()
    slack = encoder.positive_scalar_round_trip_slack(
        data, mode, positive_scalar_encoding=encoding_type, allow_lut=False
    )
    assert slack is not None and slack > 0.0, name

    group = memory_group()
    encoder.encode(
        data,
        group,
        "s",
        SemanticType.POSITIVE_SCALAR,
        mode=mode,
        positive_scalar_encoding=encoding_type,
        allow_lut=False,
        deduplicate=False,
    )
    assert group["s"].attrs["encoding"]["name"] == expected_encoding
    decoded = ArrayDecoder().decode(group["s"], group)
    upward = decoded.astype(np.float64) - data.astype(np.float64)
    assert float(upward.max()) > 0.0, name
    assert float(upward.max()) <= slack, name


@pytest.mark.parametrize("mode", list(EncodingMode))
def test_positive_scalar_slack_reports_exact_encoder_exits(mode: EncodingMode) -> None:
    encoder = ArrayEncoder()
    constant = np.full(4001, 2.5, dtype=np.float32)
    if mode == EncodingMode.CUSTOM:
        assert encoder.positive_scalar_round_trip_slack(constant, mode) is None
        return
    assert encoder.positive_scalar_round_trip_slack(constant, mode) is None

    if mode == EncodingMode.PRECISION:
        continuous = np.linspace(0.1, 5.0, 4001, dtype=np.float32)
        assert encoder.positive_scalar_round_trip_slack(continuous, mode) is None


def test_positive_scalar_slack_honours_the_lut_gate() -> None:
    encoder = ArrayEncoder()
    palette = np.linspace(0.1, 5.0, 200, dtype=np.float32)
    data = np.resize(palette, 4001)
    assert encoder.encodes_as_lut(data, SemanticType.POSITIVE_SCALAR)
    assert encoder.positive_scalar_round_trip_slack(data, EncodingMode.AUTO) is None
    assert (
        encoder.positive_scalar_round_trip_slack(
            data, EncodingMode.AUTO, allow_lut=False
        )
        is not None
    )
