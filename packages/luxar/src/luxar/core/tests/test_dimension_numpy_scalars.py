"""A Dimension built from numpy scalars must still serialise.

``Dimension.to_dict`` emits its scalar fields verbatim, so a producer that
derives values from an array -- the obvious way to do it -- used to write
values ``json.dumps`` refuses with "Object of type float32 is not JSON
serializable".

That failure is nastier than it sounds: zarr raises it while SAVING, by which
point the compiler has already created the store, so the run dies leaving a
4 KB directory containing nothing but ``zarr.json``. ``demo_ppi_flow_field``
shipped in exactly that state -- a scene that looked present on disk and held
nothing. Coercion happens in ``__post_init__`` so every producer is covered,
not just the one demo that was noticed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from luxar.core.dimensions import Dimension


def test_float_range_from_numpy_is_json_serialisable() -> None:
    d = Dimension("x", range=(np.float32(-1.5), np.float32(2.5)))
    assert isinstance(d.range[0], float) and not isinstance(d.range[0], np.generic)
    json.dumps(d.to_dict())  # would raise TypeError before the coercion


def test_integer_range_stays_integral() -> None:
    """`.item()` preserves integrality -- an index range must not become float."""
    d = Dimension("c", range=(np.int64(0), np.int64(7)), display=False, discrete=True)
    assert d.range == (0, 7)
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in d.range)
    json.dumps(d.to_dict())


def test_step_and_scale_are_coerced_too() -> None:
    """to_dict emits these verbatim as well, so they carry the same hazard."""
    d = Dimension("t", range=(0.0, 1.0), step=np.float64(0.25), scale=np.float32(2.0))
    assert isinstance(d.step, float) and not isinstance(d.step, np.generic)
    assert isinstance(d.scale, float) and not isinstance(d.scale, np.generic)
    json.dumps(d.to_dict())


def test_numpy_boolean_fields_are_json_serialisable() -> None:
    d = Dimension(
        "t",
        range=(0, 5),
        display=np.bool_(False),
        discrete=np.bool_(True),
        cyclic=np.bool_(True),
        spatial=np.bool_(False),
    )
    values = (d.display, d.discrete, d.cyclic, d.spatial)
    assert values == (False, True, True, False)
    assert all(isinstance(value, bool) for value in values)
    assert not any(isinstance(value, np.generic) for value in values)
    json.dumps(d.to_dict())


@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.float64, np.int32, np.uint8]
)
def test_every_common_scalar_dtype_round_trips(dtype) -> None:
    d = Dimension("x", range=(dtype(0), dtype(3)))
    assert json.loads(json.dumps(d.to_dict()))["range"] == [0, 3]


def test_plain_python_values_are_untouched() -> None:
    """The coercion must not disturb the ordinary path."""
    d = Dimension("x", range=(0, 10), step=2, scale=1.0)
    assert d.range == (0, 10) and d.step == 2 and d.scale == 1.0
