"""``ordering`` is a reserved, writer-stamped node attr (issue #1221).

The geometry writers stamp ``group.attrs["ordering"]`` authoritatively from the
compiler's ``ordering_method``. A caller-supplied ``ordering=`` used to be
re-persisted over that stamp, desyncing the attr from how the arrays are
actually sorted. These tests lock in that:

* passing ``ordering=`` to ``add_points`` / ``add_lines`` / ``add_gsplats``
  is rejected up front as a reserved attr; and
* with the kwarg omitted, the stamped ``ordering`` attr on disk equals the
  compiler's ``ordering_method`` for every geometry type.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.io.compiler import LuxarZarrCompiler


def _points() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(0, 50, size=(64, 3)).astype(np.float32)


def _lines() -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    vertices = rng.uniform(0, 50, size=(64, 3)).astype(np.float32)
    widths = np.full(64, 0.5, dtype=np.float32)
    return vertices, widths


def _gsplats() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    n = 64
    centers = rng.uniform(0, 50, size=(n, 3)).astype(np.float32)
    amplitudes = rng.uniform(0.1, 1.0, size=(n,)).astype(np.float32)
    cholesky = np.zeros((n, 6), dtype=np.float32)
    cholesky[:, [0, 2, 5]] = rng.uniform(0.5, 2.0, size=(n, 3))
    return centers, amplitudes, cholesky


# Every caller value is rejected key-first, so a genuine method ("morton"), the
# "index off" sentinel ("none"), and an unrecognised string ("zzz") are all the
# same class of bug the issue lists — none may reach disk and overwrite the
# stamp. ``match`` pins the collided attr AND the ordering_method hint (the
# error must point migrating callers at the real knob), not just any
# "reserved" text.
_BAD_ORDERINGS = ["morton", "none", "zzz"]
_REJECTED = r"ordering.*reserved.*ordering_method"


@pytest.mark.parametrize("bad_value", _BAD_ORDERINGS)
def test_add_points_ordering_kwarg_is_rejected(bad_value: str) -> None:
    positions = _points()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scene.luxar.zarr"
        with LuxarZarrCompiler(path) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match=_REJECTED):
                scene.add_points("p", positions, ordering=bad_value)


@pytest.mark.parametrize("bad_value", _BAD_ORDERINGS)
def test_add_lines_ordering_kwarg_is_rejected(bad_value: str) -> None:
    vertices, widths = _lines()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scene.luxar.zarr"
        with LuxarZarrCompiler(path) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match=_REJECTED):
                scene.add_lines("l", vertices, widths, ordering=bad_value)


@pytest.mark.parametrize("bad_value", _BAD_ORDERINGS)
def test_add_gsplats_ordering_kwarg_is_rejected(bad_value: str) -> None:
    centers, amplitudes, cholesky = _gsplats()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scene.luxar.zarr"
        with LuxarZarrCompiler(path) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match=_REJECTED):
                scene.add_gsplats(
                    "g",
                    centers=centers,
                    amplitudes=amplitudes,
                    cholesky_factors=cholesky,
                    ordering=bad_value,
                )


@pytest.mark.parametrize("ordering_method", ["morton", "hilbert"])
def test_points_stamp_matches_compiler_ordering(ordering_method: str) -> None:
    positions = _points()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scene.luxar.zarr"
        with LuxarZarrCompiler(path, ordering_method=ordering_method) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("p", positions)
        node = zarr.open_group(str(path), mode="r")["p"]
        assert node.attrs["ordering"] == ordering_method


@pytest.mark.parametrize("ordering_method", ["morton", "hilbert"])
def test_lines_stamp_matches_compiler_ordering(ordering_method: str) -> None:
    vertices, widths = _lines()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scene.luxar.zarr"
        with LuxarZarrCompiler(path, ordering_method=ordering_method) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines("l", vertices, widths)
        node = zarr.open_group(str(path), mode="r")["l"]
        assert node.attrs["ordering"] == ordering_method


@pytest.mark.parametrize("ordering_method", ["morton", "hilbert"])
def test_gsplats_stamp_matches_compiler_ordering(ordering_method: str) -> None:
    centers, amplitudes, cholesky = _gsplats()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scene.luxar.zarr"
        with LuxarZarrCompiler(path, ordering_method=ordering_method) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats(
                "g",
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
            )
        node = zarr.open_group(str(path), mode="r")["g"]
        assert node.attrs["ordering"] == ordering_method
