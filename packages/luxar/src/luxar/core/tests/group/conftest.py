"""Helpers shared by the ``group/`` test packages.

Plain functions rather than fixtures, imported explicitly (``from ..conftest
import …``) — the convention the ``gsplats/`` test trees already use for shared
test utilities. Defined here so the partition/ and lod/ halves of the #1437
pre-split-gate suite state the same scene setup once; nothing here is autouse,
so the existing test modules in these directories are unaffected.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np

from luxar import Dimension, Dimensions, LuxarZarrCompiler


def make_3d_dims() -> Dimensions:
    """Three displayed spatial dimensions — the plainest scene a leaf can join."""
    return Dimensions(
        [
            Dimension("X", display=True),
            Dimension("Y", display=True),
            Dimension("Z", display=True),
        ]
    )


def random_positions(n: int, seed: int) -> np.ndarray:
    """``(n, 3)`` float32 coordinates in ``[0, 100)``, reproducible per seed."""
    rng = np.random.default_rng(seed)
    return (rng.random((n, 3)) * 100.0).astype(np.float32)


def cholesky_rows(n: int) -> np.ndarray:
    """``(n, 6)`` packed identity Cholesky factors for a 3-D gsplat node."""
    return np.tile(np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1))


def open_scene(tmp_path: Any, filename: str) -> Tuple[LuxarZarrCompiler, Any, str]:
    """A fresh compiler + 3-D scene under ``tmp_path``; returns it with its path."""
    path = str(tmp_path / filename)
    compiler = LuxarZarrCompiler(path)
    scene = compiler.create_scene(dimensions=make_3d_dims())
    return compiler, scene, path


def assert_uniform(actual: Any, expected: Any, count: int, atol: float = 5e-3) -> None:
    """Assert a decoded broadcast channel is ``expected`` for all ``count`` elements.

    ``count`` is not optional on purpose: the decoder expands a stored broadcast
    on some channels and keeps the ``(1, c)`` row on others, so the expectation is
    stated per element — and an all-close against a broadcast target passes
    VACUOUSLY on an empty array, which is exactly the failure a count catches.
    """
    arr = np.asarray(actual, dtype=np.float64)
    want = np.asarray(expected, dtype=np.float64)
    assert arr.shape[0] in (1, count), (
        f"expected a per-element ({count},…) or broadcast (1,…) channel, "
        f"got shape {arr.shape}"
    )
    np.testing.assert_allclose(arr, np.broadcast_to(want, arr.shape), atol=atol)


def refusal(call: Callable[[], Any]) -> Exception:
    """Run ``call``, require it to raise, and return the exception.

    Deliberately catches ``Exception`` rather than ``ValueError``: the point of a
    parity assertion is that the two paths agree on the TYPE too, so narrowing
    here would hide exactly the divergence being tested.
    """
    try:
        call()
    except Exception as exc:
        return exc
    raise AssertionError("expected the call to be refused, but it succeeded")


def assert_same_refusal(flat: Exception, split: Exception) -> None:
    """Assert a split path refused an input the same way the plain leaf did.

    Exact message equality, not a substring: the messages ARE byte-identical
    today (the shared validator formats them once), so anything weaker would let
    a future wording divergence — or an exception-type change on one path —
    through. A loose ``match="colors"`` is especially weak here, since the
    unrelated "Cannot specify both 'colors' and 'colormap'" satisfies it.
    """
    assert type(split) is type(flat), (
        f"split path raised {type(split).__name__} where the flat path raised "
        f"{type(flat).__name__}: {split!r} vs {flat!r}"
    )
    assert str(split) == str(flat), (
        f"split path message differs from the flat one:\n"
        f"  flat:  {str(flat)!r}\n"
        f"  split: {str(split)!r}"
    )
