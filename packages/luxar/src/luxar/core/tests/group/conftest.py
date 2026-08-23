"""Helpers shared by the ``group/`` test packages.

Plain functions rather than fixtures, imported explicitly (``from ..conftest
import …``) — the convention the ``gsplats/`` test trees already use for shared
test utilities. Defined here so the partition/ and lod/ halves of the #1437
pre-split-gate suite state the same scene setup once; nothing here is autouse,
so the existing test modules in these directories are unaffected.
"""

from __future__ import annotations

from typing import Any, Callable, Set, Tuple

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


def cholesky_rows_nd(n: int, ndim: int) -> np.ndarray:
    """``(n, ndim*(ndim+1)/2)`` packed identity Cholesky factors.

    The ``ndim``-general form of :func:`cholesky_rows` (which it reproduces
    exactly at ``ndim=3``), needed by the #1446 tests: those feed a 4-column
    centers array to a 3-dimension scene, and ``GSplatData`` validates the
    Cholesky width against the centers width before the scene ever sees it — so
    a 6-wide row would fail for the wrong reason.
    """
    tril = np.tril(np.eye(ndim, dtype=np.float32))
    return np.tile(tril[np.tril_indices(ndim)], (n, 1))


def grid_mesh(side: int) -> Tuple[np.ndarray, np.ndarray]:
    """A welded ``side x side`` triangulated plane: ``(vertices, faces)``.

    The Mesh counterpart of :func:`random_positions`. Regular rather than random
    because a mesh's two structural paths both need real structure to chew on —
    a BSP cut needs vertices that separate spatially, and the decimator needs
    clusters that actually merge — where a point cloud only needs coordinates.
    Vertices are ``(side*side, 3)`` float32, faces ``(2*(side-1)**2, 3)`` uint32.
    """
    axis = np.arange(side, dtype=np.float32)
    gx, gy = np.meshgrid(axis, axis)
    vertices = np.stack(
        [gx.ravel(), gy.ravel(), np.zeros(side * side, dtype=np.float32)], axis=1
    ).astype(np.float32)
    faces = [
        tri
        for i in range(side - 1)
        for j in range(side - 1)
        for tri in (
            (i * side + j, i * side + j + 1, (i + 1) * side + j),
            (i * side + j + 1, (i + 1) * side + j + 1, (i + 1) * side + j),
        )
    ]
    return vertices, np.asarray(faces, dtype=np.uint32)


def bad_ndim_positions(n: int, seed: int, ndim: int = 4) -> np.ndarray:
    """``(n, ndim)`` coordinates — one column too many for a 3-D scene (#1446)."""
    rng = np.random.default_rng(seed)
    return (rng.random((n, ndim)) * 100.0).astype(np.float32)


def open_ranged_scene(
    tmp_path: Any, filename: str
) -> Tuple[LuxarZarrCompiler, Any, str]:
    """Like :func:`open_scene`, but every dimension declares ``range=(0, 10)``.

    Data from :func:`random_positions` spans ``[0, 100)``, so each dimension is
    out of its declared range and the per-dimension ``UserWarning`` in
    ``validate_data_dimensions`` fires — which is how the #1446 controls count
    those warnings.
    """
    path = str(tmp_path / filename)
    compiler = LuxarZarrCompiler(path)
    scene = compiler.create_scene(
        dimensions=Dimensions(
            [
                Dimension("X", range=(0.0, 10.0), display=True),
                Dimension("Y", range=(0.0, 10.0), display=True),
                Dimension("Z", range=(0.0, 10.0), display=True),
            ]
        )
    )
    return compiler, scene, path


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


def count_range_warnings(records: Any) -> int:
    """How many of ``records`` are the per-dimension out-of-range ``UserWarning``.

    Used by both halves of the #1446 suite to assert the hoisted count check did
    not multiply the warning half of ``validate_data_dimensions``.
    """
    return sum(1 for r in records if "outside declared range" in str(r.message))


# --------------------------------------------------------------------------
# Shared by the two halves of the #1471 labels-refusal suite (lod/ + partition/)
# --------------------------------------------------------------------------

#: Element count both halves build their labelled fixtures around.
N_LABELLED = 8
LABELS = [f"l{i}" for i in range(N_LABELLED)]
KEYS = [f"k{i}" for i in range(N_LABELLED)]
IMAGE_LABELS = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(N_LABELLED)]

#: ``(kwarg name, call kwargs, the writer attr that proves it reached disk)``.
#: Parametrizing over this is what keeps the two channels' verdicts identical —
#: ``image_labels`` was the one the original fix nearly missed, since it is not a
#: named kwarg anywhere above the leaf adders either.
LABEL_KWARGS = [
    ("labels", {"labels": LABELS}, "has_labels"),
    ("image_labels", {"image_labels": IMAGE_LABELS}, "has_image_labels"),
    ("keys", {"keys": KEYS}, "has_keys"),
]


# --------------------------------------------------------------------------
# Shared by the two halves of the #1489 colour-DTYPE suite (lod/ + partition/)
# --------------------------------------------------------------------------


def int64_rgb(n: int) -> np.ndarray:
    """``(n, 3)`` red as **int64** — a colour dtype no writer can store (#1489).

    Exactly what ``np.tile([255, 0, 0], (n, 1))`` gives you on Linux, which is
    why this is the realistic spelling of the bug rather than a contrived dtype:
    a COLOR array may be floating point, or integer uint8/uint16, and nothing
    else. The encoder always refused it; until the rule moved into
    ``validate_colors_for_writing`` it refused MID-write, one dataset after the
    positions.
    """
    return np.tile([255, 0, 0], (n, 1)).astype(np.int64)


def finalized_group_keys(compiler: Any, path: str) -> Set[str]:
    """Finalize and return the scene's top-level group names.

    The second half of every #1471 stranding assertion: a wrapper written and
    then abandoned mid-write is not merely present in the live store, it SURVIVES
    ``finalize()`` into the delivered scene.
    """
    import zarr

    compiler.finalize()
    return set(zarr.open_group(path, mode="r").group_keys())


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
