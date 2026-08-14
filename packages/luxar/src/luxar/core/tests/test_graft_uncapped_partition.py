"""Grafting a partition that declares NO per-part cap (``max_elements = 0``).

``max_elements`` is a per-part CAP, so only a capped splitter sets it: uniform
tiling and the BSP rules behind ``--parts`` / ``--max-elements``. A CONTENT-tiled
fit (``gsplat fit --tiling content``, and the content plans ``batch-fit`` builds)
balances its boxes by feature density instead, so it leaves the field at the
:class:`~luxar.gsplats.tree.GSplatPartition` default of ``0``.

``Node.add_partition_group`` requires ``max_elements >= 1``, so the graft path
used to raise ``ValueError: max_elements must be an int >= 1, got 0`` — i.e.
every content-tiled fit was ungraftable and could not reach a scene at all.
The graft now derives the honest value (the largest part) for an uncapped
source; a capped one must still pass its own value through untouched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import AdditiveSubLOD
from luxar.gsplats.io.save_gsplats import write_gsplats_tree
from luxar.gsplats.tree import GSplatLeaf, GSplatPartition


def _leaf(n: int, offset: float) -> GSplatLeaf:
    """A compact cluster of ``n`` splats, translated clear of its siblings."""
    rng = np.random.default_rng(int(offset))
    centers = (offset + rng.uniform(0, 5, size=(n, 3))).astype(np.float32)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    return GSplatLeaf(
        additive_sublods=[
            AdditiveSubLOD(
                centers=centers,
                amplitudes=np.ones(n, dtype=np.float32),
                cholesky_factors=chol,
            )
        ]
    )


def _uncapped_partition_file(dirpath: Path, sizes: tuple[int, ...]) -> Path:
    """Write a ``kind=partition`` whose ``max_elements`` is left at 0.

    This is the on-disk shape a content-tiled fit produces: parts of unequal,
    density-driven sizes and no declared cap.
    """
    node = GSplatPartition(
        children=[_leaf(n, offset=100.0 * i) for i, n in enumerate(sizes)]
    )
    assert node.max_elements == 0, "fixture must exercise the uncapped default"
    path = dirpath / "content.gsplats.zarr"
    write_gsplats_tree(
        path, node, ordering="none", encoding_mode=EncodingMode.PRECISION
    )
    return path


def _graft(path: Path, out: Path) -> dict:
    """Graft *path* into a scene; return the partition wrapper's attrs."""
    dims = Dimensions(
        [
            Dimension("Z", display=True, range=(0.0, 400.0)),
            Dimension("Y", display=True, range=(0.0, 400.0)),
            Dimension("X", display=True, range=(0.0, 400.0)),
        ]
    )
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        scene.add_gsplats_from_file(name="parts", path=str(path))
    return dict(zarr.open_group(str(out), mode="r")["parts"].attrs)


def test_uncapped_partition_grafts_with_derived_cap(tmp_path: Path) -> None:
    """An uncapped (content-tiled) partition grafts, capped by its largest part."""
    sizes = (12, 47, 30)
    src = _uncapped_partition_file(tmp_path / "src", sizes)

    attrs = _graft(src, tmp_path / "scene.luxar.zarr")

    assert attrs["kind"] == "partition"
    # The derived cap is the largest part — not the total, not a constant, and
    # never below 1 (which is what the validator rejects).
    assert attrs["max_elements"] == max(sizes)


def test_declared_cap_is_not_overwritten(tmp_path: Path) -> None:
    """Sensitivity control: a partition that DOES declare a cap keeps it.

    Without this, a graft that simply always derived the cap would pass the test
    above while silently discarding a real ``--max-elements`` / uniform-tiling
    value.
    """
    src_dir = tmp_path / "capped"
    src_dir.mkdir()
    path = src_dir / "capped.gsplats.zarr"
    declared = 999  # deliberately unequal to any part size and to the total
    node = GSplatPartition(
        children=[_leaf(n, offset=100.0 * i) for i, n in enumerate((12, 47))],
        max_elements=declared,
    )
    write_gsplats_tree(
        path, node, ordering="none", encoding_mode=EncodingMode.PRECISION
    )

    attrs = _graft(path, tmp_path / "scene2.luxar.zarr")

    assert attrs["max_elements"] == declared


@pytest.mark.parametrize("sizes", [(1,), (5, 5)])
def test_derived_cap_is_always_at_least_one(
    tmp_path: Path, sizes: tuple[int, ...]
) -> None:
    """Edge sizes still satisfy the validator's ``>= 1`` invariant."""
    src = _uncapped_partition_file(tmp_path / f"src{len(sizes)}", sizes)
    attrs = _graft(src, tmp_path / f"scene{len(sizes)}.luxar.zarr")
    assert attrs["max_elements"] >= 1
