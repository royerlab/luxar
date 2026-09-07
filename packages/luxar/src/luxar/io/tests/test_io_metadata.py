import json

import zarr

from luxar._zarr_compat import ZARR_FORMAT
from luxar.io.reader import DEFAULT_COMP
from luxar.utils.scenes import create_lorenz_attractor


def test_compressor_and_format(tmp_path) -> None:
    store = tmp_path / "meta.luxar.zarr"
    create_lorenz_attractor(store, n_points=100)

    root = zarr.open_group(store, mode="r")
    # Public API, and cross-checked on disk. `root._version` was a zarr-2 private
    # attribute that zarr 3 does not have; `metadata.zarr_format` is the supported
    # spelling. The on-disk half is the stronger one — it asserts what actually
    # LANDED, which is what the viewer reads, rather than what the in-memory
    # handle believes.
    #
    # Pinned to `ZARR_FORMAT` rather than to a literal 2 or 3: the assertion
    # being made is "the writer emits the format it says it emits, and both
    # documents agree", which is what can actually regress. A literal would have
    # to be edited again on the next format bump, and — worse — an edit that
    # changed only the literal would still pass while the two halves disagreed.
    assert root.metadata.zarr_format == ZARR_FORMAT
    if ZARR_FORMAT == 2:
        assert json.loads((store / ".zgroup").read_text())["zarr_format"] == 2
        assert not (store / "zarr.json").exists()
    else:
        assert json.loads((store / "zarr.json").read_text())["zarr_format"] == 3
        assert not (store / ".zgroup").exists()

    from luxar.conftest import array_compressor
    from luxar.encoding.compression import resolve_compressor

    positions = root["LorenzAttractor"]["positions"]
    comp = array_compressor(positions)
    assert comp is not None, "positions must be compressed, not stored raw"
    # DEFAULT_COMP is the width-aware sentinel: the stored compressor must
    # match the per-dtype policy for the array's actual stored dtype. The
    # policy is stated in numcodecs terms, and `array_compressor` normalises
    # format 3's codec objects back to them, so the SAME expectation holds for
    # both formats — the measured shuffle policy is format-independent.
    expected = resolve_compressor(DEFAULT_COMP, positions.dtype)
    assert comp.cname == expected.cname
    assert comp.clevel == expected.clevel
    assert comp.shuffle == expected.shuffle


def test_every_geometry_writer_stamps_ndim(tmp_path) -> None:
    """``ndim`` lands in group attrs for every geometry type, not just some.

    The viewer's points chunk-index loader cross-checks ``chunk_bounds``
    dimensionality against this attr and skips the check when it is absent, so a
    writer that forgets the stamp silently disables a correctness gate. Asserted
    across all four types in one place so a new geometry cannot omit it.
    """
    import numpy as np

    from luxar.core.dimensions import Dimensions
    from luxar.io import LuxarZarrCompiler

    rng = np.random.RandomState(0)
    n = 32
    store = tmp_path / "ndim.luxar.zarr"

    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points("pts", rng.rand(n, 3).astype(np.float32))
        scene.add_lines(
            "lns",
            rng.rand(n, 3).astype(np.float32),
            widths=np.full(n, 0.1, dtype=np.float32),
        )
        scene.add_gsplats(
            "spl",
            centers=rng.rand(n, 3).astype(np.float32),
            amplitudes=np.ones(n, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1)
            ),
        )
        scene.add_mesh(
            "msh",
            vertices=rng.rand(n, 3).astype(np.float32),
            faces=np.arange(n - (n % 3), dtype=np.uint32).reshape(-1, 3),
        )

    root = zarr.open_group(store, mode="r")
    stamped = {
        name: dict(root[name].attrs).get("ndim")
        for name in ("pts", "lns", "spl", "msh")
    }
    assert stamped == {"pts": 3, "lns": 3, "spl": 3, "msh": 3}, stamped
