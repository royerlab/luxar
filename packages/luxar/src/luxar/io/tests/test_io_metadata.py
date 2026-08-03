import zarr

from luxar.typing_utils.config import DEFAULT_COMP
from luxar.utils.demos import create_lorenz_attractor


def test_compressor_and_format(tmp_path) -> None:
    store = tmp_path / "meta.luxar.zarr"
    create_lorenz_attractor(store, n_points=100)

    root = zarr.open_group(store, "r")
    assert root._version == 2

    comp = root["LorenzAttractor"]["positions"].compressor
    from numcodecs import Blosc

    from luxar.encoding.compression import resolve_compressor

    assert isinstance(comp, Blosc)
    # DEFAULT_COMP is the width-aware sentinel: the stored compressor must
    # match the per-dtype policy for the array's actual stored dtype.
    expected = resolve_compressor(
        DEFAULT_COMP, root["LorenzAttractor"]["positions"].dtype
    )
    assert comp.cname == expected.cname
    assert comp.clevel == expected.clevel
    assert comp.shuffle == expected.shuffle


def test_every_geometry_writer_stamps_ndim(tmp_path) -> None:
    """``ndim`` lands in group attrs for every geometry type, not just some.

    The viewer's points chunk-index loader cross-checks ``chunk_bounds``
    dimensionality against this attr and skips the check when it is absent, so a
    writer that forgets the stamp silently disables a correctness gate. Asserted
    across all three types in one place so a new geometry cannot omit it.
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

    root = zarr.open_group(store, "r")
    stamped = {name: dict(root[name].attrs).get("ndim") for name in ("pts", "lns", "spl")}
    assert stamped == {"pts": 3, "lns": 3, "spl": 3}, stamped
