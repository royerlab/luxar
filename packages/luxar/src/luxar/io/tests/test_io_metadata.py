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
