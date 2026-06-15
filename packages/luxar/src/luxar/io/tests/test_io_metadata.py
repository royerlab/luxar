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

    assert isinstance(comp, Blosc)
    assert comp.cname == DEFAULT_COMP.cname
