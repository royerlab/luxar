import zarr

from luxar import Scene
from luxar._io import DEFAULT_COMP


def test_compressor_and_format(tmp_path):
    store = tmp_path / "meta.zarr"
    Scene.random_demo(store, n=100)

    root = zarr.open_group(store, "r")
    assert root._version == 2

    comp = root["LorenzAttractor"]["positions"].compressor
    from numcodecs import Blosc

    assert isinstance(comp, Blosc)
    assert comp.cname == DEFAULT_COMP.cname
