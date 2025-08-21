import numpy as np
import zarr

from luxar.demos import create_lorenz_attractor


def test_random_demo_roundtrip(tmp_path):
    """Lorenz attractor demo writes root attrs & point dataset correctly."""
    store = tmp_path / "demo.zarr"
    n = 7_777
    create_lorenz_attractor(store, n_points=n)

    root = zarr.open_group(store, mode="r")
    # ---- root attrs
    assert root.attrs["luxar_version"] == "0.2"
    assert root.attrs["units"] == "metre"

    # ---- hierarchy
    assert [name for name, _ in root.groups()] == ["LorenzAttractor"]
    grp = root["LorenzAttractor"]
    assert grp.attrs["type"] == "points"
    assert grp.attrs["n_points"] == n

    # ---- datasets / metadata
    pos = grp["positions"]
    assert pos.shape == (n, 3)
    assert pos.dtype == np.float32
    # Chunk size is min(array_size, default_chunk_size)
    assert pos.chunks[0] == min(n, 32_768)

    col = grp["colors"]
    assert col.shape == (n, 3)
    assert col.dtype == np.float32  # HDR colors are now float32
