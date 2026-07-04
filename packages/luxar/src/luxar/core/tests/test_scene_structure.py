import numpy as np
import zarr

from luxar.utils.demos import create_lorenz_attractor


def test_random_demo_roundtrip(tmp_path) -> None:
    """Lorenz attractor demo writes root attrs & point dataset correctly."""
    store = tmp_path / "demo.luxar.zarr"
    n = 7_777
    create_lorenz_attractor(store, n_points=n)

    root = zarr.open_group(store, mode="r")
    # ---- root attrs
    assert root.attrs["luxar_version"] == "0.1"
    # Units are now specified per-dimension via scene_dimensions, not globally

    # ---- hierarchy
    assert [name for name, _ in root.groups()] == ["LorenzAttractor"]
    grp = root["LorenzAttractor"]
    assert grp.attrs["type"] == "points"
    assert grp.attrs["n_points"] == n

    # ---- datasets / metadata
    pos = grp["positions"]
    assert pos.shape == (n, 3)
    # AUTO stores positions as uint16 per-axis fixed-point (linear_perchannel_u16),
    # decoded to float32 on load; PRECISION / large-extent stays float32.
    assert pos.dtype in (np.uint16, np.float32)
    # Chunk size is determined by spatial ordering system or defaults to min(n, 32_768)
    # With spatial ordering enabled, chunk size may be smaller for better query performance
    assert pos.chunks[0] > 0
    assert pos.chunks[0] <= 32_768  # Should not exceed max chunk size

    col = grp["colors"]
    assert col.shape == (n, 3)
    # AUTO mode will convert SDR colors to uint8 for efficiency
    assert col.dtype in [
        np.float32,
        np.uint8,
    ]  # Can be either depending on dtype config
