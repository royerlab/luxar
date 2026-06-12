import numpy as np
import zarr

from luxar import Dimensions, LuxarZarrCompiler


def test_incremental_build(tmp_path) -> None:
    """Building the graph node‑by‑node flushes immediately to disk."""
    store = tmp_path / "inc.luxar.zarr"

    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # parent group with transform
        T = np.identity(4, np.float32).ravel().tolist()
        g = scene.add_group("Parent", transform=T)

        # first child
        p1 = np.random.rand(100, 3).astype(np.float32)
        scene.add_points("One", p1, parent=g)

        # second child
        p2 = p1 * 0.1
        scene.add_points("Two", p2, parent=g)

    # ---- reopen & inspect
    root = zarr.open_group(store, "r")

    # .groups() now yields (name, group) tuples
    assert {name for name, _ in root.groups()} == {"Parent"}
    assert {name for name, _ in root["Parent"].groups()} == {"One", "Two"}

    # walk() returns depth‑first pairs
    path_depths = [(d, n.name) for d, n in scene.walk()]
    assert path_depths == [
        (0, "Scene"),
        (1, "Parent"),
        (2, "One"),
        (2, "Two"),
    ]
