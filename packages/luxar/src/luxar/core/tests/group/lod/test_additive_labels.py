"""Hover-overlay injection for LABELLED additive-LOD ladders (issue #1422).

A ladder writes its labels as one CSR on the parent node, so the adders must
notify the scene exactly as the flat path does — otherwise a ladder-only scene
gets no ``overlays/__hover_text`` and hover picking stays off entirely.
"""

import numpy as np
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler


def _make_3d_dims() -> Dimensions:
    return Dimensions(
        [
            Dimension("X", display=True),
            Dimension("Y", display=True),
            Dimension("Z", display=True),
        ]
    )


def _random_positions(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n, 3)) * 100.0).astype(np.float32)


def _has_hover_overlay(path: str) -> bool:
    store = zarr.open_group(path, mode="r")
    if "overlays" not in store:
        return False
    return "__hover_text" in store["overlays"]


def test_points_ladder_only_scene_gets_hover_overlay(tmp_path):
    path = str(tmp_path / "points_ladder.luxar.zarr")
    n_points = 400
    positions = _random_positions(n_points, seed=101)

    compiler = LuxarZarrCompiler(path)
    scene = compiler.create_scene(dimensions=_make_3d_dims())
    scene.add_points(
        "ladder",
        positions,
        labels=[f"p{i}" for i in range(n_points)],
        additive_lod=True,
    )
    compiler.finalize()

    # The ladder must have actually fired (otherwise this tests the flat path).
    store = zarr.open_group(path, mode="r")
    assert int(store["ladder"].attrs["n_additive_sublods"]) > 1

    assert _has_hover_overlay(path)
    hover_attrs = dict(store["overlays"]["__hover_text"].attrs)
    assert hover_attrs["hover"] is True
    assert hover_attrs["text"] == "{hover_label}"


def test_lines_ladder_only_scene_gets_hover_overlay(tmp_path):
    path = str(tmp_path / "lines_ladder.luxar.zarr")
    n_vertices = 300
    vertices = _random_positions(n_vertices, seed=102)

    compiler = LuxarZarrCompiler(path)
    scene = compiler.create_scene(dimensions=_make_3d_dims())
    scene.add_lines(
        "ladder",
        vertices,
        widths=0.2,
        line_type="segments",
        labels=[f"v{i}" for i in range(n_vertices)],
        additive_lod=True,
    )
    compiler.finalize()

    store = zarr.open_group(path, mode="r")
    assert int(store["ladder"].attrs["n_additive_sublods"]) > 1

    assert _has_hover_overlay(path)


def test_unlabelled_ladder_gets_no_hover_overlay(tmp_path):
    """The notify must be conditional on labels being present."""
    path = str(tmp_path / "unlabelled.luxar.zarr")
    positions = _random_positions(400, seed=103)

    compiler = LuxarZarrCompiler(path)
    scene = compiler.create_scene(dimensions=_make_3d_dims())
    scene.add_points("ladder", positions, additive_lod=True)
    compiler.finalize()

    store = zarr.open_group(path, mode="r")
    assert int(store["ladder"].attrs["n_additive_sublods"]) > 1

    assert not _has_hover_overlay(path)
