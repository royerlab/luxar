"""Serialization tests for the Cosmicflows galaxy destination."""

from __future__ import annotations

import numpy as np
import zarr

from luxar.demos.demo_cosmicflows_laniakea import (
    PRESETS,
    BasinLineData,
    GalaxyData,
    write_laniakea_scene,
)


def _decode_strings(node, channel: str) -> list[str]:
    offsets = np.asarray(node[f"{channel}_offsets"][:]).astype(int)
    data = bytes(np.asarray(node[f"{channel}_bytes"][:]).tobytes())
    return [
        data[offsets[i] : offsets[i + 1]].decode("utf-8")
        for i in range(len(offsets) - 1)
    ]


def test_galaxy_keys_stay_aligned_with_basin_labels(tmp_path) -> None:
    galaxies = GalaxyData(
        positions=np.array([[0, 0, 0], [10, 20, 30]], dtype=np.float32),
        basin_ids=np.array([1, 7], dtype=np.int16),
        radii=np.ones(2, dtype=np.float32),
        colors=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        pgc=np.array([4, 12345], dtype=np.int64),
    )
    output = tmp_path / "cosmicflows.luxar.zarr"

    write_laniakea_scene(output, galaxies, [], "preview", PRESETS["preview"])

    root = zarr.open_group(str(output), mode="r")
    galaxy_nodes = [
        root[name]
        for name in root.keys()
        if hasattr(root[name], "attrs")
        and dict(root[name].attrs).get("link")
        == "https://ned.ipac.caltech.edu/byname?objname={hover_key}"
    ]
    assert len(galaxy_nodes) == 1
    node = galaxy_nodes[0]
    attrs = dict(node.attrs)
    assert attrs["has_keys"] is True
    assert _decode_strings(node, "key") == ["PGC4", "PGC12345"]
    assert _decode_strings(node, "label") == ["Basin 1", "Basin 7"]


def _integrator_valid_mask(
    n_lines: int, n_steps: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """The shape the integrator actually produces: a contiguous valid PREFIX.

    ``valid[:, 0]`` is set for every seed and a streamline that fails a step is
    retired permanently (``active[bad_idx] = False``), so the only gaps are
    trailing. Returns ``(valid, valid_counts)``.
    """
    rng = np.random.default_rng(seed)
    lengths = rng.integers(2, n_steps + 1, size=n_lines)
    valid = np.arange(n_steps)[None, :] < lengths[:, None]
    return valid, valid.sum(axis=1)


def test_basin_indices_can_carry_a_streaming_ladder() -> None:
    """The precondition the basin nodes' ``additive_lod=`` depends on.

    ``add_lines`` RAISES for indexed lines whose components are not ascending
    simple paths, because the multi-LOD writer rebuilds edges by chaining each
    component in ascending vertex order. If the integrator or the index
    construction ever changes shape, this fails here with a readable reason
    instead of breaking the scene build.
    """
    from luxar.core.group.lod.lines import indexed_components_are_chains
    from luxar.demos.demo_cosmicflows_laniakea import build_basin_line_data

    n_lines, n_steps = 400, 60
    valid, valid_counts = _integrator_valid_mask(n_lines, n_steps)
    rng = np.random.default_rng(1)
    positions = rng.normal(size=(n_lines, n_steps, 3)).astype(np.float32) * 100.0
    seed_basins = rng.integers(1, 5, size=n_lines).astype(np.int16)

    basins = build_basin_line_data(positions, valid, seed_basins, valid_counts)
    assert basins, "no basins were built"
    for basin in basins:
        assert indexed_components_are_chains(
            int(basin.vertices.shape[0]),
            np.asarray(basin.segments, dtype=np.intp).reshape(-1, 2),
        ), f"basin {basin.basin_id} would have its edges fabricated by the ladder"


def test_an_interior_gap_would_still_be_safe() -> None:
    """The caveat that looks like a hazard and is not.

    An interior invalid timestep cannot occur (each row is a contiguous prefix),
    but if one did, ``segment_mask = (start >= 0) & (end >= 0)`` drops the
    crossing segment — so the two index-contiguous runs become two distinct
    CONNECTED components and the writer chains each separately. No edge is
    invented across the gap. Pinned because assuming otherwise would argue for
    refusing a ladder these nodes can safely carry.
    """
    from luxar.core.group.lod.lines import indexed_components_are_chains
    from luxar.demos.demo_cosmicflows_laniakea import build_basin_line_data

    n_lines, n_steps = 50, 21
    valid = np.ones((n_lines, n_steps), dtype=bool)
    valid[:, 10] = False  # an interior hole in every row
    rng = np.random.default_rng(2)
    positions = rng.normal(size=(n_lines, n_steps, 3)).astype(np.float32) * 100.0
    seed_basins = np.ones(n_lines, dtype=np.int16)

    basins = build_basin_line_data(
        positions, valid, seed_basins, valid.sum(axis=1)
    )
    assert basins
    for basin in basins:
        assert indexed_components_are_chains(
            int(basin.vertices.shape[0]),
            np.asarray(basin.segments, dtype=np.intp).reshape(-1, 2),
        )


def test_basin_streamlines_carry_a_streaming_ladder(tmp_path) -> None:
    """End-to-end: the written scene has additive rungs and no substitutive levels.

    Test shape adapted from the substitutive attempt in #2297, which asserted the
    on-disk tree rather than only a precondition — the right thing to check, since
    an `additive_lod=` that silently collapses leaves no other trace.

    The fixture must exceed the ladder's ~39,062-vertex first rung, or the ladder
    legitimately collapses to a single level and the assertion would be vacuous.
    """

    n_lines, n_steps = 2_000, 50
    n_vertices = n_lines * n_steps
    idx = np.arange(n_vertices, dtype=np.uint32).reshape(n_lines, n_steps)
    segments = np.stack([idx[:, :-1], idx[:, 1:]], axis=-1).reshape(-1, 2)
    rng = np.random.default_rng(3)
    vertices = (rng.normal(size=(n_vertices, 3)) * 100.0).astype(np.float32)

    galaxies = GalaxyData(
        positions=np.array([[0, 0, 0]], dtype=np.float32),
        basin_ids=np.array([1], dtype=np.int16),
        radii=np.ones(1, dtype=np.float32),
        colors=np.ones((1, 3), dtype=np.float32),
        pgc=np.array([4], dtype=np.int64),
    )
    basin = BasinLineData(
        basin_id=1,
        vertices=vertices,
        segments=segments,
        streamline_count=n_lines,
    )
    output = tmp_path / "cosmicflows_lod.luxar.zarr"

    write_laniakea_scene(output, galaxies, [basin], "preview", PRESETS["preview"])

    root = zarr.open_group(str(output), mode="r")
    node = root["Basin 1 streamlines"]
    attrs = dict(node.attrs)

    # A prefix ladder, NOT a substitutive group: no `kind`, no child_N levels,
    # and crucially no synthesised gsplats where the ribbons should be.
    assert attrs.get("kind") != "lod"
    children = sorted(node.group_keys())
    assert children, "no ladder rungs were written"
    assert all(name.startswith("additive_") for name in children), children
    assert int(attrs.get("n_additive_sublods", 1)) > 1

    for name in children:
        assert dict(node[name].attrs).get("type") != "gsplats", (
            f"{name} holds synthesised gsplats; a prefix ladder must keep the "
            "original Lines geometry"
        )
