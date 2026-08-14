"""Tests for the GEFF (cell-lineage graph) reader.

Fixtures write GEFF stores by hand, on top of the zarr-v3 writers in the io test
suite, because the installed zarr is pinned below 3 and cannot emit a v3 store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pytest

from luxar.gsplats.interop.geff import TrackingGraph, read_geff
from luxar.io.tests.test_zarr_v3 import write_v3_array, write_v3_group


def write_geff(
    path: Path,
    *,
    node_ids: Sequence[int],
    t: Sequence[int],
    z: Sequence[float],
    y: Sequence[float],
    x: Sequence[float],
    edges: Sequence[Sequence[int]],
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    unit: Optional[str] = "micrometer",
    geff_attr: bool = True,
    skip_axis: Optional[str] = None,
) -> Path:
    """Write a minimal but spec-shaped GEFF store."""
    axes = [{"name": "t", "type": "time", "unit": None, "scale": 1.0}]
    for name, s in zip(("z", "y", "x"), scale):
        axes.append({"name": name, "type": "space", "unit": unit, "scale": s})
    attributes = (
        {"geff": {"geff_version": "1.1", "directed": True, "axes": axes}}
        if geff_attr
        else {"something_else": True}
    )
    write_v3_group(path, attributes=attributes)

    nodes = write_v3_group(path / "nodes")
    write_v3_array(
        nodes / "ids", np.asarray(node_ids, dtype=np.uint64), (max(1, len(node_ids)),)
    )
    props = write_v3_group(nodes / "props")
    for name, values in (("t", t), ("z", z), ("y", y), ("x", x)):
        if name == skip_axis:
            continue
        axis_group = write_v3_group(props / name)
        arr = np.asarray(values, dtype=np.int64 if name == "t" else np.float64)
        write_v3_array(axis_group / "values", arr, (max(1, len(arr)),))

    edge_group = write_v3_group(path / "edges")
    edge_arr = np.asarray(edges, dtype=np.uint64).reshape(-1, 2)
    write_v3_array(edge_group / "ids", edge_arr, (max(1, edge_arr.shape[0]), 2))
    write_v3_group(edge_group / "props")
    return path


@pytest.fixture
def simple_lineage(tmp_path: Path) -> Path:
    """One founder that divides at t=2 into two daughters, plus a lone cell.

    10 -- 11 -- 12 -- 13   (daughter A)
             \\-- 14 -- 15   (daughter B)
    90 -- 91               (an unrelated single-cell track)
    """
    return write_geff(
        tmp_path / "s.geff",
        node_ids=[10, 11, 12, 13, 14, 15, 90, 91],
        t=[0, 1, 2, 3, 3, 4, 0, 1],
        z=[1, 1, 1, 1, 2, 2, 9, 9],
        y=[0, 1, 2, 3, 2, 2, 20, 21],
        x=[0, 0, 0, 0, 1, 2, 30, 30],
        edges=[(10, 11), (11, 12), (12, 13), (12, 14), (14, 15), (90, 91)],
        scale=(2.0, 0.5, 0.5),
    )


# =============================================================================
# Reading
# =============================================================================


def test_read_geff_basic_fields(simple_lineage: Path) -> None:
    g = read_geff(simple_lineage)
    assert isinstance(g, TrackingGraph)
    assert g.n_nodes == 8
    assert g.n_edges == 6
    assert g.n_timepoints == 5
    np.testing.assert_array_equal(g.node_ids, [10, 11, 12, 13, 14, 15, 90, 91])
    np.testing.assert_array_equal(g.t, [0, 1, 2, 3, 3, 4, 0, 1])
    assert g.positions.shape == (8, 3)
    # positions are (z, y, x) in VOXEL units, straight from the store
    np.testing.assert_array_equal(g.positions[0], [1, 0, 0])
    np.testing.assert_array_equal(g.positions[6], [9, 20, 30])
    assert g.scale == (2.0, 0.5, 0.5)
    assert g.units == ("micrometer", "micrometer", "micrometer")


def test_positions_um_applies_the_axis_scale(simple_lineage: Path) -> None:
    """The voxel->um conversion is what puts markers on the nuclei they annotate."""
    g = read_geff(simple_lineage)
    um = g.positions_um()
    np.testing.assert_allclose(um[6], [9 * 2.0, 20 * 0.5, 30 * 0.5])
    np.testing.assert_allclose(um, g.positions * np.array([2.0, 0.5, 0.5]))


def test_repr_summarises_the_graph(simple_lineage: Path) -> None:
    text = repr(read_geff(simple_lineage))
    assert "nodes=8" in text
    assert "edges=6" in text
    assert "t=0..4" in text
    assert "lineages=2" in text
    assert "divisions=1" in text


def test_index_of_maps_sparse_ids(simple_lineage: Path) -> None:
    """Challenge ids are ``global_t * 1e9 + cell_id``, so never assume 0..N-1."""
    g = read_geff(simple_lineage)
    assert g.index_of() == {10: 0, 11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 90: 6, 91: 7}


def test_edge_indices_are_positional(simple_lineage: Path) -> None:
    g = read_geff(simple_lineage)
    np.testing.assert_array_equal(
        g.edge_indices(), [[0, 1], [1, 2], [2, 3], [2, 4], [4, 5], [6, 7]]
    )


def test_edges_naming_absent_nodes_are_dropped(tmp_path: Path) -> None:
    """A crop of a bigger movie can reference cells outside its own bounds."""
    path = write_geff(
        tmp_path / "s.geff",
        node_ids=[1, 2],
        t=[0, 1],
        z=[0, 0],
        y=[0, 1],
        x=[0, 0],
        edges=[(1, 2), (2, 777), (555, 1)],
    )
    g = read_geff(path)
    assert g.n_edges == 3  # stored edges are kept verbatim
    np.testing.assert_array_equal(g.edge_indices(), [[0, 1]])  # only resolvable ones


# =============================================================================
# Graph structure
# =============================================================================


def test_divisions_finds_the_dividing_cell(simple_lineage: Path) -> None:
    g = read_geff(simple_lineage)
    np.testing.assert_array_equal(g.divisions(), [2])  # node id 12, index 2


def test_lineage_ids_group_whole_trees(simple_lineage: Path) -> None:
    """A founder and every descendant share one id, so one lineage gets one colour."""
    g = read_geff(simple_lineage)
    lineage = g.lineage_ids()
    assert lineage.shape == (8,)
    # nodes 10..15 are one lineage (including both daughters); 90/91 another
    assert len(set(lineage[:6].tolist())) == 1
    assert len(set(lineage[6:].tolist())) == 1
    assert lineage[0] != lineage[6]
    # ids are dense and ordered by first appearance, so they index a palette
    assert sorted(set(lineage.tolist())) == [0, 1]
    assert lineage[0] == 0


def test_every_edge_stays_inside_one_lineage(simple_lineage: Path) -> None:
    g = read_geff(simple_lineage)
    lineage = g.lineage_ids()
    for src, dst in g.edge_indices():
        assert lineage[src] == lineage[dst]


def test_isolated_nodes_get_their_own_lineage(tmp_path: Path) -> None:
    path = write_geff(
        tmp_path / "s.geff",
        node_ids=[1, 2, 3],
        t=[0, 1, 0],
        z=[0, 0, 5],
        y=[0, 1, 5],
        x=[0, 0, 5],
        edges=[(1, 2)],
    )
    lineage = read_geff(path).lineage_ids()
    assert sorted(set(lineage.tolist())) == [0, 1]
    assert lineage[0] == lineage[1] != lineage[2]


def test_merge_node_is_not_reported_as_a_division(tmp_path: Path) -> None:
    """Two parents into one child is a merge (in-degree 2), not a division."""
    path = write_geff(
        tmp_path / "s.geff",
        node_ids=[1, 2, 3],
        t=[0, 0, 1],
        z=[0, 0, 0],
        y=[0, 1, 2],
        x=[0, 0, 0],
        edges=[(1, 3), (2, 3)],
    )
    g = read_geff(path)
    assert g.divisions().size == 0
    assert len(set(g.lineage_ids().tolist())) == 1  # still one connected component


def test_graph_with_no_edges(tmp_path: Path) -> None:
    path = write_geff(
        tmp_path / "s.geff",
        node_ids=[1, 2],
        t=[0, 0],
        z=[0, 1],
        y=[0, 1],
        x=[0, 1],
        edges=[],
    )
    g = read_geff(path)
    assert g.n_nodes == 2
    assert g.edge_indices().shape == (0, 2)
    assert g.divisions().size == 0
    assert sorted(set(g.lineage_ids().tolist())) == [0, 1]


def test_long_chain_has_no_divisions(tmp_path: Path) -> None:
    n = 40
    path = write_geff(
        tmp_path / "s.geff",
        node_ids=list(range(n)),
        t=list(range(n)),
        z=[0.0] * n,
        y=list(range(n)),
        x=[0.0] * n,
        edges=[(i, i + 1) for i in range(n - 1)],
    )
    g = read_geff(path)
    assert g.divisions().size == 0
    assert len(set(g.lineage_ids().tolist())) == 1
    assert g.n_timepoints == n


# =============================================================================
# Error surface
# =============================================================================


def test_missing_geff_attribute_is_rejected(tmp_path: Path) -> None:
    path = write_geff(
        tmp_path / "s.geff",
        node_ids=[1],
        t=[0],
        z=[0],
        y=[0],
        x=[0],
        edges=[],
        geff_attr=False,
    )
    with pytest.raises(ValueError, match="no 'geff' attribute"):
        read_geff(path)


def test_missing_coordinate_property_is_rejected(tmp_path: Path) -> None:
    path = write_geff(
        tmp_path / "s.geff",
        node_ids=[1, 2],
        t=[0, 1],
        z=[0, 0],
        y=[0, 1],
        x=[0, 0],
        edges=[(1, 2)],
        skip_axis="z",
    )
    with pytest.raises(ValueError, match=r"missing node properties \['z'\]"):
        read_geff(path)


def test_pointing_at_an_array_is_rejected(tmp_path: Path) -> None:
    array = write_v3_array(tmp_path / "a", np.zeros((2,), np.uint8), (2,))
    with pytest.raises(ValueError, match="not a GEFF group"):
        read_geff(array)


def test_axes_without_scale_default_to_one(tmp_path: Path) -> None:
    """A store that omits per-axis scale must read as voxel units, not crash."""
    path = write_geff(
        tmp_path / "s.geff", node_ids=[1], t=[0], z=[3], y=[4], x=[5], edges=[]
    )
    meta = json.loads((path / "zarr.json").read_text())
    for axis in meta["attributes"]["geff"]["axes"]:
        axis.pop("scale", None)
        axis["unit"] = None
    (path / "zarr.json").write_text(json.dumps(meta))
    g = read_geff(path)
    assert g.scale == (1.0, 1.0, 1.0)
    assert g.units == (None, None, None)
    np.testing.assert_array_equal(g.positions_um()[0], [3, 4, 5])
