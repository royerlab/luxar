"""Tests for substitutive-LOD on Lines (coarse levels = synthesised gsplats).

Mirror of ``test_substitutive_points.py``: each segment is lifted to isotropic
"bead" gaussians, reduced by the gsplat substitutive pipeline, and assembled as
a ``kind=lod`` Group whose finest child is the original Lines node.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.core.group.lod.lines import resolve_substitutive_axis_lines
from luxar.io.compiler import LuxarZarrCompiler


def _segments(n_seg, seed=0):
    rng = np.random.default_rng(seed)
    p0 = rng.uniform(0, 60, (n_seg, 3))
    d = rng.normal(0, 5, (n_seg, 3))
    verts = np.empty((n_seg * 2, 3), np.float32)
    verts[0::2] = p0
    verts[1::2] = p0 + d
    return verts


def _build(tmp_path, *, n_seg=1500, line_type="segments", levels=2, widths=0.8, **kw):
    out = tmp_path / "t.luxar.zarr"
    verts = _segments(n_seg)
    colors = np.random.default_rng(1).uniform(0, 1, (verts.shape[0], 3)).astype(np.float32)
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_lines("curves", verts, widths, colors=colors, line_type=line_type,
                        substitutive_lod=dict(levels=levels, device="cpu", seed=0, **kw))
    return zarr.open(str(out), mode="r")["curves"], verts.shape[0]


class TestResolveSubstitutiveAxisLines:
    def test_shares_points_vocabulary(self) -> None:
        assert resolve_substitutive_axis_lines(None) is None
        assert resolve_substitutive_axis_lines(False) is None
        r = resolve_substitutive_axis_lines(True)
        assert r["compression_factor"] == 4 and r["levels"] == 3 and r["method"] == "auto"
        assert resolve_substitutive_axis_lines(dict(K=8, n_lods=2))["compression_factor"] == 8

    def test_unknown_key_mentions_lines(self) -> None:
        with pytest.raises(ValueError, match="Lines"):
            resolve_substitutive_axis_lines(dict(bogus=1))

    def test_non_ascending_min_pixel_sizes_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            resolve_substitutive_axis_lines(dict(min_pixel_sizes=[0.0, 50.0, 10.0]))


class TestAddLinesSubstitutiveLod:
    def test_group_is_kind_lod_lines(self, tmp_path) -> None:
        grp, _ = _build(tmp_path)
        assert grp.attrs["kind"] == "lod"
        assert grp.attrs["display_type"] == "lines"
        assert grp.attrs["selector"] == "pixel_size"
        assert grp.attrs["default_level"] == 0

    def test_finest_is_lines_coarse_are_gsplats(self, tmp_path) -> None:
        grp, _ = _build(tmp_path, levels=2)
        children = sorted(k for k in grp.keys() if k.startswith("child_"))
        assert children == ["child_0", "child_1", "child_2"]  # 2 gsplat + 1 lines
        types = [grp[c].attrs["type"] for c in children]
        assert types[:2] == ["gsplats", "gsplats"]
        assert types[2] == "lines"

    def test_min_pixel_size_monotone_coarsest_zero(self, tmp_path) -> None:
        grp, _ = _build(tmp_path, levels=2)
        mps = [float(grp[f"child_{i}"].attrs["min_pixel_size"]) for i in range(3)]
        assert mps[0] == 0.0
        assert all(mps[i] < mps[i + 1] for i in range(len(mps) - 1))

    def test_finest_carries_all_vertices(self, tmp_path) -> None:
        grp, n_verts = _build(tmp_path, levels=2)
        finest = grp["child_2"]
        assert finest.attrs["type"] == "lines"
        assert finest.attrs["n_vertices"] == n_verts

    @pytest.mark.parametrize("line_type", ["segments", "polyline", "loop"])
    def test_line_types_build_a_group(self, tmp_path, line_type) -> None:
        grp, _ = _build(tmp_path, n_seg=400, line_type=line_type, levels=2)
        assert grp.attrs["kind"] == "lod"
        assert grp.attrs["display_type"] == "lines"

    def test_position_bounds_backfilled(self, tmp_path) -> None:
        grp, _ = _build(tmp_path)
        assert "position_bounds" in grp.attrs


class TestSubstitutiveLinesGuards:
    def test_mutually_exclusive_with_additive(self, tmp_path) -> None:
        out = tmp_path / "t.luxar.zarr"
        verts = _segments(50)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="mutually exclusive"):
                scene.add_lines("c", verts, 1.0, line_type="segments",
                                additive_lod=True, substitutive_lod=True)

    def test_mutually_exclusive_with_partition(self, tmp_path) -> None:
        out = tmp_path / "t.luxar.zarr"
        verts = _segments(50)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="partition.*substitutive_lod"):
                scene.add_lines("c", verts, 1.0, line_type="segments",
                                partition=True, substitutive_lod=True)

    def test_scalars_plus_colormap_bakes_colors(self, tmp_path) -> None:
        out = tmp_path / "t.luxar.zarr"
        verts = _segments(1500)
        scalars = np.random.default_rng(0).uniform(0, 1, verts.shape[0]).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines("c", verts, 0.8, scalars=scalars, colormap="viridis",
                            line_type="segments",
                            substitutive_lod=dict(levels=2, device="cpu", seed=0))
        grp = zarr.open(str(out), mode="r")["c"]
        assert grp.attrs["kind"] == "lod" and grp.attrs["display_type"] == "lines"
        assert grp["child_0"].attrs["type"] == "gsplats"
        assert bool(grp["child_0"].attrs.get("has_colors")) is True
        finest = grp["child_2"]
        assert bool(finest.attrs.get("has_scalars")) is True
        assert finest.attrs.get("colormap") == "viridis"

    def test_scalars_without_colormap_raises(self, tmp_path) -> None:
        out = tmp_path / "t.luxar.zarr"
        verts = _segments(50)
        scalars = np.random.default_rng(0).uniform(0, 1, verts.shape[0]).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="colormap"):
                scene.add_lines("c", verts, 1.0, scalars=scalars, line_type="segments",
                                substitutive_lod=True)
