"""Tests for the attrs-only tree reader (audit A14-02).

Two properties matter, and they pull against each other:

1. it must decode NOTHING — that is the entire reason it exists; and
2. it must agree with the full reader — a cheap reader that quietly disagrees
   with the expensive one is worse than the expensive one, because `gsplat info`
   would then print confident wrong numbers.

The agreement test earned its place immediately: it caught this reader taking a
leaf's splat count from the finest rung of an additive ladder (1,250) instead of
summing the ladder (5,000), which no amount of re-reading the diff had shown.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from luxar._zarr_compat import open_group
from luxar.encoding import ArrayDecoder
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.load_gsplats import load_gsplat_node
from luxar.gsplats.io.tree_summary import read_gsplat_tree_summary
from luxar.gsplats.tree import iter_leaves, node_ndim, total_splats


def _flat(n: int, ndim: int = 3, *, colors: bool = False) -> GSplatData:
    rng = np.random.default_rng(n * 10 + ndim)
    tri = ndim * (ndim + 1) // 2
    chol = np.zeros((n, tri), dtype=np.float32)
    # Packed lower-triangular diagonal indices: 0, 2, 5, 9, ...
    chol[:, [i * (i + 3) // 2 for i in range(ndim)]] = 1.0
    kwargs = {}
    if colors:
        kwargs["colors"] = rng.random((n, 3)).astype(np.float32)
    return GSplatData(
        centers=(rng.random((n, ndim)) * 50).astype(np.float32),
        amplitudes=rng.random(n).astype(np.float32),
        cholesky_factors=chol,
        **kwargs,
    )


@pytest.fixture
def flat_store(tmp_path):
    path = tmp_path / "flat.gsplats.zarr"
    _flat(400).save(path)
    return path


@pytest.fixture
def partition_store(tmp_path):
    """A real kind=partition tree, built through the public partition API."""
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree
    from luxar.gsplats.lod import RecipeParams, build_recipe

    out = tmp_path / "part.gsplats.zarr"
    node = build_recipe(_flat(400), "tiles", RecipeParams(max_elements=120))
    write_gsplats_tree(out, node)
    return out


class TestAgreesWithTheFullReader:
    """The cheap reader and the expensive one must report the same shape."""

    @pytest.mark.parametrize("ndim", [2, 3, 4])
    def test_flat_leaf(self, tmp_path, ndim):
        path = tmp_path / f"leaf{ndim}.gsplats.zarr"
        _flat(250, ndim).save(path)
        node, _ = load_gsplat_node(path)
        summary = read_gsplat_tree_summary(open_group(str(path), mode="r"))

        assert summary.kind == "leaf"
        assert summary.ndim == node_ndim(node)
        assert summary.n_splats == total_splats(node)
        assert summary.leaf_counts == tuple(leaf.n_splats for leaf in iter_leaves(node))

    def test_leaf_with_colors(self, tmp_path):
        path = tmp_path / "colors.gsplats.zarr"
        _flat(300, colors=True).save(path)
        node, _ = load_gsplat_node(path)
        summary = read_gsplat_tree_summary(open_group(str(path), mode="r"))
        assert summary.n_splats == total_splats(node)

    def test_partition(self, partition_store):
        node, _ = load_gsplat_node(partition_store)
        summary = read_gsplat_tree_summary(open_group(str(partition_store), mode="r"))

        assert summary.kind == "partition"
        assert len(summary.children) == len(node.children)
        assert summary.ndim == node_ndim(node)
        assert summary.n_splats == total_splats(node)
        assert summary.leaf_counts == tuple(leaf.n_splats for leaf in iter_leaves(node))

    def test_partition_with_deduplicated_centers(self, tmp_path):
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import GSplatPartition

        first = _flat(200)
        second = _flat(200)
        second.centers = first.centers.copy()
        out = tmp_path / "deduplicated.gsplats.zarr"
        write_gsplats_tree(out, GSplatPartition(children=[first.tree, second.tree]))

        root = open_group(str(out), mode="r")
        assert root["part_1/centers"].shape[0] == 0
        assert root["part_1/centers"].attrs["encoding"]["name"] == "array_ref"

        node, _ = load_gsplat_node(out)
        summary = read_gsplat_tree_summary(root)
        assert summary.n_splats == total_splats(node) == 400
        assert summary.leaf_counts == (200, 200)

    def test_substitutive_lod(self, tmp_path):
        from luxar.gsplats.lod import RecipeParams, build_recipe

        out = tmp_path / "levels.gsplats.zarr"
        build_recipe(
            _flat(400),
            "levels",
            RecipeParams(compression_factor=4, levels=2),
        ).save(out)

        node, _ = load_gsplat_node(out)
        summary = read_gsplat_tree_summary(open_group(str(out), mode="r"))
        assert summary.kind == "lod"
        assert summary.n_splats == total_splats(node)
        assert summary.leaf_counts == tuple(leaf.n_splats for leaf in iter_leaves(node))

    def test_adaptive_nested_tree(self, tmp_path):
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.lod import RecipeParams, build_recipe

        out = tmp_path / "adaptive.gsplats.zarr"
        node = build_recipe(
            _flat(400),
            "adaptive",
            RecipeParams(max_elements=120, compression_factor=4, levels=2),
        )
        write_gsplats_tree(out, node)

        root = open_group(str(out), mode="r")
        assert any(root[name].attrs.get("kind") for name in root)
        node, _ = load_gsplat_node(out)
        summary = read_gsplat_tree_summary(root)
        assert summary.n_splats == total_splats(node)
        assert summary.leaf_counts == tuple(leaf.n_splats for leaf in iter_leaves(node))

    def test_additive_ladder_sums_its_rungs(self, tmp_path):
        """A ladder's rungs are increments, so the finest alone undercounts.

        The bug this pins: taking `sublods[-1]` reported 1,250 against the full
        reader's 5,000 on a four-rung store.
        """
        from luxar.gsplats.lod import RecipeParams, build_recipe

        out = tmp_path / "stream.gsplats.zarr"
        build_recipe(_flat(400), "stream", RecipeParams(n_lods=4)).save(out)

        node, _ = load_gsplat_node(out)
        summary = read_gsplat_tree_summary(open_group(str(out), mode="r"))
        assert summary.n_splats == total_splats(node)
        # And it really is a ladder, or the test would pass vacuously.
        assert next(iter(iter_leaves(node))).n_additive_sublods > 1


class TestDecodesNothing:
    """The whole point: metadata only, never a chunk."""

    def test_reading_a_summary_calls_no_decoder(self, partition_store, monkeypatch):
        calls = []
        real = ArrayDecoder.decode
        monkeypatch.setattr(
            ArrayDecoder,
            "decode",
            lambda self, array, root=None, *a, **k: (
                calls.append(1),
                real(self, array, root, *a, **k),
            )[1],
        )

        summary = read_gsplat_tree_summary(open_group(str(partition_store), mode="r"))

        assert summary.n_splats > 0, "guard: a summary of nothing decodes nothing too"
        assert calls == [], (
            f"summary decoded {len(calls)} array(s); it must decode none"
        )

    def test_full_read_decodes_for_comparison(self, partition_store, monkeypatch):
        """Control arm: the SAME probe shows the full reader decoding.

        Without this, `calls == []` above could mean the probe was never wired
        up rather than that nothing was decoded.
        """
        calls = []
        real = ArrayDecoder.decode
        monkeypatch.setattr(
            ArrayDecoder,
            "decode",
            lambda self, array, root=None, *a, **k: (
                calls.append(1),
                real(self, array, root, *a, **k),
            )[1],
        )

        load_gsplat_node(partition_store)
        assert calls, "the probe never fired — it is not attached to the real decoder"


class TestEstimatedDecodedBytes:
    def test_scales_with_the_number_of_splats(self, tmp_path):
        small = tmp_path / "small.gsplats.zarr"
        large = tmp_path / "large.gsplats.zarr"
        _flat(200).save(small)
        _flat(2_000).save(large)

        s = read_gsplat_tree_summary(open_group(str(small), mode="r"))
        big = read_gsplat_tree_summary(open_group(str(large), mode="r"))
        assert big.estimated_decoded_bytes > 5 * s.estimated_decoded_bytes

    def test_counts_the_float32_decode_not_the_stored_dtype(self, flat_store):
        """Quantized channels decode to float32 whatever they are stored as.

        Estimating from the stored `itemsize` would understate a uint8 encoding
        by 4x — the wrong direction for a memory warning.
        """
        summary = read_gsplat_tree_summary(open_group(str(flat_store), mode="r"))
        n, ndim = summary.n_splats, summary.ndim
        tri = ndim * (ndim + 1) // 2
        floor = n * (ndim + 1 + tri) * 4  # centers + amplitudes + cholesky, float32
        assert summary.estimated_decoded_bytes >= floor

    def test_counts_broadcast_amplitudes_at_decoded_size(self, tmp_path):
        data = _flat(1_000)
        data.amplitudes[:] = 1.0
        out = tmp_path / "uniform.gsplats.zarr"
        data.save(out)

        root = open_group(str(out), mode="r")
        assert root["amplitudes"].shape == (1,)
        assert root["amplitudes"].attrs["encoding"]["name"] == "broadcasted"
        summary = read_gsplat_tree_summary(root)
        node, _ = load_gsplat_node(out)
        leaf = next(iter(iter_leaves(node)))
        decoded_bytes = sum(
            array.nbytes
            for sublod in leaf.additive_sublods
            for array in (
                sublod.centers,
                sublod.amplitudes,
                sublod.cholesky_factors,
            )
        )
        assert summary.estimated_decoded_bytes == decoded_bytes

    def test_counts_row_lut_colors_at_decoded_shape(self, tmp_path):
        data = _flat(1_000, colors=True)
        palette = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        data.colors[:] = palette[np.arange(data.n_splats) % len(palette)]
        out = tmp_path / "row-lut-colors.gsplats.zarr"
        data.save(out)

        root = open_group(str(out), mode="r")
        encoding = root["colors"].attrs["encoding"]
        assert root["colors"].shape == (data.n_splats,)
        assert encoding["lut_mode"] == "row"
        assert encoding["original_shape"] == [data.n_splats, 3]

        summary = read_gsplat_tree_summary(root)
        node, _ = load_gsplat_node(out)
        leaf = next(iter(iter_leaves(node)))
        decoded_bytes = sum(
            array.nbytes
            for sublod in leaf.additive_sublods
            for array in (
                sublod.centers,
                sublod.amplitudes,
                sublod.cholesky_factors,
                sublod.colors,
            )
            if array is not None
        )
        assert summary.estimated_decoded_bytes == decoded_bytes


class TestRejectsWhatIsNotANode:
    def test_raises_for_a_group_that_is_not_a_gsplat_node(self, tmp_path):
        path = tmp_path / "empty.zarr"
        group = open_group(str(path), mode="w")
        group.attrs["something"] = "else"
        with pytest.raises(ValueError, match="not a gsplat node"):
            read_gsplat_tree_summary(open_group(str(path), mode="r"))


class TestLargeDecodeWarning:
    def test_default_filters_show_the_warning(self, flat_store, monkeypatch):
        import importlib

        load_module = importlib.import_module("luxar.gsplats.io.load_gsplats")

        monkeypatch.setattr(load_module, "_DECODE_WARN_BYTES", 1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.resetwarnings()
            load_gsplat_node(flat_store)

        assert len(caught) == 1
        assert caught[0].category is UserWarning
        assert "will materialise about" in str(caught[0].message)

    def test_warning_filters_cannot_break_the_load(self, flat_store, monkeypatch):
        import importlib

        load_module = importlib.import_module("luxar.gsplats.io.load_gsplats")

        monkeypatch.setattr(load_module, "_DECODE_WARN_BYTES", 1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            node, _ = load_gsplat_node(flat_store)
        assert total_splats(node) == 400
