"""Kaggle arXiv-embeddings demo: warm-cache build (#1107) and full-corpus scaling (#1919).

PART ONE — warm-cache scene build without the Points-LOD dependencies (issue #1107).

The Kaggle arXiv-embeddings demo advertises instant cached re-runs, but
``generate_paper_landscape`` used to request substitutive Points LOD
unconditionally. Building that LOD imports ``luxar.gsplats.lod`` (torch
coarsening kernels) whose additive sibling imports ``scipy.sparse`` at module
load — so on a torch/scipy-free machine WITH a complete ``arxiv_kaggle`` cache
the scene build crashed with ``ModuleNotFoundError`` instead of producing a
viewable scene. The fix routes the request through
``luxar.demos.substitutive_lod_or_flat``, which falls back to a flat point cloud
with a degradation notice when either module is missing.

PART TWO — the data path that lets the demo show the WHOLE 3,286,365-paper
corpus instead of its oldest 500,000 (issue #1919): uniform random sampling
in place of a date-ordered prefix, blockwise vector decode, streamed PCA
pre-reduction, preprint-server metadata, and point radii scaled to the
cloud's own measured spacing rather than to a constant tuned at one N.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from luxar._zarr_compat import read_node_attrs
from luxar.demos import demo_arxiv_embeddings_kaggle as demo
from luxar.demos.demo_arxiv_embeddings_kaggle import generate_paper_landscape


def _points_node(scene) -> "object":
    """Open the flat Points leaf of a built scene."""
    import zarr

    return zarr.open_group(str(scene), mode="r")["arxiv_papers_kaggle"]


def _decoded_colors(scene) -> np.ndarray:
    """Per-point float RGB, decoded through the stored LUT.

    ``colors`` lands on disk as ``lut_uint8`` indices, and the writer also
    Hilbert-reorders the points — so a raw read is neither RGB nor in input
    order. Decoding gives values comparable against the module's colour
    constants; the tests below are written to be order-independent.
    """
    node = _points_node(scene)
    array = node["colors"]
    encoding = dict(array.attrs)["encoding"]
    assert encoding["name"] == "lut_uint8", encoding["name"]
    lut = np.asarray(encoding["lut"], dtype=np.float32)
    return lut[np.asarray(array[:])]


def _fake_bundle(n: int = 40) -> dict:
    """A valid warm-cache bundle: the exact keys/shapes consumed downstream.

    ``years`` spans a real range so the year→color normalization is non-trivial;
    ``categories`` are arbitrary strings (looked up with a fallback); ``titles``
    are real strings (sliced for hover labels).
    """
    rng = np.random.default_rng(0)
    positions = rng.standard_normal((n, 3)).astype(np.float32)
    categories = [["cs.LG", "physics", "math.AT"][i % 3] for i in range(n)]
    years = [2010 + (i % 10) for i in range(n)]
    titles = [f"Paper number {i}" for i in range(n)]
    return {
        "positions": positions,
        "categories": categories,
        "years": years,
        "titles": titles,
        # Measured by the compute step; drives the radius ramp.
        "median_nn": 0.02,
    }


def _install_warm_cache(monkeypatch) -> None:
    """Simulate a complete warm cache: ``cache_computed`` never calls compute_fn.

    Returning a fabricated bundle avoids torch, the network, and ~/.cache — the
    scene build is exercised in isolation, exactly as a cached repeat run hits it.
    """

    def fake_cache_computed(name, key, compute_fn, **kwargs):  # noqa: ANN001
        return _fake_bundle()

    monkeypatch.setattr(demo, "cache_computed", fake_cache_computed)


class TestCompleteCacheRunsWithoutLodDeps:
    """A warm-cache scene build must succeed with torch/scipy unavailable."""

    @pytest.mark.parametrize("blocked", ["torch", "scipy"])
    def test_flat_scene_built_when_lod_dep_missing(
        self, blocked, monkeypatch, capsys, tmp_path
    ) -> None:
        _install_warm_cache(monkeypatch)
        # A None entry makes is_installed() False AND makes a fresh
        # ``import <blocked>`` raise — the exact issue #1107 repro, even on a
        # machine where the package IS installed.
        monkeypatch.setitem(sys.modules, blocked, None)

        out = tmp_path / "arxiv_papers_kaggle.luxar.zarr"
        n = generate_paper_landscape(out, sample_size=40)

        # The scene was written (no crash). The return value is N, not 2×N.
        assert n == 40
        assert out.exists()

        # The degradation notice named the blocked module.
        stdout = capsys.readouterr().out
        assert "skipping Points LOD" in stdout
        assert blocked in stdout

        # A FLAT Points leaf was written — not an LOD group. (Node: arxiv_papers_kaggle.)
        assert (out / "arxiv_papers_kaggle" / "positions").exists()
        attrs = read_node_attrs(out / "arxiv_papers_kaggle")
        assert attrs is not None, "the kaggle papers node must carry attributes"
        assert attrs.get("kind") != "lod"

    def test_lod_group_built_when_deps_present(
        self, monkeypatch, capsys, tmp_path
    ) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("scipy")
        _install_warm_cache(monkeypatch)

        out = tmp_path / "arxiv_papers_kaggle.luxar.zarr"
        n = generate_paper_landscape(out, sample_size=40)

        assert n == 40
        assert out.exists()

        # No degradation notice.
        stdout = capsys.readouterr().out
        assert "skipping Points LOD" not in stdout

        # A substitutive-LOD group was written: kind=lod with child_N levels and
        # NO top-level positions leaf.
        attrs = read_node_attrs(out / "arxiv_papers_kaggle")
        assert attrs is not None, "the kaggle papers node must carry attributes"
        assert attrs.get("kind") == "lod"
        assert (out / "arxiv_papers_kaggle" / "child_0").exists()
        assert not (out / "arxiv_papers_kaggle" / "positions").exists()


# =============================================================================
# Full-corpus scaling (issue #1919)
# =============================================================================


class TestSelectSampleIsRandomNotAPrefix:
    """``papers.csv`` is ID-sorted, so a prefix is a date slice, not a sample."""

    def test_none_and_oversized_select_everything(self) -> None:
        assert np.array_equal(demo.select_sample(7, None), np.arange(7))
        assert np.array_equal(demo.select_sample(7, 7), np.arange(7))
        assert np.array_equal(demo.select_sample(7, 99), np.arange(7))

    def test_subset_is_exact_sized_unique_and_ascending(self) -> None:
        idx = demo.select_sample(10_000, 250, seed=3)
        assert len(idx) == 250
        assert len(np.unique(idx)) == 250
        # Ascending keeps the downstream memmap gather sequential.
        assert np.all(np.diff(idx) > 0)
        assert idx.min() >= 0 and idx.max() < 10_000

    def test_seed_is_reproducible_and_seeds_differ(self) -> None:
        a = demo.select_sample(10_000, 250, seed=3)
        b = demo.select_sample(10_000, 250, seed=3)
        c = demo.select_sample(10_000, 250, seed=4)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_subset_spans_the_corpus_rather_than_its_head(self) -> None:
        """The regression guard for the bug this replaced.

        ``pd.read_csv(nrows=N)`` returned rows ``0..N-1``, i.e. the OLDEST 15% of
        a date-ordered corpus. A uniform sample must instead reach the far end:
        with 5,000 of 1,000,000 drawn uniformly, the largest index is above 99.9%
        of the corpus with overwhelming probability (P(max < 0.999 N) = 0.999^5000
        ~ 7e-3), whereas a prefix would top out at 4,999.
        """
        idx = demo.select_sample(1_000_000, 5_000, seed=0)
        assert idx.max() > 999_000
        # ...and it is spread, not clustered: every decile is represented.
        deciles = np.unique(idx // 100_000)
        assert len(deciles) == 10


class TestParseArgs:
    def test_default_processes_the_whole_corpus(self) -> None:
        assert demo.parse_args([]) == (None, 128, "auto", 0)

    def test_all_options_are_parsed(self) -> None:
        assert demo.parse_args(
            ["--sample=250000", "--pca-dim=64", "--device=gpu", "--seed=7"]
        ) == (250_000, 64, "gpu", 7)
        assert demo.parse_args(["--sample=all"]) == (None, 128, "auto", 0)

    def test_invalid_device_is_rejected_before_data_loading(self) -> None:
        with pytest.raises(ValueError, match="device must be one of"):
            demo.parse_args(["--device=tpu"])


class TestResolvePaperMetadata:
    """arXiv rows come from Cornell; preprint-server rows describe themselves."""

    LOOKUP = {"0704.0001": {"category": "hep-ph", "title": "Diphotons", "year": 2007}}

    def test_arxiv_row_uses_the_snapshot_and_strips_the_subcategory(self) -> None:
        lookup = {"2101.00001": {"category": "cs.LG", "title": "T", "year": 2021}}
        titles, cats, years = demo.resolve_paper_metadata(
            ["2101.00001"], ["arxiv"], lookup
        )
        assert (titles, cats, years) == (["T"], ["cs"], [2021])

    def test_preprint_servers_are_their_own_category_not_other(self) -> None:
        ids = ["10.1101/2020.03.03.20030890", "10.1101/2019.12.01.111111"]
        _, cats, _ = demo.resolve_paper_metadata(ids, ["medrxiv", "biorxiv"], {})
        assert cats == ["medrxiv", "biorxiv"]
        # Both colours exist, so the legend can name them.
        for cat in cats:
            assert cat in demo.CATEGORY_COLORS

    def test_year_is_read_from_the_preprint_doi(self) -> None:
        _, _, years = demo.resolve_paper_metadata(
            ["10.1101/2020.03.03.20030890", "10.64898/2025.12.05.25341689"],
            ["biorxiv", "medrxiv"],
            {},
        )
        assert years == [2020, 2025]

    def test_undated_legacy_accession_reports_zero_not_a_guess(self) -> None:
        """``10.1101/001891`` carries no date; inventing one would be a lie."""
        _, cats, years = demo.resolve_paper_metadata(
            ["10.1101/001891"], ["biorxiv"], {}
        )
        assert cats == ["biorxiv"]
        assert years == [0]

    def test_unmatched_arxiv_row_falls_back_to_other(self) -> None:
        _, cats, years = demo.resolve_paper_metadata(["9901.00001"], ["arxiv"], {})
        assert cats == ["other"]
        assert years == [0]

    def test_empty_selection_returns_empty_metadata(self) -> None:
        assert demo.resolve_paper_metadata([], [], {}) == ([], [], [])


class TestStreamingPcaReduction:
    """The 40 GB of vectors are decoded blockwise and reduced without a copy."""

    @staticmethod
    def _write_zip(path, vectors: np.ndarray, ids: list[str]) -> None:
        import zipfile

        with zipfile.ZipFile(path, "w") as zf:
            rows = "\n".join(f"{i},{pid},arxiv" for i, pid in enumerate(ids))
            zf.writestr("papers.csv", "index,id,journal\n" + rows + "\n")
            zf.writestr("vectors.dat", vectors.astype("<f4").tobytes())

    def _fixture(self, tmp_path, n=400, dim=None):
        dim = dim if dim is not None else demo.EMBEDDING_DIM
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((n, dim)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        zip_path = tmp_path / "emb.zip"
        self._write_zip(zip_path, vectors, [f"07{i:04d}.0001" for i in range(n)])
        return zip_path, vectors

    def test_row_count_is_derived_from_the_member_size(self, tmp_path) -> None:
        zip_path, vectors = self._fixture(tmp_path, n=37)
        assert demo.vector_row_count(zip_path) == 37

    def test_truncated_member_is_rejected_not_rounded_down(self, tmp_path) -> None:
        import zipfile

        zip_path = tmp_path / "bad.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("vectors.dat", b"\x00" * (demo.EMBEDDING_DIM * 4 + 5))
        with pytest.raises(ValueError, match="truncated"):
            demo.vector_row_count(zip_path)

    def test_papers_csv_is_read_row_aligned_with_the_vectors(self, tmp_path) -> None:
        zip_path, vectors = self._fixture(tmp_path, n=25)
        ids, journals = demo.read_paper_index(zip_path)
        assert len(ids) == len(vectors) == 25
        assert ids[0] == "070000.0001"
        assert set(journals) == {"arxiv"}

    def test_blockwise_decode_reproduces_every_row_exactly(self, tmp_path) -> None:
        """``np.frombuffer`` over blocks must equal the source, boundaries included."""
        zip_path, vectors = self._fixture(tmp_path, n=201)
        seen = np.zeros_like(vectors)

        def on_block(start, block):
            seen[start : start + len(block)] = block

        monkey_block = 16
        original = demo.STREAM_BLOCK_ROWS
        try:
            demo.STREAM_BLOCK_ROWS = monkey_block  # force many partial blocks
            got = demo._stream_vectors(zip_path, len(vectors), on_block)
        finally:
            demo.STREAM_BLOCK_ROWS = original
        assert got == len(vectors)
        assert np.array_equal(seen, vectors)

    def test_pca_matrix_is_cached_and_second_call_does_not_restream(
        self, tmp_path
    ) -> None:
        zip_path, vectors = self._fixture(tmp_path, n=400)
        cache = tmp_path / "cache"

        reduced = demo.build_pca_matrix(zip_path, cache, pca_dim=8, seed=0)
        assert reduced.shape == (400, 8)
        assert reduced.dtype == np.float32
        assert (cache / "pca8_all.npy").exists()
        assert not list(cache.glob("*.tmp")), "no partial cache may survive"

        # A second call must be served from disk: break the ZIP and it still works.
        zip_path.unlink()
        again = demo.build_pca_matrix(zip_path, cache, pca_dim=8, seed=0)
        assert np.array_equal(np.asarray(again), np.asarray(reduced))

    def test_pca_preserves_neighbourhood_structure(self, tmp_path) -> None:
        """The point of PCA here is that UMAP still sees the same neighbours.

        Two tight clusters in 3072-D must stay separated after the projection —
        if they did not, the reduction would be destroying the very structure the
        demo visualizes.
        """
        rng = np.random.default_rng(1)
        dim = demo.EMBEDDING_DIM
        centers = rng.standard_normal((2, dim)).astype(np.float32)
        vectors = np.concatenate(
            [
                c + 0.01 * rng.standard_normal((150, dim)).astype(np.float32)
                for c in centers
            ]
        )
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        zip_path = tmp_path / "clusters.zip"
        self._write_zip(zip_path, vectors, [f"1{i:06d}" for i in range(len(vectors))])

        reduced = np.asarray(
            demo.build_pca_matrix(zip_path, tmp_path / "c", pca_dim=8, seed=0)
        )
        a, b = reduced[:150], reduced[150:]
        within = np.linalg.norm(a - a.mean(axis=0), axis=1).mean()
        between = np.linalg.norm(a.mean(axis=0) - b.mean(axis=0))
        assert between > 10 * within


class TestMedianNearestNeighborDistance:
    """Radii are sized from this, so it must be the spacing of the WHOLE cloud."""

    def test_unit_lattice_reports_unit_spacing(self) -> None:
        g = np.arange(12, dtype=np.float32)
        pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
        assert demo.median_nearest_neighbor_distance(pts, n_probe=500) == pytest.approx(
            1.0, rel=1e-6
        )

    def test_denser_cloud_reports_smaller_spacing(self) -> None:
        g = np.arange(12, dtype=np.float32)
        pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
        dense = demo.median_nearest_neighbor_distance(pts * 0.25, n_probe=500)
        coarse = demo.median_nearest_neighbor_distance(pts, n_probe=500)
        assert dense == pytest.approx(coarse * 0.25, rel=1e-6)

    def test_degenerate_cloud_is_zero_not_an_error(self) -> None:
        assert (
            demo.median_nearest_neighbor_distance(np.zeros((1, 3), np.float32)) == 0.0
        )

    def test_probe_subsampling_does_not_inflate_the_estimate(self) -> None:
        """Sampling the QUERIES is unbiased; sampling the TREE would not be."""
        rng = np.random.default_rng(0)
        pts = rng.random((20_000, 3)).astype(np.float32)
        full = demo.median_nearest_neighbor_distance(pts, n_probe=20_000)
        probed = demo.median_nearest_neighbor_distance(pts, n_probe=500, seed=7)
        assert probed == pytest.approx(full, rel=0.15)


class TestRadiiTrackCloudDensity:
    """Regression guard for the hand-tuned-constant failure (issue #1919, 2e)."""

    @staticmethod
    def _max_radius(monkeypatch, tmp_path, median_nn, name):
        """Build a scene and return its ``max_radius`` — a real float.

        The per-point ``radii`` array is quantized to uint8 on write; the node
        attribute is the unquantized top of the ramp.
        """
        bundle = _fake_bundle()
        bundle["median_nn"] = median_nn
        monkeypatch.setattr(demo, "cache_computed", lambda *a, **k: bundle)
        monkeypatch.setattr(demo, "substitutive_lod_or_flat", lambda spec: None)
        out = tmp_path / f"{name}.luxar.zarr"
        demo.generate_paper_landscape(out, sample_size=40)
        attrs = read_node_attrs(out / "arxiv_papers_kaggle")
        assert attrs is not None
        return float(attrs["max_radius"])

    def test_halving_the_spacing_halves_the_radii(self, monkeypatch, tmp_path) -> None:
        coarse = self._max_radius(monkeypatch, tmp_path, 0.020, "coarse")
        dense = self._max_radius(monkeypatch, tmp_path, 0.010, "dense")
        assert dense == pytest.approx(coarse * 0.5, rel=1e-4)

    def test_radii_stay_below_the_spacing_so_spheres_do_not_merge(
        self, monkeypatch, tmp_path
    ) -> None:
        median_nn = 0.011
        top = self._max_radius(monkeypatch, tmp_path, median_nn, "spacing")
        assert top < median_nn
        assert top > 0.5 * median_nn

    def test_a_bundle_without_the_measurement_still_builds(
        self, monkeypatch, tmp_path
    ) -> None:
        """A v1-shaped bundle has no ``median_nn``; fall back, do not crash."""
        top = self._max_radius(monkeypatch, tmp_path, 0.0, "legacy")
        assert top == pytest.approx(0.016, rel=1e-3)


class TestUndatedPapersAreNotPaintedAsDated:
    def test_undated_points_take_the_neutral_colour(self, monkeypatch, tmp_path):
        bundle = _fake_bundle(n=12)
        bundle["years"] = [0, 0] + [2010 + i for i in range(10)]
        bundle["median_nn"] = 0.01
        monkeypatch.setattr(demo, "cache_computed", lambda *a, **k: bundle)
        monkeypatch.setattr(demo, "substitutive_lod_or_flat", lambda spec: None)

        out = tmp_path / "undated.luxar.zarr"
        demo.generate_paper_landscape(out, sample_size=12)

        colors = _decoded_colors(out)
        assert len(colors) == 24  # 12 papers x 2 colorings

        # EXACTLY the two undated papers wear the neutral colour, and only in
        # the year view — the category view paints them by category. Counting
        # rather than indexing keeps this independent of the Hilbert reorder.
        neutral = np.all(np.isclose(colors, demo.UNKNOWN_YEAR_COLOR, atol=1e-3), axis=1)
        assert int(neutral.sum()) == 2

        # The neutral colour must not collide with the "unmatched" grey, or the
        # two would be indistinguishable in the legend.
        assert not np.allclose(
            demo.UNKNOWN_YEAR_COLOR, demo.CATEGORY_COLORS["other"], atol=1e-3
        )


class TestUmapDeviceSelection:
    def test_gpu_without_cuml_fails_loudly(self, monkeypatch) -> None:
        monkeypatch.setattr(demo, "_cuml_umap", lambda: None)
        with pytest.raises(RuntimeError, match="cuML"):
            demo.reduce_embeddings_umap(np.zeros((4, 3), np.float32), device="gpu")

    def test_unknown_device_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="auto/cpu/gpu"):
            demo.reduce_embeddings_umap(np.zeros((4, 3), np.float32), device="tpu")

    def test_cpu_never_reaches_for_cuml(self, monkeypatch) -> None:
        called = []
        monkeypatch.setattr(demo, "_cuml_umap", lambda: called.append(1) or None)

        class _Reducer:
            def fit_transform(self, x):
                return np.zeros((len(x), 3), np.float32)

        monkeypatch.setattr(
            demo,
            "require_module",
            lambda name: type("M", (), {"UMAP": lambda **k: _Reducer()}),
        )
        demo.reduce_embeddings_umap(np.zeros((4, 8), np.float32), device="cpu")
        assert called == []
