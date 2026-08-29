"""Kaggle arXiv-embeddings demo: warm-cache build (#1107) and full-corpus scaling (#1919).

PART ONE — warm-cache scene build without the Points-LOD dependencies (issue #1107).

The Kaggle arXiv-embeddings demo advertises instant cached re-runs, but
``generate_paper_landscape`` used to request substitutive Points LOD
unconditionally. Building that LOD imports ``luxar.gsplats.lod`` (torch
coarsening kernels) whose additive sibling imports ``scipy.sparse`` at module
load — so on a torch/scipy-free machine WITH a complete ``arxiv_kaggle`` cache
the scene build crashed with ``ModuleNotFoundError`` instead of producing a
viewable scene. The first fix routed the request through
``luxar.demos.substitutive_lod_or_flat``, which DEGRADED to a flat point cloud
with a notice. The scene now carries an additive ladder rather than substitutive
levels, and the additive write path imports neither module — so the contract
holds with the ladder intact and the structure is identical either way.

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
from luxar.encoding.decoder import ArrayDecoder


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


def _decoded_radii(scene) -> np.ndarray:
    """Per-point radii decoded through the viewer's real array decoder."""
    node = _points_node(scene)
    return ArrayDecoder().decode(node["radii"], node)


def _overlay_attrs(scene) -> list[dict]:
    """Attributes for every authored overlay in storage order."""
    import zarr

    overlays = zarr.open_group(str(scene), mode="r")["overlays"]
    return [dict(overlays[name].attrs) for name in sorted(overlays.group_keys())]


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
    # A mix of every id shape the real corpus holds, so the DOI keys built from
    # this bundle exercise both branches of `paper_doi` rather than one.
    ids = [
        [
            "2101.12345",  # arXiv, new style
            "hep-th/9901001",  # arXiv, old style — contains a slash
            "10.1101/2020.03.03.20030890",  # medRxiv, already a DOI
            "10.1101/001891",  # legacy bioRxiv accession
        ][i % 4]
        for i in range(n)
    ]
    return {
        "positions": positions,
        "categories": categories,
        "years": years,
        "titles": titles,
        "ids": ids,
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
    """A warm-cache scene build must succeed with torch/scipy unavailable.

    It used to succeed by DEGRADING: ``substitutive_lod_or_flat`` printed a
    notice and wrote a flat, ladderless leaf. Now the scene carries an additive
    ladder instead of substitutive levels, and the additive write path imports
    neither module — so the contract holds with the ladder INTACT, which is the
    stronger property and what these tests assert.
    """

    @pytest.mark.parametrize("blocked", ["torch", "scipy"])
    def test_full_scene_built_when_former_lod_dep_missing(
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

        # No degradation, because there is nothing left to degrade.
        stdout = capsys.readouterr().out
        assert "skipping Points LOD" not in stdout

        # Identical structure to the deps-present run below: this is the point.
        self._assert_laddered_points_node(out)

    def test_same_structure_when_deps_present(
        self, monkeypatch, capsys, tmp_path
    ) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("scipy")
        _install_warm_cache(monkeypatch)

        out = tmp_path / "arxiv_papers_kaggle.luxar.zarr"
        n = generate_paper_landscape(out, sample_size=40)

        assert n == 40
        assert out.exists()
        stdout = capsys.readouterr().out
        assert "skipping Points LOD" not in stdout
        self._assert_laddered_points_node(out)

    @staticmethod
    def _assert_laddered_points_node(out) -> None:
        """A Points leaf with a streaming ladder, and no substitutive levels."""
        node = out / "arxiv_papers_kaggle"
        attrs = read_node_attrs(node)
        assert attrs is not None, "the kaggle papers node must carry attributes"
        # NOT a kind=lod group, and no coarse replacement levels.
        assert attrs.get("kind") != "lod"
        assert not (node / "child_0").exists()
        # At this sample size (40 x 2 colorings = 80) the ladder collapses to a
        # single rung, so assert the node is a real Points leaf rather than
        # counting rungs — the rung schedule itself is tested in
        # utils/tests/test_lod_breakpoints.py.
        assert (node / "positions").exists()


class TestScenePresentation:
    def test_title_does_not_claim_the_full_corpus_for_a_sample(
        self, monkeypatch, tmp_path
    ) -> None:
        _install_warm_cache(monkeypatch)
        out = tmp_path / "sampled.luxar.zarr"

        demo.generate_paper_landscape(out, sample_size=40)

        title = next(
            attrs["text"]
            for attrs in _overlay_attrs(out)
            if attrs.get("anchor") == "top-left"
        )
        assert title == "arXiv · bioRxiv · medRxiv Papers"
        assert "3.3M" not in title

    def test_preprint_servers_stay_in_the_category_legend_below_the_top_ten(
        self, monkeypatch, tmp_path
    ) -> None:
        bundle = _fake_bundle(n=40)
        bundle["categories"] = (
            ["cat0"] * 26
            + [f"cat{i}" for i in range(1, 12)]
            + ["biorxiv", "medrxiv", "other"]
        )
        monkeypatch.setattr(demo, "cache_computed", lambda *a, **k: bundle)
        out = tmp_path / "legend.luxar.zarr"

        demo.generate_paper_landscape(out, sample_size=40)

        legend = next(
            attrs["html"]
            for attrs in _overlay_attrs(out)
            if attrs.get("visible_range") == {"coloring": 0}
        )
        assert "> biorxiv</div>" in legend
        assert "> medrxiv</div>" in legend


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

    def test_a_numeric_whole_corpus_is_flagged_as_a_duplicate_bundle(
        self, capsys
    ) -> None:
        """Same result as `--sample=all`, different cache key — so say so.

        The key is the SPELLING: the corpus size is unknown until the ZIP is
        opened, and opening it before the cache lookup would cost a warm-bundle
        run a 30 GB download. The hint is what stops the habit instead.
        """
        assert np.array_equal(demo.select_sample(1_000, 1_000), np.arange(1_000))
        out = capsys.readouterr().out
        assert "whole corpus" in out and "--sample=all" in out

    def test_a_real_subset_says_nothing(self, capsys) -> None:
        demo.select_sample(1_000, 250)
        assert "whole corpus" not in capsys.readouterr().out

    def test_sample_none_says_nothing(self, capsys) -> None:
        demo.select_sample(1_000, None)
        assert "whole corpus" not in capsys.readouterr().out

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


class TestResolvePaperMetadata:
    """arXiv rows come from Cornell; preprint-server rows describe themselves."""

    LOOKUP = {"0704.0001": {"category": "hep-ph", "title": "Diphotons", "year": 2007}}

    def test_arxiv_row_uses_the_snapshot_and_strips_the_subcategory(self) -> None:
        lookup = {"2101.00001": {"category": "cs.LG", "title": "T", "year": 2021}}
        titles, cats, years = demo.resolve_paper_metadata(
            ["2101.00001"], ["arxiv"], lookup
        )
        assert (titles, cats, years) == (["T"], ["cs"], [2021])

    def test_year_comes_from_the_id_not_the_snapshots_update_date(self) -> None:
        """The regression guard for a 400,803-paper mislabel.

        The Cornell snapshot's ``year`` is ``update_date`` — the last revision —
        and nothing in it predates 2007. A 1992 paper revised in 2015 must read
        1992, or the Year view collapses 35 years of arXiv into 19.
        """
        lookup = {"hep-th/9201001": {"category": "hep-th", "title": "T", "year": 2015}}
        _, _, years = demo.resolve_paper_metadata(["hep-th/9201001"], ["arxiv"], lookup)
        assert years == [1992]

    def test_snapshot_year_is_the_fallback_for_an_unparseable_id(self) -> None:
        lookup = {"weird-id": {"category": "cs", "title": "T", "year": 2019}}
        _, _, years = demo.resolve_paper_metadata(["weird-id"], ["arxiv"], lookup)
        assert years == [2019]

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

    def test_unmatched_arxiv_row_falls_back_to_other_but_keeps_its_year(self) -> None:
        """No snapshot entry still leaves the ID, which carries the date."""
        _, cats, years = demo.resolve_paper_metadata(["9901.00001"], ["arxiv"], {})
        assert cats == ["other"]
        assert years == [1999]


class TestArxivSubmissionYear:
    """Both arXiv ID styles decode; nothing else is mistaken for one."""

    @pytest.mark.parametrize(
        "paper_id,year",
        [
            ("0704.0001", 2007),  # first new-style month
            ("2511.02119", 2025),
            ("9108.0001", 1991),  # arXiv's opening month, 4-digit serial
            ("hep-lat/0506004", 2005),
            ("math.AG/0601001", 2006),
            ("cond-mat/9910001", 1999),
            ("astro-ph/9108001", 1991),
        ],
    )
    def test_known_ids(self, paper_id, year) -> None:
        assert demo.arxiv_submission_year(paper_id) == year

    @pytest.mark.parametrize(
        "not_arxiv",
        [
            "10.1101/2020.03.03.20030890",  # a preprint DOI, handled elsewhere
            "10.1101/001891",
            "",
            "0704",
            "0704.1",  # serial too short
            "0700.0001",  # month 00
            "0713.0001",  # month 13
            "notanid",
        ],
    )
    def test_rejects_non_arxiv_ids(self, not_arxiv) -> None:
        assert demo.arxiv_submission_year(not_arxiv) == 0

    def test_the_1991_pivot_is_exact(self) -> None:
        """91..99 are 19xx and 00..90 are 20xx — check both sides of the seam."""
        assert demo.arxiv_submission_year("9101.0001") == 1991
        assert demo.arxiv_submission_year("9012.0001") == 2090
        assert demo.arxiv_submission_year("0001.0001") == 2000

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

    def test_interrupted_basis_write_does_not_poison_the_cache(
        self, monkeypatch, tmp_path
    ) -> None:
        zip_path, _ = self._fixture(tmp_path, n=400)
        cache = tmp_path / "cache"
        basis_path = cache / "pca8_basis.npz"

        def interrupted_savez(destination, **_arrays) -> None:
            if hasattr(destination, "write"):
                destination.write(b"partial")
            else:
                destination.write_bytes(b"partial")
            raise OSError("simulated interruption")

        with monkeypatch.context() as patch:
            patch.setattr(demo.np, "savez", interrupted_savez)
            with pytest.raises(OSError, match="simulated interruption"):
                demo.build_pca_matrix(zip_path, cache, pca_dim=8, seed=0)

        assert not basis_path.exists(), "an incomplete basis must never look cached"

        reduced = demo.build_pca_matrix(zip_path, cache, pca_dim=8, seed=0)
        assert reduced.shape == (400, 8)
        assert basis_path.exists()
        assert not list(cache.glob("*.tmp")), "retry must replace partial temp files"

    def test_a_truncated_pass_caches_no_basis_and_no_projection(
        self, tmp_path, monkeypatch
    ) -> None:
        """A short stream yields a PREFIX, and `papers.csv` is date-ordered.

        The subsample would then be uniform over an arbitrary date slice rather
        than the corpus, and — because the basis is CACHED — every later
        projection would inherit that skew silently. Nothing may survive.
        """
        zip_path, _ = self._fixture(tmp_path, n=400)
        cache = tmp_path / "short"

        real_stream = demo._stream_vectors
        monkeypatch.setattr(
            demo,
            "_stream_vectors",
            lambda z, n, cb: real_stream(z, n // 2, cb),
        )

        with pytest.raises(ValueError, match="truncated"):
            demo.build_pca_matrix(zip_path, cache, pca_dim=8, seed=0)

        leftovers = sorted(p.name for p in cache.iterdir()) if cache.exists() else []
        assert leftovers == [], f"a truncated pass left {leftovers} behind"

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

    def test_a_degenerate_cloud_without_spacing_still_builds(
        self, monkeypatch, tmp_path
    ) -> None:
        """A degenerate cloud has no measurable spacing; fall back, do not crash."""
        top = self._max_radius(monkeypatch, tmp_path, 0.0, "degenerate")
        assert top == pytest.approx(0.016, rel=1e-3)


class TestUndatedPapersAreNotPaintedAsDated:
    def test_undated_points_take_the_neutral_colour(self, monkeypatch, tmp_path):
        bundle = _fake_bundle(n=12)
        bundle["years"] = [0, 0] + [2010] * 5 + [2020] * 5
        bundle["median_nn"] = 0.01
        monkeypatch.setattr(demo, "cache_computed", lambda *a, **k: bundle)

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

        radii = _decoded_radii(out)
        unique, counts = np.unique(radii, return_counts=True)
        assert len(unique) == 3
        assert counts.tolist() == [10, 4, 10]


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


class TestBundleCacheVersion:
    """The bundle stores `years`, so changing how they are derived invalidates it.

    A v2 bundle carries `update_date` years and a v1 bundle is a date-ordered
    prefix; either would be served silently to anyone with a warm cache.
    """

    def test_the_bundle_version_is_bumped_past_the_update_date_bundles(
        self, monkeypatch, tmp_path
    ) -> None:
        seen = {}

        def spy(name, key, compute_fn, **kwargs):
            seen.update(name=name, key=key, version=kwargs.get("version"))
            return _fake_bundle()

        monkeypatch.setattr(demo, "cache_computed", spy)
        demo.generate_paper_landscape(tmp_path / "v.luxar.zarr", sample_size=40)

        assert seen["name"] == "arxiv_kaggle"
        assert seen["version"] >= 3, (
            "v1 (prefix) and v2 (update_date years) bundles must be unreachable"
        )
        assert seen["key"] == "umap3d_n40_pca128_seed0"


class TestParseArgs:
    """Values are validated at parse time, not behind the 30 GB streaming pass."""

    def test_defaults_are_the_whole_corpus_on_auto(self) -> None:
        assert demo.parse_args([]) == (None, demo.DEFAULT_PCA_DIM, "auto", 0)

    @pytest.mark.parametrize("spelling", ["all", "full", "ALL", "FULL"])
    def test_all_and_full_both_mean_the_whole_corpus(self, spelling) -> None:
        assert demo.parse_args([f"--sample={spelling}"])[0] is None

    def test_every_flag_together(self) -> None:
        got = demo.parse_args(
            ["--no-serve", "--sample=5", "--pca-dim=32", "--device=cpu", "--seed=2"]
        )
        assert got == (5, 32, "cpu", 2)

    def test_device_is_case_insensitive(self) -> None:
        assert demo.parse_args(["--device=GPU"])[2] == "gpu"

    @pytest.mark.parametrize(
        "arg",
        [
            "--device=tpu",
            "--device=",
            "--sample=0",  # used to build an empty scene with no error at all
            "--sample=-5",
            "--sample=abc",
            "--pca-dim=0",
            "--pca-dim=-3",
            "--seed=x",
            "--seed=-1",  # np.random.default_rng rejects it, much later
        ],
    )
    def test_unusable_values_raise_here(self, arg) -> None:
        with pytest.raises(ValueError):
            demo.parse_args([arg])

    @pytest.mark.parametrize("pca_dim", [0, demo.EMBEDDING_DIM + 1])
    def test_pca_dim_outside_the_embedding_width_is_rejected(self, pca_dim) -> None:
        """PCA cannot yield more components than the data has features, and the
        fit that would discover this sits behind a full streaming pass."""
        with pytest.raises(ValueError, match="pca-dim must be between"):
            demo.parse_args([f"--pca-dim={pca_dim}"])

    def test_all_options_together_are_parsed(self) -> None:
        assert demo.parse_args(
            ["--sample=250000", "--pca-dim=64", "--device=gpu", "--seed=7"]
        ) == (250_000, 64, "gpu", 7)

    def test_unknown_flags_are_ignored(self) -> None:
        """`luxar demo run` forwards its own flags (e.g. --no-serve)."""
        assert demo.parse_args(["--no-serve", "--whatever"]) == (
            None,
            demo.DEFAULT_PCA_DIM,
            "auto",
            0,
        )


class TestMainReportsBadFlagsCleanly:
    """A typo is a user error, not a crash — same shape as a build failure."""

    @pytest.mark.parametrize("arg", ["--device=tpu", "--sample=0", "--pca-dim=-1"])
    def test_bad_flag_exits_two_with_a_message_not_a_traceback(
        self, arg, monkeypatch, capsys
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["demo", arg])
        # Nothing may be built: the failure must happen before any work.
        monkeypatch.setattr(
            demo,
            "generate_paper_landscape",
            lambda *a, **k: pytest.fail("the demo started work on an invalid flag"),
        )
        with pytest.raises(SystemExit) as exc:
            demo.main()
        assert exc.value.code == 2
        assert "Error:" in capsys.readouterr().out

    def test_whole_corpus_banner_warns_before_starting_work(
        self, monkeypatch, capsys, tmp_path
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["demo", "--no-serve"])
        monkeypatch.setattr(demo, "get_demos_output_dir", lambda: tmp_path)
        monkeypatch.setattr(demo, "generate_paper_landscape", lambda *a, **k: 1)

        demo.main()

        out = capsys.readouterr().out
        assert "~39 GB" in out
        assert "hours on CPU" in out
        assert "--sample=N" in out


class TestPaperDoi:
    """`paper_doi` maps any row of `papers.csv` to a resolvable DOI (#1917).

    The corpus mixes two identifier namespaces, and a Points node carries a
    single `link` template — so one function has to normalise both onto the one
    resolver that serves all three preprint servers.
    """

    @pytest.mark.parametrize(
        ("paper_id", "expected"),
        [
            # arXiv rows carry a bare id and get arXiv's retroactively minted DOI.
            ("2101.12345", "10.48550/arXiv.2101.12345"),
            # Old-style ids contain a slash. It survives verbatim here; the
            # VIEWER percent-encodes it at click time, and doi.org resolves the
            # encoded form (checked against the live resolver).
            ("hep-th/9901001", "10.48550/arXiv.hep-th/9901001"),
            ("math.AT/0309136", "10.48550/arXiv.math.AT/0309136"),
            # bioRxiv / medRxiv rows are already DOIs and must pass through
            # untouched — prefixing one would produce a DOI that resolves to
            # nothing.
            ("10.1101/2020.03.03.20030890", "10.1101/2020.03.03.20030890"),
            ("10.1101/001891", "10.1101/001891"),
            ("10.64898/2025.12.05.25341689", "10.64898/2025.12.05.25341689"),
        ],
    )
    def test_maps_every_id_shape(self, paper_id: str, expected: str) -> None:
        assert demo.paper_doi(paper_id) == expected

    def test_never_double_prefixes(self) -> None:
        """Idempotent on anything already a DOI — the failure that would make
        every bioRxiv link dead while every arXiv link kept working, so half the
        corpus would look fine."""
        for pid in ("10.1101/001891", "10.48550/arXiv.2101.12345"):
            assert demo.paper_doi(demo.paper_doi(pid)) == pid

    def test_dispatches_on_the_id_not_the_journal_column(self) -> None:
        """`journal` is defaulted to "arxiv" when blank and bucketed to "other"
        when unrecognised, so it cannot be trusted to say what an id IS."""
        assert demo.paper_doi("10.1101/001891").startswith("10.1101/")


class TestColdBundleKeepsTheIds:
    """The COMPUTE side stores the ids the warm path reads back.

    Every other test here fakes `cache_computed`, so `_compute_bundle` never
    runs and nothing notices if it stops storing `ids` — a cold run would then
    build a bundle that produces no links, permanently, with the warm-cache
    tests still green. (Measured: deleting the `"ids"` line survived the whole
    file.) Every dependency of that closure is a module-level function, so the
    cold path can be exercised with stubs and no dataset.
    """

    def _run_compute(self, monkeypatch, n_rows: int = 6) -> dict:
        ids = [
            "2101.12345",
            "hep-th/9901001",
            "10.1101/2020.03.03.20030890",
            "10.1101/001891",
            "1234.5678",
            "10.64898/2025.12.05.25341689",
        ][:n_rows]
        journals = ["arxiv", "arxiv", "medrxiv", "biorxiv", "arxiv", "medrxiv"][:n_rows]

        monkeypatch.setattr(demo, "ensure_embeddings_zip", lambda: "unused.zip")
        monkeypatch.setattr(demo, "ensure_metadata_lookup", lambda: {})
        monkeypatch.setattr(demo, "read_paper_index", lambda _zip: (ids, journals))
        monkeypatch.setattr(
            demo,
            "build_pca_matrix",
            lambda *a, **k: np.zeros((n_rows, 4), dtype=np.float32),
        )
        monkeypatch.setattr(
            demo,
            "reduce_embeddings_umap",
            lambda emb, **k: np.zeros((len(emb), 3), dtype=np.float32),
        )
        monkeypatch.setattr(
            demo, "median_nearest_neighbor_distance", lambda *a, **k: 0.02
        )

        # Capture the closure `cache_computed` would have called, then run it —
        # this IS the cold path, minus the 40 GB.
        captured = {}

        def capture(name, key, compute_fn, **kwargs):  # noqa: ANN001
            captured["bundle"] = compute_fn()
            return captured["bundle"]

        monkeypatch.setattr(demo, "cache_computed", capture)
        return captured

    def test_bundle_stores_the_selected_ids(self, monkeypatch, tmp_path) -> None:
        captured = self._run_compute(monkeypatch)
        out = tmp_path / "arxiv_papers_kaggle.luxar.zarr"
        generate_paper_landscape(out, sample_size=None)

        bundle = captured["bundle"]
        assert "ids" in bundle, "the compute step dropped the ids again"
        # One id per position, in the same order — the pairing the DOI depends on.
        assert len(bundle["ids"]) == len(bundle["positions"])
        assert bundle["ids"][0] == "2101.12345"
        assert bundle["ids"][2] == "10.1101/2020.03.03.20030890"

    def test_a_cold_run_produces_links(self, monkeypatch, tmp_path) -> None:
        """The end the user actually sees: a first run links, without needing a
        second one to warm anything."""
        import zarr

        self._run_compute(monkeypatch)
        out = tmp_path / "arxiv_papers_kaggle.luxar.zarr"
        generate_paper_landscape(out, sample_size=None)

        root = zarr.open_group(str(out), mode="r")["arxiv_papers_kaggle"]
        nodes = [dict(root.attrs)] + [
            dict(root[name].attrs)
            for name in sorted(root.keys())
            if hasattr(root[name], "attrs")
        ]
        assert any(a.get("link") == "https://doi.org/{hover_key}" for a in nodes)
        assert any(a.get("has_keys") for a in nodes)


class TestDoiLinks:
    """The built scene carries the click-through, or cleanly carries none."""

    @staticmethod
    def _keyed_leaf(scene):
        """The node actually holding the keys CSR.

        With the LOD deps present the demo writes a ``kind=lod`` group whose
        COARSE children are merged gsplats and whose finest child is the
        original point cloud; keys live on the finest child alone, exactly as
        labels do. Without those deps it writes a flat leaf and the keys sit on
        the node itself. Search rather than hard-code, so this test says
        "wherever the keys are" instead of pinning one of the two layouts.
        """
        import zarr

        root = zarr.open_group(str(scene), mode="r")["arxiv_papers_kaggle"]
        if dict(root.attrs).get("has_keys"):
            return root
        for name in sorted(root.keys()):
            child = root[name]
            if hasattr(child, "attrs") and dict(child.attrs).get("has_keys"):
                return child
        raise AssertionError("no node in the scene carries a keys CSR")

    def _decode_keys(self, node) -> list[str]:
        offsets = np.asarray(node["key_offsets"][:])
        data = bytes(np.asarray(node["key_bytes"][:]).tobytes())
        return [
            data[offsets[i] : offsets[i + 1]].decode("utf-8")
            for i in range(len(offsets) - 1)
        ]

    def test_scene_carries_the_doi_template_and_keys(
        self, monkeypatch, tmp_path
    ) -> None:
        _install_warm_cache(monkeypatch)
        out = tmp_path / "arxiv_papers_kaggle.luxar.zarr"
        generate_paper_landscape(out, sample_size=40)

        leaf = self._keyed_leaf(out)
        attrs = dict(leaf.attrs)
        assert attrs["link"] == "https://doi.org/{hover_key}"
        assert attrs["copy"] == "{hover_key}"
        assert attrs["has_keys"] is True

        keys = self._decode_keys(leaf)
        # Stacked twice (Category and Year views), so one key per stacked point.
        assert len(keys) == 80
        # Both branches of paper_doi are present and correct on disk.
        assert "10.48550/arXiv.2101.12345" in keys
        assert "10.48550/arXiv.hep-th/9901001" in keys
        assert "10.1101/001891" in keys
        # Nothing was double-prefixed on the way through.
        assert not any(k.startswith("10.48550/arXiv.10.") for k in keys)

    def test_coarse_lod_levels_carry_the_template_but_no_keys(
        self, monkeypatch, tmp_path
    ) -> None:
        """Which is why the empty-substitution rule matters here.

        `link` is a non-compositing attr, so it is copied onto every LOD child,
        but keys only reach the finest one. A pick on a coarse merged level
        therefore resolves no key, and the viewer suppresses a link whose
        template has an empty substitution rather than opening
        ``https://doi.org/``. Pinned because the alternative — stripping the
        template from coarse levels — would silently disable clicking on the
        levels a user sees first.
        """
        import zarr

        _install_warm_cache(monkeypatch)
        out = tmp_path / "arxiv_papers_kaggle.luxar.zarr"
        generate_paper_landscape(out, sample_size=40)

        root = zarr.open_group(str(out), mode="r")["arxiv_papers_kaggle"]
        if dict(root.attrs).get("kind") != "lod":
            pytest.skip("LOD deps unavailable; the demo wrote a flat leaf")

        coarse = [
            dict(root[name].attrs)
            for name in sorted(root.keys())
            if hasattr(root[name], "attrs")
            and not dict(root[name].attrs).get("has_keys")
        ]
        assert coarse, "expected at least one coarse level"
        for attrs in coarse:
            assert attrs["link"] == "https://doi.org/{hover_key}"
            assert not attrs.get("has_keys")

    def test_a_bundle_without_ids_falls_back_to_a_label_search(
        self, monkeypatch, capsys, tmp_path
    ) -> None:
        """A cache written before ids were stored degrades to a label search.

        Rebuilding that bundle costs a 40 GB PCA stream plus a UMAP over 3.29M
        points, so it is still honoured rather than invalidated — the missing
        field must degrade, never raise.

        What changed is WHAT it degrades to. Dropping the link attributes
        entirely left every point hovering a title and doing nothing on click,
        which is indistinguishable from a demo that never had a destination and
        is invisible to every other check: the scene is valid, the labels are
        real. The title is a usable query even when the DOI is unavailable, so
        the fallback is a search rather than silence.
        """
        import zarr

        legacy = _fake_bundle()
        del legacy["ids"]
        monkeypatch.setattr(demo, "cache_computed", lambda *a, **k: legacy)

        out = tmp_path / "arxiv_papers_kaggle.luxar.zarr"
        assert generate_paper_landscape(out, sample_size=40) == 40

        # No node anywhere may claim a link it cannot fill. Checked across the
        # whole subtree rather than on the root, because with the LOD deps
        # present the root is a `kind=lod` wrapper and the real nodes are its
        # children.
        root = zarr.open_group(str(out), mode="r")["arxiv_papers_kaggle"]
        nodes = [("root", dict(root.attrs))] + [
            (name, dict(root[name].attrs))
            for name in sorted(root.keys())
            if hasattr(root[name], "attrs")
        ]
        # No node may claim a DOI it cannot fill, and none may carry keys —
        # but a labelled node must offer the title-search fallback instead of
        # nothing at all.
        linked = 0
        for name, attrs in nodes:
            assert not attrs.get("has_keys"), name
            link = attrs.get("link")
            if link is None:
                continue
            assert "doi.org" not in link, name
            assert link == "https://scholar.google.com/scholar?q={hover_label}", name
            assert attrs.get("copy") == "{hover_label}", name
            linked += 1
        assert linked, "expected the fallback search link on at least one node"

        # The scene is otherwise intact: labels still work, points still exist.
        assert any(attrs.get("has_labels") for _, attrs in nodes)
        assert "using Scholar label search" in capsys.readouterr().out
