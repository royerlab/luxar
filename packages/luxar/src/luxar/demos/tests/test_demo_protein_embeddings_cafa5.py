"""Tests for the CAFA5 demo's cluster naming and UMAP cache handling.

The demo used to label its ten k-means regions ``Cluster 0`` ... ``Cluster 9``:
the Kaggle bundle's ``*_train_terms.tsv`` covers PDB-style entries that do not
intersect the UniProt accessions in ``train_ids.npy`` at all, so the GO path
never matched and both colorings fell through to anonymous cluster indices.
Names now come from UniProt keyword enrichment, which puts three requirements
under test: the enrichment must pick the *over-represented* keyword rather than
the most common one, it must refuse to name a cluster that has no such keyword,
and the UMAP cache must carry the accessions the naming needs.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
from pathlib import Path

import numpy as np
import pytest

from luxar.demos import demo_protein_embeddings_cafa5 as demo
from luxar.demos.demo_protein_embeddings_cafa5 import (
    UNNAMED_CLUSTER_LABEL,
    disambiguate_cluster_names,
    enriched_cluster_names,
    load_accessions,
    load_cached_umap,
    select_bundle_files,
)

# Categories mirroring the UniProt vocabulary: "Technical term" entries are the
# ones the demo must ignore however common they are.
CATEGORIES = {
    "Hydrolase": "Molecular function",
    "Transit peptide": "Domain",
    "Nucleus": "Cellular component",
    "Receptor": "Molecular function",
    "Reference proteome": "Technical term",
    "3D-structure": "Technical term",
}


def _accessions(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i:03d}" for i in range(n)]


class TestEnrichedClusterNames:
    def test_picks_the_enriched_keyword_not_the_most_common_one(self) -> None:
        """A keyword on *every* protein carries no information about a cluster.

        ``Nucleus`` is universal here, so it must lose to ``Hydrolase``, which
        is confined to cluster 0, even though ``Nucleus`` is strictly more
        frequent inside that cluster.
        """
        cluster_a, cluster_b = _accessions("A", 100), _accessions("B", 100)
        keywords = {a: ["Nucleus", "Hydrolase"] for a in cluster_a}
        keywords.update({b: ["Nucleus"] for b in cluster_b})

        named = enriched_cluster_names(
            {0: cluster_a, 1: cluster_b}, keywords, CATEGORIES
        )

        assert named[0][0] == "Hydrolase"
        assert named[0][1] == 2.0  # present in 100% of the cluster, 50% overall
        assert named[1][0] == UNNAMED_CLUSTER_LABEL

    def test_technical_keywords_never_name_a_cluster(self) -> None:
        """``Reference proteome`` is a statement about UniProt, not biology."""
        cluster_a, cluster_b = _accessions("A", 100), _accessions("B", 100)
        keywords = {a: ["Reference proteome", "3D-structure"] for a in cluster_a}
        keywords.update({b: [] for b in cluster_b})

        named = enriched_cluster_names(
            {0: cluster_a, 1: cluster_b}, keywords, CATEGORIES
        )

        assert named[0][0] == UNNAMED_CLUSTER_LABEL
        assert named[1][0] == UNNAMED_CLUSTER_LABEL

    def test_unstructured_cluster_is_not_given_a_meaningless_name(self) -> None:
        """Below ``MIN_KEYWORD_LIFT`` the top keyword says nothing.

        Cluster 1 is a 55/45 mixture of the same two keywords as cluster 0 — its
        best lift is ~1.1, which would have produced a confidently wrong legend
        entry. It must read ``Mixed`` instead.
        """
        cluster_a = _accessions("A", 100)
        cluster_b = _accessions("B", 100)
        keywords = {
            a: ["Hydrolase"] if i < 50 else ["Receptor"]
            for i, a in enumerate(cluster_a)
        }
        keywords.update(
            {
                b: ["Hydrolase"] if i < 55 else ["Receptor"]
                for i, b in enumerate(cluster_b)
            }
        )

        named = enriched_cluster_names(
            {0: cluster_a, 1: cluster_b}, keywords, CATEGORIES
        )

        assert named[0] == (UNNAMED_CLUSTER_LABEL, 0.0)
        assert named[1] == (UNNAMED_CLUSTER_LABEL, 0.0)

    def test_rare_keyword_cannot_name_a_cluster(self) -> None:
        """Enormous lift on three proteins is noise, not a cluster identity."""
        cluster_a, cluster_b = _accessions("A", 100), _accessions("B", 100)
        keywords = {a: [] for a in cluster_a + cluster_b}
        for a in cluster_a[:3]:
            keywords[a] = ["Transit peptide"]

        named = enriched_cluster_names(
            {0: cluster_a, 1: cluster_b}, keywords, CATEGORIES
        )

        assert named[0][0] == UNNAMED_CLUSTER_LABEL

    def test_two_clusters_never_share_a_name(self) -> None:
        """The legend has one row per cluster; duplicate names would alias them.

        Both clusters here are 100% ``Hydrolase``, so on its own merits that is
        each one's top-scoring keyword — an independent per-cluster argmax would
        print it twice. Cluster 1 additionally carries ``Receptor`` (40%, 4x
        enriched, but a lower score than a 100%-prevalent keyword), so it must
        cede ``Hydrolase`` and fall back to it.
        """
        clusters = {i: _accessions(chr(65 + i), 100) for i in range(4)}
        keywords: dict[str, list[str]] = {}
        for a in clusters[0]:
            keywords[a] = ["Hydrolase"]
        for i, a in enumerate(clusters[1]):
            keywords[a] = ["Hydrolase", "Receptor"] if i < 40 else ["Hydrolase"]
        for a in clusters[2] + clusters[3]:
            keywords[a] = []

        named = enriched_cluster_names(clusters, keywords, CATEGORIES)

        assert named[0][0] == "Hydrolase"
        assert named[1][0] == "Receptor"

    def test_no_annotations_at_all_yields_no_names(self) -> None:
        """Offline / unresolvable accessions must not fabricate names."""
        clusters = {0: _accessions("A", 50), 1: _accessions("B", 50)}
        keywords: dict[str, list[str]] = {}

        named = enriched_cluster_names(clusters, keywords, CATEGORIES)

        assert named == {
            0: (UNNAMED_CLUSTER_LABEL, 0.0),
            1: (UNNAMED_CLUSTER_LABEL, 0.0),
        }


class TestNameClustersFallback:
    def test_unreachable_uniprot_falls_back_to_generic_labels(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The demo must stay runnable offline — just unable to name anything."""

        def _boom(*_args, **_kwargs):
            raise OSError("no network")

        monkeypatch.setattr(demo, "fetch_keyword_categories", _boom)

        names = demo.name_clusters(
            np.array([0, 0, 1, 1, 2, 2]),
            ["P1", "P2", "P3", "P4", "P5", "P6"],
            tmp_path,
        )

        assert names == ["Cluster 0", "Cluster 1", "Cluster 2"]


class TestUmapCache:
    def test_cache_round_trips_accessions(self, tmp_path: Path) -> None:
        cache = tmp_path / "umap.npz"
        positions = np.arange(6, dtype=np.float32).reshape(2, 3)
        np.savez(
            cache,
            positions=positions,
            protein_ids=np.array(["P1", "P2"], dtype=object),
        )

        loaded = load_cached_umap(cache)

        assert loaded is not None
        np.testing.assert_array_equal(loaded[0], positions)
        assert loaded[1] == ["P1", "P2"]

    def test_legacy_cache_is_upgraded_from_train_ids(self, tmp_path: Path) -> None:
        """A pre-accession cache is repaired rather than thrown away.

        Recomputing a 142k-protein UMAP costs tens of minutes, and for an
        unsampled run the cached point order *is* the ``train_ids.npy`` order.
        """
        cache = tmp_path / "umap.npz"
        positions = np.arange(6, dtype=np.float32).reshape(2, 3)
        np.savez(
            cache, positions=positions, functions=np.array(["a", "b"], dtype=object)
        )
        ids_path = tmp_path / "train_ids.npy"
        np.save(ids_path, np.array(["P1", "P2"]))

        loaded = load_cached_umap(cache, legacy_ids_path=ids_path)

        assert loaded is not None
        assert loaded[1] == ["P1", "P2"]
        # Upgrade is persisted, so the recovery happens once and not every run.
        assert "protein_ids" in np.load(cache, allow_pickle=True).files

    def test_legacy_cache_with_mismatched_ids_is_rejected(self, tmp_path: Path) -> None:
        """A sampled run's cache cannot be repaired — recompute instead of
        silently pairing coordinates with the wrong proteins."""
        cache = tmp_path / "umap.npz"
        np.savez(cache, positions=np.zeros((2, 3), dtype=np.float32))
        ids_path = tmp_path / "train_ids.npy"
        np.save(ids_path, np.array(["P1", "P2", "P3"]))

        assert load_cached_umap(cache, legacy_ids_path=ids_path) is None

    def test_legacy_cache_refuses_a_pickled_accessions_file(
        self, tmp_path: Path
    ) -> None:
        import pickle

        marker = tmp_path / "executed"

        class Exploit:
            def __reduce__(self):
                return (pathlib.Path.touch, (marker,))

        ids_path = tmp_path / "train_ids.npy"
        with ids_path.open("wb") as fh:
            np.lib.format.write_array_header_1_0(
                fh, {"descr": "|O", "fortran_order": False, "shape": (1,)}
            )
            pickle.dump(Exploit(), fh)

        cache = tmp_path / "legacy.npz"
        np.savez(cache, positions=np.zeros((1, 3), dtype=np.float32))

        with pytest.raises(ValueError, match="refusing to unpickle downloaded file"):
            load_cached_umap(cache, legacy_ids_path=ids_path)
        assert not marker.exists(), "the legacy repair path executed the payload"

    def test_mismatched_stored_accessions_are_rejected(self, tmp_path: Path) -> None:
        """A cache whose two arrays disagree cannot be trusted to pair up.

        The legacy repair path checks this; so must the ordinary one, or naming
        happily reports the wrong protein for every point it can still index.
        """
        cache = tmp_path / "umap.npz"
        np.savez(
            cache,
            positions=np.zeros((3, 3), dtype=np.float32),
            protein_ids=np.array(["P1", "P2"], dtype=object),
        )

        assert load_cached_umap(cache) is None

    def test_upgrade_does_not_clobber_the_cache_when_the_write_fails(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The repair rewrites a cache that is already good.

        Writing it in place would trade tens of minutes of UMAP for a truncated
        archive that raises on every later load — so the upgrade stages and
        renames, and a failed rename leaves the original readable.
        """
        cache = tmp_path / "umap.npz"
        positions = np.arange(6, dtype=np.float32).reshape(2, 3)
        np.savez(
            cache, positions=positions, functions=np.array(["a", "b"], dtype=object)
        )
        ids_path = tmp_path / "train_ids.npy"
        np.save(ids_path, np.array(["P1", "P2"]))

        def _fail(_src, _dst):
            raise OSError("rename failed")

        monkeypatch.setattr(demo.os, "replace", _fail)
        with pytest.raises(OSError, match="rename failed"):
            load_cached_umap(cache, legacy_ids_path=ids_path)

        with np.load(cache, allow_pickle=True) as survived:
            np.testing.assert_array_equal(survived["positions"], positions)

    def test_atomic_save_leaves_no_staging_file(self, tmp_path: Path) -> None:
        """``np.savez`` appends ``.npz`` to a *path*, so the staging write has to
        go through a handle or the rename would never find its source."""
        target = tmp_path / "nested" / "umap.npz"
        demo._save_npz_atomic(target, positions=np.zeros((2, 3), dtype=np.float32))

        assert target.exists()
        assert not list(tmp_path.rglob("*.part*"))

    def test_missing_cache_returns_none(self, tmp_path: Path) -> None:
        assert load_cached_umap(tmp_path / "absent.npz") is None


class TestDisambiguateClusterNames:
    def test_repeated_names_are_suffixed_with_the_cluster_index(self) -> None:
        """Several clusters can be ``Mixed``; the legend needs one row each.

        The suffix is the cluster INDEX, not a running counter, so a legend row,
        a hover label and the console log all point at the same cluster.
        """
        assert disambiguate_cluster_names(["Signal", "Mixed", "Nucleus", "Mixed"]) == [
            "Signal",
            "Mixed (1)",
            "Nucleus",
            "Mixed (3)",
        ]

    def test_unique_names_are_left_alone(self) -> None:
        names = ["Signal", "Nucleus", "Mixed"]
        assert disambiguate_cluster_names(names) == names

    def test_result_is_always_distinct(self) -> None:
        out = disambiguate_cluster_names(["Mixed"] * 5)
        assert len(set(out)) == 5


class TestSelectBundleFiles:
    """The accessions file must be chosen by MATCHING the embedding count.

    Picking it by file size (as an earlier revision did) is only right while
    every candidate shares a string width: a wider dtype makes a shorter array
    the largest file, which would pair every coordinate with the wrong protein
    and turn naming into confident nonsense.
    """

    def _bundle(self, root: Path, *, n_main: int = 40, n_subset: int = 5) -> None:
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "train_embeddings.npy", np.zeros((n_main, 4), dtype=np.float32))
        np.save(root / "train_ids.npy", np.array([f"P{i:05d}" for i in range(n_main)]))
        sub = root / "subset"
        sub.mkdir(exist_ok=True)
        np.save(sub / "train_embeddings.npy", np.zeros((n_subset, 4), dtype=np.float32))
        np.save(sub / "train_ids.npy", np.array([f"Q{i}" for i in range(n_subset)]))

    def test_picks_the_pair_that_matches(self, tmp_path: Path) -> None:
        self._bundle(tmp_path)
        embeddings, ids = select_bundle_files(tmp_path)
        assert embeddings.name == "train_embeddings.npy"
        assert embeddings.parent == tmp_path
        assert ids is not None and len(np.load(ids)) == 40

    def test_a_wider_but_shorter_ids_file_does_not_win(self, tmp_path: Path) -> None:
        """The regression this function exists for.

        ``decoy_train_ids.npy`` is the LARGEST ids file on disk (30 very long
        strings) but has the wrong length; size-based selection would take it.
        """
        self._bundle(tmp_path)
        np.save(
            tmp_path / "decoy_train_ids.npy",
            np.array(["X" * 400 for _ in range(30)]),
        )
        decoy = tmp_path / "decoy_train_ids.npy"
        real = tmp_path / "train_ids.npy"
        assert decoy.stat().st_size > real.stat().st_size, "decoy must be bigger"

        _embeddings, ids = select_bundle_files(tmp_path)

        assert ids == real

    def test_no_matching_ids_file_returns_none(self, tmp_path: Path) -> None:
        """Better to report "cannot name" than to mispair silently."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        np.save(tmp_path / "train_embeddings.npy", np.zeros((40, 4), dtype=np.float32))
        np.save(tmp_path / "train_ids.npy", np.array(["a", "b", "c"]))

        _embeddings, ids = select_bundle_files(tmp_path)

        assert ids is None

    def test_missing_embeddings_raise(self, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError):
            select_bundle_files(tmp_path)
        np.save(tmp_path / "unrelated.npy", np.zeros(3))
        with pytest.raises(FileNotFoundError, match="train_embeddings"):
            select_bundle_files(tmp_path)


class TestKeywordCacheDurability:
    """A cache is derived data: a truncated one must be refetched, not trusted.

    Writing it in place meant a run killed mid-write left JSON that raises on
    every later read — which naming caught and turned into a permanent, silent
    downgrade to generic `Cluster N` labels with nothing pointing at the file.
    """

    def test_truncated_cache_is_discarded_and_refetched(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        cache = tmp_path / demo.KEYWORD_CACHE_FILENAME
        cache.write_text('{"P1": ["Hydrolase"], "P2": ["Nu')
        monkeypatch.setattr(
            demo, "_uniprot_keyword_batch", lambda batch: {a: ["Signal"] for a in batch}
        )

        keywords = demo.fetch_uniprot_keywords(["P1", "P2"], tmp_path)

        assert keywords == {"P1": ["Signal"], "P2": ["Signal"]}
        assert json.loads(cache.read_text()) == keywords

    def test_non_dict_cache_is_discarded(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        path.write_text("[1, 2, 3]")
        assert demo._read_json_cache(path) is None

    def test_write_leaves_no_staging_file(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "c.json"
        demo._write_json_atomic(target, {"a": [1]})
        assert json.loads(target.read_text()) == {"a": [1]}
        assert not list(tmp_path.rglob("*.part"))

    def test_destination_is_untouched_until_the_rename(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The point of staging: a failed write must not damage the old cache.

        Discriminates against writing in place — an in-place ``write_text`` never
        reaches ``os.replace``, so the destination would already be clobbered by
        the time the failure surfaced.
        """
        target = tmp_path / "c.json"
        target.write_text('{"previous": ["value"]}')

        def _fail(_src, _dst):
            raise OSError("rename failed")

        monkeypatch.setattr(demo.os, "replace", _fail)
        with pytest.raises(OSError, match="rename failed"):
            demo._write_json_atomic(target, {"new": ["value"]})

        assert json.loads(target.read_text()) == {"previous": ["value"]}


class TestKeywordVocabulary:
    """The category vocabulary is what keeps `3D-structure` from naming a cluster.

    Silently caching an empty vocabulary would disable the filter entirely, so
    every provenance keyword would become an eligible name.
    """

    def test_empty_vocabulary_raises_and_is_not_cached(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            demo, "_http_get_paged", lambda url, **_kw: ("header\n", "")
        )

        with pytest.raises(ValueError, match="empty keyword vocabulary"):
            demo.fetch_keyword_categories(tmp_path)

        assert not (tmp_path / "uniprot_keyword_categories.json").exists()

    def test_pages_are_followed_and_never_repeated(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A cursor that never advances must terminate, not spin forever."""
        seen: list[str] = []

        def _paged(url, **_kw):
            seen.append(url)
            # Always advertise the SAME url as "next" — a server-side bug that an
            # unbounded `while url` loop would follow indefinitely.
            return (
                "id\tname\tcategory\nKW-1\tHydrolase\tMolecular function\n",
                '<https://stuck>; rel="next"',
            )

        monkeypatch.setattr(demo, "_http_get_paged", _paged)
        categories = demo.fetch_keyword_categories(tmp_path)

        assert categories == {"Hydrolase": "Molecular function"}
        assert len(seen) == 2, (
            f"followed {len(seen)} pages, expected to stop at the repeat"
        )

    def test_page_cap_bounds_a_runaway_cursor(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        counter = {"n": 0}

        def _paged(_url, **_kw):
            counter["n"] += 1
            return (
                f"id\tname\tcategory\nKW-{counter['n']}\tK{counter['n']}\tLigand\n",
                f'<https://page/{counter["n"]}>; rel="next"',
            )

        monkeypatch.setattr(demo, "_http_get_paged", _paged)
        categories = demo.fetch_keyword_categories(tmp_path)

        assert counter["n"] == demo.KEYWORD_VOCABULARY_MAX_PAGES
        assert len(categories) == demo.KEYWORD_VOCABULARY_MAX_PAGES

    def test_a_failing_batch_is_skipped_not_raised(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Partial coverage still names the landscape; losing it all does not."""
        monkeypatch.setattr(demo, "UNIPROT_BACKOFF_SECONDS", 0.0)
        calls: list[int] = []

        def _flaky(batch):
            calls.append(len(batch))
            if "P1" in batch:
                raise OSError("rate limited")
            return {a: ["Signal"] for a in batch}

        monkeypatch.setattr(demo, "_uniprot_keyword_batch", _flaky)

        keywords = demo.fetch_uniprot_keywords(["P1", "P2"], tmp_path, batch_size=1)

        # P1's batch exhausted its attempts and was skipped; P2 still resolved.
        assert "P1" not in keywords
        assert keywords["P2"] == ["Signal"]
        assert calls.count(1) == demo.UNIPROT_MAX_ATTEMPTS + 1


class TestLegacyKeywordCache:
    """A v1 cache must not be consulted, because it cannot answer the question.

    v1 wrote ``[]`` both for an accession UniProt did not resolve and for one that
    resolved carrying no keywords. Naming now has to tell those apart — the first
    must leave the cluster's denominator, the second is real evidence — and a
    cached accession is never re-requested, so trusting a v1 file would keep
    diluting enrichment on every machine that ran the demo before the fix.
    """

    def test_v1_cache_file_is_ignored(self, tmp_path: Path, monkeypatch) -> None:
        legacy = tmp_path / "uniprot_keywords.json"
        legacy.write_text(json.dumps({"P1": [], "P2": []}))
        requested: list[list[str]] = []

        def _batch(batch):
            requested.append(sorted(batch))
            return {"P1": None, "P2": []}

        monkeypatch.setattr(demo, "_uniprot_keyword_batch", _batch)

        keywords = demo.fetch_uniprot_keywords(["P1", "P2"], tmp_path)

        # Both accessions were re-requested despite being present in the v1 file,
        # and the ambiguity is resolved: P1 unmapped (None), P2 annotated-but-bare.
        assert requested == [["P1", "P2"]]
        assert keywords == {"P1": None, "P2": []}
        assert legacy.read_text() == json.dumps({"P1": [], "P2": []})

    def test_versioned_cache_is_reused(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / demo.KEYWORD_CACHE_FILENAME).write_text(
            json.dumps({"P1": ["Hydrolase"], "P2": None})
        )

        def _never(_batch):  # pragma: no cover - must not be reached
            raise AssertionError("cached accessions must not be re-requested")

        monkeypatch.setattr(demo, "_uniprot_keyword_batch", _never)

        assert demo.fetch_uniprot_keywords(["P1", "P2"], tmp_path) == {
            "P1": ["Hydrolase"],
            "P2": None,
        }


class TestUnmappedAccessions:
    """UniProt answers for a SUBSET of what it is asked — obsolete, demerged and
    deleted accessions are simply absent from the result stream.

    The lookup must record that absence distinguishably from an entry that
    genuinely carries no keywords: the former is absence of evidence and has to
    leave the cluster's denominator, the latter is evidence and stays. Both are
    cached, so neither is ever re-requested.
    """

    def test_unmapped_accession_is_none_not_an_empty_list(self) -> None:
        rows = [
            "Entry\tKeywords",
            "P00001\tHydrolase;Nucleus",
            "P00002\t",
        ]

        parsed = demo._parse_keyword_rows(["P00001", "P00002", "P99999"], rows)

        assert parsed == {
            "P00001": ["Hydrolase", "Nucleus"],
            "P00002": [],  # a real entry with no keywords
            "P99999": None,  # never came back
        }

    def test_empty_result_stream_is_reported_not_indexed(self) -> None:
        """A truncated/blank stream must raise the error naming already handles."""
        with pytest.raises(ValueError, match="empty result stream"):
            demo._parse_keyword_rows(["P1"], [])

    def test_a_refused_submission_raises_valueerror(self, monkeypatch) -> None:
        """UniProt refuses a batch with 200-plus-error-payload, not a 4xx.

        The lookup retries and then skips a ``ValueError``, and naming falls back
        on one; a bare ``KeyError`` from ``payload["jobId"]`` matches neither and
        would abort the whole demo.
        """
        monkeypatch.setattr(
            demo.urllib.request,
            "urlopen",
            lambda *_a, **_kw: contextlib.closing(
                io.BytesIO(b'{"messages": ["Invalid request"]}')
            ),
        )

        with pytest.raises(ValueError, match="refused the batch"):
            demo._uniprot_keyword_batch(["P1"])

    def test_unmapped_members_do_not_dilute_their_cluster(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The reason the ``None`` marker exists, end to end through the cache.

        Cluster 0 is 12 hydrolases plus 108 accessions UniProt does not know.
        Counting those 108 as "annotated, no keywords" puts Hydrolase at 10% of
        the cluster against 5% pooled — lift 1.83, below the threshold — and the
        real signal is lost. Dropping them lifts it to 9.3x and the cluster earns
        its name.
        """
        monkeypatch.setattr(demo, "fetch_keyword_categories", lambda _dir: CATEGORIES)

        def _batch(batch):
            """UniProt's real shape: only the ids it knows come back."""
            hits = {"A": ["Hydrolase"], "B": ["Nucleus"]}
            return {a: hits.get(a[0]) for a in batch}

        monkeypatch.setattr(demo, "_uniprot_keyword_batch", _batch)
        protein_ids = (
            [f"A{i}" for i in range(12)]
            + [f"X{i}" for i in range(108)]
            + [f"B{i}" for i in range(100)]
        )
        cluster_ids = np.array([0] * 120 + [1] * 100)

        names = demo.name_clusters(cluster_ids, protein_ids, tmp_path)

        assert names[0] == "Hydrolase"
        # The unmapped ids are cached as null, so a later run does not re-ask.
        cached = json.loads((tmp_path / demo.KEYWORD_CACHE_FILENAME).read_text())
        assert cached["X0"] is None

    def test_a_wholly_unmapped_sample_falls_back_to_generic_labels(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Nothing resolving means the demo cannot name anything — say `Cluster N`
        rather than a legend of 14 identical `Mixed` rows."""
        monkeypatch.setattr(demo, "fetch_keyword_categories", lambda _dir: CATEGORIES)
        monkeypatch.setattr(
            demo, "_uniprot_keyword_batch", lambda batch: dict.fromkeys(batch)
        )

        names = demo.name_clusters(
            np.array([0, 0, 1, 1]), ["P1", "P2", "P3", "P4"], tmp_path
        )

        assert names == ["Cluster 0", "Cluster 1"]


class TestGenerateLandscape:
    def test_a_missing_bundle_does_not_defeat_a_complete_cache(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The UMAP cache hit is meant to be a complete early exit.

        The bundle is only consulted to recover accessions for a cache written
        before they were stored; resolving it unconditionally let a bundle that
        had gone missing abort a run whose cache already carried everything.
        """
        monkeypatch.setattr(demo.Path, "home", lambda: tmp_path)
        cache_dir = tmp_path / ".cache" / "luxar" / "protein_embeddings"
        # Present but empty: the download is skipped, the .npy files are not there.
        (cache_dir / "cafa5_data").mkdir(parents=True)

        rng = np.random.default_rng(0)
        positions = rng.normal(size=(40, 3)).astype(np.float32)
        np.savez(
            cache_dir / "umap_all.npz",
            positions=positions,
            protein_ids=np.array(
                [f"{demo.SYNTHETIC_ID_PREFIX}{i}" for i in range(40)], dtype=object
            ),
        )

        def _boom(*_args, **_kwargs):
            raise AssertionError("UniProt must not be contacted for stand-in ids")

        monkeypatch.setattr(demo, "fetch_keyword_categories", _boom)
        monkeypatch.setattr(demo, "fetch_uniprot_keywords", _boom)

        output = tmp_path / "protein_landscape.luxar.zarr"
        n_proteins = demo.generate_protein_landscape(output, sample_size=None)

        assert n_proteins == 40
        assert output.exists()


class TestSyntheticAccessions:
    def test_synthetic_ids_do_not_author_dead_search_links(self, capsys) -> None:
        attrs = demo._uniprot_link_attrs(
            [f"{demo.SYNTHETIC_ID_PREFIX}{i}" for i in range(3)]
        )

        assert attrs == {}
        assert "no reliable per-protein link" in capsys.readouterr().out

    def test_synthetic_ids_skip_the_lookup_entirely(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """With no accessions file the ids are stand-ins, not proteins.

        Submitting 22k of them costs minutes and can only come back empty, so
        naming must not reach the network at all.
        """

        def _boom(*_args, **_kwargs):
            raise AssertionError("UniProt must not be contacted for stand-in ids")

        monkeypatch.setattr(demo, "fetch_keyword_categories", _boom)
        monkeypatch.setattr(demo, "fetch_uniprot_keywords", _boom)
        protein_ids = [f"{demo.SYNTHETIC_ID_PREFIX}{i}" for i in range(4)]

        names = demo.name_clusters(np.array([0, 0, 1, 1]), protein_ids, tmp_path)

        assert names == ["Cluster 0", "Cluster 1"]

    def test_nothing_resolving_falls_back_to_generic_labels(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(demo, "fetch_keyword_categories", lambda _dir: CATEGORIES)
        monkeypatch.setattr(demo, "fetch_uniprot_keywords", lambda _acc, _dir: {})

        names = demo.name_clusters(
            np.array([0, 0, 1, 1]), ["a", "b", "c", "d"], tmp_path
        )

        assert names == ["Cluster 0", "Cluster 1"]


# --------------------------------------------------------------------------- #
# Accession loading must not execute the bundle's code
# --------------------------------------------------------------------------- #


def test_accessions_load_from_a_fixed_width_array_without_pickle(tmp_path) -> None:
    """The no-pickle path: nothing from the file is ever executed."""
    path = tmp_path / "train_ids.npy"
    np.save(path, np.array(["P12345", "Q67890", "A0A123"], dtype="<U10"))

    assert load_accessions(path) == ["P12345", "Q67890", "A0A123"]


def test_accessions_load_from_a_scalar_fixed_width_array(tmp_path) -> None:
    path = tmp_path / "train_ids.npy"
    np.save(path, np.array("P12345", dtype="<U10"))

    assert load_accessions(path) == ["P12345"]


def test_accessions_refuse_an_object_array(tmp_path) -> None:
    path = tmp_path / "train_ids.npy"
    np.save(path, np.array(["P12345", "Q67890"], dtype=object), allow_pickle=True)

    with pytest.raises(ValueError, match="refusing to unpickle downloaded file"):
        load_accessions(path)


def test_accessions_refuse_a_bare_pickle_without_unsafe_advice(tmp_path) -> None:
    import pickle

    marker = tmp_path / "executed"

    class Exploit:
        def __reduce__(self):
            return (pathlib.Path.touch, (marker,))

    path = tmp_path / "train_ids.npy"
    path.write_bytes(pickle.dumps(Exploit()))

    with pytest.raises(ValueError, match="refusing to unpickle downloaded file"):
        load_accessions(path)
    assert not marker.exists(), "the payload ran — the restriction is not holding"


def test_a_malicious_accessions_pickle_is_refused_not_executed(tmp_path) -> None:
    """The reason this loader exists.

    The CAFA5 bundle comes from a Kaggle dataset its owner can replace at any
    time, and the download is verified by size, not content. ``np.load(...,
    allow_pickle=True)`` on that file runs whatever it says to run; this exact
    payload executes if a caller enables pickle. The loader refuses the object
    array instead, before any accession is read.
    """
    import pickle

    marker = tmp_path / "executed"

    class Exploit:
        def __reduce__(self):
            return (pathlib.Path.touch, (marker,))

    path = tmp_path / "train_ids.npy"
    with path.open("wb") as fh:
        np.lib.format.write_array_header_1_0(
            fh, {"descr": "|O", "fortran_order": False, "shape": (1,)}
        )
        pickle.dump(Exploit(), fh)

    with pytest.raises(ValueError, match="refusing to unpickle downloaded file"):
        load_accessions(path)
    assert not marker.exists(), "the payload ran — the restriction is not holding"
