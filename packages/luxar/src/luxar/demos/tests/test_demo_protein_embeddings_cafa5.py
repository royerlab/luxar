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

from pathlib import Path

import numpy as np

from luxar.demos import demo_protein_embeddings_cafa5 as demo
from luxar.demos.demo_protein_embeddings_cafa5 import (
    UNNAMED_CLUSTER_LABEL,
    enriched_cluster_names,
    load_cached_umap,
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

    def test_missing_cache_returns_none(self, tmp_path: Path) -> None:
        assert load_cached_umap(tmp_path / "absent.npz") is None
