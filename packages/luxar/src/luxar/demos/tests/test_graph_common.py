"""Unit tests for the shared graph-demo pipeline (``_graph_common``).

The three network demos (CAIDA AS topology, HuRI interactome, PPI flow field)
each carried a private copy of this machinery; unifying them is only safe if the
copies' behaviour is pinned somewhere. These tests do that on tiny synthetic
frames — no network, no GPU, no cache:

* ``download_file`` — the cache-hit short circuit (which must not touch the
  network at all), the promotion of the staged temp file, the default timeout,
  the cleanup of a partial download after a mid-stream failure (the leak this
  unification fixed), and the isolation of that staging from a concurrent run
  sharing the same cache directory and destination.
* ``load_hgnc`` — the ``Approved``-only filter, the chromosome prefix pulled out
  of a ``location`` like ``17q21.31``, the ``"?"`` fallback (including for a TSV
  with no ``location`` column at all, where an index-misaligned fallback used to
  produce ``NaN`` and crash the demos downstream), and first-wins on a duplicate
  Ensembl id.
* ``load_huri_edges`` — non-ENSG rows dropped with or without a header line,
  self-edges dropped, pairs with an unmapped endpoint dropped, and ``(A, B)`` /
  ``(B, A)`` collapsed to one undirected row.
* ``filter_to_lcc`` — largest component only, edges outside it dropped,
  deterministic node order, chromosome attachment with ``"?"`` for an unmapped
  symbol.
* ``build_adjacency`` / ``compute_communities`` — the ``columns`` argument
  really selects the endpoint columns (both the ``sym_*`` and ``asn_*``
  spellings the demos use), plus symmetry, size-descending community order,
  determinism, and the per-demo console wording carried by ``unit_label`` /
  ``summary_suffix``.
* ``nodes_hash`` — deterministic, order-sensitive, content-sensitive (it is a
  layout-cache key, so a collision across node sets would serve a stale layout).

``requests`` is faked with a tiny context-manager object; everything else runs
for real.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np
import pytest
from arbol import Arbol

# pandas ships in the ``demos`` extra, not core, and the module needs it at
# import time — skip cleanly when it is absent (matches the sibling demo tests).
pd = pytest.importorskip("pandas")

from luxar.demos import _graph_common  # noqa: E402
from luxar.demos._graph_common import (  # noqa: E402
    build_adjacency,
    build_community_legend,
    compute_communities,
    download_file,
    filter_to_lcc,
    load_hgnc,
    load_huri_edges,
    nodes_hash,
)


@pytest.fixture(autouse=True)
def _pin_arbol_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin Arbol's depth limit so console assertions don't depend on import order.

    ``Arbol.max_depth`` is global and every imported demo module sets it, so
    whichever ran last wins. These helpers print one or two ``asection`` levels
    deep; pin the limit rather than bank on the lowest value a demo happens to
    set.
    """
    monkeypatch.setattr(Arbol, "max_depth", 100)


# =============================================================================
# Fake requests (external dependency; nothing here may hit the network)
# =============================================================================


class _FakeResponse:
    """Minimal context-manager stand-in for a streamed ``requests`` response."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        content_length: Optional[int] = None,
        fail_after: Optional[int] = None,
    ) -> None:
        self._chunks = chunks
        self._fail_after = fail_after
        self.headers: dict[str, str] = {}
        if content_length is not None:
            self.headers["content-length"] = str(content_length)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 1 << 20) -> Iterator[bytes]:
        for i, chunk in enumerate(self._chunks):
            if self._fail_after is not None and i == self._fail_after:
                raise RuntimeError("connection reset mid-stream")
            yield chunk


class _InterleavingResponse(_FakeResponse):
    """A response that runs ``between`` once, mid-stream.

    Stands in for a second run of the same demo starting while ours is still
    streaming — the overlap that a destination-derived staging name turns into a
    collision.
    """

    def __init__(self, chunks: list[bytes], *, between: Callable[[], None]) -> None:
        super().__init__(chunks)
        self._between = between

    def iter_content(self, chunk_size: int = 1 << 20) -> Iterator[bytes]:
        for i, chunk in enumerate(self._chunks):
            if i == 1:
                self._between()
            yield chunk


class _GetSpy:
    """Records every ``requests.get`` call and replays a scripted response."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self._response


def _patch_get(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> _GetSpy:
    spy = _GetSpy(response)
    monkeypatch.setattr(_graph_common.requests, "get", spy)
    return spy


def _explode(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("requests.get must not be called")


# =============================================================================
# Tiny synthetic graphs
# =============================================================================


def _edges(pairs: list[tuple[str, str]], columns: tuple[str, str]) -> Any:
    """An edge frame with the given endpoint column names."""
    col_a, col_b = columns
    return pd.DataFrame({col_a: [a for a, _ in pairs], col_b: [b for _, b in pairs]})


def _hgnc(mapping: dict[str, str]) -> Any:
    """An HGNC-shaped frame: indexed by Ensembl id, symbol + chromosome columns."""
    symbols = list(mapping)
    return pd.DataFrame(
        {"symbol": symbols, "chromosome": [mapping[s] for s in symbols]},
        index=[f"ENSG{i:011d}" for i in range(len(symbols))],
    )


# =============================================================================
# download_file
# =============================================================================


class TestDownloadFile:
    def test_cached_file_short_circuits_without_any_request(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(_graph_common.requests, "get", _explode)
        dest = tmp_path / "data.tsv"
        dest.write_bytes(b"already here")

        download_file("http://example.invalid/data.tsv", dest, "data")

        assert dest.read_bytes() == b"already here"
        assert "Using cached data.tsv" in capsys.readouterr().out
        # The early return happens before ``mkdtemp``, so a cache hit stages
        # nothing: no staging dir to leak if the process dies right here.
        assert list(tmp_path.iterdir()) == [dest]

    def test_empty_cached_file_is_refetched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A zero-byte file is a failed earlier run, not a cache hit.
        dest = tmp_path / "data.tsv"
        dest.write_bytes(b"")
        _patch_get(monkeypatch, _FakeResponse([b"real body"]))

        download_file("http://example.invalid/data.tsv", dest, "data")

        assert dest.read_bytes() == b"real body"

    def test_successful_download_writes_bytes_and_leaves_no_part(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        body = [b"chunk-one", b"chunk-two"]
        _patch_get(monkeypatch, _FakeResponse(body, content_length=18))
        dest = tmp_path / "nested" / "deeper" / "data.tsv"

        download_file("http://example.invalid/data.tsv", dest, "the data (~1 MB)")

        assert dest.read_bytes() == b"".join(body)
        assert list(dest.parent.iterdir()) == [dest]  # no leftover .part
        out = capsys.readouterr().out
        assert "Downloading the data (~1 MB)" in out
        assert "✓ Saved data.tsv" in out

    def test_destination_parent_directory_is_created(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_get(monkeypatch, _FakeResponse([b"body"]))
        dest = tmp_path / "brand" / "new" / "tree" / "data.tsv"
        assert not dest.parent.exists()

        download_file("http://example.invalid/data.tsv", dest, "data")

        assert dest.parent.is_dir()
        assert dest.read_bytes() == b"body"

    def test_midstream_failure_reraises_and_removes_the_partial(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The copies this helper replaced left ``<dest>.part`` behind here.
        _patch_get(
            monkeypatch,
            _FakeResponse([b"first", b"second", b"third"], fail_after=1),
        )
        dest = tmp_path / "data.tsv"

        with pytest.raises(RuntimeError, match="connection reset mid-stream"):
            download_file("http://example.invalid/data.tsv", dest, "data")

        assert not dest.exists()
        assert list(tmp_path.iterdir()) == []

    def test_staging_does_not_touch_another_runs_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A concurrent run of the same demo stages its own file in the shared
        # cache dir. Ours must neither write over it nor delete it on the way
        # out — hence the private per-invocation staging directory.
        _patch_get(monkeypatch, _FakeResponse([b"ours"]))
        dest = tmp_path / "data.tsv"
        foreign = tmp_path / "data.tsv.part"
        foreign.write_bytes(b"another run's in-flight body")

        download_file("http://example.invalid/data.tsv", dest, "data")

        assert dest.read_bytes() == b"ours"
        assert foreign.read_bytes() == b"another run's in-flight body"
        assert {p.name for p in tmp_path.iterdir()} == {dest.name, foreign.name}

    def test_a_concurrent_runs_failure_does_not_break_our_promotion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Both protein demos default to the same cache dir and filenames, so two
        # overlapping runs share ``dest``. With a destination-derived ``.part``
        # name, the failing run's cleanup unlinked the other's staging file and
        # its promotion then raised FileNotFoundError.
        dest = tmp_path / "data.tsv"

        def other_run_fails() -> None:
            with pytest.raises(RuntimeError, match="connection reset mid-stream"):
                download_file("http://example.invalid/data.tsv", dest, "data")

        queue: list[_FakeResponse] = [
            _InterleavingResponse([b"one", b"two"], between=other_run_fails),
            _FakeResponse([b"theirs", b"more"], fail_after=1),
        ]
        monkeypatch.setattr(
            _graph_common.requests,
            "get",
            lambda _url, **_kwargs: queue.pop(0),
        )

        download_file("http://example.invalid/data.tsv", dest, "data")

        assert dest.read_bytes() == b"onetwo"
        assert list(tmp_path.iterdir()) == [dest]  # neither run left residue

    def test_default_timeout_is_the_slow_archive_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = _patch_get(monkeypatch, _FakeResponse([b"body"]))

        download_file("http://example.invalid/data.tsv", tmp_path / "data.tsv", "data")

        assert spy.calls[0]["timeout"] == 180.0
        assert spy.calls[0]["stream"] is True

    def test_timeout_override_is_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = _patch_get(monkeypatch, _FakeResponse([b"body"]))

        download_file(
            "http://example.invalid/data.tsv",
            tmp_path / "data.tsv",
            "data",
            timeout=5.0,
        )

        assert spy.calls[0]["timeout"] == 5.0

    def test_empty_chunks_are_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Keep-alive chunks are empty and must not end up in the file.
        _patch_get(monkeypatch, _FakeResponse([b"a", b"", b"b"]))
        dest = tmp_path / "data.tsv"

        download_file("http://example.invalid/data.tsv", dest, "data")

        assert dest.read_bytes() == b"ab"


# =============================================================================
# load_hgnc
# =============================================================================


class TestLoadHgnc:
    """The Ensembl→(symbol, chromosome) table the two protein demos consume."""

    @staticmethod
    def _write(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
        """An HGNC-shaped TSV; the column set is whatever the rows carry."""
        path = tmp_path / "hgnc.tsv"
        pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
        return path

    def test_only_approved_rows_survive(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [
                {
                    "ensembl_gene_id": "ENSG1",
                    "symbol": "AAA",
                    "location": "1p36.33",
                    "status": "Approved",
                },
                {
                    "ensembl_gene_id": "ENSG2",
                    "symbol": "BBB",
                    "location": "2q11.2",
                    "status": "Entry Withdrawn",
                },
            ],
        )

        df = load_hgnc(path)

        assert list(df.index) == ["ENSG1"]
        assert list(df.columns) == ["symbol", "chromosome"]
        assert df.loc["ENSG1", "symbol"] == "AAA"

    def test_chromosome_is_the_prefix_of_the_location(self, tmp_path: Path) -> None:
        # "17q21.31" → "17", "Xp22.2" → "X"; anything the regex cannot start on
        # (and a blank cell) falls back to "?".
        path = self._write(
            tmp_path,
            [
                {
                    "ensembl_gene_id": f"ENSG{i}",
                    "symbol": sym,
                    "location": loc,
                    "status": "Approved",
                }
                for i, (sym, loc) in enumerate(
                    [
                        ("AAA", "17q21.31"),
                        ("BBB", "Xp22.2"),
                        ("CCC", "MT"),
                        ("DDD", "unplaced"),
                        ("EEE", ""),
                    ]
                )
            ],
        )

        df = load_hgnc(path)

        assert list(df["chromosome"]) == ["17", "X", "MT", "?", "?"]

    def test_duplicate_ensembl_ids_keep_the_first(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [
                {
                    "ensembl_gene_id": "ENSG1",
                    "symbol": "FIRST",
                    "location": "1p36.33",
                    "status": "Approved",
                },
                {
                    "ensembl_gene_id": "ENSG1",
                    "symbol": "SECOND",
                    "location": "9q34.3",
                    "status": "Approved",
                },
            ],
        )

        df = load_hgnc(path)

        assert list(df.index) == ["ENSG1"]
        assert df.loc["ENSG1", "symbol"] == "FIRST"

    def test_rows_without_an_id_or_symbol_are_dropped(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [
                {
                    "ensembl_gene_id": "ENSG1",
                    "symbol": "AAA",
                    "location": "1p36.33",
                    "status": "Approved",
                },
                {
                    "ensembl_gene_id": "",
                    "symbol": "BBB",
                    "location": "2q11.2",
                    "status": "Approved",
                },
                {
                    "ensembl_gene_id": "ENSG3",
                    "symbol": "",
                    "location": "3p21.1",
                    "status": "Approved",
                },
            ],
        )

        df = load_hgnc(path)

        assert list(df.index) == ["ENSG1"]

    def test_missing_location_column_yields_question_marks_not_nan(
        self, tmp_path: Path
    ) -> None:
        # The frame is filtered before the chromosome column is derived, so its
        # index is sparse. A fallback built on a fresh RangeIndex aligned by
        # label and injected NaN for every surviving row outside that range —
        # which then crashed ``_chrom_sort_key`` in the interactome demo and
        # labelled nodes "chrnan" in the flow-field one.
        path = self._write(
            tmp_path,
            [
                {
                    "ensembl_gene_id": "ENSG1",
                    "symbol": "AAA",
                    "status": "Entry Withdrawn",
                },
                {"ensembl_gene_id": "ENSG2", "symbol": "BBB", "status": "Approved"},
                {"ensembl_gene_id": "ENSG3", "symbol": "CCC", "status": "Approved"},
            ],
        )

        df = load_hgnc(path)

        assert list(df.index) == ["ENSG2", "ENSG3"]
        assert list(df["chromosome"]) == ["?", "?"]
        assert not df["chromosome"].isna().any()


# =============================================================================
# load_huri_edges
# =============================================================================


class TestLoadHuriEdges:
    """The undirected, symbol-level edge frame built from the HuRI dump."""

    #: ``_hgnc`` indexes on ENSG{i:011d}, so these map to AAA / BBB / CCC.
    ENSG = ["ENSG00000000000", "ENSG00000000001", "ENSG00000000002"]
    HGNC = _hgnc({"AAA": "1", "BBB": "X", "CCC": "7"})

    @staticmethod
    def _write(
        tmp_path: Path, lines: list[tuple[str, str]], name: str = "huri.tsv"
    ) -> Path:
        """A headerless two-column HuRI-style TSV."""
        path = tmp_path / name
        path.write_text("".join(f"{a}\t{b}\n" for a, b in lines))
        return path

    def test_non_ensg_rows_are_dropped(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [
                (self.ENSG[0], self.ENSG[1]),
                ("not-a-gene", self.ENSG[1]),
                (self.ENSG[0], "also-not-a-gene"),
            ],
        )

        df = load_huri_edges(path, self.HGNC)

        assert len(df) == 1
        assert list(df.itertuples(index=False, name=None)) == [("AAA", "BBB")]

    def test_a_header_line_is_dropped_like_any_other_non_ensg_row(
        self, tmp_path: Path
    ) -> None:
        # The dump ships with and without a header; the ENSG prefix test is what
        # makes both work, so no ``header`` guessing is needed.
        with_header = self._write(
            tmp_path, [("gene_a", "gene_b"), (self.ENSG[0], self.ENSG[1])]
        )
        headerless = self._write(
            tmp_path, [(self.ENSG[0], self.ENSG[1])], name="bare.tsv"
        )

        # Anchored on the expected content, not only on the two frames agreeing:
        # an unconditionally empty result would satisfy the comparison alone.
        headed = load_huri_edges(with_header, self.HGNC)
        assert list(headed.itertuples(index=False, name=None)) == [("AAA", "BBB")]
        assert headed.equals(load_huri_edges(headerless, self.HGNC))

    def test_self_edges_are_dropped(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [(self.ENSG[0], self.ENSG[0]), (self.ENSG[0], self.ENSG[1])],
        )

        df = load_huri_edges(path, self.HGNC)

        assert list(df.itertuples(index=False, name=None)) == [("AAA", "BBB")]

    def test_pairs_with_an_unmapped_endpoint_are_dropped(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [
                (self.ENSG[0], self.ENSG[1]),
                (self.ENSG[0], "ENSG00000009999"),  # not in the HGNC table
            ],
        )

        df = load_huri_edges(path, self.HGNC)

        assert list(df.itertuples(index=False, name=None)) == [("AAA", "BBB")]

    def test_reversed_pairs_collapse_to_one_undirected_row(
        self, tmp_path: Path
    ) -> None:
        path = self._write(
            tmp_path,
            [(self.ENSG[1], self.ENSG[0]), (self.ENSG[0], self.ENSG[1])],
        )

        df = load_huri_edges(path, self.HGNC)

        # Canonicalized low→high, so the pair is stored once, sorted.
        assert list(df.itertuples(index=False, name=None)) == [("AAA", "BBB")]
        assert list(df.index) == [0]


# =============================================================================
# filter_to_lcc
# =============================================================================


class TestFilterToLcc:
    """Two components in, only the larger out."""

    @pytest.fixture(autouse=True)
    def _needs_networkx(self) -> None:
        pytest.importorskip("networkx")

    @staticmethod
    def _graph() -> tuple[Any, Any]:
        # LCC = {AAA, BBB, CCC} (triangle); {YYY, ZZZ} is the smaller component.
        edges = _edges(
            [("CCC", "BBB"), ("BBB", "AAA"), ("AAA", "CCC"), ("ZZZ", "YYY")],
            ("sym_a", "sym_b"),
        )
        # CCC is deliberately absent from HGNC -> chromosome "?".
        hgnc = _hgnc({"AAA": "1", "BBB": "X", "YYY": "7", "ZZZ": "MT"})
        return edges, hgnc

    def test_keeps_only_the_largest_component(self) -> None:
        edges, hgnc = self._graph()

        nodes, node_df, kept = filter_to_lcc(edges, hgnc)

        assert nodes == ["AAA", "BBB", "CCC"]
        assert list(node_df["symbol"]) == ["AAA", "BBB", "CCC"]

    def test_edges_outside_the_lcc_are_dropped(self) -> None:
        edges, hgnc = self._graph()

        _nodes, _node_df, kept = filter_to_lcc(edges, hgnc)

        assert len(kept) == 3
        assert set(kept["sym_a"]) | set(kept["sym_b"]) == {"AAA", "BBB", "CCC"}
        # The index is reset, so downstream positional lookups stay valid.
        assert list(kept.index) == [0, 1, 2]

    def test_node_order_is_sorted_and_deterministic(self) -> None:
        # Node order seeds the layout, so it must not follow set iteration.
        edges, hgnc = self._graph()

        first, _df, _e = filter_to_lcc(edges, hgnc)
        second, _df2, _e2 = filter_to_lcc(edges, hgnc)

        assert first == sorted(first)
        assert first == second

    def test_chromosomes_are_attached_with_question_mark_fallback(self) -> None:
        edges, hgnc = self._graph()

        _nodes, node_df, _kept = filter_to_lcc(edges, hgnc)

        assert list(node_df["chromosome"]) == ["1", "X", "?"]


# =============================================================================
# build_adjacency
# =============================================================================


class TestBuildAdjacency:
    @pytest.fixture(autouse=True)
    def _needs_scipy(self) -> None:
        pytest.importorskip("scipy")

    def test_symmetric_with_one_entry_per_undirected_edge(self) -> None:
        nodes = ["A", "B", "C"]
        edges = _edges([("A", "B"), ("B", "C")], ("sym_a", "sym_b"))

        mat = build_adjacency(nodes, edges, columns=("sym_a", "sym_b"))
        dense = mat.toarray()

        assert mat.shape == (3, 3)
        assert np.array_equal(dense, dense.T)
        assert np.array_equal(
            dense,
            np.array(
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32
            ),
        )
        assert mat.nnz == 4  # two edges, both directions

    def test_duplicate_edges_stay_unweighted(self) -> None:
        # sum_duplicates() would give 2.0 without the min-clamp.
        nodes = ["A", "B"]
        edges = _edges([("A", "B"), ("A", "B")], ("sym_a", "sym_b"))

        dense = build_adjacency(nodes, edges, columns=("sym_a", "sym_b")).toarray()

        assert dense.max() == 1.0

    def test_columns_argument_selects_the_endpoint_columns(self) -> None:
        # The AS-topology demo's frame is keyed on asn_a/asn_b, and carries a
        # third column the adjacency must ignore.
        nodes = ["1", "2", "3"]
        edges = _edges([("1", "2"), ("2", "3")], ("asn_a", "asn_b"))
        edges["rel"] = [-1, 0]

        by_asn = build_adjacency(nodes, edges, columns=("asn_a", "asn_b")).toarray()
        by_sym = build_adjacency(
            ["1", "2", "3"],
            _edges([("1", "2"), ("2", "3")], ("sym_a", "sym_b")),
            columns=("sym_a", "sym_b"),
        ).toarray()

        assert np.array_equal(by_asn, by_sym)

    def test_wrong_columns_raise_rather_than_silently_mislabel(self) -> None:
        nodes = ["1", "2"]
        edges = _edges([("1", "2")], ("asn_a", "asn_b"))

        with pytest.raises(KeyError):
            build_adjacency(nodes, edges, columns=("sym_a", "sym_b"))


# =============================================================================
# compute_communities
# =============================================================================


class TestComputeCommunities:
    @pytest.fixture(autouse=True)
    def _needs_networkx(self) -> None:
        pytest.importorskip("networkx")

    #: A 4-clique and a 3-clique, disjoint: Louvain can only find two.
    BIG = ["a1", "a2", "a3", "a4"]
    SMALL = ["b1", "b2", "b3"]

    @classmethod
    def _two_cliques(cls, columns: tuple[str, str] = ("sym_a", "sym_b")) -> Any:
        pairs = [
            (u, v)
            for group in (cls.BIG, cls.SMALL)
            for i, u in enumerate(group)
            for v in group[i + 1 :]
        ]
        return _edges(pairs, columns)

    def test_two_disjoint_cliques_land_in_two_communities(self) -> None:
        nodes = self.BIG + self.SMALL

        comms = compute_communities(
            nodes, self._two_cliques(), columns=("sym_a", "sym_b")
        )

        assert len(comms) == len(nodes)
        assert set(np.unique(comms)) == {0, 1}
        big_ids = {int(comms[nodes.index(n)]) for n in self.BIG}
        small_ids = {int(comms[nodes.index(n)]) for n in self.SMALL}
        assert len(big_ids) == 1 and len(small_ids) == 1
        assert big_ids != small_ids

    def test_communities_are_ordered_by_size_descending(self) -> None:
        nodes = self.BIG + self.SMALL

        comms = compute_communities(
            nodes, self._two_cliques(), columns=("sym_a", "sym_b")
        )

        # Community 0 must be the 4-clique, not whichever Louvain found first.
        assert {int(comms[nodes.index(n)]) for n in self.BIG} == {0}
        assert {int(comms[nodes.index(n)]) for n in self.SMALL} == {1}

    def test_result_is_deterministic_across_calls(self) -> None:
        # The pinned Louvain seed is what makes the demos' colors reproducible.
        nodes = self.BIG + self.SMALL
        edges = self._two_cliques()

        first = compute_communities(nodes, edges, columns=("sym_a", "sym_b"))
        second = compute_communities(nodes, edges, columns=("sym_a", "sym_b"))

        assert np.array_equal(first, second)

    def test_isolated_node_still_gets_a_community(self) -> None:
        # add_nodes_from(nodes) is what keeps a degree-0 node out of the -1 fill.
        nodes = self.BIG + self.SMALL + ["lonely"]

        comms = compute_communities(
            nodes, self._two_cliques(), columns=("sym_a", "sym_b")
        )

        assert comms.min() >= 0

    def test_columns_argument_selects_the_endpoint_columns(self) -> None:
        nodes = self.BIG + self.SMALL

        by_asn = compute_communities(
            nodes, self._two_cliques(("asn_a", "asn_b")), columns=("asn_a", "asn_b")
        )
        by_sym = compute_communities(
            nodes, self._two_cliques(), columns=("sym_a", "sym_b")
        )

        assert np.array_equal(by_asn, by_sym)

    def test_default_console_wording(self, capsys: pytest.CaptureFixture[str]) -> None:
        nodes = self.BIG + self.SMALL

        compute_communities(nodes, self._two_cliques(), columns=("sym_a", "sym_b"))

        out = capsys.readouterr().out
        assert "2 communities detected" in out
        assert "#0: 4 nodes" in out
        assert "#1: 3 nodes" in out

    def test_unit_label_and_summary_suffix_carry_the_caller_wording(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The AS-topology demo counts "ASes" and prints a bare "N communities".
        nodes = self.BIG + self.SMALL

        compute_communities(
            nodes,
            self._two_cliques(("asn_a", "asn_b")),
            columns=("asn_a", "asn_b"),
            unit_label="ASes",
            summary_suffix="",
        )

        out = capsys.readouterr().out
        assert "#0: 4 ASes" in out
        assert "nodes" not in out
        assert "2 communities" in out
        assert "communities detected" not in out


# =============================================================================
# nodes_hash
# =============================================================================


class TestNodesHash:
    def test_same_list_hashes_the_same(self) -> None:
        assert nodes_hash(["A", "B", "C"]) == nodes_hash(["A", "B", "C"])

    def test_different_nodes_hash_differently(self) -> None:
        # A collision here would serve a cached layout for the wrong node set.
        assert nodes_hash(["A", "B", "C"]) != nodes_hash(["A", "B", "D"])
        assert nodes_hash(["A", "B"]) != nodes_hash(["A", "B", "C"])

    def test_order_matters(self) -> None:
        # Coordinates are cached positionally, so a reordering invalidates them.
        assert nodes_hash(["A", "B"]) != nodes_hash(["B", "A"])

    def test_is_a_short_hex_digest(self) -> None:
        digest = nodes_hash(["A", "B"])
        assert len(digest) == 16
        assert all(c in "0123456789abcdef" for c in digest)


# =============================================================================
# build_community_legend
# =============================================================================


class TestBuildCommunityLegend:
    def test_lists_communities_largest_first_with_counts(self) -> None:
        communities = np.array([1, 1, 1, 0, 0, 2], dtype=np.int32)

        html = build_community_legend(communities)

        assert "Community (3)" in html
        assert "#1 (3)" in html
        assert html.index("#1 (3)") < html.index("#0 (2)") < html.index("#2 (1)")

    def test_top_n_truncates_with_a_more_line(self) -> None:
        communities = np.arange(5, dtype=np.int32)

        html = build_community_legend(communities, top_n=2)

        assert html.count("white-space:nowrap") == 2
        assert "... +3 more" in html
