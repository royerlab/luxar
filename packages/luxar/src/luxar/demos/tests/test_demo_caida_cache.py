"""Tests for the CAIDA AS-topology demo's warm-run caching.

Every warm run of this demo used to make two network GETs just to *discover*
which CAIDA snapshot is current, then re-parse both snapshot files, re-filter to
the largest connected component and re-run Louvain — ~12 s of deterministic
recompute, and impossible offline. These tests pin the caching that removes it:

* ``ensure_data`` — a fresh ``snapshots.json`` memo means NO network call at all
  (the acceptance criterion: a fully-cached run works with networking disabled);
  a stale memo or ``refresh=True`` re-discovers and rewrites the memo; a
  corrupt/invalid memo reads as absent. A discovery failure falls back on
  whatever COMPLETE snapshot pair is cached — the memo's if we have both of its
  files, else the newest pair on disk even with no memo at all (the state every
  user who ran this demo before the memo existed is in) — and propagates only
  when there is no complete pair, i.e. a genuine first run.
* ``load_pipeline`` — the derived bundle (parse → LCC + tier-1 → Louvain →
  degrees) round-trips through the cache to objects identical to the uncached
  chain's, down to dtypes and plain-``str`` edge columns, and a second call does
  not re-parse. Keyed on the two snapshot FILENAMES, so a new monthly release
  gets its own entry instead of clobbering the previous one.
* ``_prune_superseded_snapshots`` — superseded raw snapshots and their derived
  bundles are removed; the memo, the layout cache and unrecognised files are not.
* ``compute_layout`` — cached under a key that IS its input identity (node hash +
  edge count), so two node sets get two cache files and each is reused, and a
  rewired graph with the same nodes is not served last month's geometry.
* ``parse_as_org`` — the section markers CAIDA really emits carry NO space after
  the colon (``# format:org_id|…``). Matching only the spaced spelling yields an
  empty table and hence ``unknown``/``??`` for every AS — a silent failure the
  demo renders straight through, so the fixture below uses the real spelling and
  a second case pins that the spaced variant still parses.
* ``main()`` end to end on a warm cache — the issue's actual acceptance
  criterion: no network, no re-parse, no Louvain, no layout recompute. Plus the
  flag wiring (``--refresh-snapshots`` / ``--keep-snapshots`` /
  ``--recompute-pipeline``).

No network: ``requests.get`` is replaced in every test that could reach it, and
the warm-run tests (including the COLD half of the ``main()`` one, which needs
none either) make any call a hard failure. ``main()`` is exercised only with
``--no-serve`` (so no viewer is launched). The expensive layout body is
monkeypatched, so nothing here needs umap-learn.
"""

from __future__ import annotations

import bz2
import gzip
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from arbol import Arbol

# pandas ships in the ``demos`` extra, not core, and the demo needs it at import
# time — skip cleanly when it is absent (matches the sibling demo tests).
pd = pytest.importorskip("pandas")

from luxar.demos import demo_caida_as_topology as demo  # noqa: E402

REL_NAME = "20250401.as-rel2.txt.bz2"
ORG_NAME = "20250301.as-org2info.txt.gz"

#: Two triangles joined by one peer edge, plus a two-node component that the
#: LCC filter must drop. Provider-customer edges make AS1 and AS10 tier-1.
REL_LINES = [
    "1|2|-1",
    "1|3|-1",
    "2|3|0",
    "10|11|-1",
    "10|12|-1",
    "11|12|0",
    "3|10|0",
    "100|101|-1",
]

#: CAIDA's REAL section markers: ``# format:`` with no space after the colon.
#: Anything looser here would let the demo's detection regress unnoticed.
ORG_TEXT = """# format:org_id|changed|org_name|country|source
ORG-A|20250101|Alpha Networks|US|ARIN
ORG-B|20250101|Beta Telecom|DE|RIPE
# format:aut|changed|aut_name|org_id|opaque_id|source
1|20250101|AS1|ORG-A|x|ARIN
2|20250101|AS2|ORG-B|x|RIPE
3|20250101|AS3|ORG-A|x|ARIN
10|20250101|AS10|ORG-B|x|RIPE
11|20250101|AS11|ORG-A|x|ARIN
12|20250101|AS12|ORG-B|x|RIPE
100|20250101|AS100|ORG-A|x|ARIN
101|20250101|AS101|ORG-B|x|RIPE
"""


@pytest.fixture(autouse=True)
def _pin_arbol_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin Arbol's depth limit so console assertions don't depend on import order.

    ``Arbol.max_depth`` is global and every imported demo module sets it, so
    whichever ran last wins (same reason as ``test_graph_common``).
    """
    monkeypatch.setattr(Arbol, "max_depth", 100)


# =============================================================================
# Fixture builders (tiny but REAL snapshot files)
# =============================================================================


def _write_snapshot_pair(
    cache_dir: Path, rel_name: str = REL_NAME, org_name: str = ORG_NAME
) -> tuple[Path, Path]:
    """Write a tiny but genuinely-formatted ``.bz2`` / ``.gz`` snapshot pair."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    rel_path = cache_dir / rel_name
    org_path = cache_dir / org_name
    with bz2.open(rel_path, "wt", encoding="utf-8") as f:
        f.write("# a comment line\n")
        f.write("\n".join(REL_LINES) + "\n")
    with gzip.open(org_path, "wt", encoding="utf-8") as f:
        f.write(ORG_TEXT)
    return rel_path, org_path


def _write_memo(
    cache_dir: Path,
    checked_at: float,
    rel_name: str = REL_NAME,
    org_name: str = ORG_NAME,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / demo.SNAPSHOT_MEMO_FILENAME
    path.write_text(
        json.dumps({"as_rel": rel_name, "as_org": org_name, "checked_at": checked_at}),
        encoding="utf-8",
    )
    return path


class _IndexResponse:
    """Stand-in for the CAIDA directory-listing response ``_find_latest_file`` reads."""

    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _StreamResponse:
    """Stand-in for the streamed body ``download_file`` consumes."""

    def __init__(self, body: bytes) -> None:
        self._body = body
        self.headers = {"content-length": str(len(body))}

    def __enter__(self) -> "_StreamResponse":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 1 << 20) -> Any:
        yield self._body


def _patch_index(
    monkeypatch: pytest.MonkeyPatch,
    rel_names: list[str],
    org_names: list[str],
    *,
    file_body: bytes | None = None,
) -> list[str]:
    """Serve directory listings for the two index URLs; record the URLs hit.

    ``file_body`` additionally serves that payload for any snapshot-file URL, for
    the tests that must let a download actually happen.
    """
    seen: list[str] = []

    def fake_get(url: str, **_kwargs: Any) -> Any:
        seen.append(url)
        if url == demo.AS_REL_INDEX_URL:
            names = rel_names
        elif url == demo.AS_ORG_INDEX_URL:
            names = org_names
        elif file_body is not None:
            return _StreamResponse(file_body)
        else:  # pragma: no cover - a URL no test scripts
            raise AssertionError(f"unexpected GET {url}")
        return _IndexResponse("".join(f'<a href="{n}">{n}</a>' for n in names))

    monkeypatch.setattr(demo.requests, "get", fake_get)
    return seen


def _forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any HTTP call becomes a test failure.

    ``requests`` is one module object, so patching it through ``demo.requests``
    also covers ``_graph_common.download_file``.
    """

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("network access on a fully-cached run")

    monkeypatch.setattr(demo.requests, "get", _boom)


# =============================================================================
# ensure_data — snapshot-discovery memo
# =============================================================================


class TestEnsureDataMemo:
    def test_warm_run_makes_no_network_call_at_all(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # The acceptance criterion: both snapshots cached and a fresh memo means
        # a run that works with networking disabled.
        cache = tmp_path / "caida"
        rel_path, org_path = _write_snapshot_pair(cache)
        _write_memo(cache, checked_at=1_000_000.0)
        _forbid_network(monkeypatch)

        got_rel, got_org = demo.ensure_data(cache, now=1_000_000.0 + 3600)

        assert (got_rel, got_org) == (rel_path, org_path)
        out = capsys.readouterr().out
        assert "Skipped" in out
        assert REL_NAME in out and ORG_NAME in out

    def test_stale_memo_triggers_rediscovery_and_is_rewritten(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = tmp_path / "caida"
        new_rel = "20250501.as-rel2.txt.bz2"
        _write_snapshot_pair(cache, rel_name=new_rel)
        _write_memo(cache, checked_at=0.0)  # far older than the TTL
        seen = _patch_index(monkeypatch, [REL_NAME, new_rel], [ORG_NAME])

        now = demo.SNAPSHOT_DISCOVERY_TTL_SECONDS * 10.0
        rel_path, org_path = demo.ensure_data(cache, now=now)

        assert seen == [demo.AS_REL_INDEX_URL, demo.AS_ORG_INDEX_URL]
        assert rel_path.name == new_rel and org_path.name == ORG_NAME
        memo = demo._load_snapshot_memo(cache)
        assert memo == {"as_rel": new_rel, "as_org": ORG_NAME, "checked_at": now}

    def test_refresh_rediscovers_despite_a_fresh_memo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = tmp_path / "caida"
        new_rel = "20250501.as-rel2.txt.bz2"
        _write_snapshot_pair(cache, rel_name=new_rel)
        _write_memo(cache, checked_at=1_000_000.0)
        seen = _patch_index(monkeypatch, [REL_NAME, new_rel], [ORG_NAME])

        rel_path, _org = demo.ensure_data(cache, refresh=True, now=1_000_000.0 + 60)

        assert len(seen) == 2
        assert rel_path.name == new_rel
        assert demo._load_snapshot_memo(cache)["as_rel"] == new_rel

    def test_discovery_failure_falls_back_to_the_memoized_pair(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import requests

        cache = tmp_path / "caida"
        rel_path, org_path = _write_snapshot_pair(cache)
        _write_memo(cache, checked_at=0.0)  # STALE — discovery is attempted

        def offline(*_a: Any, **_k: Any) -> Any:
            raise requests.ConnectionError("name or service not known")

        monkeypatch.setattr(demo.requests, "get", offline)

        got_rel, got_org = demo.ensure_data(
            cache, now=demo.SNAPSHOT_DISCOVERY_TTL_SECONDS * 10.0
        )

        assert (got_rel, got_org) == (rel_path, org_path)
        out = capsys.readouterr().out
        assert "discovery failed" in out.lower()
        assert "memoized" in out.lower()
        # The fallback must NOT restamp the memo: doing so would freeze a
        # possibly-superseded pair for another whole TTL while looking healthy.
        assert demo._load_snapshot_memo(cache)["checked_at"] == 0.0

    def test_a_future_timestamp_is_not_treated_as_fresh(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A skewed clock (fresh VM, restored cache) would otherwise pin one
        # snapshot pair for as long as the stamp stays in the future.
        cache = tmp_path / "caida"
        _write_snapshot_pair(cache)
        _write_memo(cache, checked_at=5_000_000_000.0)
        seen = _patch_index(monkeypatch, [REL_NAME], [ORG_NAME])

        demo.ensure_data(cache, now=1_000_000.0)

        assert len(seen) == 2  # re-discovered rather than trusted
        assert demo._load_snapshot_memo(cache)["checked_at"] == 1_000_000.0

    @pytest.mark.parametrize("stamp", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_timestamp_reads_as_absent(
        self, tmp_path: Path, stamp: str
    ) -> None:
        # ``json`` parses these happily, and either would defeat the freshness
        # comparison (NaN makes every comparison False, inf pins forever).
        cache = tmp_path / "caida"
        cache.mkdir(parents=True)
        (cache / demo.SNAPSHOT_MEMO_FILENAME).write_text(
            '{"as_rel": "%s", "as_org": "%s", "checked_at": %s}'
            % (REL_NAME, ORG_NAME, stamp),
            encoding="utf-8",
        )

        assert demo._load_snapshot_memo(cache) is None

    @pytest.mark.parametrize("missing", ["rel", "org", "empty_rel"])
    def test_a_fresh_memo_with_a_missing_snapshot_falls_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
    ) -> None:
        # The memo is written BEFORE the downloads, so an interrupted first run
        # leaves a fresh memo naming a file that is absent (or zero-byte). The
        # shortcut must not skip discovery there — the network is needed anyway,
        # and a raw connection error out of ``download_file`` is a bad symptom.
        cache = tmp_path / "caida"
        rel_path, org_path = _write_snapshot_pair(cache)
        if missing == "rel":
            rel_path.unlink()
        elif missing == "org":
            org_path.unlink()
        else:
            rel_path.write_bytes(b"")  # a failed earlier download
        _write_memo(cache, checked_at=1_000_000.0)
        seen = _patch_index(
            monkeypatch, [REL_NAME], [ORG_NAME], file_body=b"refetched body"
        )

        got_rel, got_org = demo.ensure_data(cache, now=1_000_000.0 + 60)

        assert demo.AS_REL_INDEX_URL in seen and demo.AS_ORG_INDEX_URL in seen, (
            "discovery was skipped despite a missing snapshot"
        )
        assert got_rel.stat().st_size > 0 and got_org.stat().st_size > 0

    def test_offline_fallback_prefers_a_complete_local_pair(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # An interrupted run leaves a FRESH memo naming files it never finished
        # downloading. Offline, adopting those names hands back a raw
        # ConnectionError — even though the previous release sits complete on
        # disk and a fully warm offline run was possible.
        import requests

        cache = tmp_path / "caida"
        old_rel = "20250101.as-rel2.txt.bz2"
        old_org = "20250101.as-org2info.txt.gz"
        _write_snapshot_pair(cache, rel_name=old_rel, org_name=old_org)
        _write_memo(cache, checked_at=0.0, rel_name=REL_NAME, org_name=ORG_NAME)

        def offline(*_a: Any, **_k: Any) -> Any:
            raise requests.ConnectionError("name or service not known")

        monkeypatch.setattr(demo.requests, "get", offline)

        rel_path, org_path = demo.ensure_data(
            cache, now=demo.SNAPSHOT_DISCOVERY_TTL_SECONDS * 10.0
        )

        assert (rel_path.name, org_path.name) == (old_rel, old_org)
        out = capsys.readouterr().out
        assert "newest complete pair" in out
        assert old_rel in out

    def test_offline_run_with_a_cached_pair_and_no_memo_succeeds(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # The state EVERY pre-existing user is in: both snapshots cached from a
        # run before the memo existed, so there is no snapshots.json. Whether an
        # offline run works must depend on the DATA being cached, not on a memo
        # happening to exist — a stale memo naming these same two files works.
        import requests

        cache = tmp_path / "caida"
        rel_path, org_path = _write_snapshot_pair(cache)
        assert not (cache / demo.SNAPSHOT_MEMO_FILENAME).exists()

        def offline(*_a: Any, **_k: Any) -> Any:
            raise requests.ConnectionError("name or service not known")

        monkeypatch.setattr(demo.requests, "get", offline)

        got_rel, got_org = demo.ensure_data(cache, now=1_000.0)

        assert (got_rel, got_org) == (rel_path, org_path)
        out = capsys.readouterr().out
        assert "newest complete pair" in out

    def test_the_newest_rel_and_org_are_chosen_independently(
        self, tmp_path: Path
    ) -> None:
        # The two series have different cadences, so a live discovery round picks
        # each independently; the local fallback must match that.
        cache = tmp_path / "caida"
        cache.mkdir(parents=True)
        for name in (
            "20250101.as-rel2.txt.bz2",
            "20250401.as-rel2.txt.bz2",
            "20250201.as-org2info.txt.gz",
            "20250301.as-org2info.txt.gz",
        ):
            (cache / name).write_bytes(b"x")
        (cache / "20250501.as-rel2.txt.bz2").write_bytes(b"")  # incomplete: skipped

        assert demo._newest_complete_local_pair(cache) == (
            "20250401.as-rel2.txt.bz2",
            "20250301.as-org2info.txt.gz",
        )

    def test_no_complete_local_pair_keeps_the_memo_names(self, tmp_path: Path) -> None:
        # Only one half on disk is not a usable pair, so the memo names stand and
        # the "needs network" download path still applies.
        cache = tmp_path / "caida"
        cache.mkdir(parents=True)
        (cache / "20250101.as-rel2.txt.bz2").write_bytes(b"x")

        assert demo._newest_complete_local_pair(cache) is None
        assert demo._newest_complete_local_pair(tmp_path / "absent") is None

    def test_an_unwritable_cache_dir_does_not_abort_the_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A read-only or other-user-owned --cache-dir still serves a fine run;
        # failing to record the memo must not throw away completed downloads.
        import os

        if os.geteuid() == 0:  # pragma: no cover - root ignores mode bits
            pytest.skip("root can write to a read-only directory")

        cache = tmp_path / "caida"
        rel_path, org_path = _write_snapshot_pair(cache)
        _patch_index(monkeypatch, [REL_NAME], [ORG_NAME])
        cache.chmod(0o500)
        try:
            got_rel, got_org = demo.ensure_data(cache, now=1_000.0)
        finally:
            cache.chmod(0o700)

        assert (got_rel, got_org) == (rel_path, org_path)
        assert not (cache / demo.SNAPSHOT_MEMO_FILENAME).exists()
        out = capsys.readouterr().out
        assert demo.SNAPSHOT_MEMO_FILENAME in out
        assert "could not write" in out.lower()

    @pytest.mark.parametrize("seeded", ["nothing", "rel_only"])
    def test_discovery_failure_with_no_complete_pair_propagates(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        seeded: str,
    ) -> None:
        # A genuine first run (or a half-populated cache): there is nothing to
        # fall back on, so the network error must surface with the useful message.
        import requests

        cache = tmp_path / "caida"
        cache.mkdir(parents=True)
        if seeded == "rel_only":
            (cache / REL_NAME).write_bytes(b"one half is not a pair")

        def offline(*_a: Any, **_k: Any) -> Any:
            raise requests.ConnectionError("name or service not known")

        monkeypatch.setattr(demo.requests, "get", offline)

        with pytest.raises(requests.ConnectionError):
            demo.ensure_data(cache, now=1_000.0)

        assert "needs network access" in capsys.readouterr().out

    def test_empty_index_without_a_memo_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``_find_latest_file`` raises RuntimeError when the listing has no match;
        # with no memo there is nothing to fall back on.
        cache = tmp_path / "caida"
        _patch_index(monkeypatch, [], [])

        with pytest.raises(RuntimeError, match="No files matching"):
            demo.ensure_data(cache, now=1_000.0)

    @pytest.mark.parametrize(
        "body",
        [
            "not json at all",
            json.dumps(["a", "list"]),
            json.dumps({"as_rel": REL_NAME, "as_org": ORG_NAME}),  # no checked_at
            json.dumps({"as_rel": "garbage.txt", "as_org": ORG_NAME, "checked_at": 1}),
            json.dumps({"as_rel": REL_NAME, "as_org": REL_NAME, "checked_at": 1}),
            json.dumps({"as_rel": REL_NAME, "as_org": ORG_NAME, "checked_at": "now"}),
        ],
    )
    def test_invalid_memo_reads_as_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str
    ) -> None:
        cache = tmp_path / "caida"
        _write_snapshot_pair(cache)
        (cache / demo.SNAPSHOT_MEMO_FILENAME).write_text(body, encoding="utf-8")

        assert demo._load_snapshot_memo(cache) is None

        # ...and therefore discovery is attempted rather than a bogus name used.
        seen = _patch_index(monkeypatch, [REL_NAME], [ORG_NAME])
        rel_path, org_path = demo.ensure_data(cache, now=1_234.0)

        assert len(seen) == 2
        assert (rel_path.name, org_path.name) == (REL_NAME, ORG_NAME)
        assert demo._load_snapshot_memo(cache) is not None

    def test_missing_memo_file_reads_as_absent(self, tmp_path: Path) -> None:
        assert demo._load_snapshot_memo(tmp_path / "nope") is None


# =============================================================================
# load_pipeline — the derived bundle
# =============================================================================


class TestLoadPipeline:
    @pytest.fixture(autouse=True)
    def _needs_networkx(self) -> None:
        pytest.importorskip("networkx")

    @staticmethod
    def _uncached(rel_path: Path, org_path: Path) -> tuple[Any, ...]:
        """The chain ``main()`` used to run inline, for comparison."""
        org_df = demo.parse_as_org(org_path)
        edges_raw = demo.parse_as_rel(rel_path)
        nodes, node_df, edges, tier1 = demo.filter_to_lcc(edges_raw, org_df)
        communities = demo.compute_communities(
            nodes,
            edges,
            columns=("asn_a", "asn_b"),
            unit_label="ASes",
            summary_suffix="",
        )
        degrees = demo._node_degrees(nodes, edges)
        return nodes, node_df, edges, tier1, communities, degrees

    def test_bundle_round_trip_matches_the_uncached_chain(self, tmp_path: Path) -> None:
        cache = tmp_path / "caida"
        rel_path, org_path = _write_snapshot_pair(cache)
        exp_nodes, exp_node_df, exp_edges, exp_t1, exp_comms, exp_deg = self._uncached(
            rel_path, org_path
        )

        # The synthetic graph is non-trivial: two triangles joined by a peer
        # edge, one dropped component, two tier-1s.
        assert exp_nodes == ["1", "2", "3", "10", "11", "12"]
        assert list(exp_t1) == [True, False, False, True, False, False]

        def check(result: tuple[Any, ...], label: str) -> None:
            nodes, node_df, edges, tier1, comms, degrees = result

            assert nodes == exp_nodes, label
            assert all(type(a) is str for a in nodes), label
            np.testing.assert_array_equal(tier1, exp_t1)
            np.testing.assert_array_equal(comms, exp_comms)
            np.testing.assert_array_equal(degrees, exp_deg)
            assert comms.dtype == exp_comms.dtype == np.int32, label
            assert degrees.dtype == exp_deg.dtype == np.int32, label
            assert tier1.dtype == np.bool_, label

            assert list(node_df.columns) == list(exp_node_df.columns), label
            for col in exp_node_df.columns:
                assert list(node_df[col]) == list(exp_node_df[col]), (col, label)
            # The REAL org names and countries must reach the nodes. If the
            # section markers stopped matching, every row here would be
            # "unknown"/"??" — silent in the render, so assert it somewhere.
            assert list(node_df["org_name"]) == [
                "Alpha Networks",
                "Beta Telecom",
                "Alpha Networks",
                "Beta Telecom",
                "Alpha Networks",
                "Beta Telecom",
            ], label
            assert list(node_df["country"]) == ["US", "DE", "US", "DE", "US", "DE"]
            assert all(type(v) is str for v in node_df["org_name"]), label

            assert list(edges.columns) == ["asn_a", "asn_b", "rel"], label
            assert list(edges["asn_a"]) == list(exp_edges["asn_a"]), label
            assert list(edges["asn_b"]) == list(exp_edges["asn_b"]), label
            assert list(edges["rel"]) == list(exp_edges["rel"]), label
            assert list(edges.index) == list(range(len(exp_edges))), label
            # Plain Python str, so ``idx[a]`` / ``value_counts`` / ``isin`` and
            # the hover-label formatting behave exactly as before.
            assert all(type(v) is str for v in edges["asn_a"]), label
            assert all(type(v) is str for v in edges["asn_b"]), label
            # An int column whose values compare with the -1 / 0 masks downstream.
            rel = edges["rel"].to_numpy()
            assert rel.dtype.kind == "i", label
            assert set(np.unique(rel)) <= {-1, 0}, label
            assert int((rel == -1).sum()) == 4 and int((rel == 0).sum()) == 3, label

        # A cache MISS returns the freshly built value, which never went through
        # pickle — so the same assertions have to hold for the cache-SERVED one.
        check(demo.load_pipeline(rel_path, org_path, cache), "fresh")
        assert list(cache.glob("pipeline_*.pkl")), "nothing was cached"
        check(demo.load_pipeline(rel_path, org_path, cache), "unpickled")

    def test_second_call_reuses_the_cache_without_reparsing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = tmp_path / "caida"
        rel_path, org_path = _write_snapshot_pair(cache)
        first = demo.load_pipeline(rel_path, org_path, cache)

        def exploding(_path: Path) -> Any:
            raise AssertionError("re-parsed a snapshot that was already bundled")

        monkeypatch.setattr(demo, "parse_as_rel", exploding)

        second = demo.load_pipeline(rel_path, org_path, cache)

        assert second[0] == first[0]
        np.testing.assert_array_equal(second[4], first[4])

        # recompute=True does go back to the builder.
        with pytest.raises(AssertionError, match="re-parsed"):
            demo.load_pipeline(rel_path, org_path, cache, recompute=True)

    def test_a_new_snapshot_pair_gets_its_own_cache_entry(self, tmp_path: Path) -> None:
        # Monthly rollover: the new pair must recompute, and the old pair's
        # bundle must survive (keyed on filenames, not overwritten).
        cache = tmp_path / "caida"
        rel_a, org_a = _write_snapshot_pair(cache)
        rel_b, org_b = _write_snapshot_pair(
            cache, rel_name="20250501.as-rel2.txt.bz2", org_name=ORG_NAME
        )

        demo.load_pipeline(rel_a, org_a, cache)
        pkl_a = {p.name for p in cache.glob("pipeline_*.pkl")}
        demo.load_pipeline(rel_b, org_b, cache)
        pkl_ab = {p.name for p in cache.glob("pipeline_*.pkl")}

        assert len(pkl_a) == 1
        assert pkl_a < pkl_ab and len(pkl_ab) == 2
        assert any(rel_a.name in n for n in pkl_ab)
        assert any(rel_b.name in n for n in pkl_ab)

    def test_a_new_org_snapshot_alone_also_gets_its_own_entry(
        self, tmp_path: Path
    ) -> None:
        # The common real case: as-org2info is republished on its own slower
        # cadence, so only the second half of the key changes.
        cache = tmp_path / "caida"
        rel_a, org_a = _write_snapshot_pair(cache)
        new_org = "20250501.as-org2info.txt.gz"
        _rel_b, org_b = _write_snapshot_pair(cache, rel_name=REL_NAME, org_name=new_org)

        demo.load_pipeline(rel_a, org_a, cache)
        demo.load_pipeline(rel_a, org_b, cache)

        names = {p.name for p in cache.glob("pipeline_*.pkl")}
        assert len(names) == 2
        assert any(org_a.name in n for n in names)
        assert any(new_org in n for n in names)


# =============================================================================
# parse_as_org — the section markers CAIDA really writes
# =============================================================================


class TestParseAsOrgSections:
    """A marker mismatch is silent: an empty table renders as unknown/??."""

    @staticmethod
    def _parse(tmp_path: Path, text: str) -> Any:
        path = tmp_path / "20250301.as-org2info.txt.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(text)
        return demo.parse_as_org(path)

    def test_real_unspaced_markers_parse(self, tmp_path: Path) -> None:
        df = self._parse(tmp_path, ORG_TEXT)

        assert df.loc["1", "org_name"] == "Alpha Networks"
        assert df.loc["1", "country"] == "US"
        assert df.loc["2", "country"] == "DE"

    def test_spaced_markers_still_parse(self, tmp_path: Path) -> None:
        # Mirrors / older copies write "# format: org_id"; both must work, so a
        # future edit cannot fix one spelling by breaking the other.
        spaced = ORG_TEXT.replace("# format:", "# format: ")
        assert "# format: org_id" in spaced

        df = self._parse(tmp_path, spaced)

        assert df.loc["1", "org_name"] == "Alpha Networks"
        assert df.loc["1", "country"] == "US"

    def test_an_unmatched_marker_yields_an_empty_table(self, tmp_path: Path) -> None:
        # The anti-case, pinning what "silent" means: no recognisable section
        # header at all and every row is skipped.
        df = self._parse(tmp_path, ORG_TEXT.replace("# format:", "# columns:"))

        assert len(df) == 0


# =============================================================================
# _prune_superseded_snapshots
# =============================================================================


class TestPruneSupersededSnapshots:
    DATES = ["20250101", "20250201", "20250301", "20250401"]

    def _populate(self, cache: Path) -> None:
        cache.mkdir(parents=True, exist_ok=True)
        for date in self.DATES:
            (cache / f"{date}.as-rel2.txt.bz2").write_bytes(b"rel" * 100)
            (cache / f"{date}.as-org2info.txt.gz").write_bytes(b"org" * 100)
            (
                cache
                / f"pipeline_{date}.as-rel2.txt.bz2_{date}.as-org2info.txt.gz_v1.pkl"
            ).write_bytes(b"bundle")
        _write_memo(cache, checked_at=1.0)
        (cache / "layout3d_deadbeefdeadbeef_v1.pkl").write_bytes(b"coords")
        (cache / "notes.txt").write_bytes(b"unrecognised, keep")

    def test_keeps_the_newest_releases_and_the_current_pair(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cache = tmp_path / "caida"
        self._populate(cache)

        demo._prune_superseded_snapshots(
            cache,
            keep=2,
            current=("20250401.as-rel2.txt.bz2", "20250301.as-org2info.txt.gz"),
        )

        names = {p.name for p in cache.iterdir()}
        for date in ("20250301", "20250401"):
            assert f"{date}.as-rel2.txt.bz2" in names
            assert f"{date}.as-org2info.txt.gz" in names
        for date in ("20250101", "20250201"):
            assert f"{date}.as-rel2.txt.bz2" not in names
            assert f"{date}.as-org2info.txt.gz" not in names
            # The derived bundle of a removed snapshot goes with it.
            assert not any(date in n for n in names)
        # Never the memo, never a layout cache, never an unrecognised file.
        assert demo.SNAPSHOT_MEMO_FILENAME in names
        assert "layout3d_deadbeefdeadbeef_v1.pkl" in names
        assert "notes.txt" in names

        out = capsys.readouterr().out
        assert "20250101.as-rel2.txt.bz2" in out
        assert "Reclaimed" in out

    def test_an_older_current_file_is_kept_even_below_the_keep_window(
        self, tmp_path: Path
    ) -> None:
        # as-org2info is released less often than as-rel2, so the pair in use can
        # name a date outside the newest ``keep``. Deleting it would break the run.
        cache = tmp_path / "caida"
        self._populate(cache)

        demo._prune_superseded_snapshots(
            cache,
            keep=1,
            current=("20250401.as-rel2.txt.bz2", "20250101.as-org2info.txt.gz"),
        )

        names = {p.name for p in cache.iterdir()}
        assert "20250101.as-org2info.txt.gz" in names
        assert "20250101.as-rel2.txt.bz2" in names  # same date group
        assert "20250401.as-rel2.txt.bz2" in names
        assert "20250201.as-rel2.txt.bz2" not in names
        assert "20250301.as-rel2.txt.bz2" not in names

    def test_keep_below_one_is_clamped_to_one(self, tmp_path: Path) -> None:
        # ``current`` is deliberately an OLDER pair, so keep=0 (clamped to 1 →
        # newest date + current) and keep=2 give different answers: without the
        # clamp, ``sorted(...)[:0]`` would be empty and 20250401 would be deleted.
        cache = tmp_path / "caida"
        self._populate(cache)

        demo._prune_superseded_snapshots(
            cache,
            keep=0,
            current=("20250101.as-rel2.txt.bz2", "20250101.as-org2info.txt.gz"),
        )

        names = {p.name for p in cache.iterdir()}
        assert "20250401.as-rel2.txt.bz2" in names  # the newest, via the clamp
        assert "20250101.as-rel2.txt.bz2" in names  # the current pair
        assert "20250201.as-rel2.txt.bz2" not in names
        assert "20250301.as-rel2.txt.bz2" not in names

    def test_derived_bundle_naming_either_date_is_removed(self, tmp_path: Path) -> None:
        # A real bundle names the rel and org files it was built from, and their
        # dates differ. It is unusable once EITHER raw file goes.
        cache = tmp_path / "caida"
        self._populate(cache)
        crossed = (
            cache / "pipeline_20250401.as-rel2.txt.bz2_"
            "20250101.as-org2info.txt.gz_v1.pkl"
        )
        crossed.write_bytes(b"bundle")

        demo._prune_superseded_snapshots(
            cache,
            keep=2,
            current=("20250401.as-rel2.txt.bz2", "20250401.as-org2info.txt.gz"),
        )

        # 20250401 survives, 20250101 does not — so the crossed bundle goes.
        assert (cache / "20250401.as-rel2.txt.bz2").exists()
        assert not crossed.exists()

    def test_quarantined_and_temp_bundle_siblings_are_removed(
        self, tmp_path: Path
    ) -> None:
        # ``cache_computed`` can leave a ``.pkl.corrupt`` (quarantine) or a
        # ``.pkl.tmp`` (interrupted atomic write) — each a full-size bundle.
        cache = tmp_path / "caida"
        self._populate(cache)
        stem = "pipeline_20250101.as-rel2.txt.bz2_20250101.as-org2info.txt.gz_v1.pkl"
        (cache / f"{stem}.corrupt").write_bytes(b"quarantined")
        (cache / f"{stem}.tmp").write_bytes(b"interrupted")

        demo._prune_superseded_snapshots(
            cache,
            keep=2,
            current=("20250401.as-rel2.txt.bz2", "20250401.as-org2info.txt.gz"),
        )

        assert not any("20250101" in p.name for p in cache.iterdir())

    def test_a_bundle_with_both_dates_superseded_is_removed_once(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Collected once per superseded date, so without a dedup the second
        # unlink prints a spurious "Skipped … No such file" and the cleanup
        # looks like it failed.
        cache = tmp_path / "caida"
        self._populate(cache)
        both = (
            cache / "pipeline_20250101.as-rel2.txt.bz2_"
            "20250201.as-org2info.txt.gz_v1.pkl"
        )
        both.write_bytes(b"bundle")

        demo._prune_superseded_snapshots(
            cache,
            keep=2,
            current=("20250401.as-rel2.txt.bz2", "20250401.as-org2info.txt.gz"),
        )

        assert not both.exists()
        out = capsys.readouterr().out
        assert "Skipped" not in out
        assert out.count(both.name) == 1

    def test_another_demos_layout_npz_is_never_touched(self, tmp_path: Path) -> None:
        # ``--cache-dir`` can point us at a directory another demo owns, and
        # ``demo_huri_interactome`` keeps its LIVE layout cache in exactly this
        # file. Pruning anything we do not recognise would destroy it.
        cache = tmp_path / "caida"
        self._populate(cache)
        foreign = cache / "layout_3d.npz"
        foreign.write_bytes(b"another demo's warm layout")

        demo._prune_superseded_snapshots(
            cache,
            keep=2,
            current=("20250401.as-rel2.txt.bz2", "20250401.as-org2info.txt.gz"),
        )

        assert foreign.read_bytes() == b"another demo's warm layout"

    def test_a_file_that_vanishes_underfoot_does_not_abort_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A concurrent second copy of the demo can unlink the same victim. Dying
        # here would waste a run that has already paid for its download.
        cache = tmp_path / "caida"
        self._populate(cache)
        real_unlink = Path.unlink
        hit = {"n": 0}

        def flaky_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
            hit["n"] += 1
            if hit["n"] == 1:
                raise FileNotFoundError(f"gone: {self.name}")
            real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", flaky_unlink)

        demo._prune_superseded_snapshots(
            cache,
            keep=2,
            current=("20250401.as-rel2.txt.bz2", "20250401.as-org2info.txt.gz"),
        )

        # The rest of the sweep still happened.
        assert hit["n"] > 1
        assert not any("20250201" in p.name for p in cache.iterdir())

    def test_nothing_to_prune_is_silent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cache = tmp_path / "caida"
        _write_snapshot_pair(cache)
        _write_memo(cache, checked_at=1.0)

        demo._prune_superseded_snapshots(cache, keep=2, current=(REL_NAME, ORG_NAME))
        demo._prune_superseded_snapshots(  # a cache dir that does not exist
            tmp_path / "absent", keep=2, current=(REL_NAME, ORG_NAME)
        )

        assert (cache / REL_NAME).exists() and (cache / ORG_NAME).exists()
        assert "Pruning" not in capsys.readouterr().out


# =============================================================================
# compute_layout — cached on the node hash
# =============================================================================


class TestComputeLayoutCache:
    def test_two_node_sets_get_two_cache_files_and_each_is_reused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = tmp_path / "caida"
        edges = pd.DataFrame({"asn_a": ["1"], "asn_b": ["2"], "rel": [0]})
        calls: list[int] = []

        def fake_coords(nodes: list[str], _edges: Any) -> np.ndarray:
            calls.append(len(nodes))
            return np.full((len(nodes), 3), len(nodes), dtype=np.float32)

        monkeypatch.setattr(demo, "_compute_layout_coords", fake_coords)

        a = ["1", "2"]
        b = ["1", "2", "3"]
        first_a = demo.compute_layout(a, edges, cache_dir=cache, recompute=False)
        again_a = demo.compute_layout(a, edges, cache_dir=cache, recompute=False)
        first_b = demo.compute_layout(b, edges, cache_dir=cache, recompute=False)

        assert calls == [2, 3]  # one computation per node set, no recompute for a
        np.testing.assert_array_equal(first_a, again_a)
        assert first_a.shape == (2, 3) and first_b.shape == (3, 3)

        from luxar.demos._graph_common import nodes_hash

        names = {p.name for p in cache.glob("layout3d_*.pkl")}
        assert names == {
            f"layout3d_{nodes_hash(a)}_e1_v{demo.LAYOUT_CACHE_VERSION}.pkl",
            f"layout3d_{nodes_hash(b)}_e1_v{demo.LAYOUT_CACHE_VERSION}.pkl",
        }

        # recompute=True bypasses the cache for that node set only.
        demo.compute_layout(a, edges, cache_dir=cache, recompute=True)
        assert calls == [2, 3, 2]

    def test_a_changed_edge_set_is_not_served_last_months_geometry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The layout is computed from the ADJACENCY. A release whose LCC keeps the
        # same node set but rewires it must not silently reuse the old coords, so
        # the edge count is part of the key.
        cache = tmp_path / "caida"
        nodes = ["1", "2", "3"]
        few = pd.DataFrame({"asn_a": ["1"], "asn_b": ["2"], "rel": [0]})
        more = pd.DataFrame({"asn_a": ["1", "2"], "asn_b": ["2", "3"], "rel": [0, -1]})
        calls: list[int] = []

        def fake_coords(ns: list[str], es: Any) -> np.ndarray:
            calls.append(len(es))
            return np.full((len(ns), 3), len(es), dtype=np.float32)

        monkeypatch.setattr(demo, "_compute_layout_coords", fake_coords)

        first = demo.compute_layout(nodes, few, cache_dir=cache, recompute=False)
        second = demo.compute_layout(nodes, more, cache_dir=cache, recompute=False)
        again = demo.compute_layout(nodes, few, cache_dir=cache, recompute=False)

        assert calls == [1, 2]  # recomputed for the new edge set, then reused
        assert first[0, 0] == 1.0 and second[0, 0] == 2.0
        np.testing.assert_array_equal(first, again)
        assert len({p.name for p in cache.glob("layout3d_*.pkl")}) == 2


# =============================================================================
# main() — the acceptance criteria, end to end
# =============================================================================


class TestMainWarmRun:
    """What the issue actually asks for, asserted through the entry point.

    Pinning the pieces individually is not enough: ``main()`` could go on running
    the old inline chain and every other test here would stay green.
    """

    @pytest.fixture(autouse=True)
    def _needs_networkx(self) -> None:
        pytest.importorskip("networkx")

    @staticmethod
    def _fake_coords(nodes: list[str], _edges: Any) -> np.ndarray:
        """Stand-in for the spectral+UMAP body (keeps umap-learn out of the test)."""
        rng = np.random.default_rng(0)
        return rng.standard_normal((len(nodes), 3)).astype(np.float32)

    def _argv(self, cache: Path) -> list[str]:
        return ["demo_caida_as_topology", "--no-serve", "--cache-dir", str(cache)]

    def test_second_run_needs_no_network_and_recomputes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time as _time

        cache = tmp_path / "caida"
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _write_snapshot_pair(cache)
        _write_memo(cache, checked_at=_time.time())

        monkeypatch.setattr(demo, "get_demos_output_dir", lambda: out_dir)
        monkeypatch.setattr(demo, "_compute_layout_coords", self._fake_coords)
        monkeypatch.setattr(sys, "argv", self._argv(cache))
        # Forbidden for BOTH runs: the cold one needs no network either (fresh
        # memo + both snapshots on disk), and a cold run that quietly reached
        # publicdata.caida.org would download 6 MB, run a real 80k-node Louvain,
        # and let the warm assertions below pass for the wrong reason.
        _forbid_network(monkeypatch)

        # First (cold) run populates both derived caches.
        demo.main()
        scene = out_dir / "caida_as_topology.luxar.zarr"
        assert scene.exists()
        assert list(cache.glob("pipeline_*.pkl")) and list(cache.glob("layout3d_*.pkl"))

        # Second run: nothing may re-parse, re-run Louvain or recompute the
        # layout either.
        def _boom(name: str) -> Any:
            def fail(*_a: Any, **_k: Any) -> Any:
                raise AssertionError(f"{name} ran on a fully-warm run")

            return fail

        monkeypatch.setattr(demo, "parse_as_rel", _boom("parse_as_rel"))
        monkeypatch.setattr(demo, "parse_as_org", _boom("parse_as_org"))
        monkeypatch.setattr(demo, "compute_communities", _boom("compute_communities"))
        monkeypatch.setattr(
            demo, "_compute_layout_coords", _boom("_compute_layout_coords")
        )
        shutil.rmtree(scene)

        demo.main()

        assert scene.exists()

    def test_flags_reach_the_helpers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = tmp_path / "caida"
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        rel_path, org_path = _write_snapshot_pair(cache)
        seen: dict[str, Any] = {}

        def fake_ensure(cache_dir: Path, *, refresh: bool = False, **_k: Any) -> Any:
            seen["refresh"] = refresh
            seen["cache_dir"] = cache_dir
            return rel_path, org_path

        def fake_prune(_cache_dir: Path, *, keep: int, current: Any) -> None:
            seen["keep"] = keep
            seen["current"] = current

        def fake_load(
            _rel: Path, _org: Path, _cache: Path, *, recompute: bool = False
        ) -> Any:
            seen["recompute_pipeline"] = recompute
            nodes = ["1", "2"]
            node_df = pd.DataFrame(
                {
                    "asn": nodes,
                    "org_name": ["A", "B"],
                    "country": ["US", "DE"],
                    "tier1": np.array([True, False]),
                }
            )
            edges = pd.DataFrame({"asn_a": ["1"], "asn_b": ["2"], "rel": [0]})
            return (
                nodes,
                node_df,
                edges,
                np.array([True, False]),
                np.zeros(2, np.int32),
                np.ones(2, np.int32),
            )

        def fake_layout(
            nodes: list[str], _edges: Any, cache_dir: Path, recompute: bool
        ) -> np.ndarray:
            seen["recompute_layout"] = recompute
            return np.zeros((len(nodes), 3), dtype=np.float32)

        monkeypatch.setattr(demo, "get_demos_output_dir", lambda: out_dir)
        monkeypatch.setattr(demo, "ensure_data", fake_ensure)
        monkeypatch.setattr(demo, "_prune_superseded_snapshots", fake_prune)
        monkeypatch.setattr(demo, "load_pipeline", fake_load)
        monkeypatch.setattr(demo, "compute_layout", fake_layout)

        # Defaults first.
        monkeypatch.setattr(sys, "argv", self._argv(cache))
        demo.main()
        assert seen["cache_dir"] == cache
        assert seen["refresh"] is False
        assert seen["recompute_pipeline"] is False
        assert seen["recompute_layout"] is False
        assert seen["keep"] == demo.DEFAULT_KEEP_SNAPSHOTS
        assert seen["current"] == (rel_path.name, org_path.name)

        # ...then every flag at once.
        monkeypatch.setattr(
            sys,
            "argv",
            self._argv(cache)
            + [
                "--refresh-snapshots",
                "--recompute-pipeline",
                "--recompute-layout",
                "--keep-snapshots",
                "5",
            ],
        )
        demo.main()
        assert seen["refresh"] is True
        assert seen["recompute_pipeline"] is True
        assert seen["recompute_layout"] is True
        assert seen["keep"] == 5
