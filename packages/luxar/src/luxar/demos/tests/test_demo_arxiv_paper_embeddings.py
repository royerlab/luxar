"""Regression tests for ``search_papers_by_field`` citation-count filtering.

The Semantic Scholar API can return an explicit JSON ``null`` for
``citationCount``. ``paper.get("citationCount", 0)`` only substitutes the
default when the KEY IS MISSING; a present ``None`` value slips through and
``None >= min_citations`` raises ``TypeError``. That crash is not caught by the
surrounding ``except requests.exceptions.RequestException`` handler, so it
aborts a fresh (uncached) demo run. These tests pin the null-safe behaviour:
null, missing, and explicit ``0`` are all treated as zero citations.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
import pytest

from luxar.demos import demo_arxiv_paper_embeddings as demo
from luxar.demos.demo_arxiv_paper_embeddings import (
    generate_paper_landscape,
    search_papers_by_field,
)


class _FakeResponse:
    """Minimal stand-in for ``requests.Response`` used by the demo."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:  # no-op: always "200 OK"
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def _install_fake_get(monkeypatch, first_payload: dict[str, Any]) -> None:
    """Patch ``requests.get`` to return ``first_payload`` once, then drain.

    The demo loops ``while len(papers) < limit`` and paginates via ``offset``,
    only stopping when the API returns an empty/short ``data`` list (or the
    limit is reached). Returning the fixture on the first call and an empty
    ``{"data": []}`` afterwards makes the loop terminate immediately. ``sleep``
    is also stubbed so the 0.5s inter-request pause never runs.
    """
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):  # noqa: ANN001 - test stub
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(first_payload)
        return _FakeResponse({"data": []})

    monkeypatch.setattr(demo.requests, "get", fake_get)
    monkeypatch.setattr(demo.time, "sleep", lambda *_args, **_kwargs: None)


def test_null_citation_count_does_not_crash_and_counts_as_zero(monkeypatch) -> None:
    """A paper with ``citationCount: None`` must not raise and reads as 0."""
    payload = {
        "data": [
            {
                "paperId": "p-null",
                "title": "Null citation paper",
                "abstract": "A real abstract.",
                "citationCount": None,
            }
        ]
    }
    _install_fake_get(monkeypatch, payload)

    # min_citations=0: a zero-citation paper is included (no TypeError).
    papers = search_papers_by_field("cs", limit=10, min_citations=0)
    assert [p["paperId"] for p in papers] == ["p-null"]


def test_null_citation_count_excluded_above_threshold(monkeypatch) -> None:
    """Treated as 0, a null-citation paper fails a positive threshold."""
    payload = {
        "data": [
            {
                "paperId": "p-null",
                "title": "Null citation paper",
                "abstract": "A real abstract.",
                "citationCount": None,
            }
        ]
    }
    _install_fake_get(monkeypatch, payload)

    papers = search_papers_by_field("cs", limit=10, min_citations=1)
    assert papers == []


def test_missing_zero_and_real_counts_are_consistent(monkeypatch) -> None:
    """Missing key and explicit 0 both read as zero; a real count passes."""
    payload = {
        "data": [
            {
                "paperId": "p-missing",
                "title": "Missing citationCount",
                "abstract": "Abstract A.",
                # citationCount key intentionally absent
            },
            {
                "paperId": "p-zero",
                "title": "Explicit zero",
                "abstract": "Abstract B.",
                "citationCount": 0,
            },
            {
                "paperId": "p-high",
                "title": "Well cited",
                "abstract": "Abstract C.",
                "citationCount": 42,
            },
        ]
    }
    _install_fake_get(monkeypatch, payload)

    # min_citations=1 excludes both zero-equivalent papers, keeps the real one.
    ids = {
        p["paperId"] for p in search_papers_by_field("cs", limit=10, min_citations=1)
    }
    assert ids == {"p-high"}

    # min_citations=0 keeps all three.
    _install_fake_get(monkeypatch, payload)
    ids_all = {
        p["paperId"] for p in search_papers_by_field("cs", limit=10, min_citations=0)
    }
    assert ids_all == {"p-missing", "p-zero", "p-high"}


# =============================================================================
# Warm-cache scene build without the Points-LOD dependencies (issue #1107)
# =============================================================================
#
# The demo advertises "repeat runs are instant" but used to request substitutive
# Points LOD unconditionally. Building that LOD imports ``luxar.gsplats.lod``
# (torch coarsening kernels) whose additive sibling imports ``scipy.sparse`` at
# module load — so on a torch/scipy-free machine WITH a complete cache the scene
# build crashed with ``ModuleNotFoundError`` instead of producing a viewable
# scene. The fix routes the request through
# ``luxar.demos.substitutive_lod_or_flat``, which falls back to a flat point
# cloud with a degradation notice when either module is missing.

# Two real field names so FIELD_COLORS lookups resolve during coloring.
_FIELDS = ["Computer Science", "Physics"]


def _fake_bundle(n: int = 40) -> dict:
    """A valid warm-cache bundle: the exact keys/shapes consumed downstream."""
    rng = np.random.default_rng(0)
    embeddings_3d = rng.standard_normal((n, 3)).astype(np.float32)
    fields = [_FIELDS[i % len(_FIELDS)] for i in range(n)]
    citations = [int(i) for i in range(n)]
    papers_clean = [
        {
            "title": f"Paper {i}",
            "field": fields[i],
            "citations": citations[i],
            "year": 2015 + (i % 8),
        }
        for i in range(n)
    ]
    return {
        "papers_clean": papers_clean,
        "embeddings_3d": embeddings_3d,
        "fields": fields,
        "citations": citations,
    }


def _install_warm_cache(monkeypatch) -> None:
    """Simulate a complete warm cache: ``cache_computed`` never calls compute_fn.

    Returning a fabricated bundle avoids torch, the network, and ~/.cache — the
    scene build is exercised in isolation, exactly as a cached repeat run would
    hit it.
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

        out = tmp_path / "arxiv_papers.luxar.zarr"
        n = generate_paper_landscape(out, fields=_FIELDS, papers_per_field=10)

        # The scene was written (no crash).
        assert n == 40
        assert out.exists()

        # The degradation notice named the blocked module.
        stdout = capsys.readouterr().out
        assert "skipping Points LOD" in stdout
        assert blocked in stdout

        # A FLAT Points leaf was written — not an LOD group.
        assert (out / "papers" / "positions").exists()
        zattrs = json.loads((out / "papers" / ".zattrs").read_text())
        assert zattrs.get("kind") != "lod"

    def test_lod_group_built_when_deps_present(
        self, monkeypatch, capsys, tmp_path
    ) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("scipy")
        _install_warm_cache(monkeypatch)

        out = tmp_path / "arxiv_papers.luxar.zarr"
        n = generate_paper_landscape(out, fields=_FIELDS, papers_per_field=10)

        assert n == 40
        assert out.exists()

        # No degradation notice.
        stdout = capsys.readouterr().out
        assert "skipping Points LOD" not in stdout

        # A substitutive-LOD group was written: kind=lod with child_N levels and
        # NO top-level positions leaf.
        zattrs = json.loads((out / "papers" / ".zattrs").read_text())
        assert zattrs.get("kind") == "lod"
        assert (out / "papers" / "child_0").exists()
        assert not (out / "papers" / "positions").exists()
