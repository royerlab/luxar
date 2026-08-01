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

from typing import Any

from luxar.demos import demo_arxiv_paper_embeddings as demo
from luxar.demos.demo_arxiv_paper_embeddings import search_papers_by_field


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
