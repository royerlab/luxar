"""Warm-cache scene build without the Points-LOD dependencies (issue #1107).

The Kaggle arXiv-embeddings demo advertises instant cached re-runs, but
``generate_paper_landscape`` used to request substitutive Points LOD
unconditionally. Building that LOD imports ``luxar.gsplats.lod`` (torch
coarsening kernels) whose additive sibling imports ``scipy.sparse`` at module
load — so on a torch/scipy-free machine WITH a complete ``arxiv_kaggle`` cache
the scene build crashed with ``ModuleNotFoundError`` instead of producing a
viewable scene. The fix routes the request through
``luxar.demos.substitutive_lod_or_flat``, which falls back to a flat point cloud
with a degradation notice when either module is missing.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from luxar.demos import demo_arxiv_embeddings_kaggle as demo
from luxar.demos.demo_arxiv_embeddings_kaggle import generate_paper_landscape


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
        zattrs = json.loads((out / "arxiv_papers_kaggle" / ".zattrs").read_text())
        assert zattrs.get("kind") != "lod"

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
        zattrs = json.loads((out / "arxiv_papers_kaggle" / ".zattrs").read_text())
        assert zattrs.get("kind") == "lod"
        assert (out / "arxiv_papers_kaggle" / "child_0").exists()
        assert not (out / "arxiv_papers_kaggle" / "positions").exists()
