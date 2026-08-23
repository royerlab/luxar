"""Unit tests for the built-scene credit check.

Built on synthetic stores in ``tmp_path`` rather than on ``datasets/demos``, so
the gate is exercised on a checkout with no generated scenes — the same reason
``check_demo_ladders`` is unit-tested rather than relying on built data.

The cases are the ones that actually occurred or plausibly could:
  * a store that agrees with its demo
  * a store rebuilt without its citation  (the desi_galaxies regression)
  * a store carrying a superseded credit  (drift after a credit is corrected)
  * a procedural demo whose store stays uncredited
  * both zarr layouts, because probing the wrong metadata document is its own
    recurring bug
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from check_scene_credits import compare, main  # noqa: E402

CITED = {
    "short": "Tully et al. 2023 (Cosmicflows-4)",
    "doi": "10.3847/1538-4357/ac94d8",
}


def _v3_store(root: Path, name: str, attrs: dict) -> Path:
    store = root / f"{name}.luxar.zarr"
    store.mkdir(parents=True)
    (store / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": attrs})
    )
    return store


def _v2_store(root: Path, name: str, attrs: dict) -> Path:
    store = root / f"{name}.luxar.zarr"
    store.mkdir(parents=True)
    (store / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (store / ".zattrs").write_text(json.dumps(attrs))
    return store


@pytest.mark.parametrize("build", [_v3_store, _v2_store], ids=["v3", "v2"])
def test_agreeing_store_is_clean(tmp_path: Path, build) -> None:
    store = build(tmp_path, "ok", {"citation": CITED})
    assert compare(store, CITED) is None


@pytest.mark.parametrize("build", [_v3_store, _v2_store], ids=["v3", "v2"])
def test_missing_citation_is_caught(tmp_path: Path, build) -> None:
    """The desi_galaxies regression: rebuilt from a checkout predating wiring."""
    store = build(tmp_path, "uncredited", {"content_hash": "abc"})
    problem = compare(store, CITED)
    assert problem is not None and "carries no citation" in problem


def test_superseded_credit_is_caught(tmp_path: Path) -> None:
    """A store still asserting the credit a later correction replaced."""
    store = _v3_store(tmp_path, "stale", {"citation": {"short": "Tully et al. 2014"}})
    problem = compare(store, CITED)
    assert problem is not None
    assert "Tully et al. 2014" in problem and "Cosmicflows-4" in problem


def test_procedural_store_without_a_citation_is_clean(tmp_path: Path) -> None:
    store = _v3_store(tmp_path, "synthetic", {"content_hash": "abc"})
    assert compare(store, None) is None


def test_procedural_store_asserting_a_credit_is_caught(tmp_path: Path) -> None:
    """A synthetic scene must not claim someone else's work."""
    store = _v3_store(tmp_path, "synthetic", {"citation": {"short": "Someone 2020"}})
    problem = compare(store, None)
    assert problem is not None and "does not declare" in problem


def test_unreadable_store_is_reported_not_skipped(tmp_path: Path) -> None:
    store = tmp_path / "empty.luxar.zarr"
    store.mkdir()
    assert "no readable root metadata" in (compare(store, CITED) or "")


def test_stale_consolidated_attrs_do_not_override_live_v2_attrs(tmp_path: Path) -> None:
    store = _v2_store(tmp_path, "legacy", {"citation": CITED})
    (store / ".zmetadata").write_text(
        json.dumps({"zarr_consolidated_format": 1, "metadata": {".zattrs": {}}})
    )
    assert compare(store, CITED) is None


def test_no_built_scenes_is_a_clean_no_op(tmp_path: Path, capsys) -> None:
    """A checkout without generated scenes must not fail the gate."""
    assert main(["--demos-dir", str(tmp_path)]) == 0
    assert "nothing to check" in capsys.readouterr().out


def test_exit_code_is_non_zero_when_a_store_contradicts_its_demo(
    tmp_path: Path, capsys
) -> None:
    store = _v3_store(tmp_path, "cosmicflows_laniakea_full", {"content_hash": "x"})
    # Addressed by path, so the demo lookup resolves through `outputs`.
    code = main([str(store)])
    out = capsys.readouterr().out
    assert code == 1, out
    assert "carries no citation" in out
