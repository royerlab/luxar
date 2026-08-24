"""Tests for scripts/reencode_gsplat_demos.py (the LFS baseline re-encoder).

Only the sidecar-pairing refusal is covered: re-encoding a fit reorders its
splats, which silently invalidates a per-splat ``.npz`` sidecar indexed
positionally against it — the bug of #1670, which this script's ``--apply`` path
would otherwise recreate every time it ran. Both halves are tested: the predicate
that names a refused dataset, and the EFFECT in ``main`` — printing a refusal and
then re-encoding the dataset anyway would be the whole bug back.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_MOD_PATH = Path(__file__).resolve().parents[2] / "scripts" / "reencode_gsplat_demos.py"
_spec = importlib.util.spec_from_file_location("reencode_gsplat_demos", _MOD_PATH)
assert _spec is not None and _spec.loader is not None
rg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rg)


def test_sidecar_paired_datasets_are_refused() -> None:
    for dataset, fit in (
        ("gsplats_ct_totalsegmentator", "ct_atlas.gsplats.zarr.zip"),
        ("gsplats_visible_human_head", "vh_head.gsplats.zarr.zip"),
    ):
        zip_path = rg.DATA_DIR / dataset / fit
        refusal = rg.sidecar_pair_refusal(zip_path)
        assert refusal is not None, f"{dataset} must be refused (its sidecar)"
        assert dataset in refusal
        assert "1670" in refusal


def test_ordinary_datasets_are_not_refused() -> None:
    ok = rg.DATA_DIR / "gsplats_milkyway_dust" / "milkyway_dust.gsplats.zarr.zip"
    assert rg.sidecar_pair_refusal(ok) is None


def test_the_refused_set_matches_the_datasets_that_ship_an_npz() -> None:
    """Spell out the coupling, so a NEW sidecar-bearing dataset cannot slip in.

    Enumerated through the SCRIPT's own ``DATA_DIR.glob("gsplats_*/*.zip")`` —
    the set of dirs it will actually visit — rather than every child of
    ``DATA_DIR``, so an unrelated directory can never redden this. The assertion
    is bidirectional, as the name says: a refused dir that ships no sidecar has
    no reason to be excluded either.
    """
    if not rg.DATA_DIR.exists():  # pragma: no cover - source checkouts have it
        pytest.skip(f"packaged demo data absent at {rg.DATA_DIR}")
    visited = {z.parent for z in rg.DATA_DIR.glob("gsplats_*/*.zip")}
    if not visited:
        # The GSplat archives are manifest-hosted rather than in-repo, so there is
        # no on-disk set to compare the constant against. The coupling itself stays
        # covered by the synthetic-tree tests below.
        pytest.skip("no in-repo GSplat archives to enumerate")
    with_sidecar = {d.name for d in visited if any(d.glob("*.npz"))}
    assert with_sidecar == set(rg.SIDECAR_PAIRED_DIRS), (
        "SIDECAR_PAIRED_DIRS must name exactly the visited datasets that ship a "
        "per-splat .npz sidecar next to their fit: re-encoding a missing one "
        "would silently misindex its sidecar, and a spurious one is refused for "
        "no reason"
    )


def _fake_data_tree(tmp_path: Path) -> Path:
    """A ``DATA_DIR``-shaped tree: two sidecar-paired dirs and one ordinary one."""
    data = tmp_path / "data"
    for dataset, fit in (
        ("gsplats_ct_totalsegmentator", "ct_atlas.gsplats.zarr.zip"),
        ("gsplats_visible_human_head", "vh_head.gsplats.zarr.zip"),
        ("gsplats_plain", "plain.gsplats.zarr.zip"),
    ):
        d = data / dataset
        d.mkdir(parents=True)
        # > 1 KB, else the script skips it as an unmaterialized LFS pointer.
        (d / fit).write_bytes(b"\0" * 2048)
    return data


def test_main_never_reencodes_a_sidecar_paired_dataset(
    tmp_path, monkeypatch, capsys
) -> None:
    """The refusal must SKIP the dataset, not merely print about it.

    Covering only ``sidecar_pair_refusal`` left the ``continue`` in ``main``
    untested: dropping it printed the very same warning and then re-encoded the
    fit anyway — reordering its splats and recreating #1670 on every run.
    """
    data = _fake_data_tree(tmp_path)
    monkeypatch.setattr(rg, "REPO", tmp_path)
    monkeypatch.setattr(rg, "DATA_DIR", data)
    processed: list[str] = []

    def _record(src_zip: Path, out_zip: Path, recipe: str, tmp: Path) -> dict[str, Any]:
        processed.append(src_zip.parent.name)
        return {"kind": "leaf", "old_bytes": 2048, "new_bytes": 1024}

    monkeypatch.setattr(rg, "is_bundle", lambda src: False)
    monkeypatch.setattr(rg, "process_single_zip", _record)
    monkeypatch.setattr(
        rg,
        "process_bundle_zip",
        lambda *a, **k: pytest.fail("no bundle in this fixture"),
    )
    monkeypatch.setattr(sys, "argv", ["reencode_gsplat_demos", "--apply"])

    assert rg.main() == 0
    out = capsys.readouterr().out

    assert processed == ["gsplats_plain"], (
        "a sidecar-paired dataset was re-encoded despite the refusal"
    )
    assert "SKIP (sidecar-paired dataset)" in out
    # The ordinary dataset is still processed — the skip is scoped, not a stop.
    assert "OK  gsplats_plain" in out
