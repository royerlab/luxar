"""Tests for scripts/reencode_gsplat_demos.py (the LFS baseline re-encoder).

Only the sidecar-pairing refusal is covered: re-encoding a fit reorders its
splats, which silently invalidates a per-splat ``.npz`` sidecar indexed
positionally against it — the bug of #1670, which this script's ``--apply`` path
would otherwise recreate every time it ran.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

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

    Every dataset dir in the packaged demo data that holds both a ``.zip`` fit and
    a ``.npz`` sidecar must be on the refusal list. Skipped when the data tree is
    absent (an installed wheel / a checkout without demo data).
    """
    if not rg.DATA_DIR.exists():  # pragma: no cover - source checkouts have it
        return
    with_sidecar = {
        d.name
        for d in rg.DATA_DIR.iterdir()
        if d.is_dir() and any(d.glob("*.npz")) and any(d.glob("*.zip"))
    }
    assert with_sidecar <= set(rg.SIDECAR_PAIRED_DIRS), (
        "a dataset ships a per-splat .npz sidecar next to its fit but is not in "
        "SIDECAR_PAIRED_DIRS — re-encoding it would silently misindex the sidecar"
    )
