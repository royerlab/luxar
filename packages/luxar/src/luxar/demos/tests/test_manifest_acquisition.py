"""Every hosted gsplat dataset must say what it was fitted FROM, or why it cannot.

The Zenodo records quote two compression figures: against the raw voxels of the
declared source grid, and against that source **as stored**. The first is
stamped per archive at fit time. The second cannot be: a dataset is often
several archives (one per channel) fitted from one file, so only the dataset as
a whole is comparable to the acquisition as a whole.

Hence this second, dataset-level declaration. It is also where a dataset says
that no honest ratio exists -- for a fit that resizes, repacks, or converts
colour, the stored size measures something other than compression, and quoting
it would repeat the error already corrected for nexrad.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_MANIFEST = Path(__file__).resolve().parents[1] / "data_manifest.json"


def _datasets() -> dict[str, Any]:
    return json.loads(_MANIFEST.read_text())["datasets"]


#: Hosted datasets that ship no Gaussian splats, so no fit denominator applies.
#:
#: The selector below is FAIL-CLOSED against this list rather than keyed on a
#: `gsplats` name prefix: `h2afva` is a gsplat timelapse whose manifest key
#: carries no such prefix, and a prefix test let the largest hosted dataset of
#: the set out of this gate entirely, untracked. Anything hosted and not named
#: here is in scope, so a new dataset joins the gate instead of escaping it on a
#: naming accident.
_NOT_GSPLAT_DATASETS = {
    "desi_galaxies": "DESI DR1 large-scale-structure catalogue, shipped as points",
    "census_umap_1m": "precomputed CELLxGENE Census UMAP coordinates (npz)",
    "3d_umap_coords_human": "precomputed multiome peak UMAP coordinates (parquet)",
    "3d_umap_coords_mouse": "precomputed multiome peak UMAP coordinates (parquet)",
}


def _ships_gsplats(entry: dict[str, Any]) -> bool:
    """Whether a manifest entry's own archives are Gaussian splats."""
    files = list(entry.get("files", []))
    for variant in entry.get("variants", {}).values():
        files.extend(variant.get("files", []))
    return any(".gsplats.zarr" in f.get("name", "") for f in files)


def _hosted() -> dict[str, Any]:
    return {
        name: entry
        for name, entry in _datasets().items()
        if entry.get("bucket") == "zenodo"
    }


def _hosted_gsplat_datasets() -> dict[str, Any]:
    """Zenodo-hosted gsplat datasets -- the ones a record will describe."""
    return {
        name: entry
        for name, entry in _hosted().items()
        if name not in _NOT_GSPLAT_DATASETS
    }


#: Hosted gsplat datasets with no acquisition block yet. SHRINKS to empty.
#:
#: An entry here is work, not an exemption: the dataset will go to Zenodo with
#: only the raw-voxel ratio until it either declares a measurable source or
#: states why none is comparable.
#: `gsplats_flylight_mcfo` is deliberately absent: it is `bucket=local-compute`
#: (built on the user's machine, not redistributable), so no record describes it
#: and it owes no denominator. It needs no naming here either — the selector is
#: by bucket and fail-closed, so if it is ever promoted to `zenodo` it joins the
#: hosted set with no acquisition block and this gate goes red on its own.
_NOT_YET_DECLARED = {
    # Refit still in flight.
    "gsplats_flylight_mcfo_63x",
    # Cache-only datasets: hosted, but no in-repo copy to probe, so their
    # acquisitions are declared when their sources are next opened.
    "gsplats_3d_drosophila_gastrulation",
    "gsplats_3d_h2afva_stack",
    "gsplats_3d_h2afva_decimation",
    "gsplats_4d_neuromast_2ch",
}

#: Datasets whose `stored_bytes` is still to be measured. SHRINKS to empty.
#:
#: Measuring means opening the acquisition, which for several of these is a
#: multi-gigabyte download, so it happens during the refit when the source is
#: already in hand rather than as a separate pass.
_STORED_BYTES_PENDING = {
    # Behind an authenticated Kaggle competition endpoint, so sizing the
    # chunk-compressed source stores is a ~4 GB download rather than a probe.
    "gsplats_cell_tracking",
    "gsplats_cryoem_virus",
    "gsplats_milkyway_dust",
    "gsplats_kidney",
    "gsplats_cells3d",
    "gsplats_opencell_map4",
    "gsplats_multichannel",
    "gsplats_dapi",
    "gsplats_celegans",
    "h2afva",
}


def test_every_hosted_gsplat_dataset_declares_an_acquisition() -> None:
    missing = {
        name
        for name, entry in _hosted_gsplat_datasets().items()
        if "acquisition" not in entry
    }
    assert missing <= _NOT_YET_DECLARED, (
        "a hosted gsplat dataset has no acquisition block:\n  "
        + "\n  ".join(sorted(missing - _NOT_YET_DECLARED))
        + "\nAdd acquisition=dict(...) to its row in scripts/gen_data_manifest.py "
        "and regenerate, giving either a measurable source or the reason none is "
        "comparable."
    )


def test_the_pending_lists_shrink_and_do_not_go_stale() -> None:
    """A dataset that has been wired must leave the list.

    Without this the lists quietly become permanent exemptions, which is
    precisely what they must not be.
    """
    hosted = _hosted_gsplat_datasets()
    for name in sorted(_NOT_YET_DECLARED):
        assert name in hosted, f"{name} is no longer a hosted gsplat dataset — drop it"
        assert "acquisition" not in hosted[name], (
            f"{name} now declares an acquisition — remove it from _NOT_YET_DECLARED"
        )
    for name in sorted(_STORED_BYTES_PENDING):
        assert name in hosted, f"{name} is no longer a hosted gsplat dataset — drop it"
        acq = hosted[name].get("acquisition")
        assert acq is not None, f"{name} lost its acquisition block"
        assert acq.get("stored_bytes") is None, (
            f"{name} now has stored_bytes — remove it from _STORED_BYTES_PENDING"
        )


def test_a_comparable_acquisition_is_measured_or_pending() -> None:
    """`comparable: true` is a promise of a denominator, not a mood."""
    for name, entry in sorted(_hosted_gsplat_datasets().items()):
        acq = entry.get("acquisition")
        if acq is None or not acq.get("comparable"):
            continue
        if name in _STORED_BYTES_PENDING:
            continue
        stored = acq.get("stored_bytes")
        assert isinstance(stored, int) and stored > 0, (
            f"{name}: comparable acquisition needs a positive stored_bytes, "
            f"got {stored!r}"
        )


def test_an_incomparable_acquisition_says_why() -> None:
    """Refusing to quote a ratio is a claim, and it has to be argued.

    The bar is a stated mechanism -- what the fit does to the data that makes
    the stored size measure something else -- not "it is complicated".
    """
    for name, entry in sorted(_hosted_gsplat_datasets().items()):
        acq = entry.get("acquisition")
        if acq is None or acq.get("comparable"):
            continue
        reason = acq.get("reason", "")
        assert len(reason) > 60, (
            f"{name}: say what the stored size would actually be measuring "
            f"(got {reason!r})"
        )
        assert "stored_bytes" not in acq, (
            f"{name}: an incomparable acquisition must not carry a denominator — "
            "it would get quoted"
        )


def test_every_acquisition_describes_its_source() -> None:
    for name, entry in sorted(_hosted_gsplat_datasets().items()):
        acq = entry.get("acquisition")
        if acq is None:
            continue
        description = acq.get("description", "")
        assert len(description) > 20, (
            f"{name}: acquisition needs a description naming the actual file or "
            f"store (got {description!r})"
        )


def test_the_non_gsplat_exclusions_do_not_go_stale() -> None:
    """The one way out of this gate is a named list, and it has to stay true."""
    hosted = _hosted()
    for name, reason in sorted(_NOT_GSPLAT_DATASETS.items()):
        assert name in hosted, f"{name} is no longer a hosted dataset — drop it"
        assert len(reason) > 20, f"{name}: say what it ships instead of splats"
        assert not _ships_gsplats(hosted[name]), (
            f"{name} ships a .gsplats.zarr archive — it owes a denominator, so "
            "remove it from _NOT_GSPLAT_DATASETS"
        )


def test_the_detector_actually_finds_datasets() -> None:
    """A selector that matched nothing would pass every assertion above."""
    hosted = _hosted_gsplat_datasets()
    assert len(hosted) >= 15, f"only found {len(hosted)} hosted gsplat datasets"
    # And it is not keyed on the name: `h2afva` is a gsplat timelapse whose key
    # has no `gsplats` prefix, and a regression to prefix matching would drop
    # the largest hosted dataset of the set without a word.
    assert any(not name.startswith("gsplats") for name in hosted), (
        "every selected dataset is `gsplats*`-named — the selector has fallen "
        "back to the name prefix and lets h2afva out"
    )
    declared = [n for n, e in hosted.items() if "acquisition" in e]
    assert len(declared) >= 10, f"only {len(declared)} declare an acquisition"
    assert any(
        not e["acquisition"].get("comparable")
        for e in hosted.values()
        if "acquisition" in e
    ), "no incomparable acquisition found — the reason assertions are vacuous"
