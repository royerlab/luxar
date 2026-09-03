"""What may be hosted, under which record, and the rail against publishing.

A Zenodo record carries ONE license field, so datasets are grouped into records
by license family and each dataset keeps its own true license alongside. That
model is only as good as its enforcement: a ShareAlike dataset filed under the
CC-BY record, or a non-redistributable one given a record at all, is a
licensing error that is invisible until someone downloads it.

The publication rail is the other half. Whether records go public is the
maintainer's call, made by hand on Zenodo, so a published record must carry the
evidence Zenodo minted at that transition rather than relying on discipline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_MANIFEST = Path(__file__).resolve().parents[1] / "data_manifest.json"

#: Licenses permissive enough to sit inside a CC-BY record. Public-domain
#: dedications qualify: a record cannot be more permissive than its strictest
#: item, and PD is looser than CC-BY, so the record's CC-BY field is a floor the
#: per-dataset license field records the truth beneath.
_CC_BY_COMPATIBLE = {
    "cc-by-4.0",
    "cc-by-3.0",
    "cc0-1.0",
    "pd-noaa",
    "pd-nlm",
    "public-domain",
}

#: Additionally allowed inside a ShareAlike record.
_CC_BY_SA_COMPATIBLE = _CC_BY_COMPATIBLE | {"cc-by-sa-4.0", "cc-by-sa-3.0"}

_RECORD_COMPATIBLE = {
    "cc-by-4.0": _CC_BY_COMPATIBLE,
    "cc-by-sa-4.0": _CC_BY_SA_COMPATIBLE,
}


def _manifest() -> dict[str, Any]:
    return json.loads(_MANIFEST.read_text())


def _hosted() -> dict[str, Any]:
    return {
        name: entry
        for name, entry in _manifest()["datasets"].items()
        if entry.get("bucket") == "zenodo"
    }


def test_publication_is_recorded_with_its_evidence() -> None:
    """Publishing is the maintainer's decision, made by hand, on Zenodo.

    This test used to assert every record was still a draft, so nothing in the
    repository could flip the flag and any attempt landed here first. It said
    that if publication genuinely happened, this was the place to record it
    deliberately. THE MAINTAINER PUBLISHED ALL FOUR RECORDS ON 2026-09-02, so
    this is that deliberate record.

    The guard is kept rather than dropped, by demanding EVIDENCE instead of a
    fixed answer: a published record must carry a concept DOI, which Zenodo
    mints only at publication and which therefore cannot be derived from a
    deposition id. Setting `published` without one now fails, so the flag still
    cannot be flipped speculatively -- while a genuinely published record passes.
    """
    records = _manifest()["records"]
    assert {name for name, record in records.items() if record.get("published")} >= {
        "cc-by",
        "cc-by-sa",
        "h2afva",
        "droso-timelapse",
    }

    for name, record in records.items():
        if not record.get("published"):
            continue
        concept = record.get("zenodo_conceptdoi")
        assert concept and re.fullmatch(r"10\.5281/zenodo\.\d+", concept), (
            f"record {name!r} is marked published but carries no concept DOI. "
            "Zenodo mints that identifier at publication, so its absence means "
            "the flag was set without the record actually being public."
        )
        assert concept != record.get("zenodo_doi"), (
            f"record {name!r} lists its version DOI as the concept DOI; the "
            "concept DOI is version-independent and is a different identifier"
        )


def test_every_hosted_dataset_belongs_to_a_real_record() -> None:
    records = _manifest()["records"]
    for name, entry in sorted(_hosted().items()):
        record = entry.get("record")
        assert record, f"{name} is hosted on Zenodo but names no record"
        assert record in records, f"{name} names unknown record {record!r}"


def test_no_dataset_is_more_restrictive_than_its_record() -> None:
    """The record's single license field must not overstate what it may grant.

    The failure this catches is a ShareAlike dataset inside the CC-BY record:
    the record would offer it under terms its source does not allow.
    """
    records = _manifest()["records"]
    for name, entry in sorted(_hosted().items()):
        record_license = records[entry["record"]]["license"]
        allowed = _RECORD_COMPATIBLE.get(record_license)
        assert allowed is not None, (
            f"{name}: record license {record_license!r} has no compatibility rule — "
            "add one rather than letting it pass unchecked"
        )
        assert entry.get("license") in allowed, (
            f"{name}: license {entry.get('license')!r} cannot be offered under a "
            f"{record_license} record. Move it to a record whose family fits, or "
            "stop hosting it."
        )


def test_nothing_non_redistributable_is_hosted() -> None:
    """`redistribute=False` and `bucket=zenodo` are a contradiction."""
    for name, entry in sorted(_hosted().items()):
        assert entry.get("redistribute") is not False, (
            f"{name} is marked non-redistributable yet hosted on Zenodo"
        )
        assert entry.get("license") not in ("conflict", "none", None), (
            f"{name} is hosted with license {entry.get('license')!r} — an unresolved "
            "or absent license must not be uploaded"
        )


def test_no_noncommercial_dataset_is_hosted() -> None:
    """NonCommercial terms are incompatible with both record licenses.

    `milky_way_gaia_3m` is the live example (CC BY-NC 3.0 IGO) and is correctly
    `local-compute`; this keeps it that way.
    """
    for name, entry in sorted(_hosted().items()):
        license_id = str(entry.get("license", ""))
        assert "-nc-" not in license_id and not license_id.endswith("-nc"), (
            f"{name} carries NonCommercial terms ({license_id}) and cannot be "
            "redistributed under either record"
        )


def test_every_hosted_dataset_credits_its_source() -> None:
    for name, entry in sorted(_hosted().items()):
        for field in ("source", "attribution"):
            value = entry.get(field, "")
            assert len(value) > 10, (
                f"{name}: {field} is required for a hosted dataset (got {value!r})"
            )


def test_the_selector_actually_finds_hosted_datasets() -> None:
    """A selector that matched nothing would pass every assertion above."""
    hosted = _hosted()
    assert len(hosted) >= 20, f"only found {len(hosted)} hosted datasets"
    # And the non-hosted side is non-empty, or the redistribution rules are moot.
    all_datasets = _manifest()["datasets"]
    assert len(all_datasets) > len(hosted), (
        "no non-hosted datasets — check the bucket field"
    )
