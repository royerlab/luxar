"""Tests for the offline record-attribution drift audit.

The fixtures reproduce the real wordings verbatim so the tests pin the rule
rather than the current contents of the repository, which will change when the
Zenodo record is corrected and re-captured.

Both directions are pinned deliberately. The maintainer's ruling is that the
2016 paper describes the *instrument and the method*, not the imaging, so
"the instrument is described in Royer et al." must stay clean forever while
"the imaging is described in Royer et al." must always be caught.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_script(name: str) -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution because `dataclasses` resolves the postponed
    # annotations of a frozen dataclass through `sys.modules`. Whatever was
    # registered under that name is put back: another test module may be
    # holding the object that lives there.
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return module


CHECKER = _load_script("check_record_attribution")
CHECKER_PATH = Path(__file__).resolve().parents[1] / "check_record_attribution.py"

UNPUBLISHED_ATTRIBUTION = (
    "Acquired in Philipp J. Keller's lab at HHMI Janelia Research Campus, where "
    "L. A. Royer was then a postdoctoral fellow; used with permission "
    "(CC BY 4.0). The splat fits were computed later at CZ Biohub SF. The "
    "imaging itself is unpublished; for the SiMView instrument see Royer et al., "
    "Nat. Biotechnol. 34, 1267-1278 (2016), doi:10.1038/nbt.3708."
)
# The live droso-timelapse bullet and record paragraph, verbatim.
DROSO_BULLET = (
    "<li><code>drosophila_embryogenesis_500tp.gsplats.zarr.zip</code> (863.8 MB, "
    "cc-by-4.0). Attribution: Acquired in the laboratory of Philipp J. Keller at "
    "HHMI Janelia Research Campus. Imaging described in Royer et al., Nature "
    "Biotechnology 34, 1267-1278 (2016), doi:10.1038/nbt.3708.</li>"
)
DROSO_RECORD_PARAGRAPH = (
    "<p>The imaging is described in Royer et al., <em>Nature Biotechnology</em> "
    "34, 1267-1278 (2016), doi:10.1038/nbt.3708, CC BY 4.0. Users of this record "
    "should cite that publication in addition to this record.</p>"
)
# The intro sentence whose "published" is about the ARCHIVE, not the imaging.
DROSO_INTRO_PARAGRAPH = (
    "<p>A <em>Drosophila melanogaster</em> embryo imaged through early "
    "embryogenesis on a SiMView-type multi-view light-sheet microscope, with "
    "nuclei labelled by a histone H2A variant fusion (<code>w; His2Av::mRFP1; "
    "+</code>), fitted to 3D Gaussian splats per timepoint and published as a "
    "single 4D archive for the Luxar viewer.</p>"
)
# The correct framing on the gastrulation still: the paper is the METHOD.
GASTRULATION_BULLET = (
    "<li><code>gsplats_3d_drosophila_gastrulation</code> (2.2 MB, cc-by-4.0). "
    "Attribution: Acquired in the laboratory of Philipp J. Keller at HHMI "
    "Janelia Research Campus. Imaging method: Royer, L. A., Lemon, W. C., "
    "Chhetri, R. K., Wan, Y., Coleman, M., Myers, E. W., Keller, P. J. Adaptive "
    "light-sheet microscopy for long-term, high-resolution imaging in living "
    "organisms. Nature Biotechnology 34, 1267-1278 (2016).</li>"
)
# The correct framing on the neuromast timelapse.
NEUROMAST_BULLET = (
    "<li><code>gsplats_4d_neuromast_2ch</code> (136.4 MB, cc-by-4.0). "
    "Attribution: Adrian Jacobo (Chan Zuckerberg Biohub San Francisco); used "
    "with permission (CC BY 4.0). This imaging is unpublished; for the biology "
    "of the organ it shows, see the data author's work on lateral-line sensory "
    "organs: Erzberger, A., Jacobo, A., Dasgupta, A. &amp; Hudspeth, A. J. "
    "Mechanochemical symmetry breaking during morphogenesis of lateral-line "
    "sensory organs. Nature Physics 16, 949-957 (2020).</li>"
)

# Every one of these says a publication covers the INSTRUMENT, the METHOD or
# nothing at all. They are the maintainer's plausible one-sentence corrections,
# so a report that flagged them would stay red no matter what was written.
CORRECT_FRAMINGS = (
    "The imaging instrument is described in Royer et al., Nature Biotechnology "
    "34 (2016).",
    "The imaging instrument used for this dataset is described in Royer et al.",
    "The imaging microscope used throughout the experiment is described in "
    "Royer et al.",
    "The imaging method used for every timepoint here is described in Royer et al.",
    "The imaging protocol used for these embryos is described in Royer et al.",
    "The imaging pipeline that produced these splats is described in Royer et al.",
    "The microscope used for this imaging is described in Royer et al., Nature "
    "Biotechnology 34 (2016).",
    "Imaging method described in Royer et al., Nature Biotechnology 34 (2016).",
    "Imaging was carried out as described in Royer et al., Nature Biotechnology "
    "34 (2016).",
    # The standard methods-section phrasing. An adverb between "as" and the verb
    # is normal, and anchoring the manner clause on a bare trailing "as" told
    # the maintainer to go and edit prose that was already right.
    "Imaging was performed as previously described in Royer et al.",
    "Imaging performed on the SiMView instrument described in Royer et al. (2016).",
    "The imaging has not been published in any article; the instrument is "
    "described in Royer et al.",
    "The imaging has not under any circumstances ever been published in any article.",
    "The imaging data are not published in any paper.",
    # The archive, not the imaging, is the thing that was published.
    "The imaging was fitted and published as a single 4D archive.",
)
# Every one of these makes the IMAGING the thing a publication covers. The
# first two are long because the live records write inline genotype, species
# and acquisition strings into the same clause.
DRIFTING_FRAMINGS = (
    "Imaging of this His2Av::mRFP1 Drosophila melanogaster embryo is described "
    "in Royer et al.",
    "The imaging, which was acquired over four hours of development in 2013, is "
    "described in Royer et al.",
    "The imaging is documented in Royer et al.",
    "Imaging described by Royer et al., Nature Biotechnology 34 (2016).",
)
# Real claims about the imaging that an earlier version of the rule silently
# cleared, because a guard reached past the thing it was written for. They are
# pinned so that surface cannot come back.
RECOVERED_DRIFTING_FRAMINGS = (
    # The gap between "imaging" and the verb is exactly where a locative or
    # partitive modifier OF the imaging sits, so artefact nouns in it are not
    # replacement subjects. Listing them cleared all four of these.
    "The imaging in this archive is described in Royer et al. (2016).",
    "The imaging in this record is published in Royer et al.",
    "The imaging and the splat fits are both described in Royer et al.",
    "The imaging of these records is described in Royer et al.",
    # Only "not" and "never" negate the publication verb; "no"/"nor"/"without"
    # reach a modifier of the imaging instead.
    "The imaging, with no post-processing, is described in Royer et al.",
    "The imaging without further processing is described in Royer et al.",
    "The imaging, with no deconvolution nor denoising, is described in Royer et al.",
    # A manner adverbial runs into the verb; a parenthetical is bounded by its
    # own commas, so widening the manner clause must stop at one.
    "The imaging is, as we note, described in Royer et al.",
    # A replacement subject or denial only clears the claim when it runs into
    # the publication verb. Appositives and parentheticals still describe the
    # imaging, so nouns and negations inside them must not suppress the claim.
    "The imaging, acquired on the SiMView microscope, is described in Royer et al.",
    "The imaging, a light-sheet microscopy timelapse, is described in Royer et al.",
    "The imaging, produced with the standard acquisition protocol, is described "
    "in Royer et al.",
    "The imaging, which does not include the fixed-cutoff pass, is described in "
    "Royer et al.",
    "The imaging, not yet reprocessed, is described in Royer et al.",
)
# Clean only because "imaging" and the verb are more than a sentence-window
# apart with a different subject in between; widening the window flags it.
DISTANT_UNRELATED_CLAIM = (
    "The imaging captured an entire Drosophila melanogaster embryo through "
    "gastrulation, germband extension and dorsal closure, and the optics it ran "
    "on are described in Royer et al."
)


def _write(
    tmp_path: Path, datasets: dict[str, Any], snapshots: dict[str, str]
) -> tuple[Path, Path]:
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text(json.dumps({"datasets": datasets}), encoding="utf-8")
    snapshot_dir = tmp_path / "zenodo_record_text"
    snapshot_dir.mkdir()
    for record, body in snapshots.items():
        (snapshot_dir / f"{record}.html").write_text(body, encoding="utf-8")
    return manifest, snapshot_dir


def _run(
    tmp_path: Path, datasets: dict[str, Any], snapshots: dict[str, str]
) -> tuple[list[Any], str, int]:
    manifest, snapshot_dir = _write(tmp_path, datasets, snapshots)
    findings = CHECKER.audit(manifest, snapshot_dir)
    return findings, CHECKER.render(findings), CHECKER.exit_status(findings)


def _droso_dataset() -> dict[str, Any]:
    return {
        "bucket": "zenodo",
        "record": "droso-timelapse",
        "attribution": UNPUBLISHED_ATTRIBUTION,
        "files": [{"name": "drosophila_embryogenesis_500tp.gsplats.zarr.zip"}],
    }


def _droso_bullet(sentence: str) -> str:
    """The live bullet with its provenance claim replaced."""
    return (
        "<li><code>drosophila_embryogenesis_500tp.gsplats.zarr.zip</code> "
        "(863.8 MB, cc-by-4.0). Attribution: Acquired in the laboratory of "
        "Philipp J. Keller at HHMI Janelia Research Campus. " + sentence + "</li>"
    )


def test_record_contradicting_the_manifest_is_flagged_by_name(tmp_path: Path) -> None:
    findings, report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {"droso-timelapse": DROSO_INTRO_PARAGRAPH + DROSO_BULLET},
    )

    assert [finding.level for finding in findings] == ["STALE"]
    assert "gsplats_4d_drosophila_embryogenesis" in report
    assert "bullet: Imaging described in Royer et al." in report
    assert status == 1


def test_both_the_bullet_and_the_record_paragraph_are_quoted(tmp_path: Path) -> None:
    findings, report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {
            "droso-timelapse": DROSO_INTRO_PARAGRAPH
            + DROSO_BULLET
            + DROSO_RECORD_PARAGRAPH
        },
    )

    (finding,) = findings
    assert finding.level == "STALE"
    assert len(finding.quotes) == 2
    assert finding.quotes[0].startswith("bullet: Imaging described in Royer et al.,")
    assert finding.quotes[1].startswith(
        "record prose: The imaging is described in Royer et al.,"
    )
    assert "doi:10.1038/nbt.3708" in finding.quotes[0]
    assert status == 1


@pytest.mark.parametrize("sentence", CORRECT_FRAMINGS)
def test_a_publication_covering_the_instrument_or_method_is_never_flagged(
    tmp_path: Path, sentence: str
) -> None:
    """The whole point: the 2016 paper describes the instrument, not the data."""
    findings, report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {"droso-timelapse": DROSO_INTRO_PARAGRAPH + _droso_bullet(sentence)},
    )

    assert [finding.level for finding in findings] == ["OK"]
    assert "STALE" not in report
    assert status == 0


@pytest.mark.parametrize("sentence", DRIFTING_FRAMINGS)
def test_a_publication_covering_the_imaging_is_always_flagged(
    tmp_path: Path, sentence: str
) -> None:
    findings, _report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {"droso-timelapse": DROSO_INTRO_PARAGRAPH + _droso_bullet(sentence)},
    )

    assert [finding.level for finding in findings] == ["STALE"]
    assert status == 1


@pytest.mark.parametrize("sentence", RECOVERED_DRIFTING_FRAMINGS)
def test_a_guard_reaching_past_its_purpose_does_not_clear_a_real_claim(
    tmp_path: Path, sentence: str
) -> None:
    """False-negative surface that earned nothing, pinned so it cannot return."""
    findings, _report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {"droso-timelapse": DROSO_INTRO_PARAGRAPH + _droso_bullet(sentence)},
    )

    assert [finding.level for finding in findings] == ["STALE"]
    assert status == 1


def test_an_abbreviation_in_the_gap_still_bounds_the_clause(tmp_path: Path) -> None:
    """`no. 3` is out of scope through the clause bound, not through negation.

    The gap between "imaging" and the verb may not cross a full stop, and the
    abbreviation carries one. Recorded because it looks like the "no"/"nor"
    over-reach that was removed from `_NEGATED` and is a different bound.
    """
    findings, _report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {
            "droso-timelapse": _droso_bullet(
                "The imaging (no. 3 of the series) is described in Royer et al."
            )
        },
    )

    assert [finding.level for finding in findings] == ["OK"]
    assert status == 0


@pytest.mark.parametrize(
    "markup",
    [
        "<h3>Imaging</h3><p>Documented in Royer et al., Nature Biotechnology "
        "34 (2016).</p>",
        "<table><tr><td>imaging</td><td>SiMView</td></tr></table><p>Described "
        "in Royer et al.</p>",
        "<p>Imaging</p><p>Described in Royer et al.</p>",
        "Imaging<br>Described in Royer et al.",
    ],
)
def test_a_block_boundary_is_not_the_middle_of_a_sentence(
    tmp_path: Path, markup: str
) -> None:
    """The live records use `<h3>` headings and a `<td><code>` table.

    Joining block elements with a space alone merged a heading or a table cell
    into the paragraph after it as one sentence, which reads as a claim neither
    block makes.
    """
    findings, _report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {
            "droso-timelapse": markup
            + "<li><code>drosophila_embryogenesis_500tp.gsplats.zarr.zip</code> "
            "(863.8 MB). This imaging is unpublished.</li>"
        },
    )

    assert [finding.level for finding in findings] == ["OK"]
    assert status == 0


def test_a_claim_a_whole_clause_away_from_the_imaging_is_out_of_scope(
    tmp_path: Path,
) -> None:
    """The proximity window is the recall bound, and it is deliberate."""
    findings, _report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {"droso-timelapse": _droso_bullet(DISTANT_UNRELATED_CLAIM)},
    )

    assert [finding.level for finding in findings] == ["OK"]
    assert status == 0


def test_published_as_an_archive_is_not_a_claim_about_the_imaging(
    tmp_path: Path,
) -> None:
    """The precision trap: the ARCHIVE was published; the imaging was not."""
    findings, report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {
            "droso-timelapse": DROSO_INTRO_PARAGRAPH
            + "<li><code>drosophila_embryogenesis_500tp.gsplats.zarr.zip</code> "
            "(863.8 MB, cc-by-4.0). Attribution: Acquired in the laboratory of "
            "Philipp J. Keller at HHMI Janelia Research Campus. The imaging "
            "itself is unpublished; for the SiMView instrument see Royer et al., "
            "Nature Biotechnology 34, 1267-1278 (2016).</li>"
        },
    )

    assert [finding.level for finding in findings] == ["OK"]
    assert "STALE" not in report
    assert status == 0


@pytest.mark.parametrize(
    ("key", "bullet"),
    [
        ("gsplats_3d_drosophila_gastrulation", GASTRULATION_BULLET),
        ("gsplats_4d_neuromast_2ch", NEUROMAST_BULLET),
    ],
)
def test_correct_framing_on_the_shared_record_is_not_flagged(
    tmp_path: Path, key: str, bullet: str
) -> None:
    other = {
        "bucket": "zenodo",
        "record": "cc-by",
        "attribution": "scikit-image sample data (CC0).",
        "files": [{"name": "kidney_ch0.gsplats.zarr.zip"}],
    }
    findings, report, status = _run(
        tmp_path,
        {
            key: {
                "bucket": "zenodo",
                "record": "cc-by",
                "attribution": UNPUBLISHED_ATTRIBUTION,
                "files": [{"name": f"{key}.gsplats.zarr.zip"}],
            },
            "gsplats_kidney": other,
        },
        {"cc-by": bullet + "<li><code>gsplats_kidney</code> (2.1 MB).</li>"},
    )

    assert [finding.level for finding in findings] == ["OK"]
    assert "STALE" not in report
    assert status == 0


def test_the_ok_line_claims_only_what_was_checked(tmp_path: Path) -> None:
    """An `OK` means the contradiction was absent, not that agreement was found.

    The gastrulation bullet never says the imaging is unpublished; its
    "Imaging method: Royer, ..." framing is correct, and reporting a missing
    statement as a finding would be wrong.
    """
    findings, report, _status = _run(
        tmp_path,
        {
            "gsplats_3d_drosophila_gastrulation": {
                "bucket": "zenodo",
                "record": "cc-by",
                "attribution": UNPUBLISHED_ATTRIBUTION,
                "files": [{"name": "gsplats_3d_drosophila_gastrulation.zip"}],
            }
        },
        {"cc-by": GASTRULATION_BULLET},
    )

    (finding,) = findings
    assert finding.level == "OK"
    assert finding.message == (
        "no claim that a publication describes the imaging was found in cc-by.html"
    )
    assert report.startswith(
        "[OK]     gsplats_3d_drosophila_gastrulation (record cc-by): no claim "
        "that a publication describes the imaging was found in cc-by.html"
    )


@pytest.mark.parametrize(
    "attribution",
    [
        "Royer lab. The imaging itself is unpublished.",
        "Royer lab. This imaging is unpublished.",
        "Royer lab. The imaging remains unpublished.",
        "Royer lab. The imaging has never been separately published; for the "
        "SiMView instrument see Royer et al.",
        "Royer lab. The imaging has not been published.",
        "Royer lab. The imaging was not separately deposited.",
        "Royer lab. Unpublished imaging from the Keller lab.",
        "Royer lab. Unpublished imaging; for the instrument see Royer et al.",
    ],
)
def test_every_unpublished_imaging_phrasing_keeps_a_dataset_inspected(
    tmp_path: Path, attribution: str
) -> None:
    """A one-sided reword of the attribution must not disarm the audit."""
    dataset = _droso_dataset()
    dataset["attribution"] = attribution
    findings, _report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": dataset},
        {"droso-timelapse": DROSO_BULLET},
    )

    assert [finding.level for finding in findings] == ["STALE"]
    assert status == 1


def test_a_record_bullet_claiming_unpublished_imaging_is_inspected(
    tmp_path: Path,
) -> None:
    """The inverse drift: the record says unpublished, the manifest cites a paper."""
    dataset = _droso_dataset()
    dataset["attribution"] = (
        "Acquired in the Keller lab. Imaging described in Royer et al., Nature "
        "Biotechnology 34, 1267-1278 (2016)."
    )
    findings, report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": dataset},
        {
            "droso-timelapse": _droso_bullet(
                "This imaging is unpublished. Imaging described in Someone et "
                "al., Journal 1, 2 (2020)."
            )
        },
    )

    assert [finding.level for finding in findings] == ["STALE"]
    assert "gsplats_4d_drosophila_embryogenesis" in report
    assert status == 1


def test_dataset_without_an_unpublished_claim_is_never_inspected(
    tmp_path: Path,
) -> None:
    findings, _report, status = _run(
        tmp_path,
        {
            "gsplats_kidney": {
                "bucket": "zenodo",
                "record": "droso-timelapse",
                "attribution": "scikit-image sample data (CC0).",
                "files": [{"name": "kidney_ch0.gsplats.zarr.zip"}],
            }
        },
        {"droso-timelapse": DROSO_BULLET + DROSO_RECORD_PARAGRAPH},
    )

    # Nothing to compare at all is a broken input, not a clean bill of health.
    assert [finding.level for finding in findings] == ["CONFIG"]
    assert status == 2


def test_missing_snapshot_is_a_broken_input_not_drift(tmp_path: Path) -> None:
    findings, report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {"cc-by": GASTRULATION_BULLET},
    )

    assert [finding.level for finding in findings] == ["CONFIG"]
    assert "no captured record text to compare" in report
    # A `CONFIG` needs attention too, so it carries the remedy block.
    assert "Remedy:" in report
    assert status == 2


def test_an_uncaptured_record_holding_nothing_in_scope_is_silent(
    tmp_path: Path,
) -> None:
    """Scope is resolved before any input-level `CONFIG`.

    Adding a manifest dataset and capturing its record later is an ordinary
    sequence, and no gate asserts that every manifest record is captured. A
    CC0 dataset making no unpublished-imaging claim is not this audit's
    business, so it must not escalate the whole leg to ERROR and relabel the
    real drift on another record as "could not run".
    """
    findings, report, status = _run(
        tmp_path,
        {
            "gsplats_4d_drosophila_embryogenesis": _droso_dataset(),
            "gsplats_new_thing": {
                "bucket": "zenodo",
                "record": "not-captured-yet",
                "attribution": "scikit-image sample data (CC0).",
                "files": [{"name": "new_thing.gsplats.zarr.zip"}],
            },
        },
        {"droso-timelapse": DROSO_BULLET},
    )

    assert [(f.level, f.dataset) for f in findings] == [
        ("STALE", "gsplats_4d_drosophila_embryogenesis")
    ]
    assert "CONFIG" not in report
    assert status == 1


def test_a_bulletless_record_holding_nothing_in_scope_is_silent(
    tmp_path: Path,
) -> None:
    """The same rule for a captured record that simply has no per-file list."""
    findings, report, status = _run(
        tmp_path,
        {
            "gsplats_4d_drosophila_embryogenesis": _droso_dataset(),
            "gsplats_kidney": {
                "bucket": "zenodo",
                "record": "cc-by",
                "attribution": "scikit-image sample data (CC0).",
                "files": [{"name": "kidney_ch0.gsplats.zarr.zip"}],
            },
        },
        {"droso-timelapse": DROSO_BULLET, "cc-by": DROSO_RECORD_PARAGRAPH},
    )

    assert [(f.level, f.dataset) for f in findings] == [
        ("STALE", "gsplats_4d_drosophila_embryogenesis")
    ]
    assert "has no per-file bullets" not in report
    assert status == 1


def test_unknown_record_key_is_a_broken_input(tmp_path: Path) -> None:
    dataset = _droso_dataset()
    dataset["record"] = "not-a-record"
    findings, _report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": dataset},
        {"droso-timelapse": DROSO_BULLET},
    )

    assert [finding.level for finding in findings] == ["CONFIG"]
    assert status == 2


def test_a_dataset_with_no_record_at_all_is_reported_as_unset(tmp_path: Path) -> None:
    dataset = _droso_dataset()
    dataset["record"] = None
    findings, report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": dataset},
        {"droso-timelapse": DROSO_BULLET},
    )

    assert [finding.level for finding in findings] == ["CONFIG"]
    assert "(record (unset))" in report
    assert status == 2


def test_unparsable_manifest_is_a_broken_input(tmp_path: Path) -> None:
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text("{not json", encoding="utf-8")
    snapshot_dir = tmp_path / "zenodo_record_text"
    snapshot_dir.mkdir()

    findings = CHECKER.audit(manifest, snapshot_dir)

    assert [finding.level for finding in findings] == ["CONFIG"]
    assert CHECKER.exit_status(findings) == 2


def test_a_manifest_whose_datasets_are_not_an_object_is_a_broken_input(
    tmp_path: Path,
) -> None:
    """Well-formed JSON of the wrong shape must still honour the exit-2 contract.

    `{"datasets": "oops"}` used to escape it as an uncaught `AttributeError`,
    which the aggregator reads as a traceback rather than a `CONFIG` finding.
    """
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text(json.dumps({"datasets": "oops"}), encoding="utf-8")
    snapshot_dir = tmp_path / "zenodo_record_text"
    snapshot_dir.mkdir()

    findings = CHECKER.audit(manifest, snapshot_dir)

    assert [finding.level for finding in findings] == ["CONFIG"]
    assert "not an object" in findings[0].message
    assert CHECKER.exit_status(findings) == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("files", 5),
        ("variants", ["a"]),
        ("variants", {"v": "oops"}),
        ("variants", {"v": {"files": 5}}),
    ],
)
def test_malformed_dataset_labelling_fields_honour_the_exit_2_contract(
    tmp_path: Path, field: str, value: object
) -> None:
    """Wrong nested shapes are skipped instead of escaping as a traceback."""
    dataset = _droso_dataset()
    dataset["files"] = []
    dataset[field] = value
    findings, _report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": dataset},
        {"droso-timelapse": DROSO_BULLET},
    )

    assert [finding.level for finding in findings] == ["CONFIG"]
    assert "no bullet" in findings[0].message
    assert status == 2


def test_a_file_entry_that_is_not_an_object_does_not_break_labelling(
    tmp_path: Path,
) -> None:
    """A bare string in `files` is skipped; the sibling object still labels."""
    dataset = _droso_dataset()
    dataset["files"] = [
        "drosophila_embryogenesis_500tp.gsplats.zarr.zip",
        {"name": "sidecar.zip"},
    ]
    findings, _report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": dataset},
        {
            "droso-timelapse": DROSO_BULLET
            + "<li><code>sidecar.zip</code> (1 MB). Imaging described in Someone "
            "et al., Journal 1, 2 (2020).</li>"
        },
    )

    # The bare string names no bullet, so only the `sidecar.zip` bullet is
    # owned - the verbatim droso bullet is not scanned for this dataset.
    (finding,) = findings
    assert finding.level == "STALE"
    assert finding.quotes == (
        "bullet: Imaging described in Someone et al., Journal 1, 2 (2020).",
    )
    assert status == 1


def test_dataset_with_no_bullet_of_its_own_is_a_broken_input(tmp_path: Path) -> None:
    findings, report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {"droso-timelapse": "<li><code>some_other_file.zip</code> (1 MB).</li>"},
    )

    assert [finding.level for finding in findings] == ["CONFIG"]
    assert "no bullet in droso-timelapse.html names" in report
    assert status == 2


def test_snapshot_without_bullets_is_a_broken_input(tmp_path: Path) -> None:
    findings, report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {"droso-timelapse": DROSO_RECORD_PARAGRAPH},
    )

    assert [finding.level for finding in findings] == ["CONFIG"]
    assert "has no per-file bullets" in report
    assert status == 2


def test_ambiguous_record_prose_is_a_human_check_not_a_dataset_finding(
    tmp_path: Path,
) -> None:
    """A record hosting other datasets may legitimately describe one of those.

    It still has to be reported: the contradiction is real, only its owner is
    unknown, and dropping it silently is how the same half of the droso record
    went unexamined.
    """
    findings, report, status = _run(
        tmp_path,
        {
            "gsplats_3d_drosophila_gastrulation": {
                "bucket": "zenodo",
                "record": "cc-by",
                "attribution": UNPUBLISHED_ATTRIBUTION,
                "files": [{"name": "droso_gastrulation.gsplats.zarr.zip"}],
            },
            "gsplats_kidney": {
                "bucket": "zenodo",
                "record": "cc-by",
                "attribution": "scikit-image sample data (CC0).",
                "files": [{"name": "kidney_ch0.gsplats.zarr.zip"}],
            },
        },
        {
            "cc-by": GASTRULATION_BULLET
            + "<li><code>gsplats_kidney</code> (2.1 MB).</li>"
            + DROSO_RECORD_PARAGRAPH
        },
    )

    assert [finding.level for finding in findings] == ["OK", "HUMAN"]
    human = findings[1]
    assert human.dataset == "(record prose)"
    assert human.record == "cc-by"
    assert "record prose names no dataset" in human.message
    assert "a human must decide" in human.message
    assert human.quotes == (
        "record prose: The imaging is described in Royer et al., Nature "
        "Biotechnology 34, 1267-1278 (2016), doi:10.1038/nbt.3708, CC BY 4.0.",
    )
    # A zero exit would grade PASS through the aggregator and hide the line.
    assert status == 1
    assert "capture.py" in report


def test_record_prose_is_blamed_when_every_dataset_makes_the_claim(
    tmp_path: Path,
) -> None:
    findings, report, status = _run(
        tmp_path,
        {
            "gsplats_3d_drosophila_gastrulation": {
                "bucket": "zenodo",
                "record": "cc-by",
                "attribution": UNPUBLISHED_ATTRIBUTION,
                "files": [{"name": "droso_gastrulation.gsplats.zarr.zip"}],
            }
        },
        {"cc-by": GASTRULATION_BULLET + DROSO_RECORD_PARAGRAPH},
    )

    (finding,) = findings
    assert finding.level == "STALE"
    assert finding.quotes == (
        "record prose: The imaging is described in Royer et al., Nature "
        "Biotechnology 34, 1267-1278 (2016), doi:10.1038/nbt.3708, CC BY 4.0.",
    )
    assert status == 1
    assert "capture.py" in report


def test_non_zenodo_datasets_are_ignored(tmp_path: Path) -> None:
    findings, _report, status = _run(
        tmp_path,
        {
            "gsplats_4d_drosophila_embryogenesis": _droso_dataset(),
            "gsplats_tribolium": {
                "bucket": "local-compute",
                "record": None,
                "attribution": "The imaging itself is unpublished.",
                "files": [{"name": "tribolium.gsplats.zarr.zip"}],
            },
        },
        {"droso-timelapse": DROSO_BULLET},
    )

    assert [(f.level, f.dataset) for f in findings] == [
        ("STALE", "gsplats_4d_drosophila_embryogenesis")
    ]
    assert status == 1


def test_variant_file_names_label_a_bullet(tmp_path: Path) -> None:
    """`h2afva` labels its bullets with variant file names, not the key."""
    findings, _report, status = _run(
        tmp_path,
        {
            "h2afva": {
                "bucket": "zenodo",
                "record": "h2afva",
                "attribution": "Royer lab. The imaging itself is unpublished.",
                "dir": "h2afva",
                "variants": {
                    "51tp": {"files": [{"name": "h2afva_51tp.gsplats.zarr.zip"}]}
                },
            }
        },
        {
            "h2afva": "<li><code>h2afva_51tp.gsplats.zarr.zip</code> (1.1 GB). "
            "Imaging described in Someone et al., Journal 1, 2 (2020).</li>"
        },
    )

    assert [finding.level for finding in findings] == ["STALE"]
    assert status == 1


def test_a_bullet_belongs_to_a_dataset_only_on_an_exact_label(tmp_path: Path) -> None:
    """`h2afva` must not inherit the `h2afva_253tp` bullet by prefix."""
    findings, _report, status = _run(
        tmp_path,
        {
            "h2afva": {
                "bucket": "zenodo",
                "record": "h2afva",
                "attribution": "Royer lab. The imaging itself is unpublished.",
                "variants": {
                    "51tp": {"files": [{"name": "h2afva_51tp.gsplats.zarr.zip"}]}
                },
            }
        },
        {
            "h2afva": "<li><code>h2afva_51tp.gsplats.zarr.zip</code> (1.1 GB). "
            "This imaging is unpublished.</li>"
            "<li><code>h2afva_253tp.gsplats.zarr.zip</code> (5.9 GB). Imaging "
            "described in Someone et al., Journal 1, 2 (2020).</li>"
        },
    )

    assert [finding.level for finding in findings] == ["OK"]
    assert status == 0


def test_only_the_leading_label_claims_a_bullet(tmp_path: Path) -> None:
    """A bullet that merely mentions another file must not be blamed on it.

    The live droso bullet already carries two `<code>` spans, the second of
    which is a genotype; a sibling file name in passing is just as plausible.
    """
    findings, _report, status = _run(
        tmp_path,
        {
            "h2afva": {
                "bucket": "zenodo",
                "record": "h2afva",
                "attribution": "Royer lab. This imaging is unpublished.",
                "variants": {
                    "51tp": {"files": [{"name": "h2afva_51tp.gsplats.zarr.zip"}]}
                },
            }
        },
        {
            "h2afva": "<li><code>h2afva_51tp.gsplats.zarr.zip</code> (1.1 GB). "
            "This imaging is unpublished.</li>"
            "<li><code>h2afva_253tp.gsplats.zarr.zip</code> (5.9 GB). A strided "
            "slice of <code>h2afva_51tp.gsplats.zarr.zip</code>. Imaging "
            "described in Someone et al., Journal 1, 2 (2020).</li>"
        },
    )

    assert [finding.level for finding in findings] == ["OK"]
    assert status == 0


def test_a_bullet_carrying_attributes_is_not_leaked_into_another_dataset(
    tmp_path: Path,
) -> None:
    """`<li class="...">` is a list item too, and misparsing it misattributes."""
    findings, _report, status = _run(
        tmp_path,
        {
            "dataset_a": {
                "bucket": "zenodo",
                "record": "cc-by",
                "attribution": "Royer lab. This imaging is unpublished.",
                "files": [{"name": "a.gsplats.zarr.zip"}],
            },
            "dataset_b": {
                "bucket": "zenodo",
                "record": "cc-by",
                "attribution": "Royer lab. This imaging is unpublished.",
                "files": [{"name": "b.gsplats.zarr.zip"}],
            },
        },
        {
            "cc-by": "<li><code>a.gsplats.zarr.zip</code> (1 MB). The imaging "
            "itself is unpublished.</li>"
            '<li class="file"><code>b.gsplats.zarr.zip</code> (2 MB). Imaging '
            "described in Someone et al., Journal 1, 2 (2020).</li>"
        },
    )

    assert [(f.level, f.dataset) for f in findings] == [
        ("OK", "dataset_a"),
        ("STALE", "dataset_b"),
    ]
    assert status == 1


def test_every_bullet_a_dataset_owns_is_scanned(tmp_path: Path) -> None:
    dataset = _droso_dataset()
    dataset["files"] = [{"name": "first.zip"}, {"name": "second.zip"}]
    findings, _report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": dataset},
        {
            "droso-timelapse": "<li><code>first.zip</code> (1 MB). Imaging "
            "described in First et al., Journal 1, 2 (2019).</li>"
            "<li><code>second.zip</code> (2 MB). Imaging described in Second et "
            "al., Journal 3, 4 (2020).</li>"
        },
    )

    (finding,) = findings
    assert finding.level == "STALE"
    assert [quote.split(" in ")[1].split(" et")[0] for quote in finding.quotes] == [
        "First",
        "Second",
    ]
    assert status == 1


def test_every_contradicting_sentence_in_a_bullet_is_quoted(tmp_path: Path) -> None:
    findings, _report, _status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {
            "droso-timelapse": _droso_bullet(
                "Imaging described in First et al., Journal 1, 2 (2019). The "
                "imaging is documented in Second et al., Journal 3, 4 (2020)."
            )
        },
    )

    (finding,) = findings
    assert finding.level == "STALE"
    assert len(finding.quotes) == 2
    assert "First et al." in finding.quotes[0]
    assert "Second et al." in finding.quotes[1]


def test_html_entities_in_a_bullet_are_decoded_before_matching(tmp_path: Path) -> None:
    """Zenodo's editor emits `&nbsp;` routinely; it must not hide a claim."""
    findings, _report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {
            "droso-timelapse": _droso_bullet(
                "Imaging described&nbsp;in Royer et al., Nature Biotechnology "
                "34 (2016)."
            )
        },
    )

    assert [finding.level for finding in findings] == ["STALE"]
    assert status == 1


def test_html_entities_in_a_code_label_are_decoded(tmp_path: Path) -> None:
    """An escaped label still has to resolve to its dataset."""
    findings, _report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {
            "droso-timelapse": "<li><code>drosophila&#95;embryogenesis&#95;500tp"
            ".gsplats.zarr.zip</code> (863.8 MB). Imaging described in Royer et "
            "al., Nature Biotechnology 34 (2016).</li>"
        },
    )

    assert [finding.level for finding in findings] == ["STALE"]
    assert status == 1


def test_markup_is_stripped_before_entities_are_decoded(tmp_path: Path) -> None:
    """The live cc-by text contains a literal `&lt;dataset&gt;`.

    Decoding first turns it into a tag and the tag strip then eats the word,
    so the quote would lose the very placeholder it is about.
    """
    findings, _report, _status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {
            "droso-timelapse": _droso_bullet(
                "Imaging of the &lt;dataset&gt; crop is described in Royer et "
                "al., Nature Biotechnology 34 (2016)."
            )
        },
    )

    (finding,) = findings
    assert finding.level == "STALE"
    assert "<dataset>" in finding.quotes[0]


@pytest.mark.parametrize(
    ("sentence", "tail"),
    [
        (
            "Imaging described in Royer, L. A. Adaptive light-sheet microscopy "
            "for long-term imaging, Nature Biotechnology 34, "
            "doi:10.1038/nbt.3708.",
            "doi:10.1038/nbt.3708.",
        ),
        (
            "Imaging described in Royer et al., Nat. Biotechnol. 34, 1267-1278 (2016).",
            "Nat. Biotechnol. 34, 1267-1278 (2016).",
        ),
    ],
)
def test_a_quote_survives_the_abbreviations_in_a_citation(
    tmp_path: Path, sentence: str, tail: str
) -> None:
    """Cutting at the first period drops the reference that makes it actionable."""
    findings, _report, _status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {"droso-timelapse": _droso_bullet(sentence)},
    )

    (finding,) = findings
    assert finding.level == "STALE"
    assert finding.quotes == (f"bullet: {sentence}",)
    assert finding.quotes[0].endswith(tail)


def test_the_report_always_names_the_remedy_that_is_not_regeneration(
    tmp_path: Path,
) -> None:
    _findings, report, _status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {"droso-timelapse": DROSO_BULLET},
    )

    assert "scripts/zenodo_record_text/capture.py" in report
    assert "Never hand-edit the HTML" in report
    assert "gen_zenodo_records.py --render" in report


def test_a_clean_report_carries_no_remedy_block(tmp_path: Path) -> None:
    """Nothing needs fixing, so the report must not tell anyone to fix Zenodo."""
    _findings, report, status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {"droso-timelapse": _droso_bullet("This imaging is unpublished.")},
    )

    assert status == 0
    assert "Remedy:" not in report
    assert report.count("\n") == 1


def test_an_overlong_sentence_is_truncated_in_the_quote(tmp_path: Path) -> None:
    """A record may hold an unbroken paragraph; a report line stays readable."""
    filler = "This record holds a long unbroken account of the acquisition, "
    sentence = f"{filler * 5}and the imaging is described in Royer et al."
    assert len(sentence) > CHECKER._MAX_QUOTE_CHARS

    findings, _report, _status = _run(
        tmp_path,
        {"gsplats_4d_drosophila_embryogenesis": _droso_dataset()},
        {"droso-timelapse": _droso_bullet(sentence)},
    )

    (finding,) = findings
    assert finding.level == "STALE"
    (quote,) = finding.quotes
    assert quote.endswith(" ...")
    assert len(quote) <= len("bullet: ") + CHECKER._MAX_QUOTE_CHARS + len(" ...")


def test_report_lines_carry_a_level_marker() -> None:
    findings = [
        CHECKER.Finding("OK", "cc-by", "a", "fine"),
        CHECKER.Finding("HUMAN", "cc-by", "(record prose)", "unattributable"),
        CHECKER.Finding("STALE", "droso-timelapse", "b", "drifted"),
        CHECKER.Finding("CONFIG", "droso-timelapse", "c", "unreadable"),
    ]

    report = CHECKER.render(findings)

    marker_lines = report.splitlines()[:4]
    for line, level in zip(
        marker_lines, ("OK", "HUMAN", "STALE", "CONFIG"), strict=True
    ):
        assert line.startswith(f"[{level}]")


def test_main_prints_the_report_and_exits_on_its_own_contract() -> None:
    """The only entry point CI invokes, run as CI invokes it.

    Asserted independently of the module's own functions on purpose: comparing
    the subprocess against in-process `audit()`/`render()` calls passes under
    any breakage the two share, which is most of them.
    """
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(CHECKER_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.stderr == ""
    assert completed.returncode in {0, 1, 2}
    assert re.search(r"^\[(OK|HUMAN|STALE|CONFIG)]", completed.stdout, re.MULTILINE)


def test_the_live_repository_audit_blames_only_real_manifest_datasets() -> None:
    """Deliberately weak, and it must stay that way. Do not strengthen it.

    `scripts/tests` is in `pyproject.toml`'s `testpaths` and runs inside
    `ci.yml`'s **required** `python-tests` job, and `^scripts/zenodo_record_text/`
    is in that workflow's python diff domain. So a re-capture PR - the very
    remedy this audit prints - executes this test in a required gate. The
    wording being measured is authored on Zenodo by the maintainer and cannot
    be changed by a commit here, so any assertion about what the audit *finds*
    would put a required check under the control of third-party prose: one
    added sentence of record-level prose raises a `[HUMAN]` line and reddens
    the gate. See the module docstring of `check_record_attribution.py`.

    Assert only what this repository controls: the audit runs, it produces a
    report, its levels come from the documented vocabulary, and it blames
    nothing that is not a manifest dataset (or one of the two record-scoped
    sentinels). Not a count, not a level, not the absence of a `CONFIG`.
    """
    findings = CHECKER.audit()
    manifest = json.loads(CHECKER.MANIFEST_PATH.read_text(encoding="utf-8"))

    assert findings
    assert {finding.level for finding in findings} <= {"OK", "HUMAN", "STALE", "CONFIG"}
    nameable = set(manifest["datasets"]) | {"-", "(record prose)"}
    for finding in findings:
        named = {part.strip() for part in finding.dataset.split(",")}
        assert named <= nameable, finding
