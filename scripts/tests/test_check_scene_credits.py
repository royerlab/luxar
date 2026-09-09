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
    "license": "CC BY 4.0",
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


@pytest.mark.parametrize("field", ["doi", "license"])
def test_attribution_field_drift_is_caught(tmp_path: Path, field: str) -> None:
    carried = {**CITED, field: "wrong"}
    store = _v3_store(tmp_path, "drift", {"citation": carried})
    problem = compare(store, CITED)
    assert problem is not None
    assert field in problem and "wrong" in problem and str(CITED[field]) in problem


def test_presentational_reference_drift_is_ignored(tmp_path: Path) -> None:
    declared = {**CITED, "ref": "caption-only reference"}
    store = _v3_store(tmp_path, "reference", {"citation": CITED})
    assert compare(store, declared) is None


@pytest.mark.parametrize("carried", ["Someone 2020", ["Someone 2020"]])
@pytest.mark.parametrize("declared", [CITED, None])
def test_non_mapping_citation_is_reported(
    tmp_path: Path, carried: object, declared: dict | None
) -> None:
    store = _v3_store(tmp_path, "malformed", {"citation": carried})
    problem = compare(store, declared)
    assert problem is not None
    assert "malformed citation record" in problem and repr(carried) in problem


@pytest.mark.parametrize("carried", [{}, {"doi": CITED["doi"]}])
def test_citation_without_short_is_reported(tmp_path: Path, carried: dict) -> None:
    store = _v3_store(tmp_path, "malformed", {"citation": carried})
    problem = compare(store, CITED)
    assert problem is not None and "has no 'short'" in problem


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
    """A checkout without generated scenes must not fail the gate.

    Still exit 0 — the output directory is gitignored, so this is the normal
    state of a fresh clone and of CI. But the message must not read like a
    result: it now says INSPECTED NOTHING, because "no built demo scenes found;
    nothing to check" was indistinguishable from a clean bill of health at a
    glance (audit A9-04).
    """
    assert main(["--demos-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "INSPECTED NOTHING" in out
    assert "not a pass" in out


def test_require_scenes_turns_an_empty_inventory_into_a_failure(
    tmp_path: Path, capsys
) -> None:
    """The opt-in for callers that KNOW demos should be present.

    The default no-op is right for a fresh checkout and wrong for gallery
    generation or the pre-upload audit, where an empty inventory means the build
    did not produce what it was supposed to. `--require-scenes` is how such a
    caller says "silence here is a failure".
    """
    assert main(["--demos-dir", str(tmp_path), "--require-scenes"]) == 1
    assert "INSPECTED NOTHING" in capsys.readouterr().out


def test_exit_code_is_non_zero_when_a_store_contradicts_its_demo(
    tmp_path: Path, capsys
) -> None:
    store = _v3_store(tmp_path, "cosmicflows_laniakea_full", {"content_hash": "x"})
    # Addressed by path, so the demo lookup resolves through `outputs`.
    code = main([str(store)])
    out = capsys.readouterr().out
    assert code == 1, out
    assert "carries no citation" in out


def test_default_discovery_checks_built_demo_stores(tmp_path: Path, capsys) -> None:
    _v3_store(tmp_path, "cosmicflows_laniakea_full", {"content_hash": "x"})
    code = main(["--demos-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1, out
    assert "checked 1 built scene(s)" in out and "carries no citation" in out


def test_unknown_explicit_store_fails_instead_of_passing(
    tmp_path: Path, capsys
) -> None:
    """A named path that is not a demo output is the caller's error — exit 1.

    THIS TEST PREVIOUSLY ASSERTED EXIT 0, i.e. it pinned the fail-open as
    intended behaviour. Naming one unknown path left `targets` empty while the
    skip list stayed populated, so the empty-inventory guard did not fire, the
    compare loop ran zero times, and the run printed "checked 0 built scene(s);
    problems: 0" and exited 0 — a green tick for an inspection that never
    happened. Verified against the pre-fix script before changing it.

    A typo and a demo output renamed since the caller wrote the command both
    land here, and both deserve to be heard rather than silently tolerated.
    """
    store = _v3_store(tmp_path, "my_analysis", {"citation": CITED})
    assert main([str(store)]) == 1
    out = capsys.readouterr().out
    assert "are not demo outputs" in out
    assert "not a known demo output, skipped" in out


def test_known_explicit_archive_is_named_then_skipped(tmp_path: Path, capsys) -> None:
    """An archive is a REAL demo output this tool cannot open — still exit 0.

    Deliberately not treated as the unknown-path case above: the caller named
    the right artifact and the limitation is ours. The report must still say
    plainly that nothing was inspected, and `--require-scenes` still escalates.
    """
    store = tmp_path / "cosmicflows_laniakea_full.luxar.zarr.zip"
    store.write_bytes(b"not opened")
    assert main([str(store)]) == 0
    out = capsys.readouterr().out
    assert "archive stores are not inspected, skipped" in out
    assert "INSPECTED NOTHING" in out


def test_unknown_path_does_not_hide_a_skipped_archive(tmp_path: Path, capsys) -> None:
    """Every named input is reported even when none can be inspected."""
    archive = tmp_path / "cosmicflows_laniakea_full.luxar.zarr.zip"
    archive.write_bytes(b"not opened")
    unknown = tmp_path / "typo.luxar.zarr"

    assert main([str(unknown), str(archive)]) == 1

    out = capsys.readouterr().out
    assert "are not demo outputs" in out
    assert "archive stores are not inspected, skipped" in out
    assert "INSPECTED NOTHING" in out


def test_require_scenes_escalates_an_all_skipped_run(tmp_path: Path) -> None:
    """The archive case is tolerated by default but not when scenes are required."""
    store = tmp_path / "cosmicflows_laniakea_full.luxar.zarr.zip"
    store.write_bytes(b"not opened")
    assert main([str(store), "--require-scenes"]) == 1


def test_a_mixed_run_with_one_checkable_store_still_inspects(
    tmp_path: Path, capsys
) -> None:
    """The unknown-path failure is not a blanket ban on unrecognised arguments.

    One inspectable target is enough for the run to do its job; only a run that
    inspected NOTHING is suspect. Without this the fix would be over-broad and
    would break any caller passing a mixed glob.
    """
    good = _v3_store(
        tmp_path,
        "cosmicflows_laniakea_full",
        {
            "citation": {
                "short": (
                    "Tully et al. 2023 (Cosmicflows-4); Laniakea, Tully et al. 2014"
                ),
                "doi": "10.3847/1538-4357/ac94d8",
            }
        },
    )
    unknown = _v3_store(tmp_path, "my_analysis", {"citation": CITED})

    code = main([str(good), str(unknown)])

    out = capsys.readouterr().out
    # Exit 1 for the unknown path, but the report must show it looked at `good`
    # rather than bailing before the compare loop.
    assert code == 1
    assert "are not demo outputs" in out
    assert "checked 1 built scene(s); problems: 0; unrecognised: 1; skipped: 0" in out
