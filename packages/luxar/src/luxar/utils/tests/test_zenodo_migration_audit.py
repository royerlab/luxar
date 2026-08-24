"""Tests for scripts/zenodo_migration_audit.py (the demo-data migration audit).

The script is a standalone repo script (not part of the installed ``luxar``
package), so it is loaded from ``REPO_ROOT/scripts/`` via importlib and the whole
module is skipped on a packaged install that ships no ``scripts/`` tree.

Two things are worth guarding: that the audit RUNS end to end (its sections are
wired together by hand, and a mismatch between them is invisible until the last
line prints), and that "ready to upload" means what it says — Zenodo publication
cannot be undone, so a dataset whose declared files are only partly here, or
whose in-tree copy is an unpulled git-LFS pointer, must not be reported ready.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import urllib.error
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[6]
_SCRIPT = REPO_ROOT / "scripts" / "zenodo_migration_audit.py"

pytestmark = pytest.mark.skipif(
    not _SCRIPT.exists(),
    reason="audit script not present (packaged install without repo scripts/)",
)

_LFS_POINTER = (
    b"version https://git-lfs.github.com/spec/v1\n"
    b"oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
    b"size 12345\n"
)


def _load(repo: Path, cache: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the audit against a throwaway repo tree and cache root.

    ``REPO`` comes from the first positional argument at import time, so the argv
    patch has to be in place before the module body runs; ``CACHE`` is
    home-relative and is redirected afterwards.
    """
    monkeypatch.setattr(sys, "argv", ["zenodo_migration_audit", str(repo)])
    spec = importlib.util.spec_from_file_location("zenodo_migration_audit", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CACHE", cache)
    return module


def _write_manifest(repo: Path, datasets: dict) -> Path:
    """Write a minimal manifest into a fake repo tree; return its data dir."""
    demos = repo / "packages/luxar/src/luxar/demos"
    demos.mkdir(parents=True)
    (demos / "data_manifest.json").write_text(
        json.dumps(
            {
                "records": {
                    "cc-by": {
                        "license": "cc-by-4.0",
                        "zenodo_record": None,
                        "zenodo_doi": None,
                    }
                },
                "datasets": datasets,
            }
        )
    )
    data = demos / "data"
    data.mkdir()
    return data


def _two_file_dataset() -> dict:
    return {
        "ds": {
            "bucket": "zenodo",
            "record": "cc-by",
            "license": "cc-by-4.0",
            "dir": "ds",
            "files": [
                {"name": "a.zip", "sha256": "aa", "bytes": 1024},
                {"name": "b.zip", "sha256": "bb", "bytes": 1024},
            ],
        }
    }


def test_audit_runs_end_to_end_on_the_real_repo() -> None:
    """The whole audit prints its readiness summary and exits cleanly.

    Exit 1 is a legitimate result (undeclared files on disk); a traceback is
    not — the sections pass their results along by hand, and a signature that
    drifts from its call site only shows up here.
    """
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode in (0, 1), proc.stderr
    assert "Traceback" not in proc.stderr, proc.stderr
    assert "READINESS" in proc.stdout
    assert "datasets ready to upload now:" in proc.stdout


def test_complete_dataset_is_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    repo, cache = tmp_path / "repo", tmp_path / "cache"
    data = _write_manifest(repo, _two_file_dataset())
    (data / "ds").mkdir()
    (data / "ds" / "a.zip").write_bytes(b"real bytes")
    # The second file resolves from the cache: repo + cache together cover the
    # declared set, which is what "ready" means.
    (cache / "ds").mkdir(parents=True)
    (cache / "ds" / "b.zip").write_bytes(b"real bytes")

    audit = _load(repo, cache, monkeypatch)
    assert audit.main() == 0
    out = capsys.readouterr().out
    assert "datasets ready to upload now: 1" in out
    assert "incomplete here (NOT ready):  0" in out


def test_partial_dataset_is_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """One of two declared files present is INCOMPLETE, never ready."""
    repo, cache = tmp_path / "repo", tmp_path / "cache"
    data = _write_manifest(repo, _two_file_dataset())
    (data / "ds").mkdir()
    (data / "ds" / "a.zip").write_bytes(b"real bytes")

    audit = _load(repo, cache, monkeypatch)
    assert audit.main() == 0
    out = capsys.readouterr().out
    assert "INCOMPLETE: 1 of 2 files have bytes here" in out
    assert "datasets ready to upload now: 0" in out
    assert "incomplete here (NOT ready):  1" in out


def test_variant_files_are_looked_for_in_their_variant_subdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A variant's bytes live one level deeper, and the audit must look there.

    ``ensure_dataset`` resolves a variant under ``<dir>/<variant>/`` in-repo and
    ``<name>/<variant>/`` in the cache (h2afva's 51tp vs 253tp). Probing the
    dataset directory instead reports "BYTES NOT ON THIS MACHINE" for a file that
    is sitting right there, and counts the same file as UNDECLARED on disk.
    """
    repo, cache = tmp_path / "repo", tmp_path / "cache"
    data = _write_manifest(
        repo,
        {
            "ds": {
                "bucket": "zenodo",
                "record": "cc-by",
                "license": "cc-by-4.0",
                "dir": "ds",
                "variants": {
                    "big": {
                        "default": True,
                        "files": [{"name": "a.zip", "sha256": "aa", "bytes": 1024}],
                    }
                },
            }
        },
    )
    (data / "ds" / "big").mkdir(parents=True)
    (data / "ds" / "big" / "a.zip").write_bytes(b"real bytes")

    audit = _load(repo, cache, monkeypatch)
    assert audit.main() == 0
    out = capsys.readouterr().out
    assert "datasets ready to upload now: 1" in out
    assert "BYTES NOT ON THIS MACHINE" not in out
    assert "UNDECLARED (must be 0): 0" in out


def test_unpulled_lfs_pointer_counts_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A pointer stub is a path without bytes — nothing to upload from here."""
    repo, cache = tmp_path / "repo", tmp_path / "cache"
    data = _write_manifest(repo, _two_file_dataset())
    (data / "ds").mkdir()
    (data / "ds" / "a.zip").write_bytes(_LFS_POINTER)
    (data / "ds" / "b.zip").write_bytes(_LFS_POINTER)

    audit = _load(repo, cache, monkeypatch)
    assert audit.main() == 0
    out = capsys.readouterr().out
    assert "BYTES NOT ON THIS MACHINE" in out
    assert "datasets ready to upload now: 0" in out


# ---------------------------------------------------------------------------
# --live: the deposition comparison
#
# These exercise the pure checks directly, with no network. That is the point of
# the split: the whole comparison is a function of (manifest, depositions), so a
# tool that touches a publishable record can be tested without touching one.
# ---------------------------------------------------------------------------


def _dep(files: list[dict], desc: str = "", **meta: object) -> dict:
    """A deposition dict shaped like the API's, with publishable metadata."""
    base = {
        "title": "t",
        "creators": [{"name": "a"}],
        "license": "cc-by-4.0",
        "upload_type": "dataset",
        "description": desc,
    }
    base.update(meta)
    return {"submitted": False, "metadata": base, "files": files}


def _audit_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the audit against the real repo (these tests never read the tree)."""
    return _load(REPO_ROOT, REPO_ROOT / "nonexistent-cache", monkeypatch)


def test_pins_include_variant_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """A variant's files are pinned on the same footing as a dataset's own.

    h2afva declares its files only under ``variants``, so a pin collector that
    reads ``spec["files"]`` alone reports it as having nothing to check — the
    dataset most in need of checking.
    """
    audit = _audit_module(monkeypatch)
    pins = audit.pins_of(
        {
            "h2afva": {
                "bucket": "zenodo",
                "record": "h2afva",
                "variants": {
                    "253tp": {
                        "files": [{"name": "big.zip", "sha256": "cc", "bytes": 9}]
                    }
                },
            }
        }
    )
    assert pins == {"big.zip": ("h2afva", 9)}


def test_a_clean_record_produces_no_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The happy path is silent — otherwise every real run cries wolf."""
    audit = _audit_module(monkeypatch)
    datasets = _two_file_dataset()
    desc = (
        "<li><code>ds</code> (0.0 MB)</li><table><tr>h</tr><tr>a</tr><tr>b</tr></table>"
    )
    dep = _dep(
        [
            {"filename": "a.zip", "filesize": 1024},
            {"filename": "b.zip", "filesize": 1024},
        ],
        desc=desc,
    )
    fails, warns = audit.check_deposition(
        "cc-by", dep, audit.pins_of(datasets), audit.dataset_totals(datasets), {}
    )
    assert fails == []
    assert warns == []


def test_pin_and_record_are_compared_in_both_directions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A superseded file left ON the record and a pin never uploaded both fail.

    Checking one direction only is how the migration mislaid things twice: a
    stale object keeps a right-looking name, and a pin for an absent file reads
    as success until the download 404s after publication.
    """
    audit = _audit_module(monkeypatch)
    datasets = _two_file_dataset()
    dep = _dep(
        [
            {"filename": "a.zip", "filesize": 999},  # wrong size
            {"filename": "stray.zip", "filesize": 1},  # on record, not pinned
        ],
        desc="<table><tr>h</tr><tr>1</tr><tr>2</tr></table>",
    )
    fails, _ = audit.check_deposition(
        "cc-by", dep, audit.pins_of(datasets), audit.dataset_totals(datasets), {}
    )
    joined = "\n".join(fails)
    assert "a.zip pinned 1,024 but hosted 999" in joined
    assert "stray.zip is on the record but NOT pinned" in joined
    assert "b.zip is pinned but ABSENT from the record" in joined


def test_a_file_on_the_wrong_record_is_named_as_misplaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cross-record upload must not be misreported as merely absent elsewhere."""
    audit = _audit_module(monkeypatch)
    pins = {"a.zip": ("cc-by", 10), "b.zip": ("cc-by-sa", 20)}
    dep = _dep(
        [
            {"filename": "a.zip", "filesize": 10},
            {"filename": "b.zip", "filesize": 20},
        ],
        desc="<table><tr>h</tr><tr>a</tr><tr>b</tr></table>",
    )

    fails, _ = audit.check_deposition("cc-by", dep, pins, {}, {})

    assert "[cc-by] b.zip is on this record but pinned to cc-by-sa" in fails


def test_bucket_style_and_incomplete_file_entries_are_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zenodo response drift produces findings, never KeyError or formatting errors."""
    audit = _audit_module(monkeypatch)
    pins = {"a.zip": ("cc-by", 10), "b.zip": ("cc-by", 20)}
    dep = _dep(
        [
            {"key": "a.zip", "size": 10},
            {"filename": "b.zip"},
            {"filesize": 30},
        ],
        desc="<table><tr>h</tr><tr>a</tr><tr>b</tr><tr>unknown</tr></table>",
    )

    fails, _ = audit.check_deposition("cc-by", dep, pins, {}, {})
    joined = "\n".join(fails)

    assert "b.zip has no filesize/size" in joined
    assert "file entry has no filename/key" in joined


def test_scratch_objects_and_submission_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probe object left by a failed upload, and an already-published record.

    Both are unfixable after the fact: a published record cannot be tidied, and
    publication cannot be undone.
    """
    audit = _audit_module(monkeypatch)
    datasets = _two_file_dataset()
    dep = _dep(
        [
            {"filename": "a.zip", "filesize": 1024},
            {"filename": "b.zip", "filesize": 1024},
            {"filename": "_write_probe.bin", "filesize": 8},
        ],
        desc="<table><tr>h</tr><tr>1</tr><tr>2</tr><tr>3</tr></table>",
    )
    dep["submitted"] = True
    fails, _ = audit.check_deposition(
        "cc-by", dep, audit.pins_of(datasets), audit.dataset_totals(datasets), {}
    )
    joined = "\n".join(fails)
    assert "ALREADY SUBMITTED" in joined
    assert "scratch file on the record: _write_probe.bin" in joined


def test_a_stale_size_claim_in_the_description_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The record text is the one place a stale number gets PUBLISHED.

    ``ds`` totals 2048 B = 0.0 MB; claiming 5.0 MB is the drift this catches.
    """
    audit = _audit_module(monkeypatch)
    datasets = _two_file_dataset()
    dep = _dep(
        [
            {"filename": "a.zip", "filesize": 1024},
            {"filename": "b.zip", "filesize": 1024},
        ],
        desc="<li><code>ds</code> (5.0 MB)</li><table><tr>h</tr><tr>1</tr><tr>2</tr></table>",
    )
    fails, _ = audit.check_deposition(
        "cc-by", dep, audit.pins_of(datasets), audit.dataset_totals(datasets), {}
    )
    assert any("description says ds is 5.0 MB" in f for f in fails)


def test_description_size_claims_compare_at_one_decimal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A human-written ``5 MB`` claim is equivalent to computed ``5.0 MB``."""
    audit = _audit_module(monkeypatch)
    datasets = {
        "ds": {
            "bucket": "zenodo",
            "record": "cc-by",
            "files": [{"name": "a.zip", "sha256": "aa", "bytes": 5_000_000}],
        }
    }
    dep = _dep(
        [{"filename": "a.zip", "filesize": 5_000_000}],
        desc="<li><code>ds</code> (5 MB)</li><table><tr>h</tr><tr>a</tr></table>",
    )

    fails, _ = audit.check_deposition(
        "cc-by", dep, audit.pins_of(datasets), audit.dataset_totals(datasets), {}
    )

    assert fails == []


def test_description_size_claim_still_has_to_use_the_canonical_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Numeric equivalence cannot hide a stale MB/GB presentation."""
    audit = _audit_module(monkeypatch)
    datasets = {
        "ds": {
            "bucket": "zenodo",
            "record": "cc-by",
            "files": [{"name": "a.zip", "sha256": "aa", "bytes": 5_000_000_000}],
        }
    }
    dep = _dep(
        [{"filename": "a.zip", "filesize": 5_000_000_000}],
        desc="<li><code>ds</code> (5000 MB)</li><table><tr>h</tr><tr>a</tr></table>",
    )

    fails, _ = audit.check_deposition(
        "cc-by", dep, audit.pins_of(datasets), audit.dataset_totals(datasets), {}
    )

    assert any("description says ds is 5000 MB, actually 5.0 GB" in f for f in fails)


def test_files_with_no_description_size_claim_are_warned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed description shape cannot silently disable the size audit."""
    audit = _audit_module(monkeypatch)
    datasets = _two_file_dataset()
    dep = _dep(
        [
            {"filename": "a.zip", "filesize": 1024},
            {"filename": "b.zip", "filesize": 1024},
        ],
        desc="<table><tr>h</tr><tr>a</tr><tr>b</tr></table>",
    )

    _, warns = audit.check_deposition(
        "cc-by", dep, audit.pins_of(datasets), audit.dataset_totals(datasets), {}
    )

    assert "[cc-by] description has 0 size claims for 2 files" in warns


def test_thead_is_warned_because_zenodo_strips_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``<thead>`` survives the payload you send and vanishes on the rendered page."""
    audit = _audit_module(monkeypatch)
    datasets = _two_file_dataset()
    dep = _dep(
        [
            {"filename": "a.zip", "filesize": 1024},
            {"filename": "b.zip", "filesize": 1024},
        ],
        desc="<table><thead><tr>h</tr></thead><tr>1</tr><tr>2</tr></table>",
    )
    _, warns = audit.check_deposition(
        "cc-by", dep, audit.pins_of(datasets), audit.dataset_totals(datasets), {}
    )
    assert any("<thead>" in w for w in warns)


def test_wholesale_mismatch_is_diagnosed_as_the_wrong_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nearly every pin failing means the wrong manifest, not broken records.

    This is the check that exists because of an incident: an audit run against a
    superseded manifest reported the records stale everywhere and was believed.
    The records were ahead; the re-pin was sitting in an unmerged PR.
    """
    audit = _audit_module(monkeypatch)
    pins = {f"f{i}.zip": ("cc-by", 10) for i in range(10)}
    fails = [f"[cc-by] f{i}.zip pinned 10 but hosted 20" for i in range(10)]
    note = audit.diagnose_manifest_staleness(fails, pins)
    assert note is not None
    assert "10 of 10 pins disagree" in note
    assert "not the one that will ship" in note


def test_a_couple_of_real_failures_are_not_diagnosed_away(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The staleness hint must not fire on ordinary findings.

    Two bad files out of ten is exactly the case the gate exists to report, so
    explaining it away as "wrong manifest" would suppress the real signal.
    """
    audit = _audit_module(monkeypatch)
    pins = {f"f{i}.zip": ("cc-by", 10) for i in range(10)}
    fails = [f"[cc-by] f{i}.zip pinned 10 but hosted 20" for i in range(2)]
    assert audit.diagnose_manifest_staleness(fails, pins) is None


def test_the_audit_can_only_read_from_zenodo() -> None:
    """No mutating verb anywhere in the source, and no publish call.

    This tool is pointed at records a human is about to publish by hand, so
    "read-only" has to be a property of the file rather than an intention. Same
    guard as ``scripts/zenodo_upload_draft.py`` carries for its own narrow write.
    """
    src = _SCRIPT.read_text()
    for forbidden in ('"POST"', "'POST'", '"PUT"', "'PUT'", '"DELETE"', "'DELETE'"):
        assert forbidden not in src, f"mutating verb {forbidden} in the audit"
    assert "/actions/publish" not in src
    assert src.count('method="GET"') >= 1
    assert "data=" not in src


def test_live_without_a_token_does_not_reach_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """``--live`` with no token reports and stops, rather than half-running.

    It returns non-zero: a live check that silently did not happen is worse than
    one that failed, because the caller is about to publish.
    """
    repo = tmp_path / "repo"
    data = _write_manifest(repo, _two_file_dataset())
    for n in ("a.zip", "b.zip"):
        (data / "ds").mkdir(exist_ok=True)
        (data / "ds" / n).write_bytes(b"x" * 1024)
    monkeypatch.setenv("ZENODO_TOKEN", "")
    monkeypatch.delenv("ZENODO_TOKEN", raising=False)
    audit = _load(repo, tmp_path / "cache", monkeypatch)
    monkeypatch.setattr(sys, "argv", ["zenodo_migration_audit", str(repo), "--live"])

    def _boom(*a: object, **k: object) -> dict:
        raise AssertionError("fetch attempted without a token")

    monkeypatch.setattr(audit, "fetch_deposition", _boom)
    assert audit.main() == 1
    assert "refusing to run --live without ZENODO_TOKEN" in capsys.readouterr().out


def test_fetch_deposition_reports_http_errors_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _audit_module(monkeypatch)
    error = urllib.error.HTTPError(
        "https://example.invalid", 401, "Unauthorized", {}, None
    )
    monkeypatch.setattr(
        audit.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(error)
    )

    with pytest.raises(SystemExit, match="HTTP 401 Unauthorized.*deposit:write scope"):
        audit.fetch_deposition("21912280", "token")


def test_live_refuses_to_skip_manifest_records_without_a_deposition_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    repo = tmp_path / "repo"
    _write_manifest(repo, {})
    manifest = repo / "packages/luxar/src/luxar/demos/data_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["records"] = {
        "cc-by": {"zenodo_record": "123"},
        "cc-by-sa": {"zenodo_record": None},
    }
    manifest.write_text(json.dumps(payload))
    monkeypatch.setenv("ZENODO_TOKEN", "token")
    audit = _load(repo, tmp_path / "cache", monkeypatch)
    monkeypatch.setattr(sys, "argv", ["zenodo_migration_audit", str(repo), "--live"])
    monkeypatch.setattr(
        audit,
        "fetch_deposition",
        lambda dep_id, token: _dep([], desc="<table><tr>header</tr></table>"),
    )

    assert audit.main() == 1
    output = capsys.readouterr().out
    skipped = "[cc-by-sa] manifest has no zenodo_record; live check did not run"
    assert skipped in output
    assert output.index("LIVE DEPOSITIONS vs MANIFEST") < output.index(skipped)
    assert "all pins match the live records" not in output


def test_live_fails_when_a_pin_names_an_unknown_record(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    audit = _audit_module(monkeypatch)
    manifest = {
        "records": {"cc-by": {"zenodo_record": "123"}},
        "datasets": {
            "checked": {
                "bucket": "zenodo",
                "record": "cc-by",
                "files": [{"name": "a.zip", "bytes": 1024}],
            },
            "typo": {
                "bucket": "zenodo",
                "record": "cc-by-40",
                "files": [{"name": "b.zip", "bytes": 2048}],
            },
        },
    }
    depositions = {
        "cc-by": _dep(
            [{"filename": "a.zip", "filesize": 1024}],
            desc=(
                "<li><code>checked</code> (0.0 MB)</li>"
                "<table><tr>h</tr><tr>a</tr></table>"
            ),
        )
    }

    assert audit._audit_live_depositions(manifest, depositions, []) == 1
    output = capsys.readouterr().out
    assert "records checked: 1   pins: 1" in output
    assert "FAIL [cc-by-40] no fetched deposition for pinned files: b.zip" in output
    assert "all pins match the live records" not in output


def test_the_live_check_compares_hosted_sizes_not_in_repo_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact shape #1734 lands in: `bytes` local, `hosted_bytes` the record's.

    Splitting `sha256` without splitting `bytes` left this comparing the wrong
    end of the contract, reporting a mismatch for a file that is correct — on
    EVERY diverged dataset, which with 21 of them would also trip the
    "wrong manifest" heuristic and tell someone holding the right manifest that
    they are holding the wrong one.
    """
    audit = _audit_module(monkeypatch)
    datasets = {
        "gsplats_multichannel": {
            "bucket": "zenodo",
            "record": "cc-by",
            "files": [
                {
                    "name": "blastocyst_ch0.gsplats.zarr.zip",
                    "sha256": "1" * 64,  # the in-repo copy
                    "bytes": 196680,
                    "hosted_sha256": "7" * 64,  # what the record serves
                    "hosted_bytes": 186483,
                }
            ],
        }
    }
    dep = _dep(
        [{"filename": "blastocyst_ch0.gsplats.zarr.zip", "filesize": 186483}],
        desc=(
            "<li><code>gsplats_multichannel</code> (0.2 MB)</li>"
            "<table><tr>h</tr><tr>1</tr></table>"
        ),
    )
    fails, warns = audit.check_deposition(
        "cc-by", dep, audit.pins_of(datasets), audit.dataset_totals(datasets), {}
    )
    assert fails == [], fails
    assert warns == []


def test_a_description_claim_is_checked_against_the_hosted_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record's prose describes the record's files, so its sizes are hosted ones.

    Same bug as the pin comparison, second site: summing `bytes` would check the
    published claim against bytes the record does not serve.
    """
    audit = _audit_module(monkeypatch)
    datasets = {
        "ds": {
            "bucket": "zenodo",
            "record": "cc-by",
            "files": [
                {
                    "name": "a.zip",
                    "sha256": "1" * 64,
                    "bytes": 9_000_000,  # in-repo
                    "hosted_sha256": "7" * 64,
                    "hosted_bytes": 5_000_000,  # the record's -> "5.0 MB"
                }
            ],
        }
    }
    dep = _dep(
        [{"filename": "a.zip", "filesize": 5_000_000}],
        desc="<li><code>ds</code> (5.0 MB)</li><table><tr>h</tr><tr>1</tr></table>",
    )
    fails, _ = audit.check_deposition(
        "cc-by", dep, audit.pins_of(datasets), audit.dataset_totals(datasets), {}
    )
    assert fails == [], fails


def test_an_entry_with_no_hosted_keys_still_uses_its_local_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Most datasets never diverged and carry no hosted keys at all.

    They must keep working off `sha256`/`bytes`, or preferring the hosted side
    would break the majority to fix the minority.
    """
    audit = _audit_module(monkeypatch)
    datasets = _two_file_dataset()
    dep = _dep(
        [
            {"filename": "a.zip", "filesize": 1024},
            {"filename": "b.zip", "filesize": 1024},
        ],
        desc=(
            "<li><code>ds</code> (0.0 MB)</li>"
            "<table><tr>h</tr><tr>a</tr><tr>b</tr></table>"
        ),
    )
    fails, warns = audit.check_deposition(
        "cc-by", dep, audit.pins_of(datasets), audit.dataset_totals(datasets), {}
    )
    assert fails == []
    assert warns == []
