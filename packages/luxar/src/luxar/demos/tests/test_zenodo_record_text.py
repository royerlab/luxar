"""The generated record text must not claim more than the archives support.

A Zenodo record is the one place where a wrong number is *published* rather than
merely wrong, so the generator derives every figure from the manifest and the
archives' own stamps. These tests cover the ways that can go wrong quietly: an
unreadable or non-splat file being described as though it were a fit, a
publication flag leaking into the text, and a dataset silently missing from its
own record.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

#: .../packages/luxar/src/luxar/demos/tests/this_file.py -> repo root is 6 up.
_SCRIPT = Path(__file__).resolve().parents[6] / "scripts" / "gen_zenodo_records.py"


def _load_module() -> Any:
    """Import the script by path; `scripts/` is not an importable package."""
    # Deliberately an error, not a skip: a wrong path here silently turned every
    # test in this file into a pass once already.
    assert _SCRIPT.exists(), (
        f"generator not found at {_SCRIPT} — fix the path rather than skipping, "
        "or these tests all pass while checking nothing"
    )
    spec = importlib.util.spec_from_file_location("gen_zenodo_records", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["gen_zenodo_records"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen() -> Any:
    return _load_module()


@pytest.fixture(scope="module")
def manifest(gen: Any) -> dict[str, Any]:
    return json.loads(gen.MANIFEST.read_text())


def test_every_record_renders(gen: Any, manifest: dict[str, Any]) -> None:
    for key in manifest["records"]:
        text = gen.render_record(key, manifest)
        assert manifest["records"][key]["title"] in text
        assert manifest["records"][key]["zenodo_doi"] in text


def test_every_hosted_dataset_appears_in_its_record(
    gen: Any, manifest: dict[str, Any]
) -> None:
    """A dataset missing from its record would simply never be described."""
    rendered = {key: gen.render_record(key, manifest) for key in manifest["records"]}
    for name, entry in manifest["datasets"].items():
        if entry.get("bucket") != "zenodo":
            continue
        record = entry.get("record")
        assert record in rendered, f"{name} names record {record!r}"
        assert f"`{name}`" in rendered[record], (
            f"{name} is hosted under {record!r} but does not appear in its text"
        )


def test_a_draft_record_says_so(gen: Any, manifest: dict[str, Any]) -> None:
    """While `published` is false the text must carry the draft marker."""
    for key, record in manifest["records"].items():
        text = gen.render_record(key, manifest)
        if record.get("published"):
            assert "DRAFT" not in text
        else:
            assert "DRAFT" in text, f"{key} is unpublished but the text does not say so"


def test_the_text_never_contains_a_zenodo_file_url(
    gen: Any, manifest: dict[str, Any]
) -> None:
    """A file URL into an unpublished record 404s for every reader."""
    for key, record in manifest["records"].items():
        if record.get("published"):
            continue
        text = gen.render_record(key, manifest)
        assert f"zenodo.org/records/{record['zenodo_record']}/files" not in text


class TestNonSplatFilesAreNotDescribedAsFits:
    """The bug this class exists for: an `.npz` is also a zip.

    Opened blindly it parses, every field comes back defaulted, and the record
    ends up asserting "single level" and an absent PSNR for a file that is not a
    splat fit at all.
    """

    def test_an_npz_is_not_read_as_a_splat_store(
        self, gen: Any, tmp_path: Path
    ) -> None:
        import numpy as np

        path = tmp_path / "labels.npz"
        np.savez(path, labels=np.zeros((4, 4), np.uint8))
        assert gen._read_archive(path) is None

    def test_a_zarr_store_that_is_not_gsplats_is_rejected(
        self, gen: Any, tmp_path: Path
    ) -> None:
        path = tmp_path / "points.luxar.zarr.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("points.luxar.zarr/.zgroup", json.dumps({"zarr_format": 2}))
            zf.writestr(
                "points.luxar.zarr/.zattrs",
                json.dumps({"format_type": "luxar_zarr", "type": "points"}),
            )
        assert gen._read_archive(path) is None

    def test_a_gsplats_store_is_accepted(self, gen: Any, tmp_path: Path) -> None:
        """The negative tests above would pass even if nothing were readable."""
        path = tmp_path / "x.gsplats.zarr.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("x.gsplats.zarr/.zgroup", json.dumps({"zarr_format": 2}))
            zf.writestr(
                "x.gsplats.zarr/.zattrs",
                json.dumps(
                    {
                        "format_type": "gsplats_zarr",
                        "n_splats": 1234,
                        "n_additive_sublods": 4,
                    }
                ),
            )
            zf.writestr(
                "x.gsplats.zarr/fitting/.zgroup", json.dumps({"zarr_format": 2})
            )
            zf.writestr(
                "x.gsplats.zarr/fitting/.zattrs",
                json.dumps({"psnr_db": 41.3, "foreground_psnr_db": 28.2}),
            )
        info = gen._read_archive(path)
        assert info is not None
        assert info["n_splats"] == 1234
        assert info["psnr_db"] == pytest.approx(41.3)
        assert info["foreground_psnr_db"] == pytest.approx(28.2)
        assert "4 steps" in info["topology"]

    def test_a_corrupt_file_is_absent_not_fatal(self, gen: Any, tmp_path: Path) -> None:
        path = tmp_path / "truncated.gsplats.zarr.zip"
        path.write_bytes(b"not a zip at all")
        assert gen._read_archive(path) is None


class TestFiguresAreAbsentRatherThanInvented:
    def test_a_missing_number_renders_as_absent(self, gen: Any) -> None:
        assert gen._db(None) == gen._ABSENT
        assert gen._db("41.3") == gen._ABSENT, "a string must not pass as a measurement"
        assert gen._db(float("nan")) == gen._ABSENT
        assert gen._db(float("inf")) == gen._ABSENT, (
            "an infinite PSNR means an exact fit on a degenerate case, not a score"
        )
        assert gen._db(41.34) == "41.3"

    def test_a_ratio_needs_both_sides(self, gen: Any) -> None:
        assert gen._ratio(None, 100) == gen._ABSENT
        assert gen._ratio(100, None) == gen._ABSENT
        assert gen._ratio(100, 0) == gen._ABSENT, "no dividing by an empty archive"
        assert gen._ratio(35_000_000, 100_000) == "350:1"

    def test_an_incomparable_acquisition_states_the_reason(self, gen: Any) -> None:
        entry = {
            "acquisition": {
                "description": "377 colour photographs",
                "comparable": False,
                "reason": "the fit converts colour to greyscale",
            }
        }
        line = gen._acquisition_line(entry, 1_000_000)
        assert "colour to greyscale" in line
        assert ":1" not in line, "an incomparable source must not be given a ratio"

    def test_a_comparable_acquisition_quotes_the_download_ratio(self, gen: Any) -> None:
        entry = {
            "acquisition": {
                "description": "the kidney sample",
                "comparable": True,
                "stored_bytes": 21_000_000,
            }
        }
        line = gen._acquisition_line(entry, 2_100_000)
        assert "10:1" in line
        assert "as downloaded" in line

    def test_a_comparable_acquisition_awaiting_measurement_quotes_nothing(
        self, gen: Any
    ) -> None:
        entry = {
            "acquisition": {"description": "the kidney sample", "comparable": True}
        }
        line = gen._acquisition_line(entry, 2_100_000)
        assert ":1" not in line
        assert "kidney sample" in line
