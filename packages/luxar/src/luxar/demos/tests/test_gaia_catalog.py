from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import luxar.demos._gaia_catalog as catalog


def _write_fake_raw(path: Path) -> None:
    path.mkdir(parents=True)
    (path / ".zgroup").write_text("{}")
    column = path / "x_kpc"
    column.mkdir()
    (column / "0").write_bytes(b"values")


def test_build_catalog_uses_the_required_zip_member_name(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[int, float]] = []

    def fake_generate(path: Path, *, count: int, rmax_kpc: float) -> int:
        calls.append((count, rmax_kpc))
        _write_fake_raw(path)
        return count

    monkeypatch.setattr(catalog, "generate_raw_catalog", fake_generate)

    result = catalog.build_catalog(cache_dir=tmp_path, count=12, rmax_kpc=4.5)

    assert result == tmp_path / "milky_way_gaia_3m.zarr.zip"
    assert calls == [(12, 4.5)]
    assert (tmp_path / catalog.RAW_ZARR_NAME).is_dir()
    with zipfile.ZipFile(result) as archive:
        assert archive.namelist()
        assert all(
            name.startswith(f"{catalog.RAW_ZARR_NAME}/") for name in archive.namelist()
        )


def test_build_catalog_returns_an_existing_archive_untouched(
    tmp_path: Path, monkeypatch
) -> None:
    """A cached zip short-circuits the build — which is why a BROKEN cached zip
    has to be pointed at ``--recompute``, never ``--build-catalog``."""
    cached = tmp_path / f"{catalog.RAW_ZARR_NAME}.zip"
    cached.write_bytes(b"unusable but present")

    def should_not_query(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("a cached archive must not trigger the TAP query")

    monkeypatch.setattr(catalog, "generate_raw_catalog", should_not_query)

    assert catalog.build_catalog(cache_dir=tmp_path) == cached
    assert cached.read_bytes() == b"unusable but present"


def test_build_catalog_resumes_from_the_completed_raw_table(
    tmp_path: Path, monkeypatch
) -> None:
    _write_fake_raw(tmp_path / catalog.RAW_ZARR_NAME)

    def should_not_query(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("the completed raw table should skip the TAP query")

    monkeypatch.setattr(catalog, "generate_raw_catalog", should_not_query)

    result = catalog.build_catalog(cache_dir=tmp_path)

    with zipfile.ZipFile(result) as archive:
        assert f"{catalog.RAW_ZARR_NAME}/.zgroup" in archive.namelist()


def test_recompute_keeps_the_old_zip_until_the_new_one_is_complete(
    tmp_path: Path, monkeypatch
) -> None:
    cached = tmp_path / f"{catalog.RAW_ZARR_NAME}.zip"
    cached.write_bytes(b"old usable catalog")

    def fake_generate(path: Path, *, count: int, rmax_kpc: float) -> int:
        _write_fake_raw(path)
        return count

    def fail_while_archiving(_raw: Path, staged_zip: Path | None = None) -> Path:
        assert staged_zip is not None
        staged_zip.write_bytes(b"partial replacement")
        raise RuntimeError("archive interrupted")

    monkeypatch.setattr(catalog, "generate_raw_catalog", fake_generate)
    monkeypatch.setattr(catalog, "create_zip", fail_while_archiving)

    with pytest.raises(RuntimeError, match="archive interrupted"):
        catalog.build_catalog(cache_dir=tmp_path, recompute=True)

    assert cached.read_bytes() == b"old usable catalog"
    assert not list(tmp_path.glob(".*.tmp"))


def test_acknowledgement_is_printed_before_the_tap_query(monkeypatch) -> None:
    messages: list[str] = []

    class FakeResults:
        def to_pandas(self) -> pd.DataFrame:
            return pd.DataFrame({"phot_g_mean_mag": [1.0, 2.0]})

        def __len__(self) -> int:
            return 2

    class FakeJob:
        def get_results(self) -> FakeResults:
            return FakeResults()

    class FakeGaia:
        @staticmethod
        def launch_job_async(_query: str) -> FakeJob:
            assert any("DPAC" in message for message in messages)
            return FakeJob()

    monkeypatch.setattr(catalog, "aprint", lambda message="": messages.append(message))
    monkeypatch.setattr(
        catalog,
        "require_module",
        lambda module: (
            SimpleNamespace(Gaia=FakeGaia) if module == "astroquery.gaia" else None
        ),
    )

    result = catalog.fetch_stars(2)

    assert len(result) == 2
