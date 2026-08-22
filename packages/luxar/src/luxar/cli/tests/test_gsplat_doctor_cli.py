"""``luxar gsplat doctor`` at the CLI boundary.

Lives here rather than beside the doctor's own tests because the layering
contract forbids a domain package importing ``luxar.cli``. What is checked here
is only what the command adds over :func:`~luxar.gsplats.doctor.diagnose_store`:
the exit code (so it can gate a pipeline), and the JSON report.
"""

from __future__ import annotations

import json
import shutil
import struct
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar._zarr_compat import consolidate as zc_consolidate
from luxar._zarr_compat import open_group as zc_open_group
from luxar.cli import app


def _partition_without_split_planes(tmp: Path) -> Path:
    """A real BSP partition on disk with its ``bsp_tree`` stripped — i.e. exactly
    what any tiled fit written before the planes were recorded looks like."""
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree

    rng = np.random.default_rng(4)
    n = 400
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    node = GSplatData(
        centers=(rng.random((n, 3)) * 100).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, size=(n,)).astype(np.float32),
        cholesky_factors=chol,
    ).to_spatial_partition(max_elements=80)
    path = tmp / "part.gsplats.zarr"
    write_gsplats_tree(path, node)

    # Through the facade, as Luxar's in-place editors are: a plain re-open of
    # an already-consolidated store leaves a stale NESTED index behind on
    # re-consolidation (see `_zarr_compat.open_group`).
    root = zc_open_group(str(path), mode="r+")
    del root.attrs["bsp_tree"]
    zc_consolidate(root)
    return path


def _scene_without_split_planes(tmp: Path) -> Path:
    from luxar import Dimensions, LuxarZarrCompiler

    source = _partition_without_split_planes(tmp)
    path = tmp / "scene.luxar.zarr"
    with LuxarZarrCompiler(path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_gsplats_from_file("tiles", str(source))
    return path


def _corrupt_zip_payload(path: Path) -> None:
    """Damage one compressed data member without touching the archive index."""
    payload = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as archive:
        member = max(
            (
                info
                for info in archive.infolist()
                if not info.is_dir()
                and info.compress_type == zipfile.ZIP_DEFLATED
                and info.compress_size > 8
                and not info.filename.endswith(
                    ("zarr.json", ".zattrs", ".zgroup", ".zmetadata")
                )
            ),
            key=lambda info: info.compress_size,
        )
    name_length, extra_length = struct.unpack_from(
        "<HH", payload, member.header_offset + 26
    )
    data_offset = member.header_offset + 30 + name_length + extra_length
    payload[data_offset + 2] ^= 0xFF
    path.write_bytes(payload)


def test_doctor_exits_nonzero_while_a_problem_stands() -> None:
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        path = _partition_without_split_planes(Path(tmp))

        result = runner.invoke(app, ["gsplat", "doctor", str(path), "--no-info"])
        assert result.exit_code == 1, result.stdout
        assert "no split planes" in result.stdout

        fixed = runner.invoke(
            app, ["gsplat", "doctor", str(path), "--no-info", "--fix"]
        )
        assert fixed.exit_code == 0, fixed.stdout

        again = runner.invoke(app, ["gsplat", "doctor", str(path), "--no-info"])
        assert again.exit_code == 0, again.stdout
        assert "No problems found" in again.stdout


def test_doctor_writes_a_json_report() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _partition_without_split_planes(Path(tmp))
        out = Path(tmp) / "report.json"
        CliRunner().invoke(
            app, ["gsplat", "doctor", str(path), "--no-info", "--json", str(out)]
        )
        payload = json.loads(out.read_text())
        assert payload["healthy"] is False
        assert payload["findings"][0]["check"] == "split-planes"
        assert payload["findings"][0]["fixable"] is True


def test_doctor_accepts_and_repairs_a_scene_store() -> None:
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        path = _scene_without_split_planes(Path(tmp))

        result = runner.invoke(app, ["gsplat", "doctor", str(path), "--no-info"])
        assert result.exit_code == 1, result.stdout
        assert "tiles" in result.stdout
        assert "no split planes" in result.stdout

        fixed = runner.invoke(
            app, ["gsplat", "doctor", str(path), "--no-info", "--fix"]
        )
        assert fixed.exit_code == 0, fixed.stdout


def test_doctor_classifies_a_plainly_named_scene_archive_from_its_attrs() -> None:
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        path = _scene_without_split_planes(tmp_path)
        shutil.make_archive(
            str(tmp_path / "scene"), "zip", root_dir=str(tmp_path), base_dir=path.name
        )

        result = runner.invoke(app, ["gsplat", "doctor", str(tmp_path / "scene.zip")])
        assert result.exit_code == 1, result.stdout
        assert "gsplat info report does not apply" in result.stdout
        assert "tiles" in result.stdout
        assert "no split planes" in result.stdout


def test_doctor_diagnoses_archive_when_root_attr_peek_is_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luxar.gsplats.io import _archive

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        path = _partition_without_split_planes(tmp_path)
        shutil.make_archive(
            str(tmp_path / "part"), "zip", root_dir=str(tmp_path), base_dir=path.name
        )
        monkeypatch.setattr(_archive, "read_archive_root_attrs", lambda _path: {})

        result = CliRunner().invoke(
            app, ["gsplat", "doctor", str(tmp_path / "part.zip")]
        )

        assert result.exit_code == 1, result.stdout
        assert "not a Luxar scene" not in result.stdout
        assert "no split planes" in result.stdout


@pytest.mark.parametrize("name", ["broken.zip", "broken.tar.gz"])
def test_doctor_reports_corrupt_archives_without_a_traceback(name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_bytes(b"not an archive")

        result = CliRunner().invoke(app, ["gsplat", "doctor", str(path)])

        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert "❌ " in result.stdout
        assert "Traceback" not in result.stdout


def test_doctor_reports_a_truncated_tar_archive_without_aborting() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store = _partition_without_split_planes(tmp_path)
        archive = Path(
            shutil.make_archive(
                str(tmp_path / "part"),
                "gztar",
                root_dir=str(tmp_path),
                base_dir=store.name,
            )
        )
        payload = archive.read_bytes()
        archive.write_bytes(payload[: len(payload) // 2])

        result = CliRunner().invoke(app, ["gsplat", "doctor", str(archive)])

        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert "❌ " in result.stdout
        assert "Traceback" not in result.stdout


def test_doctor_reports_a_corrupt_zip_payload_without_a_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store = _partition_without_split_planes(tmp_path)
        archive = Path(
            shutil.make_archive(
                str(tmp_path / "part"),
                "zip",
                root_dir=str(tmp_path),
                base_dir=store.name,
            )
        )
        _corrupt_zip_payload(archive)

        result = CliRunner().invoke(
            app, ["gsplat", "doctor", str(archive), "--no-info"]
        )

        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert "Diagnosing:" in result.stdout
        assert "❌ " in result.stdout
        assert "Traceback" not in result.stdout
