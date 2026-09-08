"""``luxar gsplat doctor`` at the CLI boundary.

Lives here rather than beside the doctor's own tests because the layering
contract forbids a domain package importing ``luxar.cli``. What is checked here
is only what the command adds over :func:`~luxar.gsplats.doctor.diagnose_store`:
the exit code (so it can gate a pipeline), and the JSON report.
"""

from __future__ import annotations

import inspect
import json
import shutil
import struct
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
from typer.models import ParameterInfo
from typer.testing import CliRunner

from luxar._zarr_compat import consolidate as zc_consolidate
from luxar._zarr_compat import open_group as zc_open_group
from luxar.cli import app
from luxar.cli._traceback import TRACEBACK_ENV_VAR
from luxar.cli.gsplat_ops import inspect_commands
from luxar.cli.tests._testing import normalized_cli_output


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


def _partition_with_nested_provenance(tmp: Path) -> Path:
    path = _partition_without_split_planes(tmp)
    root = zc_open_group(str(path), mode="r+")
    root.require_group("fitting").attrs["part_provenance"] = [
        {
            "fitting": {
                "part_provenance": [{"coordinate": 0.0, "fitting": {"psnr_db": 40.0}}]
            },
            "coordinate": 0.0,
        },
        {
            "fitting": {
                "part_provenance": [{"coordinate": 1.0, "fitting": {"psnr_db": 41.0}}]
            },
            "coordinate": 1.0,
        },
    ]
    zc_consolidate(root)
    return path


def _legacy_gsplat_store(tmp: Path) -> Path:
    path = tmp / "legacy.gsplats.zarr"
    root = zc_open_group(str(path), mode="w")
    root.attrs["format_type"] = "gsplats_zarr"
    root.attrs["format_version"] = "2.0"
    return path


def _unreadable_gsplat_store(tmp: Path) -> Path:
    path = tmp / "unreadable.gsplats.zarr"
    root = zc_open_group(str(path), mode="w")
    root.attrs["format_type"] = "gsplats_zarr"
    root.attrs["format_version"] = "3.4"
    return path


def _scene_without_split_planes(tmp: Path) -> Path:
    from luxar import Dimensions, LuxarZarrCompiler

    source = _partition_without_split_planes(tmp)
    path = tmp / "scene.luxar.zarr"
    with LuxarZarrCompiler(path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_gsplats_from_file("tiles", str(source))

    # The graft recovers planes from these disjoint boxes; strip them so this
    # remains the scene-store counterpart of _partition_without_split_planes.
    root = zc_open_group(str(path), mode="r+")
    del root["tiles"].attrs["bsp_tree"]
    zc_consolidate(root)
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


def _mark_zip_member_encrypted(path: Path, *, metadata: bool) -> None:
    """Set the encryption flag on one member without encrypting its payload."""
    payload = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as archive:
        if metadata:
            member = min(
                (
                    info
                    for info in archive.infolist()
                    if not info.is_dir()
                    and Path(info.filename).name in {"zarr.json", ".zattrs"}
                ),
                key=lambda info: len(Path(info.filename).parts),
            )
        else:
            member = next(
                info
                for info in archive.infolist()
                if not info.is_dir()
                and info.compress_type == zipfile.ZIP_DEFLATED
                and info.compress_size > 8
                and not info.filename.endswith(
                    ("zarr.json", ".zattrs", ".zgroup", ".zmetadata")
                )
            )

    local_flags = member.header_offset + 6
    flags = struct.unpack_from("<H", payload, local_flags)[0]
    struct.pack_into("<H", payload, local_flags, flags | 1)

    central_offset = payload.find(b"PK\x01\x02")
    while central_offset >= 0:
        name_length, extra_length, comment_length = struct.unpack_from(
            "<HHH", payload, central_offset + 28
        )
        name_start = central_offset + 46
        name_end = name_start + name_length
        if payload[name_start:name_end].decode() == member.filename:
            central_flags = central_offset + 8
            struct.pack_into(
                "<H",
                payload,
                central_flags,
                struct.unpack_from("<H", payload, central_flags)[0] | 1,
            )
            break
        central_offset = payload.find(
            b"PK\x01\x02", name_end + extra_length + comment_length
        )
    else:
        raise AssertionError(f"central directory entry not found: {member.filename}")

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


def test_doctor_prints_info_before_diagnosing_a_gsplat_store() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _partition_without_split_planes(Path(tmp))

        result = CliRunner().invoke(app, ["gsplat", "doctor", str(path)])

        assert result.exit_code == 1, result.stdout
        assert result.stdout.index("DATASET INFORMATION") < result.stdout.index(
            "Diagnosing:"
        )
        assert "no split planes" in result.stdout


def test_doctor_continues_when_info_rejects_a_legacy_store() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _legacy_gsplat_store(Path(tmp))

        result = CliRunner().invoke(app, ["gsplat", "doctor", str(path)])

        assert result.exit_code == 1, result.stdout
        assert "migrate-format" in result.stdout
        assert "Info report failed; continuing to the diagnosis" in result.stdout
        assert "Diagnosing:" in result.stdout
        assert "unsupported gsplat format version '2.0'" in result.stdout
        assert "1 problem(s) outstanding" in result.stdout


def test_info_command_exits_when_report_rejects_a_legacy_store() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _legacy_gsplat_store(Path(tmp))

        result = CliRunner().invoke(app, ["gsplat", "info", str(path)])

        assert result.exit_code == 1, result.stdout
        assert "migrate-format" in result.stdout
        assert "Traceback" not in result.stdout


@pytest.mark.parametrize("traceback_enabled", [False, True])
def test_doctor_rejects_a_supported_but_unreadable_store(
    traceback_enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if traceback_enabled:
        monkeypatch.setenv(TRACEBACK_ENV_VAR, "1")
    with tempfile.TemporaryDirectory() as tmp:
        path = _unreadable_gsplat_store(Path(tmp))

        result = CliRunner().invoke(app, ["gsplat", "doctor", str(path)])

        assert result.exit_code == 1, result.stdout
        assert "Info report failed; continuing to the diagnosis" in result.stdout
        assert "Diagnosing:" in result.stdout
        assert "[/] gsplat store cannot be read" in result.stdout
        assert "missing required array 'centers'" in result.stdout
        assert "1 problem(s) outstanding" in result.stdout


def test_doctor_summarizes_nested_provenance_unless_full_is_requested() -> None:
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        path = _partition_with_nested_provenance(Path(tmp))

        summary = runner.invoke(app, ["gsplat", "doctor", str(path)])
        assert summary.exit_code == 1, summary.stdout
        assert (
            "part_provenance: 2 parts, nested component records (2 levels)"
            in summary.stdout
        )
        assert "{'fitting':" not in summary.stdout
        assert "psnr_db" not in summary.stdout

        full = runner.invoke(app, ["gsplat", "doctor", str(path), "--full-provenance"])
        assert full.exit_code == 1, full.stdout
        assert "{'fitting':" in full.stdout
        assert "psnr_db" in full.stdout
        assert (
            "part_provenance: 2 parts, nested component records (2 levels)"
            not in full.stdout
        )

        implied = runner.invoke(
            app,
            ["gsplat", "doctor", str(path), "--full-provenance", "--no-info"],
        )
        assert implied.exit_code == 1, implied.stdout
        assert "psnr_db" in implied.stdout

    help_result = runner.invoke(app, ["gsplat", "doctor", "--help"])
    assert help_result.exit_code == 0, help_result.stdout
    help_output = normalized_cli_output(help_result)
    assert "--full-provenance" in help_output
    assert "implies --info" in help_output


def test_doctor_accepts_custom_info_histogram_bins() -> None:
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        path = _partition_without_split_planes(Path(tmp))

        result = runner.invoke(
            app,
            ["gsplat", "doctor", str(path), "--histograms", "--bins", "17"],
        )

    assert result.exit_code == 1, result.stdout
    assert "No such option" not in result.stderr

    help_result = runner.invoke(app, ["gsplat", "doctor", "--help"])
    assert help_result.exit_code == 0, help_result.stdout
    help_output = normalized_cli_output(help_result)
    assert "--bins" in help_output
    assert "only applies with --histograms" in help_output


def test_doctor_passes_every_info_report_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = inspect.signature(inspect_commands._info_report)
    calls: list[inspect.BoundArguments] = []

    def capture_info_call(*args: object, **kwargs: object) -> bool:
        calls.append(signature.bind(*args, **kwargs))
        return True

    monkeypatch.setattr(inspect_commands, "_info_report", capture_info_call)

    with tempfile.TemporaryDirectory() as tmp:
        path = _partition_without_split_planes(Path(tmp))
        result = CliRunner().invoke(
            app,
            ["gsplat", "doctor", str(path), "--histograms", "--bins", "17"],
        )

    assert result.exit_code == 1, result.stdout
    assert len(calls) == 1
    bound = calls[0]
    assert set(bound.arguments) == set(signature.parameters)
    assert not any(
        isinstance(parameter.default, ParameterInfo)
        for parameter in signature.parameters.values()
    )
    assert not any(
        isinstance(value, ParameterInfo) for value in bound.arguments.values()
    )
    assert bound.arguments["show_histograms"] is True
    assert bound.arguments["bins"] == 17
    assert bound.arguments["full_provenance"] is False


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
        assert "Could not classify this store from its metadata" in result.stdout
        assert "no split planes" in result.stdout


def test_doctor_does_not_call_a_non_archive_input_an_archive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "volume.npy"
        np.save(path, np.zeros((2, 2, 2), dtype=np.float32))

        result = CliRunner().invoke(app, ["gsplat", "doctor", str(path)])

        assert result.exit_code == 1
        assert "Could not classify this store from its metadata" in result.stdout
        assert "archive index" not in result.stdout


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


@pytest.mark.parametrize("metadata", [True, False])
def test_doctor_reports_an_encrypted_zip_member_without_a_traceback(
    metadata: bool,
) -> None:
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
        _mark_zip_member_encrypted(archive, metadata=metadata)

        result = CliRunner().invoke(
            app, ["gsplat", "doctor", str(archive), "--no-info"]
        )

        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        if not metadata:
            assert "Diagnosing:" in result.stdout
        assert "encrypted, password required" in result.stdout
        assert "Traceback" not in result.stdout
