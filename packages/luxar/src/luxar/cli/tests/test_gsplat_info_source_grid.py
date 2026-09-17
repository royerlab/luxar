"""The source-grid stamp, exercised through the commands that write and read it.

These live in the CLI package rather than beside the rest of the source-grid
tests because ``luxar.gsplats`` is a domain layer and the import-linter contract
"Domain layers must not import the CLI" forbids a test there from importing
``luxar.cli`` — a rule enforced in its own CI step, not by the test run, so a
violation passes locally and fails only on the layer check.

The domain half (that the stamps are computed, and reach ``fitting/`` on disk)
is asserted in ``gsplats/tests/test_fit_source_grid_stamp.py``. What is asserted
here is the part that needs the command: that the read path SURFACES the stamp,
and that a fit driven from a file on disk records the dtype that file was
STORED in rather than the float32 the loader hands the optimiser.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from typer.testing import CliRunner  # noqa: E402

from luxar._zarr_compat import read_node_attrs
from luxar.cli import app  # noqa: E402
from luxar.cli.tests._testing import normalized_cli_output  # noqa: E402
from luxar.gsplats.fit_gsplats import fit_gaussian_splats  # noqa: E402


def _sparse_blobs(shape=(24, 32, 32), n=20, dtype=np.uint16) -> np.ndarray:
    """A sparse blob field — sparse enough that occupancy is a real number."""
    rng = np.random.default_rng(0)
    V = np.zeros(shape, dtype=np.float32)
    hi = [s - 2 for s in shape]
    for _ in range(n):
        idx = rng.integers([2] * len(shape), hi)
        V[tuple(slice(i - 1, i + 2) for i in idx)] = 3000
    return np.round(V).astype(dtype)


def test_info_surfaces_the_source_grid(tmp_path: Path) -> None:
    """A stamp nobody reads back is not worth writing."""
    V = _sparse_blobs()
    out = tmp_path / "stamped.gsplats.zarr"
    fit_gaussian_splats(
        V=V,
        seeds=120,
        n_iters=30,
        verbose=False,
        device="cpu",
        enable_dynamic_ops=False,
        napari_movie=False,
    ).save(out)

    result = CliRunner().invoke(app, ["gsplat", "info", str(out), "--no-histograms"])
    assert result.exit_code == 0, result.output
    assert "Source volume: 24 x 32 x 32 uint16" in result.output, result.output
    assert "compression:" in result.output, result.output


def test_info_summarizes_nested_part_provenance_unless_full_is_requested(
    tmp_path: Path,
) -> None:
    from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData
    from luxar.gsplats.io.save_gsplats import write_partition_streaming
    from luxar.gsplats.tree import GSplatLeaf

    records = [
        {
            "coordinate": 0.0,
            "fitting": {
                "part_provenance": [
                    {
                        "coordinate": 1.0,
                        "fitting": {
                            "part_provenance": [
                                {"coordinate": 2.0, "fitting": {"psnr_db": 40.0}}
                            ]
                        },
                    }
                ]
            },
        }
    ]
    centers = np.zeros((1, 3), dtype=np.float32)
    amplitudes = np.ones(1, dtype=np.float32)
    cholesky_factors = np.array([[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    flat = tmp_path / "flat.gsplats.zarr"
    GSplatData(
        centers=centers,
        amplitudes=amplitudes,
        cholesky_factors=cholesky_factors,
        stats={"part_provenance": records},
    ).save(flat, ordering="none")

    partition = tmp_path / "partition.gsplats.zarr"
    leaf = GSplatLeaf(
        additive_sublods=[
            AdditiveSubLOD(
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky_factors,
            )
        ]
    )
    write_partition_streaming(
        partition,
        lambda: iter([leaf]),
        fitting_info={"part_provenance": records},
        ordering="none",
    )

    for path in (flat, partition):
        summary = CliRunner().invoke(
            app, ["gsplat", "info", str(path), "--no-histograms"]
        )
        assert summary.exit_code == 0, summary.output
        summary_output = normalized_cli_output(summary)
        assert "part_provenance: 1 part, nested channels × timepoints" in summary_output
        assert "psnr_db" not in summary_output

        full = CliRunner().invoke(
            app,
            [
                "gsplat",
                "info",
                str(path),
                "--no-histograms",
                "--full-provenance",
            ],
        )
        assert full.exit_code == 0, full.output
        assert "psnr_db" in normalized_cli_output(full)

    summary_path = tmp_path / "summary.gsplats.zarr"
    GSplatData(
        centers=centers,
        amplitudes=amplitudes,
        cholesky_factors=cholesky_factors,
        stats={
            "part_provenance": [{"part_count": 47, "fitting": {"source_bytes": 3000}}]
        },
    ).save(summary_path, ordering="none")
    summary = CliRunner().invoke(
        app, ["gsplat", "info", str(summary_path), "--no-histograms"]
    )
    assert summary.exit_code == 0, summary.output
    assert "part_provenance: 47 parts" in normalized_cli_output(summary)

    help_result = CliRunner().invoke(app, ["gsplat", "info", "--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "--full-provenance" in normalized_cli_output(help_result)


def test_cli_fit_stamps_the_stored_dtype_not_the_loaders_cast(tmp_path: Path) -> None:
    """End to end through the command: a uint16 file must not record float32.

    This is the whole point of the stamp — a compression ratio quoted against
    the float working copy is exactly 2x too flattering on 16-bit data.
    """
    src = tmp_path / "vol.npy"
    V = _sparse_blobs(shape=(16, 16, 16), n=10)
    np.save(src, V)
    out = tmp_path / "cli.gsplats.zarr"
    result = CliRunner().invoke(
        app,
        [
            "gsplat",
            "fit",
            str(src),
            str(out),
            "--iters",
            "20",
            "--seeds",
            "80",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    attrs = read_node_attrs(out / "fitting")
    assert attrs["source_dtype"] == "uint16"
    assert attrs["source_bytes"] == V.size * 2


def test_info_says_when_the_source_grid_was_declared_rather_than_measured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stated denominator must read as a claim in the report, not as a
    measurement.

    ``source_declared`` is stamped when the producer preprocessed before fitting
    and named the acquisition itself. The compression line printed here rests on
    that number, so the report is the one place a reader would find out — a
    marker only visible by opening `fitting/.zattrs` is not a distinction anyone
    reading `info` gets to make.
    """
    from types import SimpleNamespace

    from luxar.cli.gsplat_ops.inspect_commands import _print_source_grid, _store_size

    store = tmp_path / "declared.gsplats.zarr"
    store.mkdir()
    (store / "chunk").write_bytes(b"\0" * 4096)
    stats = {
        "source_shape": [96, 128, 128],
        "source_voxels": 96 * 128 * 128,
        "source_dtype": "uint16",
        "source_bytes": 2 * 96 * 128 * 128,
        "fitted_shape": [24, 32, 32],
        "fitted_voxels": 24 * 32 * 32,
    }
    data = SimpleNamespace(
        stats={**stats, "source_declared": True},
        amplitudes=np.zeros(64, dtype=np.float32),
    )
    _print_source_grid(data, _store_size(store))
    out = capsys.readouterr().out
    assert "Source volume: 96 x 128 x 128 uint16" in out, out
    assert "declared by the producer" in out, out

    # The negative control: a measured grid must not be labelled a claim.
    _print_source_grid(
        SimpleNamespace(stats=stats, amplitudes=np.zeros(64, dtype=np.float32)),
        _store_size(store),
    )
    measured = capsys.readouterr().out
    assert "Source volume: 96 x 128 x 128 uint16" in measured, measured
    assert "declared" not in measured, measured


def test_an_artifact_larger_than_its_source_is_not_reported_as_0_to_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A sub-unity ratio must print as itself rather than round to "0:1".

    A store bigger than the volume it represents is a real outcome — a small
    crop with a generous seed budget — and "0:1" reads as a broken measurement
    instead of an expansion. Exercised through the printer because the ratio is
    formatted there; the stamps themselves carry no ratio.
    """
    from types import SimpleNamespace

    from luxar.cli.gsplat_ops.inspect_commands import _print_source_grid, _store_size

    store = tmp_path / "expanded.gsplats.zarr"
    store.mkdir()
    (store / "chunk").write_bytes(b"\0" * 4096)
    data = SimpleNamespace(
        stats={
            "source_shape": [8, 8, 8],
            "source_voxels": 512,
            "source_dtype": "uint16",
            "source_bytes": 1024,
            "fitted_shape": [8, 8, 8],
            "fitted_voxels": 512,
        },
        amplitudes=np.zeros(64, dtype=np.float32),
    )
    _print_source_grid(data, _store_size(store))
    out = capsys.readouterr().out
    assert "Source volume: 8 x 8 x 8 uint16" in out, out
    assert "0.25:1" in out, out


def test_both_compression_ratios_are_printed_with_their_bases_named(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two denominators, and the report must say which is which.

    ``source_bytes`` is the DECODED array while the store it is divided by is
    compressed, so that ratio credits the splats with the source codec's own
    factor — measured on the DAPI acquisition the two bases differ by ~35x. An
    unlabelled number invites reading the first as the second, so both are
    printed and each names its basis.
    """
    from types import SimpleNamespace

    from luxar.cli.gsplat_ops.inspect_commands import _print_source_grid, _store_size

    store = tmp_path / "both.gsplats.zarr"
    store.mkdir()
    (store / "chunk").write_bytes(b"\0" * 4096)
    stats = {
        "source_shape": [96, 128, 128],
        "source_voxels": 96 * 128 * 128,
        "source_dtype": "uint16",
        "source_bytes": 2 * 96 * 128 * 128,
        "fitted_shape": [24, 32, 32],
        "fitted_voxels": 24 * 32 * 32,
    }
    data = SimpleNamespace(
        stats={**stats, "source_stored_bytes": 40960},
        amplitudes=np.zeros(64, dtype=np.float32),
    )
    keys = _print_source_grid(data, _store_size(store))
    out = capsys.readouterr().out
    assert "stored source: 40.0 KB (as downloaded)" in out, out
    assert "768:1 vs raw voxels" in out, out
    assert "10:1 vs the stored source" in out, out
    # Reported here, so the catch-all metadata dump must not quote it again
    # under a raw key name with no basis attached.
    assert "source_stored_bytes" in keys, keys

    # The negative control: without the stored size there is exactly one ratio,
    # never a second one inferred from the decoded size.
    _print_source_grid(
        SimpleNamespace(stats=stats, amplitudes=np.zeros(64, dtype=np.float32)),
        _store_size(store),
    )
    one = capsys.readouterr().out
    assert "vs raw voxels" in one, one
    assert "stored source" not in one, one


def test_a_downscaled_tiled_fit_records_the_grid_it_was_given_not_the_decimated_copy(
    tmp_path: Path,
) -> None:
    """``--tiling uniform --downscale`` must not publish the working copy.

    The tiled fitter measures the array it is HANDED, and on this path the
    command decimates the volume itself before handing it over — so without a
    declaration the merged result records the decimated grid as its source, and
    quotes a compression ratio against a volume 8x smaller than the file it was
    fitted from. Worse, ``fitted_shape`` would equal it, so ``info`` could not
    even show that a decimation happened.
    """
    src = tmp_path / "vol.npy"
    V = _sparse_blobs(shape=(16, 16, 16), n=10)
    np.save(src, V)
    out = tmp_path / "tiled.gsplats.zarr"
    result = CliRunner().invoke(
        app,
        [
            "gsplat",
            "fit",
            str(src),
            str(out),
            "--tiling",
            "uniform",
            "--tile-size",
            "8",
            "--overlap",
            "2",
            "--downscale",
            "2",
            "--iters",
            "10",
            "--seeds",
            "40",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    attrs = read_node_attrs(out / "fitting")
    assert attrs["source_shape"] == [16, 16, 16], attrs
    assert attrs["source_bytes"] == V.size * 2
    assert attrs["source_declared"] is True
    # The optimiser saw the decimated grid, and the report says so.
    assert attrs["fitted_shape"] == [8, 8, 8]

    info = CliRunner().invoke(app, ["gsplat", "info", str(out), "--no-histograms"])
    assert info.exit_code == 0, info.output
    assert "Source volume: 16 x 16 x 16 uint16" in info.output, info.output
    assert "downscaled before fitting" in info.output, info.output


def test_a_single_downscaled_tile_does_not_claim_the_whole_acquisition(
    tmp_path: Path,
) -> None:
    """``--tile i/M`` writes ONE crop, and a crop's source is its own region.

    The declaration above exists because the merged result covers the volume; a
    lone tile does not, so it must keep measuring what it was handed. Without the
    distinction every tile of a distributed run would publish the whole
    acquisition as its own source.
    """
    from luxar.gsplats.tiling import compute_tile_specs

    src = tmp_path / "vol.npy"
    V = _sparse_blobs(shape=(32, 32, 32), n=16)
    np.save(src, V)
    specs = compute_tile_specs((16, 16, 16), 8, 2)
    n_tiles = len(specs)
    assert n_tiles > 1, "a single tile would cover the whole grid, proving nothing"
    out = tmp_path / "tile0.gsplats.zarr"
    result = CliRunner().invoke(
        app,
        [
            "gsplat",
            "fit",
            str(src),
            str(out),
            "--tile",
            f"0/{n_tiles}",
            "--tile-size",
            "8",
            "--overlap",
            "2",
            "--downscale",
            "2",
            "--iters",
            "10",
            "--seeds",
            "40",
            "--device",
            "cpu",
            "--allow-empty-tile",
        ],
    )
    assert result.exit_code == 0, result.output
    if not out.exists():
        pytest.skip("tile 0 windowed to empty; nothing was stamped")
    attrs = read_node_attrs(out / "fitting")
    assert attrs["psnr_db"] > 0
    assert np.isfinite(attrs["foreground_psnr_db"])
    assert attrs["source_shape"] == list(specs[0].shape), attrs
    assert "source_declared" not in attrs, attrs
