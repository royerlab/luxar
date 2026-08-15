"""A fit records the volume its splats represent.

Without these stamps a stored ``.gsplats.zarr`` cannot answer "how much did this
compress?": the source grid appears nowhere on disk, and it is not recoverable
from the producing script either, because the fitted grid is derived at run time
from downscale factors and from isotropic resampling of the voxel spacing.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from luxar.gsplats.fit_gsplats import fit_gaussian_splats  # noqa: E402

STAMPS = (
    "source_shape",
    "source_dtype",
    "source_voxels",
    "source_bytes",
    "fitted_shape",
    "fitted_voxels",
    "occupancy",
    "voxels_per_splat",
)


def _sparse_blobs(shape=(24, 32, 32), n=20, dtype=np.uint16) -> np.ndarray:
    """A sparse blob field: sparse enough that occupancy is a meaningful number."""
    rng = np.random.default_rng(0)
    V = np.zeros(shape, dtype=dtype)
    hi = [s - 2 for s in shape]
    for _ in range(n):
        idx = rng.integers([2] * len(shape), hi)
        sl = tuple(slice(i - 1, i + 2) for i in idx)
        V[sl] = 3000
    return V


def _fit(V, **kw):
    return fit_gaussian_splats(
        V=V,
        seeds=120,
        n_iters=30,
        verbose=False,
        device="cpu",
        enable_dynamic_ops=False,
        napari_movie=False,
        **kw,
    )


def test_fit_stamps_the_source_grid() -> None:
    V = _sparse_blobs()
    stats = _fit(V).stats
    for key in STAMPS:
        assert key in stats, f"missing stamp {key!r}"
    assert stats["source_shape"] == list(V.shape)
    assert stats["source_dtype"] == str(V.dtype)
    assert stats["source_voxels"] == V.size
    assert stats["source_bytes"] == V.nbytes
    # A uint16 volume must not be recorded as the float the fitter works in:
    # the byte count is the denominator of every compression ratio.
    assert stats["source_dtype"] == "uint16"
    assert 0.0 < stats["occupancy"] < 1.0
    assert stats["voxels_per_splat"] > 0


def test_downscaling_keeps_the_two_grids_apart() -> None:
    """``source_*`` is what was handed in; ``fitted_*`` is what was optimised.

    Collapsing the two would silently overstate compression by the downscale
    factor cubed — the one mistake these stamps exist to prevent.
    """
    V = _sparse_blobs(shape=(24, 32, 32))
    stats = _fit(V, downscale=2).stats
    assert stats["source_shape"] == [24, 32, 32]
    assert stats["source_voxels"] == V.size
    assert stats["fitted_shape"] != stats["source_shape"], (
        "fitted grid was not downscaled, or it was recorded as the source grid"
    )
    assert stats["fitted_voxels"] < stats["source_voxels"]


def test_grids_agree_when_nothing_is_downscaled() -> None:
    stats = _fit(_sparse_blobs()).stats
    assert stats["fitted_shape"] == stats["source_shape"]
    assert stats["fitted_voxels"] == stats["source_voxels"]


def test_occupancy_tracks_how_full_the_volume_is() -> None:
    """A denser volume must report a higher occupancy than a sparse one.

    Pinning an absolute value would only restate the fixture; the property that
    matters is that the number responds to the data.
    """
    sparse = _fit(_sparse_blobs(n=5)).stats["occupancy"]
    dense = _fit(_sparse_blobs(n=60)).stats["occupancy"]
    assert dense > sparse, f"occupancy did not respond to density: {dense} vs {sparse}"


def test_stamps_reach_the_fitting_group_on_disk(tmp_path: Path) -> None:
    """They must survive the save, not just live in the in-memory stats.

    ``split_fitting_info`` whitelists what lands in ``fitting/``; a key missing
    from that list falls through to ``pipeline/`` instead, which is where a
    reader looking for provenance would not think to look.
    """
    V = _sparse_blobs()
    out = tmp_path / "stamped.gsplats.zarr"
    if out.exists():
        shutil.rmtree(out)
    _fit(V).save(out)
    attrs = json.loads((out / "fitting" / ".zattrs").read_text())
    for key in STAMPS:
        assert key in attrs, f"{key!r} did not reach fitting/ on disk"
    assert attrs["source_bytes"] == V.nbytes

    # The point of the exercise: compression is now computable from the artifact.
    stored = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    assert stored > 0
    assert attrs["source_bytes"] / stored > 0

    # And `gsplat info` reads them back out of `fitting/` and reports them —
    # the stamp is only worth writing if the read path surfaces it.
    from typer.testing import CliRunner

    from luxar.cli import app

    result = CliRunner().invoke(app, ["gsplat", "info", str(out), "--no-histograms"])
    assert result.exit_code == 0, result.output
    assert "Source volume: 24 x 32 x 32 uint16" in result.output, result.output
    assert "compression:" in result.output, result.output


def test_a_caller_that_already_cast_can_name_the_stored_dtype() -> None:
    """``source_dtype`` overrides what ``V`` reports.

    The CLI loads through ``load_volume``, which returns float32 whatever the
    file holds — so on the path that produces essentially every stored dataset
    the fitter never sees the acquisition's element type, and measuring ``V``
    would report the float working copy (2x too many bytes for a 16-bit stack).
    """
    V = _sparse_blobs().astype(np.float32)  # as load_volume would hand it over
    stats = _fit(V, source_dtype="uint16").stats
    assert stats["source_dtype"] == "uint16"
    assert stats["source_bytes"] == stats["source_voxels"] * 2
    # Without the override the same array records its float32 self, which is
    # what makes passing it necessary rather than decorative.
    assert _fit(V).stats["source_bytes"] == stats["source_voxels"] * 4


def test_load_volume_reports_the_stored_dtype(tmp_path: Path) -> None:
    """The loader is the last place the on-disk element type exists."""
    from luxar.io.volume import load_volume

    src = tmp_path / "vol.npy"
    np.save(src, _sparse_blobs(shape=(8, 8, 8)))
    info: dict = {}
    volume = load_volume(src, info=info)
    assert volume.dtype == np.float32  # unchanged contract
    assert info["source_dtype"] == "uint16"


def test_cli_fit_stamps_the_stored_dtype_not_the_loaders_cast(tmp_path: Path) -> None:
    """End to end through the command: a uint16 file must not record float32.

    This is the whole point of the stamp — a compression ratio quoted against
    the float working copy is exactly 2x too flattering on 16-bit data.
    """
    from typer.testing import CliRunner

    from luxar.cli import app

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
    attrs = json.loads((out / "fitting" / ".zattrs").read_text())
    assert attrs["source_dtype"] == "uint16"
    assert attrs["source_bytes"] == V.size * 2


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

    from luxar.cli.gsplat_ops.inspect_commands import _print_source_grid

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
    _print_source_grid(data, store)
    out = capsys.readouterr().out
    assert "Source volume: 8 x 8 x 8 uint16" in out, out
    assert "0.25:1" in out, out
