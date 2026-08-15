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
