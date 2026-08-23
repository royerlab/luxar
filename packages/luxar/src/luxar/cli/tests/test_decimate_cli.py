"""``luxar gsplat decimate`` at the CLI layer.

The library function is covered next door
(``gsplats/lod/tests/test_decimate.py``); what only exists here is the command's
own logic — the mutually exclusive ``--target`` / ``--fraction`` pair, the
ordering-name validation, and the load → reduce → write round trip that produces
a file another command can read back.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar.cli import app
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.tests.test_arbol_warnings import _default_display_sandbox

N_SPLATS = 64


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A small flat dataset with enough splats for a visible reduction."""
    rng = np.random.default_rng(0)
    data = GSplatData(
        centers=(rng.random((N_SPLATS, 3)) * 10).astype(np.float32),
        amplitudes=rng.random(N_SPLATS).astype(np.float32),
        cholesky_factors=np.tile(
            np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (N_SPLATS, 1)
        ),
    )
    out = tmp_path / "src.gsplats.zarr"
    data.save(out)
    return out


@pytest.mark.parametrize(
    ("args", "expected"),
    [(["--target", "16"], 16), (["--fraction", "0.25"], 16)],
)
def test_decimate_writes_a_readable_flat_result(
    runner: CliRunner, source: Path, tmp_path: Path, args: list, expected: int
) -> None:
    """Both target spellings reduce and write a single flat leaf."""
    out = tmp_path / "small.gsplats.zarr"
    result = runner.invoke(app, ["gsplat", "decimate", str(source), str(out), *args])
    assert result.exit_code == 0, result.stdout
    reduced = GSplatData.load(out)
    assert reduced.n_splats == expected
    assert reduced.n_substitutive == 1  # flat, not a ladder


def test_merge_keeps_the_count_it_was_asked_for(
    runner: CliRunner, source: Path, tmp_path: Path
) -> None:
    """An explicit ``-m merge`` above the auto crossover is still honoured.

    The merge reduces by clustering, and a clustering driven by an integer
    compression factor can only halve at best — it used to answer a 48-of-64
    request with 32.
    """
    out = tmp_path / "most.gsplats.zarr"
    result = runner.invoke(
        app,
        ["gsplat", "decimate", str(source), str(out), "-n", "48", "-m", "merge"],
    )
    assert result.exit_code == 0, result.stdout
    assert GSplatData.load(out).n_splats == 48


def test_target_and_fraction_are_mutually_exclusive(
    runner: CliRunner, source: Path, tmp_path: Path
) -> None:
    out = tmp_path / "never.gsplats.zarr"
    both = runner.invoke(
        app,
        ["gsplat", "decimate", str(source), str(out), "-n", "8", "-f", "0.5"],
    )
    neither = runner.invoke(app, ["gsplat", "decimate", str(source), str(out)])
    assert both.exit_code == 1
    assert neither.exit_code == 1
    assert not out.exists()


def test_the_ignored_coarsen_dims_notice_reaches_a_cli_user(
    runner: CliRunner, source: Path, tmp_path: Path
) -> None:
    """The prefix family's dropped-knob notice is a WARNING, and must still show.

    It moved off ``aprint`` so a programmatic ``verbose=False`` call stays quiet
    (#1600 review), which only works because the CLI displays warnings: the root
    ``luxar`` callback installs ``install_arbol_warnings``, rendering them as
    arbol lines. This is the half of that trade the library test cannot see —
    without it, the fix would silence the notice for the users who need it.

    Run inside ``_default_display_sandbox``, because the hook deliberately steps
    aside for pytest's own per-test warning recorder: without it every warning
    in this process goes to the recorder and no CLI test could ever observe the
    display path a real user gets.
    """
    out = tmp_path / "small.gsplats.zarr"
    with _default_display_sandbox():
        result = runner.invoke(
            app,
            # -f 0.5 is at the crossover, so `auto` resolves to prefix and the
            # --coarsen-dims request is dropped.
            [
                "gsplat",
                "decimate",
                str(source),
                str(out),
                "-f",
                "0.5",
                "--coarsen-dims",
                "0,1",
            ],
        )
    assert result.exit_code == 0, result.stdout
    assert "coarsen_dims=[0, 1] is IGNORED" in result.stdout
    assert "method='auto' resolved to prefix" in result.stdout


def test_unknown_prefix_ordering_is_rejected(
    runner: CliRunner, source: Path, tmp_path: Path
) -> None:
    """A typo'd ordering names the valid set instead of raising three layers down."""
    out = tmp_path / "never.gsplats.zarr"
    result = runner.invoke(
        app,
        [
            "gsplat",
            "decimate",
            str(source),
            str(out),
            "-f",
            "0.5",
            "--prefix-method",
            "nonesuch",
        ],
    )
    assert result.exit_code != 0
    assert not out.exists()
