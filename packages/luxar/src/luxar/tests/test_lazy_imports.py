"""Importing the base package must stay lightweight (no eager torch/scipy).

Scene construction, dimensions, and basic zarr compilation should not pull in
the heavy gsplats subsystem. The GSplat re-exports resolve lazily (PEP 562) on
first attribute access.

The stronger contract in the second half of this module is behavioural: core
scene authoring must WORK on a plain ``pip install luxar``, with the whole
optional ``gsplats`` extra (``scipy`` *and* ``torch``) genuinely unimportable.
Every probe runs in a subprocess, both because a pristine interpreter is the
only honest way to test an import contract and because blocking the extra means
installing a ``sys.meta_path`` finder and evicting modules from ``sys.modules``,
which would poison the rest of the session in-process.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_import_luxar_does_not_import_torch() -> None:
    """A bare `import luxar` must not import torch (the gsplats-only dep)."""
    result = _run(
        "import luxar, sys; "
        "assert 'torch' not in sys.modules, 'torch imported eagerly by import luxar'; "
        "print('ok')"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_lazy_gsplat_export_is_accessible() -> None:
    """Accessing luxar.GSplatData resolves it lazily (and only then loads gsplats)."""
    result = _run(
        "import luxar; "
        "cls = luxar.GSplatData; "
        "from luxar import fit_gaussian_splats; "
        "assert cls.__name__ == 'GSplatData'; "
        "assert callable(fit_gaussian_splats); "
        "print('ok')"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_unknown_attribute_still_raises_attribute_error() -> None:
    """The lazy __getattr__ must not swallow genuinely missing attributes."""
    import luxar

    try:
        luxar.does_not_exist  # noqa: B018
    except AttributeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected AttributeError for unknown attribute")


def test_lazy_device_exports_stay_visible_to_dir() -> None:
    """`luxar.gsplats.utils.__dir__` must keep the lazy torch exports listed.

    Sphinx's ``automodule ... :members:`` collects from ``dir()``, so dropping
    ``__dir__`` would silently erase ``is_mps_available`` / ``resolve_torch_device``
    from the API reference and from ``objects.inv`` — with no warning and a green
    docs ratchet.
    """
    from luxar.gsplats import utils

    listed = dir(utils)
    assert "resolve_torch_device" in listed, listed
    assert "is_mps_available" in listed, listed
    # ...without the lazy __getattr__ swallowing genuinely missing attributes.
    with pytest.raises(AttributeError):
        utils.no_such_helper  # noqa: B018


#: Distribution roots that make up the optional ``gsplats`` extra.
_EXTRA_ROOTS = ("torch", "scipy")

# Installed first thing in the child: a meta-path finder that makes the whole
# ``gsplats`` extra unimportable, exactly as it is on a core-only install.
# Placed at the front of ``sys.meta_path`` so it wins over the real finders when
# the extra IS installed (which it is in the dev env this test normally runs in).
_BLOCK_EXTRA = """
import sys

_EXTRA_ROOTS = ("torch", "scipy")


class _GsplatsExtraBlocker:
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in _EXTRA_ROOTS:
            raise ModuleNotFoundError(f"No module named '{root}'", name=root)
        return None


for _name in [m for m in sys.modules if m.split(".")[0] in _EXTRA_ROOTS]:
    del sys.modules[_name]
sys.meta_path.insert(0, _GsplatsExtraBlocker())

import gsplats_extra_is_blocked_sentinel  # noqa: F401
"""

_SENTINEL = "import gsplats_extra_is_blocked_sentinel  # noqa: F401"

# The contract is behavioural, not "the module imports": build a real
# four-geometry scene plus a standalone ``.gsplats.zarr``, then read the latter
# back. Blocking only ``torch`` would be too weak — a module-level ``scipy``
# import anywhere in the now-unguarded ``gsplat_data`` closure would break every
# core-only install while a torch-only blocker stayed green.
_BUILD_SCENE = """
import numpy as np

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.gsplats import GSplatData

N = 64
rng = np.random.default_rng(0)
cholesky = np.zeros((N, 6), dtype=np.float32)
cholesky[:, 0] = cholesky[:, 2] = cholesky[:, 5] = 0.5
centers = rng.uniform(0, 10, (N, 3)).astype(np.float32)
amplitudes = rng.uniform(0.5, 1.5, N).astype(np.float32)

assert GSplatData.__module__ == "luxar.gsplats.gsplat_data", (
    f"GSplatData resolved to a stub in {GSplatData.__module__}"
)
data = GSplatData(
    centers=centers, amplitudes=amplitudes, cholesky_factors=cholesky
)
data.save(SPLATS_PATH)

with LuxarZarrCompiler(SCENE_PATH) as compiler:
    scene = compiler.create_scene(
        dimensions=Dimensions(
            [
                Dimension("z", unit="um", range=(0, 10)),
                Dimension("y", unit="um", range=(0, 10)),
                Dimension("x", unit="um", range=(0, 10)),
            ]
        )
    )
    scene.add_points("pts", positions=rng.uniform(0, 10, (N, 3)).astype(np.float32))
    scene.add_lines(
        "lns",
        vertices=rng.uniform(0, 10, (N, 3)).astype(np.float32),
        widths=np.full(N, 0.1, dtype=np.float32),
    )
    scene.add_gsplats(
        "gs",
        centers=rng.uniform(0, 10, (N, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.5, 1.5, N).astype(np.float32),
        cholesky_factors=cholesky,
    )
    scene.add_gsplats(
        "gs_dim_order",
        centers=centers,
        amplitudes=amplitudes,
        cholesky_factors=cholesky,
        dim_order=["x", "y", "z"],
    )
    scene.add_gsplats(
        "gs_partition",
        centers=centers,
        amplitudes=amplitudes,
        cholesky_factors=cholesky,
        partition={"parts": 4},
    )
    scene.add_gsplats_from_data("gs_data", data)
    scene.add_gsplats_from_file("gs_file", SPLATS_PATH)
    scene.add_mesh(
        "msh",
        vertices=np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
        ),
        faces=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32),
    )

# LOADING is part of the advertised core-only scope, and a save-only check would
# not notice a heavy import added inside `luxar.gsplats.io.load_gsplats`.
loaded = GSplatData.load(SPLATS_PATH)
assert loaded.n_splats == N, loaded.n_splats
assert loaded.ndim == 3, loaded.ndim
assert loaded.centers.shape == (N, 3), loaded.centers.shape
assert loaded.amplitudes.shape == (N,), loaded.amplitudes.shape
assert loaded.cholesky_factors.shape == (N, 6), loaded.cholesky_factors.shape
# The store quantizes, so compare loosely — this only has to prove that real
# values came back, not that the encoder is lossless.
np.testing.assert_allclose(np.sort(loaded.amplitudes), np.sort(amplitudes), rtol=0.05)
np.testing.assert_allclose(
    np.sort(loaded.centers, axis=0), np.sort(centers, axis=0), atol=0.05
)

print("BUILD-OK")
"""


@pytest.mark.parametrize("root", _EXTRA_ROOTS)
def test_extra_blocker_actually_blocks(root: str) -> None:
    """Guard the guard: the meta-path finder really does hide the whole extra.

    Without this, a broken blocker would let the real test pass on a dev machine
    (where torch and scipy ARE installed) while proving nothing. The first leg
    keeps the check from passing vacuously on a machine where ``root`` simply is
    not installed — then the blocker must hide that same working import.
    """
    unblocked = _run(f"import {root}; print('IMPORT-OK')")
    assert unblocked.returncode == 0, (
        f"{root} is not importable even unblocked, so blocking it proves "
        f"nothing:\n{unblocked.stdout}\n{unblocked.stderr}"
    )
    assert "IMPORT-OK" in unblocked.stdout, unblocked.stderr

    blocked = _run(_BLOCK_EXTRA.replace(_SENTINEL, f"import {root}"))
    assert blocked.returncode != 0, f"{blocked.stdout}\n{blocked.stderr}"
    assert f"No module named '{root}'" in blocked.stderr, blocked.stderr


def test_core_scene_authoring_without_gsplats_extra(tmp_path: Path) -> None:
    """Points/Lines/GSplats/Mesh + a standalone .gsplats.zarr, extra-free.

    ``scene.add_gsplats(...)`` is core authoring API, yet before #1853 it died on
    a plain ``pip install luxar``. The unconditional trigger is the compiler:
    ``write_gsplat_arrays`` (``io/_compiler/gsplat_assembly.py``) runs ``from
    ...gsplats.utils.trils import split_tril`` on every call, and importing that
    pure-NumPy submodule executes a parent ``__init__`` that eagerly pulled the
    torch-only ``device`` module. ``GSplatData`` had the same shape of problem
    one level up, behind ``luxar/gsplats/__init__.py``'s stub guard.
    """
    scene_path = tmp_path / "core_only.luxar.zarr"
    splats_path = tmp_path / "core_only.gsplats.zarr"
    script = (
        _BLOCK_EXTRA.replace(_SENTINEL, "")
        + f"\nSCENE_PATH = {str(scene_path)!r}\nSPLATS_PATH = {str(splats_path)!r}\n"
        + textwrap.dedent(_BUILD_SCENE)
    )
    result = _run(script)

    assert result.returncode == 0, (
        f"core authoring failed without the gsplats extra:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "BUILD-OK" in result.stdout, result.stderr

    # The scene was really compiled, not merely imported.
    assert scene_path.is_dir()
    written = {child.name for child in scene_path.iterdir()}
    assert {
        "pts",
        "lns",
        "gs",
        "gs_dim_order",
        "gs_partition",
        "gs_data",
        "gs_file",
        "msh",
    } <= written, written
    assert {"zarr.json", ".zgroup"} & written, written

    # And the gsplat node holds real arrays, not just a group shell.
    gs_written = {child.name for child in (scene_path / "gs").iterdir()}
    assert {"centers", "amplitudes", "cholesky_factors_diag"} <= gs_written, gs_written

    assert splats_path.is_dir()
    splat_written = {child.name for child in splats_path.iterdir()}
    assert {"zarr.json", ".zgroup"} & splat_written, splat_written
    assert {
        "amplitudes",
        "centers",
        "cholesky_factors_diag",
        "cholesky_factors_offdiag",
        "chunk_bounds",
    } <= splat_written, splat_written
