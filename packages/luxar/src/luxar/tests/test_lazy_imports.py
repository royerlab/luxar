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


@pytest.mark.parametrize(
    "module",
    [
        "luxar.demos._support.datasets.bundles",
        "luxar.demos._support.datasets.cache",
        "luxar.utils.colors",
        "luxar.demos._support.runtime.device",
        "luxar.demos._support.runtime.flags",
        "luxar.demos._support.datasets.lfs",
        "luxar.demos._support.datasets.payload_agreement",
        "luxar.demos._support.runtime.provenance",
        "luxar.utils.scenes",
        "luxar.demos._support.runtime.viewer",
        "luxar.demos._support.downloads.zip_safety",
    ],
)
def test_demo_utilities_do_not_import_cli_dependencies(module: str) -> None:
    """Shared demo helpers must not pull in the eager CLI package."""
    result = _run(
        "import importlib, sys; "
        f"importlib.import_module({module!r}); "
        "assert 'luxar.cli' not in sys.modules, 'luxar.cli imported by demo utilities'; "
        "assert 'typer' not in sys.modules, 'typer imported by demo utilities'; "
        "assert 'click' not in sys.modules, 'click imported by demo utilities'; "
        "print('ok')"
    )
    assert result.returncode == 0, f"{module}: {result.stderr}"
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


def test_fit_parameters_has_a_core_only_stub() -> None:
    """The class-API parameter export must degrade like the fitter itself."""
    result = _run(
        _BLOCK_EXTRA.replace(
            _SENTINEL,
            "from luxar.gsplats import FitParameters\n"
            "try:\n"
            "    FitParameters()\n"
            "except ImportError as exc:\n"
            "    assert 'luxar[gsplats]' in str(exc)\n"
            "else:\n"
            "    raise AssertionError('expected the optional-dependency stub')\n"
            "print('ok')",
        )
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


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
from luxar.gsplats import AdditiveSubLOD, GSplatData, SubstitutiveLevel

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
    centers=centers,
    amplitudes=amplitudes,
    cholesky_factors=cholesky,
    stats={"psnr_db": 31.0, "iterations": 2000, "n_splats": N},
)
data.save(SPLATS_PATH)
loaded_data = GSplatData.load(SPLATS_PATH)

scaled = loaded_data.scale_intensity(0.5)
np.testing.assert_allclose(scaled.amplitudes, loaded_data.amplitudes * 0.5)
filtered = loaded_data.filter_by(amplitude_min=1.0)
assert 0 < filtered.n_splats < N
cropped = loaded_data.slice_by([slice(0, 5), slice(0, 5), slice(0, 5)])
assert 0 < cropped.n_splats < N
embedded = loaded_data.embed_dimension(5.0)
np.testing.assert_allclose(embedded.centers[:, -1], 5.0)

translated = data.translate(np.ones(3, dtype=np.float32))
np.testing.assert_allclose(translated.centers, centers + 1)
transformed = data.transform(np.eye(3, dtype=np.float32) * 2)
np.testing.assert_allclose(transformed.centers, centers * 2)
centered = data.center_at_centroid()
np.testing.assert_allclose(
    centered.centers.T @ centered.amplitudes,
    np.zeros(3, dtype=np.float32),
    atol=1e-4,
)


def sublod(count):
    return AdditiveSubLOD(
        centers=centers[:count],
        amplitudes=amplitudes[:count],
        cholesky_factors=cholesky[:count],
    )


GSplatData(additive_sublods=[sublod(32), sublod(N)]).save(LADDER_PATH)
loaded_ladder = GSplatData.load(LADDER_PATH)
assert loaded_ladder.n_additive_sublods == 2
assert loaded_ladder.n_substitutive == 1
assert loaded_ladder.additive_prefix(0).n_splats == 32

stamped_rung = sublod(32)
stamped_rung.stats.update(
    energy_fraction_cum=0.5, lod_n_splats=32, lod_cumulative_n=32
)
GSplatData(additive_sublods=[stamped_rung, sublod(N)]).save(STAMPED_LADDER_PATH)
loaded_stamped_ladder = GSplatData.load(STAMPED_LADDER_PATH)
for edit in (
    lambda: loaded_stamped_ladder.scale_intensity(0.5),
    lambda: loaded_stamped_ladder.additive_prefix(0),
):
    try:
        edit()
    except ModuleNotFoundError as exc:
        assert exc.name == "scipy", exc
    else:
        raise AssertionError("stamped LOD edit unexpectedly stayed core-only")
print("STAMPED-LOD-BLOCKED")

GSplatData(
    substitutive_levels=[
        SubstitutiveLevel(additive_sublods=[sublod(16)]),
        SubstitutiveLevel(additive_sublods=[sublod(N)]),
    ]
).save(STACK_PATH)
loaded_stack = GSplatData.load(STACK_PATH)
assert loaded_stack.n_substitutive == 2
assert [level.n_additive_lods for level in loaded_stack.substitutive_levels] == [1, 1]

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
        partition={"max_elements": 16},
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
    one level up, behind ``luxar/gsplats/__init__.py``'s stub guard. Content
    edits such as ``scale_intensity`` must likewise avoid the SciPy-backed LOD
    package when an unstamped dataset has nothing to restamp (#2229).
    """
    scene_path = tmp_path / "core_only.luxar.zarr"
    splats_path = tmp_path / "core_only.gsplats.zarr"
    ladder_path = tmp_path / "core_only_ladder.gsplats.zarr"
    stamped_ladder_path = tmp_path / "stamped_ladder.gsplats.zarr"
    stack_path = tmp_path / "core_only_stack.gsplats.zarr"
    script = (
        _BLOCK_EXTRA.replace(_SENTINEL, "")
        + f"\nSCENE_PATH = {str(scene_path)!r}\n"
        + f"SPLATS_PATH = {str(splats_path)!r}\n"
        + f"LADDER_PATH = {str(ladder_path)!r}\n"
        + f"STAMPED_LADDER_PATH = {str(stamped_ladder_path)!r}\n"
        + f"STACK_PATH = {str(stack_path)!r}\n"
        + textwrap.dedent(_BUILD_SCENE)
    )
    result = _run(script)

    assert result.returncode == 0, (
        f"core authoring failed without the gsplats extra:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "BUILD-OK" in result.stdout, result.stderr
    assert "STAMPED-LOD-BLOCKED" in result.stdout, result.stderr

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

    partition_written = {
        child.name for child in (scene_path / "gs_partition").iterdir()
    }
    assert {"part_0", "part_1"} <= partition_written, partition_written
    assert "centers" not in partition_written, partition_written

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
