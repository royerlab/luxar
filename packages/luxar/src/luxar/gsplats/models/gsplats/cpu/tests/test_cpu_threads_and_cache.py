"""Thread shares, the numba cache fallback, and failure reporting."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from luxar.gsplats.models.gsplats.cpu import backend as backend_mod
from luxar.gsplats.utils import device


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(backend_mod.BACKEND_ENV, raising=False)
    monkeypatch.delenv(backend_mod.THREADS_ENV, raising=False)
    backend_mod.reset_for_tests()
    yield
    backend_mod.reset_for_tests()


# ------------------------------------------------------------- thread count


def test_an_invalid_thread_count_warns_once_however_often_it_is_read(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(backend_mod.THREADS_ENV, "lots")
    for _ in range(5):
        assert backend_mod.resolve_thread_count() == device.physical_core_count()
    assert capsys.readouterr().out.count("ignoring LUXAR_CPU_THREADS") == 1


def test_a_changed_thread_count_is_picked_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(backend_mod.THREADS_ENV, "3")
    assert backend_mod.resolve_thread_count() == 3
    monkeypatch.setenv(backend_mod.THREADS_ENV, "5")
    assert backend_mod.resolve_thread_count() == 5


def test_zero_thread_count_uses_default_and_worker_share(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(device, "physical_core_count", lambda: 16)
    monkeypatch.setattr(backend_mod, "physical_core_count", lambda: 16)
    monkeypatch.setenv("LUXAR_CPU_THREADS", "0")
    assert backend_mod.resolve_thread_count() == 16
    assert device.cpu_worker_threads_env(4) == {"LUXAR_CPU_THREADS": "4"}
    assert "ignoring LUXAR_CPU_THREADS" not in capsys.readouterr().out


@pytest.mark.parametrize(("workers", "share"), [(1, 16), (3, 5), (8, 2), (40, 1)])
def test_worker_share_splits_the_physical_cores(
    monkeypatch: pytest.MonkeyPatch, workers: int, share: int
) -> None:
    monkeypatch.setattr(device, "physical_core_count", lambda: 16)
    assert device.cpu_worker_threads_env(workers) == {"LUXAR_CPU_THREADS": str(share)}


def test_worker_share_defers_to_a_user_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LUXAR_CPU_THREADS", "6")
    assert device.cpu_worker_threads_env(4) == {}


def test_parallel_tile_and_box_workers_get_a_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luxar.gsplats.fit_tiled_parallel import _tile_worker_env
    from luxar.gsplats.planner.fit_planned_parallel import _worker_env

    monkeypatch.setattr(device, "physical_core_count", lambda: 12)
    assert _tile_worker_env(4)["LUXAR_CPU_THREADS"] == "3"
    assert _worker_env(False, 6)["LUXAR_CPU_THREADS"] == "2"
    # A single worker keeps the default (all physical cores).
    assert "LUXAR_CPU_THREADS" not in _tile_worker_env(1)
    assert "LUXAR_CPU_THREADS" not in _worker_env(True, 1)
    monkeypatch.setenv("LUXAR_CPU_THREADS", "5")
    assert _tile_worker_env(4)["LUXAR_CPU_THREADS"] == "5"  # inherited


# ------------------------------------------------------- failure reporting


def test_a_broken_numba_is_reported_as_broken_not_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luxar.gsplats.fitting.initialization import _cpu_native_hint

    def _boom() -> None:
        raise OSError("libllvmlite.so: undefined symbol")

    monkeypatch.setattr(backend_mod, "_import_kernel_module", _boom)
    assert backend_mod.get_cpu_kernels() is None
    assert not backend_mod.numba_is_missing()
    hint = _cpu_native_hint(3)
    assert "undefined symbol" in hint and "pip install numba" not in hint

    monkeypatch.setenv(backend_mod.BACKEND_ENV, "native")
    with pytest.raises(RuntimeError) as info:
        backend_mod.get_cpu_kernels()
    assert "Fix the numba installation" in str(info.value)
    assert "Install numba" not in str(info.value)


def test_a_missing_numba_still_suggests_installing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luxar.gsplats.fitting.initialization import _cpu_native_hint

    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.setitem(sys.modules, backend_mod._KERNELS_MODULE, None)
    assert backend_mod.get_cpu_kernels() is None
    assert backend_mod.numba_is_missing()
    assert "pip install numba" in _cpu_native_hint(3)


# ------------------------------------------------------------ cache fallback

# Run in a fresh interpreter: the kernel module is decorated at import time,
# and the scenario needs numba to find NO usable cache location, which the
# script forces by trimming numba's locator list before the import.
_CACHE_PROBE = textwrap.dedent(
    """
    import sys
    from numba.core import caching

    keep = {keep}
    impl = getattr(caching, "CacheImpl", None) or getattr(caching, "_CacheImpl")
    impl._locator_classes = [
        cls for cls in impl._locator_classes if cls.__name__.lstrip("_") in keep
    ]
    from luxar.gsplats.models.gsplats.cpu import backend, render_gaussians_cpu_native
    import torch

    kernels = backend.get_cpu_kernels()
    assert kernels is not None, backend.cpu_native_unavailable_reason()
    out = render_gaussians_cpu_native(
        (6, 6, 6), torch.full((1, 3), 3.0), torch.eye(3)[None], torch.ones(1)
    )
    assert float(out.max()) > 0.5
    print("CACHE_ENABLED", backend.kernel_cache_enabled())
    """
)


def _run_probe(tmp_path: Path, keep: set[str]) -> str:
    pytest.importorskip("numba")
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_CACHE_DIR"}
    env["TMPDIR"] = str(tmp_path)  # tempfile.gettempdir() for the child
    env["LUXAR_CPU_THREADS"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", _CACHE_PROBE.format(keep=sorted(keep))],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_no_cache_location_falls_back_to_a_temp_dir_cache(tmp_path: Path) -> None:
    out = _run_probe(tmp_path, {"UserProvidedCacheLocator"})
    assert "caching in" in out and "CACHE_ENABLED True" in out
    cache_dirs = list(tmp_path.glob("luxar-numba-cache-*"))
    assert cache_dirs and any(cache_dirs[0].rglob("*.nbi"))


def test_no_cache_at_all_still_runs_the_kernels(tmp_path: Path) -> None:
    out = _run_probe(tmp_path, set())
    assert "compiling without one" in out and "CACHE_ENABLED False" in out


# -------------------------------------------------------- non-finite splats


def test_non_finite_splats_are_skipped_and_reported_once(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("numba")
    from luxar.gsplats.models.gsplats.cpu import gsplat_model_cpu

    monkeypatch.setattr(gsplat_model_cpu, "_WARNED_NONFINITE", False)
    centers = np.array(
        [[3.0, 3.0, 3.0], [np.nan, 2.0, 2.0], [4.0, 4.0, 4.0]], np.float32
    )
    Ls = np.stack([np.eye(3, dtype=np.float32)] * 3)
    Ls[2, 1, 0] = np.inf
    amps = np.array([1.0, 1.0, 1.0], np.float32)
    for _ in range(3):
        out = gsplat_model_cpu.cpu_render_numpy((7, 7, 7), centers, Ls, amps)
        assert np.isfinite(out).all() and out.max() > 0.5
        _, _, d_amps = gsplat_model_cpu.cpu_render_backward_numpy(
            np.ones_like(out), centers, Ls, amps
        )
        assert d_amps[1] == d_amps[2] == 0.0 and d_amps[0] > 0.0
    out_text = capsys.readouterr().out
    assert out_text.count("non-finite parameters") == 1
    assert "skipped 2 splat(s)" in out_text
