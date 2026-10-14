"""Platform-independent tests for Metal kernel-backend selection.

No GPU needed: MPS availability, the AOT extension and the runtime compiler are
monkeypatched, so these run on every CI runner (Linux included).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from luxar.gsplats.models.gsplats.metal import backend as backend_mod
from luxar.gsplats.models.gsplats.metal import runtime_backend
from luxar.gsplats.models.gsplats.metal.backend import (
    BACKEND_ENV,
    MetalKernelBackend,
    get_metal_backend,
    metal_backend_unavailable_reason,
    requested_backend,
)


@pytest.fixture(autouse=True)
def _fresh_resolution(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(BACKEND_ENV, raising=False)
    backend_mod.reset_for_tests()
    yield
    backend_mod.reset_for_tests()


def _fake(name: str) -> MetalKernelBackend:
    return MetalKernelBackend(
        name,
        lambda *a: f"{name}-fwd",
        lambda *a: f"{name}-bwd",
        lambda *a: f"{name}-conic",
    )


def _env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mps: bool,
    aot: bool,
    runtime: bool,
) -> dict[str, int]:
    """Fake the three probes; returns call counters for the backend probes."""
    calls = {"aot": 0, "runtime": 0}
    monkeypatch.setattr(
        backend_mod,
        "mps_unavailable_reason",
        lambda: None if mps else "PyTorch cannot access an MPS device",
    )

    def fake_aot() -> tuple[MetalKernelBackend | None, str]:
        calls["aot"] += 1
        return (_fake("aot"), "") if aot else (None, "compiled extension not built")

    def fake_runtime() -> tuple[MetalKernelBackend | None, str]:
        calls["runtime"] += 1
        return (
            (_fake("runtime"), "")
            if runtime
            else (None, "torch 2.5.1 has no torch.mps.compile_shader")
        )

    monkeypatch.setattr(backend_mod, "_aot_backend", fake_aot)
    monkeypatch.setattr(backend_mod, "_runtime_backend", fake_runtime)
    return calls


# =============================================================================
# LUXAR_METAL_BACKEND parsing
# =============================================================================


@pytest.mark.parametrize(
    "raw,expected",
    [("", "auto"), ("  RUNTIME ", "runtime"), ("aot", "aot"), ("torch", "torch")],
)
def test_requested_backend_parsing(raw: str, expected: str) -> None:
    assert requested_backend({BACKEND_ENV: raw}) == expected


def test_requested_backend_unset_is_auto() -> None:
    assert requested_backend({}) == "auto"


def test_requested_backend_rejects_typos() -> None:
    with pytest.raises(ValueError, match="runtim"):
        requested_backend({BACKEND_ENV: "runtim"})


# =============================================================================
# Selection order
# =============================================================================


def test_auto_prefers_aot(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _env(monkeypatch, mps=True, aot=True, runtime=True)
    chosen = get_metal_backend()
    assert chosen is not None and chosen.name == "aot"
    assert chosen.forward_splat_3d() == "aot-fwd"
    assert calls["runtime"] == 0, "runtime must not compile when AOT is usable"


def test_auto_falls_to_runtime_without_aot(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, mps=True, aot=False, runtime=True)
    chosen = get_metal_backend()
    assert chosen is not None and chosen.name == "runtime"
    assert chosen.backward_splat_3d() == "runtime-bwd"
    assert chosen.compute_conic_metal() == "runtime-conic"


def test_auto_falls_to_torch_with_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, mps=True, aot=False, runtime=False)
    assert get_metal_backend() is None
    reason = metal_backend_unavailable_reason()
    assert reason is not None
    assert "compile_shader" in reason and "not built" in reason


def test_auto_without_mps_is_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _env(monkeypatch, mps=False, aot=True, runtime=True)
    assert get_metal_backend() is None
    assert metal_backend_unavailable_reason() == "PyTorch cannot access an MPS device"
    assert calls == {"aot": 0, "runtime": 0}


def test_force_runtime_skips_aot(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _env(monkeypatch, mps=True, aot=True, runtime=True)
    monkeypatch.setenv(BACKEND_ENV, "runtime")
    chosen = get_metal_backend()
    assert chosen is not None and chosen.name == "runtime"
    assert calls["aot"] == 0


def test_force_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, mps=True, aot=True, runtime=True)
    monkeypatch.setenv(BACKEND_ENV, "torch")
    assert get_metal_backend() is None
    assert metal_backend_unavailable_reason() == f"{BACKEND_ENV}=torch"


@pytest.mark.parametrize(
    "forced,env,match",
    [
        ("aot", dict(mps=True, aot=False, runtime=True), "not built"),
        ("runtime", dict(mps=True, aot=True, runtime=False), "compile_shader"),
        ("runtime", dict(mps=False, aot=False, runtime=True), "MPS device"),
        ("aot", dict(mps=False, aot=True, runtime=True), "MPS device"),
    ],
)
def test_forced_backend_that_cannot_run_raises(
    monkeypatch: pytest.MonkeyPatch, forced: str, env: dict[str, bool], match: str
) -> None:
    _env(monkeypatch, **env)
    monkeypatch.setenv(BACKEND_ENV, forced)
    with pytest.raises(RuntimeError, match=match):
        get_metal_backend()


def test_resolution_is_cached_and_logged_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _env(monkeypatch, mps=True, aot=False, runtime=True)
    lines: list[str] = []
    monkeypatch.setattr(backend_mod, "aprint", lambda msg: lines.append(str(msg)))
    first = get_metal_backend()
    second = get_metal_backend()
    assert first is second
    assert calls["runtime"] == 1
    assert len(lines) == 1 and "runtime" in lines[0]


def test_torch_fallback_log_names_the_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, mps=True, aot=False, runtime=False)
    lines: list[str] = []
    monkeypatch.setattr(backend_mod, "aprint", lambda msg: lines.append(str(msg)))
    get_metal_backend()
    assert len(lines) == 1
    assert "using PyTorch" in lines[0] and "compile_shader" in lines[0]


# =============================================================================
# Public package helpers
# =============================================================================


def test_is_metal_available_follows_the_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import luxar.gsplats.models.gsplats.metal as metal_pkg

    monkeypatch.setattr(metal_pkg.sys, "platform", "darwin")
    _env(monkeypatch, mps=True, aot=False, runtime=True)
    assert metal_pkg.is_metal_available()
    assert "runtime" in metal_pkg.get_metal_status()

    monkeypatch.setenv(BACKEND_ENV, "torch")
    assert not metal_pkg.is_metal_available()
    assert f"{BACKEND_ENV}=torch" in metal_pkg.get_metal_status()


def test_get_metal_status_reports_a_forced_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import luxar.gsplats.models.gsplats.metal as metal_pkg

    monkeypatch.setattr(metal_pkg.sys, "platform", "darwin")
    _env(monkeypatch, mps=True, aot=False, runtime=True)
    monkeypatch.setenv(BACKEND_ENV, "aot")
    assert "not built" in metal_pkg.get_metal_status()


# =============================================================================
# Runtime availability probe (torch version floor)
# =============================================================================


def _fake_torch_mps(monkeypatch: pytest.MonkeyPatch, **attrs: Any) -> None:
    import torch

    fake_mps = SimpleNamespace(**attrs)
    monkeypatch.setattr(torch, "mps", fake_mps)
    monkeypatch.setattr(runtime_backend.sys, "platform", "darwin")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)


def test_runtime_prefers_the_public_compiler_with_arg_casts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public, private = object(), object()
    _fake_torch_mps(monkeypatch, compile_shader=public, _compile_shader=private)
    assert runtime_backend._shader_compiler() is public
    assert runtime_backend.runtime_unavailable_reason() is None


def test_runtime_uses_the_private_compiler_on_torch_2_6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private = object()
    _fake_torch_mps(monkeypatch, _compile_shader=private)
    assert runtime_backend._shader_compiler() is private
    assert runtime_backend.runtime_unavailable_reason() is None


def test_launch_binds_every_integer_argument_as_int32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An int64 [D, H, W] would be read by ``constant uint3&`` as (D, 0, H)."""
    recorded: dict[str, Any] = {}

    def kernel(*args: Any, **kwargs: Any) -> None:
        recorded["args"] = args
        recorded.update(kwargs)

    monkeypatch.setattr(
        runtime_backend, "_library", SimpleNamespace(some_kernel=kernel)
    )
    runtime_backend._launch(
        "some_kernel", ["buf", [5, 7, 11], 3, 2.5], (1, 2), threads=192, group_size=64
    )
    assert recorded["arg_casts"] == {1: "int32", 2: "int32"}
    assert recorded["threads"] == 192 and recorded["group_size"] == 64
    assert recorded["args"] == ("buf", [5, 7, 11], 3, 2.5)


def test_splat_count_cap_keeps_the_dispatch_in_32_bits() -> None:
    cap = runtime_backend.MAX_SPLATS
    assert cap * runtime_backend.THREADGROUP_SIZE <= 2**32 - 1
    assert (cap + 1) * runtime_backend.THREADGROUP_SIZE > 2**32 - 1
    runtime_backend.check_splat_count(cap)
    with pytest.raises(ValueError, match="at most 67108863 splats"):
        runtime_backend.check_splat_count(cap + 1)


def test_runtime_unavailable_without_a_shader_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_torch_mps(monkeypatch)
    reason = runtime_backend.runtime_unavailable_reason()
    assert reason is not None and "torch >= 2.7" in reason


def test_runtime_unavailable_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_backend.sys, "platform", "linux")
    assert runtime_backend.runtime_unavailable_reason() == (
        "Metal is only available on macOS"
    )


def test_runtime_compile_error_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(source: str) -> Any:
        raise RuntimeError("program_source:12: error: boom")

    _fake_torch_mps(monkeypatch, compile_shader=broken)
    runtime_backend.reset_for_tests()
    try:
        with pytest.raises(runtime_backend.MetalCompileError, match="boom"):
            runtime_backend.prepare()
    finally:
        runtime_backend.reset_for_tests()


def test_runtime_compile_error_falls_back_in_auto_and_raises_when_forced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(source: str) -> Any:
        raise RuntimeError("error: boom")

    _fake_torch_mps(monkeypatch, compile_shader=broken)
    monkeypatch.setattr(backend_mod, "mps_unavailable_reason", lambda: None)
    monkeypatch.setattr(backend_mod, "_aot_backend", lambda: (None, "not built"))
    runtime_backend.reset_for_tests()
    try:
        assert get_metal_backend() is None
        assert "boom" in (metal_backend_unavailable_reason() or "")
        monkeypatch.setenv(BACKEND_ENV, "runtime")
        with pytest.raises(RuntimeError, match="boom"):
            get_metal_backend()
    finally:
        runtime_backend.reset_for_tests()


def test_kernel_source_ships_inside_the_package() -> None:
    """The runtime path reads kernels.metal from the installed package."""
    from importlib.resources import files

    shipped = files("luxar.gsplats.models.gsplats.metal") / "src" / "kernels.metal"
    assert shipped.is_file()
    assert Path(str(shipped)) == runtime_backend.KERNELS_SOURCE
    assert "rasterize_forward_splat_centric_3d" in shipped.read_text()


# =============================================================================
# AOT discovery never builds and never uses a stale build
# =============================================================================


def _fake_metal_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    for name in ("kernels.metal", "bindings.mm"):
        (src / name).write_text("// source\n")
    (tmp_path / "setup.py").write_text("# setup\n")
    monkeypatch.setattr(backend_mod, "_METAL_DIR", tmp_path)
    monkeypatch.setattr(backend_mod, "_SRC_DIR", src)
    return tmp_path


def test_aot_absent_reports_the_explicit_build_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_metal_dir(monkeypatch, tmp_path)
    module, why = backend_mod._import_aot()
    assert module is None
    assert "not built" in why and "build_ext --inplace" in why
    assert list(tmp_path.glob("*.so")) == [], "discovery must never build"


def test_stale_aot_build_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metal_dir = _fake_metal_dir(monkeypatch, tmp_path)
    extension = metal_dir / backend_mod._EXTENSION_PATTERN.replace("*", "darwin")
    extension.write_bytes(b"")
    metallib = metal_dir / "src" / "default.metallib"
    metallib.write_bytes(b"")
    old = 1_000_000.0
    for path in (extension, metallib):
        os.utime(path, (old, old))
    module, why = backend_mod._import_aot()
    assert module is None and "stale" in why


def test_fresh_aot_build_is_not_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metal_dir = _fake_metal_dir(monkeypatch, tmp_path)
    old = 1_000_000.0
    for path in [*metal_dir.rglob("*.*")]:
        os.utime(path, (old, old))
    extension = metal_dir / backend_mod._EXTENSION_PATTERN.replace("*", "darwin")
    extension.write_bytes(b"")
    (metal_dir / "src" / "default.metallib").write_bytes(b"")
    assert not backend_mod.extension_is_stale(extension)


def test_newer_setup_py_makes_the_build_stale_and_hint_forces_a_rebuild(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """setuptools only compares bindings.mm, so the hint must say --force."""
    metal_dir = _fake_metal_dir(monkeypatch, tmp_path)
    extension = metal_dir / backend_mod._EXTENSION_PATTERN.replace("*", "darwin")
    extension.write_bytes(b"")
    (metal_dir / "src" / "default.metallib").write_bytes(b"")
    old = 1_000_000.0
    for path in (extension, metal_dir / "src" / "bindings.mm"):
        os.utime(path, (old, old))
    assert backend_mod.extension_is_stale(extension)
    _, why = backend_mod._import_aot()
    assert "stale" in why and "build_ext --inplace --force" in why


def test_setup_py_forces_the_extension_rebuild() -> None:
    """``build_ext`` must rebuild even when only setup.py/kernels changed."""
    import ast

    tree = ast.parse((backend_mod._METAL_DIR / "setup.py").read_text())
    run = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    assert any(
        isinstance(stmt, ast.Assign)
        and isinstance(stmt.targets[0], ast.Attribute)
        and stmt.targets[0].attr == "force"
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is True
        for stmt in ast.walk(run)
    )


def test_get_metal_status_reports_a_bad_backend_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import luxar.gsplats.models.gsplats.metal as metal_pkg

    monkeypatch.setattr(metal_pkg.sys, "platform", "darwin")
    monkeypatch.setenv(BACKEND_ENV, "metalx")
    status = metal_pkg.get_metal_status()
    assert "metalx" in status and BACKEND_ENV in status


# =============================================================================
# Fitting-side fallback (initialization._try_metal_model)
# =============================================================================


def test_try_metal_model_warns_and_returns_none_without_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luxar.gsplats.fitting import initialization

    _env(monkeypatch, mps=True, aot=False, runtime=False)
    with pytest.warns(UserWarning, match="PyTorch fallback"):
        assert (
            initialization._try_metal_model(SimpleNamespace(verbose=False), {})  # type: ignore[arg-type]
            is None
        )


def test_try_metal_model_quiet_when_torch_is_forced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luxar.gsplats.fitting import initialization

    _env(monkeypatch, mps=True, aot=True, runtime=True)
    monkeypatch.setenv(BACKEND_ENV, "torch")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            initialization._try_metal_model(SimpleNamespace(verbose=False), {})  # type: ignore[arg-type]
            is None
        )


def test_try_metal_model_raises_when_forced_backend_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LUXAR_METAL_BACKEND=runtime means 'fail if runtime fails', in fitting too."""
    from luxar.gsplats.fitting import initialization

    _env(monkeypatch, mps=True, aot=True, runtime=False)
    monkeypatch.setenv(BACKEND_ENV, "runtime")
    with pytest.raises(RuntimeError, match="compile_shader"):
        initialization._try_metal_model(SimpleNamespace(verbose=False), {})  # type: ignore[arg-type]


def test_try_metal_model_builds_the_metal_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luxar.gsplats.fitting import initialization
    from luxar.gsplats.models.gsplats import metal as metal_pkg

    _env(monkeypatch, mps=True, aot=False, runtime=True)
    built: dict[str, Any] = {}

    def fake_model(**kwargs: Any) -> str:
        built.update(kwargs)
        return "metal-model"

    monkeypatch.setattr(metal_pkg, "GaussianSplatModelMetal", fake_model)
    model = initialization._try_metal_model(
        SimpleNamespace(verbose=False),  # type: ignore[arg-type]
        {"shape": (8, 8, 8)},
    )
    assert model == "metal-model" and built == {"shape": (8, 8, 8)}
