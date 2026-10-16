"""Tests for the NVRTC runtime-compilation machinery (``luxar.gsplats._nvrtc``).

Everything above the "real NVRTC" section runs anywhere -- no GPU, no CUDA
library: library location, compile-target selection, cache keying and cache
robustness, and the runtime's compile-or-load-from-cache flow against a fake
compiler. The real-NVRTC tests compile the shipped kernel sources for every
target architecture WITHOUT a GPU; they skip only when no ``libnvrtc`` exists
in the environment (CPU torch, macOS), which is a genuine environment limit.
"""

from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

import pytest

from luxar.gsplats import _nvrtc
from luxar.gsplats._nvrtc import (
    NVRTC_LIBRARY_ENV,
    CompiledProgram,
    CompileTarget,
    KernelProgram,
    Nvrtc,
    NvrtcCompileError,
    NvrtcRuntime,
    NvrtcUnavailableError,
    cache_key,
    cache_paths,
    cache_root,
    candidate_nvrtc_libraries,
    read_cached_program,
    select_target,
    write_cached_program,
)

GSPLATS = Path(_nvrtc.__file__).parent
SPLAT_SRC = GSPLATS / "models" / "gsplats" / "cuda" / "src"
NLM_SRC = GSPLATS / "preprocessing" / "cuda" / "src"


# =============================================================================
# Library location
# =============================================================================


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_override_env_comes_first(tmp_path: Path) -> None:
    cands = candidate_nvrtc_libraries(
        cuda_major=13, wheel_roots=[], environ={NVRTC_LIBRARY_ENV: "/x/libnvrtc.so.99"}
    )
    assert cands[0] == "/x/libnvrtc.so.99"


def test_wheel_layouts_cu13_and_cu12_are_found(tmp_path: Path) -> None:
    root = tmp_path / "nvidia"
    cu13 = _touch(root / "cu13" / "lib" / "libnvrtc.so.13")
    _touch(root / "cu13" / "lib" / "libnvrtc-builtins.so.13.0")  # not a candidate
    cu12 = _touch(root / "cuda_nvrtc" / "lib" / "libnvrtc.so.12")

    cands = candidate_nvrtc_libraries(cuda_major=13, wheel_roots=[root], environ={})

    assert cands.index(str(cu13)) < cands.index(str(cu12))
    assert not any("builtins" in c for c in cands)
    # Wheel hits come before the bare sonames the loader would search for.
    assert cands.index(str(cu12)) < cands.index("libnvrtc.so.13")


def test_torch_cuda_major_is_preferred(tmp_path: Path) -> None:
    root = tmp_path / "nvidia"
    cu13 = _touch(root / "cu13" / "lib" / "libnvrtc.so.13")
    cu12 = _touch(root / "cuda_nvrtc" / "lib" / "libnvrtc.so.12")

    cands = candidate_nvrtc_libraries(cuda_major=12, wheel_roots=[root], environ={})

    assert cands.index(str(cu12)) < cands.index(str(cu13))
    assert cands.index("libnvrtc.so.12") < cands.index("libnvrtc.so.13")


def test_conda_and_torch_lib_dirs_follow_the_wheels(tmp_path: Path) -> None:
    root = tmp_path / "nvidia"
    wheel = _touch(root / "cu13" / "lib" / "libnvrtc.so.13")
    conda = _touch(tmp_path / "conda" / "lib" / "libnvrtc.so.12")
    torch_lib = _touch(tmp_path / "torch" / "lib" / "libnvrtc.so.13")

    cands = candidate_nvrtc_libraries(
        cuda_major=13,
        wheel_roots=[root],
        environ={"CONDA_PREFIX": str(tmp_path / "conda")},
        torch_lib_dir=torch_lib.parent,
    )

    order = [cands.index(str(p)) for p in (wheel, conda, torch_lib)]
    assert order == sorted(order)
    assert order[-1] < cands.index("libnvrtc.so.13")  # before the loader search


def test_shortest_soname_first_and_cuda_home(tmp_path: Path) -> None:
    root = tmp_path / "nvidia"
    full = _touch(root / "cu13" / "lib" / "libnvrtc.so.13.0.88")
    short = _touch(root / "cu13" / "lib" / "libnvrtc.so.13")
    home = tmp_path / "cuda"
    home_lib = _touch(home / "lib64" / "libnvrtc.so.12")

    cands = candidate_nvrtc_libraries(
        cuda_major=13, wheel_roots=[root], environ={"CUDA_HOME": str(home)}
    )

    assert cands.index(str(short)) < cands.index(str(full))
    assert str(home_lib) in cands
    assert len(cands) == len(set(cands))


def test_no_library_anywhere_raises_unavailable() -> None:
    with pytest.raises(NvrtcUnavailableError, match="libnvrtc"):
        _nvrtc._load_first(["/nonexistent/libnvrtc.so.13", "libnvrtc-not-a-real.so"])


def test_non_linux_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(NvrtcUnavailableError, match="Linux only"):
        Nvrtc.load()


@pytest.mark.parametrize(
    "name,major",
    [("libnvrtc.so.13", 13), ("libnvrtc.so.12.8.93", 12), ("libnvrtc.so", None)],
)
def test_soname_major(name: str, major: int | None) -> None:
    assert _nvrtc._soname_major(name) == major


# =============================================================================
# Target selection
# =============================================================================


def test_known_arch_gets_sass() -> None:
    target = select_target(86, [75, 80, 86, 120])
    assert target == CompileTarget("cubin", 86)
    assert target.gpu_architecture == "sm_86"


def test_unknown_newer_arch_gets_ptx_for_highest_lower_arch() -> None:
    # A GPU newer than the NVRTC: PTX for the best arch it can JIT forward from.
    target = select_target(122, [75, 86, 90, 120, 121])
    assert target == CompileTarget("ptx", 121)
    assert target.gpu_architecture == "compute_121"
    # Between known archs: never pick a higher arch (PTX does not JIT backward).
    assert select_target(87, [75, 80, 86, 90]) == CompileTarget("ptx", 86)


def test_arch_older_than_everything_is_unavailable() -> None:
    with pytest.raises(NvrtcUnavailableError, match="sm_70"):
        select_target(70, [75, 80])


# =============================================================================
# Cache location and keying
# =============================================================================


def test_cache_root_respects_xdg(tmp_path: Path) -> None:
    assert cache_root({"XDG_CACHE_HOME": str(tmp_path)}) == tmp_path / "luxar" / "cuda"


def test_cache_root_ignores_relative_or_empty_xdg() -> None:
    default = Path.home() / ".cache" / "luxar" / "cuda"
    assert cache_root({"XDG_CACHE_HOME": "relative/dir"}) == default
    assert cache_root({}) == default


def test_cache_paths_layout(tmp_path: Path) -> None:
    binary, sidecar = cache_paths(tmp_path, (13, 0), 120, "abc", "cubin")
    assert binary == tmp_path / "nvrtc-13.0" / "sm_120" / "abc.cubin"
    assert sidecar == tmp_path / "nvrtc-13.0" / "sm_120" / "abc.json"


def _key(include_dir: Path, **overrides: object) -> str:
    args: dict[str, object] = dict(
        source='#include "k.cuh"\n',
        include_dir=include_dir,
        options=("-std=c++17",),
        name_expressions=("k<1>",),
        nvrtc_version=(13, 0),
        target=CompileTarget("cubin", 86),
    )
    args.update(overrides)
    return cache_key(**args)  # type: ignore[arg-type]


def test_cache_key_tracks_every_input(tmp_path: Path) -> None:
    (tmp_path / "k.cuh").write_text("v1")
    base = _key(tmp_path)
    assert _key(tmp_path) == base  # deterministic

    variants = [
        _key(tmp_path, source="// other\n"),
        _key(tmp_path, options=("-std=c++20",)),
        _key(tmp_path, name_expressions=("k<2>",)),
        _key(tmp_path, nvrtc_version=(13, 1)),
        _key(tmp_path, target=CompileTarget("ptx", 86)),
        _key(tmp_path, target=CompileTarget("cubin", 120)),
    ]
    assert base not in variants
    assert len(set(variants)) == len(variants)

    # Editing ANY header invalidates, even one the source does not include.
    (tmp_path / "unrelated.cuh").write_text("x")
    with_extra = _key(tmp_path)
    assert with_extra != base
    (tmp_path / "k.cuh").write_text("v2")
    assert _key(tmp_path) not in (base, with_extra)


def test_shipped_kernel_sources_hash_stably() -> None:
    a = _key(SPLAT_SRC, source='#include "kernels_core.cuh"\n')
    b = _key(SPLAT_SRC, source='#include "kernels_core.cuh"\n')
    assert a == b and len(a) == 24


# =============================================================================
# Cache robustness
# =============================================================================


def _program() -> CompiledProgram:
    return CompiledProgram(
        binary=b"\x7fELF-fake-cubin",
        kind="cubin",
        lowered_names={"k<1>": "_Z1kILi1EEvv"},
    )


def test_cache_round_trip(tmp_path: Path) -> None:
    binary, sidecar = cache_paths(tmp_path, (13, 0), 86, "key", "cubin")
    assert write_cached_program(binary, sidecar, _program())
    loaded = read_cached_program(binary, sidecar)
    assert loaded is not None
    assert loaded.binary == _program().binary
    assert loaded.lowered_names == _program().lowered_names
    assert not list(binary.parent.glob("*.tmp"))  # atomic writes leave no temp files


def test_missing_cache_reads_as_none(tmp_path: Path) -> None:
    binary, sidecar = cache_paths(tmp_path, (13, 0), 86, "key", "cubin")
    assert read_cached_program(binary, sidecar) is None


@pytest.mark.parametrize(
    "damage",
    ["truncate_binary", "garbage_json", "wrong_schema", "bad_names", "wrong_kind"],
)
def test_corrupt_cache_reads_as_none(tmp_path: Path, damage: str) -> None:
    binary, sidecar = cache_paths(tmp_path, (13, 0), 86, "key", "cubin")
    write_cached_program(binary, sidecar, _program())
    meta = json.loads(sidecar.read_text())
    if damage == "truncate_binary":
        binary.write_bytes(binary.read_bytes()[:3])
    elif damage == "garbage_json":
        sidecar.write_text("{not json")
    elif damage == "wrong_schema":
        sidecar.write_text(json.dumps({**meta, "schema": -1}))
    elif damage == "bad_names":
        sidecar.write_text(json.dumps({**meta, "lowered_names": ["k"]}))
    else:
        sidecar.write_text(json.dumps({**meta, "kind": "ptx"}))
    assert read_cached_program(binary, sidecar) is None


def test_unwritable_cache_returns_false_without_raising(tmp_path: Path) -> None:
    blocker = tmp_path / "file-not-dir"
    blocker.write_text("")
    binary, sidecar = cache_paths(blocker, (13, 0), 86, "key", "cubin")
    assert write_cached_program(binary, sidecar, _program()) is False


# =============================================================================
# Runtime compile-or-cache flow (fake compiler; no GPU)
# =============================================================================


class _FakeNvrtc:
    version = (13, 0)
    supported_archs = (75, 86, 120)

    def __init__(self) -> None:
        self.compiles = 0

    def compile(self, source, program_name, options, name_expressions, target):  # noqa: ANN001, ANN201
        self.compiles += 1
        return CompiledProgram(
            binary=f"bin:{target.gpu_architecture}".encode(),
            kind=target.kind,
            lowered_names={n: f"mangled_{i}" for i, n in enumerate(name_expressions)},
        )


def _compiled(
    runtime: NvrtcRuntime, program: KernelProgram, names: list[str]
) -> CompiledProgram:
    target = select_target(86, runtime.nvrtc.supported_archs)
    key = cache_key(
        program.source,
        program.include_dir,
        program.options,
        names,
        runtime.nvrtc.version,
        target,
    )
    return runtime._compiled(program, list(program.options), names, target, 86, key)


@pytest.fixture
def program(tmp_path: Path) -> KernelProgram:
    inc = tmp_path / "inc"
    inc.mkdir()
    (inc / "k.cuh").write_text("__global__ void k() {}")
    return KernelProgram(label="test", source='#include "k.cuh"\n', include_dir=inc)


def test_compiles_once_then_serves_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, program: KernelProgram
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    fake = _FakeNvrtc()
    first = _compiled(NvrtcRuntime(nvrtc=fake, driver=None), program, ["k"])  # type: ignore[arg-type]
    assert fake.compiles == 1
    assert list(
        (tmp_path / "cache" / "luxar" / "cuda" / "nvrtc-13.0" / "sm_86").glob("*.cubin")
    )

    # A NEW runtime (a new process) reuses the disk cache: no compile.
    fake2 = _FakeNvrtc()
    second = _compiled(NvrtcRuntime(nvrtc=fake2, driver=None), program, ["k"])  # type: ignore[arg-type]
    assert fake2.compiles == 0
    assert second.binary == first.binary and second.lowered_names == first.lowered_names


def test_corrupt_disk_cache_recompiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, program: KernelProgram
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _compiled(NvrtcRuntime(nvrtc=_FakeNvrtc(), driver=None), program, ["k"])  # type: ignore[arg-type]
    for f in (tmp_path / "cache").rglob("*.cubin"):
        f.write_bytes(b"corrupt")
    fake = _FakeNvrtc()
    result = _compiled(NvrtcRuntime(nvrtc=fake, driver=None), program, ["k"])  # type: ignore[arg-type]
    assert fake.compiles == 1
    assert result.binary == b"bin:sm_86"


def test_unwritable_cache_degrades_to_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, program: KernelProgram
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("")  # XDG_CACHE_HOME is a FILE: nothing can be created under it
    monkeypatch.setenv("XDG_CACHE_HOME", str(blocker))
    runtime = NvrtcRuntime(nvrtc=_FakeNvrtc(), driver=None)  # type: ignore[arg-type]
    first = _compiled(runtime, program, ["k"])
    second = _compiled(runtime, program, ["k"])
    assert first.binary == second.binary == b"bin:sm_86"
    assert runtime._cache_warned


def test_header_edit_invalidates_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, program: KernelProgram
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _compiled(NvrtcRuntime(nvrtc=_FakeNvrtc(), driver=None), program, ["k"])  # type: ignore[arg-type]
    assert program.include_dir is not None
    (program.include_dir / "k.cuh").write_text("__global__ void k() { /* edited */ }")
    fake = _FakeNvrtc()
    _compiled(NvrtcRuntime(nvrtc=fake, driver=None), program, ["k"])  # type: ignore[arg-type]
    assert fake.compiles == 1


class _FailingNvrtc(_FakeNvrtc):
    def __init__(self, exc: BaseException) -> None:
        super().__init__()
        self.exc = exc

    def compile(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.compiles += 1
        raise self.exc


@pytest.mark.parametrize(
    "exc,expected",
    [
        (NvrtcCompileError("NVRTC failed", "error: boom"), NvrtcCompileError),
        (OSError("headers unreadable"), NvrtcUnavailableError),
        (
            AttributeError("libcuda.so.1: undefined symbol cuModuleLoadDataEx"),
            NvrtcUnavailableError,
        ),
    ],
    ids=["compile-error", "oserror", "missing-symbol"],
)
def test_build_failures_are_memoized_and_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    program: KernelProgram,
    exc: BaseException,
    expected: type,
) -> None:
    """A failing kernel fails ONCE per process (no recompile per render call),
    and stray failures surface as NvrtcUnavailableError so auto mode falls back."""
    import torch

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=None: (8, 6))
    fake = _FailingNvrtc(exc)
    runtime = NvrtcRuntime(nvrtc=fake, driver=None)  # type: ignore[arg-type]
    for _ in range(3):
        with pytest.raises(expected):
            runtime.kernels(program, ["k"], 0)
    assert fake.compiles == 1


def test_memoized_failure_does_not_pin_caller_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, program: KernelProgram
) -> None:
    """Each re-raise must be a fresh exception: re-raising one stored instance
    grows its traceback per call, keeping every caller's locals (in production,
    a GPU output volume) alive for the life of the process."""
    import gc
    import weakref

    import torch

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=None: (8, 6))

    class FreshFailingNvrtc(_FakeNvrtc):
        # Like the real wrapper, raise a NEW exception per compile (a shared
        # instance would itself pin the first caller's frame).
        def compile(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
            self.compiles += 1
            raise NvrtcCompileError("bad kernel", "log")

    runtime = NvrtcRuntime(nvrtc=FreshFailingNvrtc(), driver=None)  # type: ignore[arg-type]

    class Payload:
        pass

    refs: list[weakref.ref[Payload]] = []

    def caller() -> None:
        # Stands in for nvrtc_backend.forward: a frame holding a big tensor.
        payload = Payload()
        refs.append(weakref.ref(payload))
        runtime.kernels(program, ["k"], 0)

    for _ in range(3):
        with pytest.raises(NvrtcCompileError, match="bad kernel") as info:
            caller()
        assert info.value.log == "log"
        del info
    gc.collect()
    assert all(r() is None for r in refs)
    stored = next(iter(runtime._failed.values()))
    assert stored.__traceback__ is None and stored.__context__ is None


def test_no_home_directory_degrades_to_memory(
    monkeypatch: pytest.MonkeyPatch, program: KernelProgram
) -> None:
    def no_home() -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(no_home))
    runtime = NvrtcRuntime(nvrtc=_FakeNvrtc(), driver=None)  # type: ignore[arg-type]
    compiled = _compiled(runtime, program, ["k"])
    assert compiled.binary == b"bin:sm_86"
    assert runtime._cache_warned


def test_runtime_error_is_memoized(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def failing_load() -> Nvrtc:
        nonlocal calls
        calls += 1
        raise NvrtcUnavailableError("no libnvrtc here")

    _nvrtc.reset_runtime_for_tests()
    monkeypatch.setattr(Nvrtc, "load", staticmethod(failing_load))
    try:
        assert _nvrtc.nvrtc_unavailable_reason() == "no libnvrtc here"
        assert _nvrtc.nvrtc_unavailable_reason() == "no libnvrtc here"
        assert calls == 1
    finally:
        _nvrtc.reset_runtime_for_tests()


def test_compile_error_carries_log() -> None:
    err = NvrtcCompileError("NVRTC failed", "k.cu(3): error: identifier undefined")
    assert "identifier undefined" in str(err)
    assert err.log.startswith("k.cu(3)")


# =============================================================================
# Real NVRTC, no GPU: the shipped sources compile for every target arch
# =============================================================================

#: Architectures the AOT build targets by default (_cuda_build.py) + Spark's sm_121.
TARGET_ARCHS = [75, 80, 86, 89, 90, 100, 120, 121]


@pytest.fixture(scope="module")
def real_nvrtc() -> Nvrtc:
    try:
        return Nvrtc.load()
    except NvrtcUnavailableError as exc:
        pytest.skip(f"libnvrtc not present in this environment: {exc}")


def _compile_all(
    nvrtc: Nvrtc, program: KernelProgram, names: list[str], arch: int
) -> CompiledProgram:
    target = select_target(arch, nvrtc.supported_archs)
    options = [*program.options, f"-I{program.include_dir}"]
    return nvrtc.compile(program.source, f"{program.label}.cu", options, names, target)


@pytest.mark.parametrize(
    "arch",
    # ~3 s per arch: one representative in the fast loop, the rest under `slow`.
    [a if a == 86 else pytest.param(a, marks=pytest.mark.slow) for a in TARGET_ARCHS],
)
def test_splatting_kernels_compile_for_arch(real_nvrtc: Nvrtc, arch: int) -> None:
    """All 28 instantiations (DIM 2-8 x fp32/fp16 x fwd/bwd) compile for ``arch``.

    An arch this NVRTC does not know compiles to PTX via the same fallback the
    runtime uses, so every arch is exercised rather than skipped.
    """
    from luxar.gsplats.models.gsplats.cuda.nvrtc_backend import PROGRAM, kernel_names

    names = [n for dim in range(2, 9) for n in kernel_names(dim)]
    compiled = _compile_all(real_nvrtc, PROGRAM, names, arch)
    assert len(compiled.lowered_names) == 28
    assert all(m.startswith("_Z") for m in compiled.lowered_names.values())
    if arch in real_nvrtc.supported_archs:
        assert compiled.kind == "cubin" and compiled.binary[:4] == b"\x7fELF"
    else:
        assert compiled.kind == "ptx" and b".entry" in compiled.binary


def test_nlm_kernels_compile(real_nvrtc: Nvrtc) -> None:
    from luxar.gsplats.preprocessing.cuda.nlm_nvrtc import (
        PROGRAM,
        SUPPORTED_PATCH_HALF,
        SUPPORTED_SEARCH_DIST,
    )

    names = [
        f"nlm_{nd}d_kernel<{ph}, {sd}>"
        for nd in (2, 3)
        for ph in SUPPORTED_PATCH_HALF
        for sd in SUPPORTED_SEARCH_DIST
    ]
    compiled = _compile_all(real_nvrtc, PROGRAM, names, 86)
    assert len(compiled.lowered_names) == 24


def test_decoy_header_in_cwd_is_not_compiled(
    real_nvrtc: Nvrtc, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The top-level include resolves through -I only, never the CWD.

    A same-named header in the working directory would otherwise be compiled
    -- and cached under the genuine source's key.
    """
    from luxar.gsplats.models.gsplats.cuda.nvrtc_backend import PROGRAM, kernel_names

    (tmp_path / "kernels_core.cuh").write_text('#error "decoy header from the CWD"\n')
    monkeypatch.chdir(tmp_path)
    compiled = _compile_all(real_nvrtc, PROGRAM, kernel_names(2), 86)
    assert len(compiled.lowered_names) == 4


def test_broken_source_reports_nvrtc_log(real_nvrtc: Nvrtc) -> None:
    program = KernelProgram(
        label="broken", source="__global__ void k() { undefined_symbol(); }\n"
    )
    with pytest.raises(NvrtcCompileError, match="undefined_symbol") as info:
        _compile_all(real_nvrtc, program, ["k"], 86)
    assert "undefined_symbol" in info.value.log


def test_nvrtc_reports_version_and_archs(real_nvrtc: Nvrtc) -> None:
    assert real_nvrtc.version[0] >= 11
    assert real_nvrtc.supported_archs
    assert isinstance(real_nvrtc.lib, ctypes.CDLL)
