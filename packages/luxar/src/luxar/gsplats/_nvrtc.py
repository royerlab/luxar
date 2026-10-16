"""Runtime CUDA kernel compilation through NVRTC and the CUDA driver API.

This is the machinery behind Luxar's "no build step" CUDA kernels. A plain
``pip install "luxar[gsplats]"`` on Linux with an NVIDIA GPU has no ``nvcc``
and often no C++ compiler, so the ahead-of-time torch extension cannot be
built. It does have:

- ``libnvrtc`` -- the CUDA runtime compiler, shipped inside the NVIDIA wheels
  that a CUDA build of torch depends on (``nvidia/cu13/lib`` for CUDA 13,
  ``nvidia/cuda_nvrtc/lib`` for CUDA 12);
- ``libcuda`` -- the driver itself.

That is enough to compile the shipped ``.cuh`` kernel sources to SASS for the
exact GPU in the machine, load them into torch's primary context, and launch
them on torch's current stream with raw ``data_ptr()`` pointers. Everything
here is ``ctypes``: no new dependency, and no torch C++ ABI to match.

Compiled binaries are cached on disk under
``$XDG_CACHE_HOME/luxar/cuda/nvrtc-<version>/sm_<arch>/<key>.cubin`` (default
``~/.cache``). The key hashes the program text, every header in the include
directory, the options and the NVRTC version, so editing a kernel invalidates
it. A corrupt or unwritable cache only costs a recompile -- it never fails a
run.

When NVRTC does not know the device's architecture (a GPU newer than the
NVRTC in the environment) the program is compiled to PTX for the highest
architecture NVRTC does know that the device can run, and the driver JITs it.

Linux only: the wheel layouts and library names searched are the Linux ones.
"""

from __future__ import annotations

import copy
import ctypes
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from arbol import aprint

#: Explicit path to a ``libnvrtc`` shared library; wins over every search.
NVRTC_LIBRARY_ENV = "LUXAR_NVRTC_LIBRARY"

#: Bump when the cache layout or the sidecar schema changes.
_CACHE_SCHEMA = 1

# CUDA driver enums used below (cuda.h).
_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
_CU_JIT_ERROR_LOG_BUFFER = 5
_CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6

#: Dynamic shared memory a kernel may use without opting in.
DEFAULT_MAX_DYNAMIC_SHARED_BYTES = 48 * 1024


class NvrtcUnavailableError(RuntimeError):
    """The NVRTC CUDA path cannot run here (library missing, unsupported GPU)."""


class NvrtcCompileError(RuntimeError):
    """NVRTC rejected a CUDA program; ``log`` holds the compiler output."""

    def __init__(self, message: str, log: str = "") -> None:
        super().__init__(f"{message}\n{log}" if log else message)
        self.log = log


class CudaDriverError(RuntimeError):
    """A CUDA driver API call returned an error."""


# =============================================================================
# Library location
# =============================================================================


def _nvidia_wheel_roots() -> list[Path]:
    """Directories of the ``nvidia`` namespace package (the CUDA wheels)."""
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    return [Path(p) for p in spec.submodule_search_locations]


def _torch_cuda_major() -> int | None:
    """CUDA major version torch was built against, or None (CPU torch / no torch)."""
    try:
        import torch
    except ImportError:
        return None
    version = getattr(torch.version, "cuda", None)
    if not version:
        return None
    try:
        return int(str(version).split(".")[0])
    except ValueError:
        return None


def _shortest_first(paths: Sequence[Path]) -> list[Path]:
    """Prefer ``libnvrtc.so.13`` over ``libnvrtc.so.13.0.88`` (same library)."""
    return sorted(paths, key=lambda p: (len(p.name), p.name))


def _torch_lib_dir() -> Path | None:
    """``torch/lib`` without importing torch (some builds bundle libnvrtc there)."""
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations))) / "lib"


def candidate_nvrtc_libraries(
    cuda_major: int | None = None,
    wheel_roots: Sequence[Path] | None = None,
    environ: dict[str, str] | None = None,
    torch_lib_dir: Path | None = None,
) -> list[str]:
    """Ordered ``libnvrtc`` candidates: override, wheels, conda/torch, system.

    Wheel candidates matching ``cuda_major`` (torch's CUDA) come first, since
    that NVRTC is the one torch's own wheels were validated against. Bare
    sonames (no ``/``) are resolved by the dynamic loader's normal search
    (``LD_LIBRARY_PATH``, ``ldconfig``).
    """
    env = os.environ if environ is None else environ
    roots = _nvidia_wheel_roots() if wheel_roots is None else list(wheel_roots)
    out: list[str] = []

    override = env.get(NVRTC_LIBRARY_ENV)
    if override:
        out.append(override)

    wheel_hits: list[tuple[int, Path]] = []
    for root in roots:
        # CUDA 13 wheels: nvidia/cu13/lib; CUDA 12 wheels: nvidia/cuda_nvrtc/lib.
        for lib_dir in sorted(root.glob("cu*/lib")) + [root / "cuda_nvrtc" / "lib"]:
            for path in _shortest_first(list(lib_dir.glob("libnvrtc.so*"))):
                major = _soname_major(path.name)
                rank = 0 if (cuda_major is not None and major == cuda_major) else 1
                wheel_hits.append((rank, path))
    out.extend(str(p) for _, p in sorted(wheel_hits, key=lambda t: t[0]))

    # Next to the wheels: a conda env's own CUDA libs, and torch's bundled libs.
    extra_dirs: list[Path] = []
    conda = env.get("CONDA_PREFIX")
    if conda:
        extra_dirs.append(Path(conda) / "lib")
    torch_lib = _torch_lib_dir() if torch_lib_dir is None else torch_lib_dir
    if torch_lib is not None:
        extra_dirs.append(torch_lib)
    for lib_dir in extra_dirs:
        hits = _shortest_first(list(lib_dir.glob("libnvrtc.so*")))
        hits.sort(key=lambda p: 0 if _soname_major(p.name) == cuda_major else 1)
        out.extend(str(p) for p in hits)

    majors = [cuda_major] if cuda_major is not None else []
    majors += [m for m in (13, 12, 11) if m not in majors]
    out.extend(f"libnvrtc.so.{m}" for m in majors)
    out.append("libnvrtc.so")

    for var in ("CUDA_HOME", "CUDA_PATH"):
        home = env.get(var)
        if home:
            out.extend(
                str(p)
                for p in _shortest_first(list(Path(home, "lib64").glob("libnvrtc.so*")))
            )
    out.extend(
        str(p)
        for p in _shortest_first(
            list(Path("/usr/local/cuda/lib64").glob("libnvrtc.so*"))
        )
    )

    return list(dict.fromkeys(out))  # de-duplicate, keeping first occurrence


def _soname_major(name: str) -> int | None:
    """``libnvrtc.so.13.0.88`` -> 13; ``libnvrtc.so`` -> None."""
    parts = name.split(".so.")
    if len(parts) != 2:
        return None
    try:
        return int(parts[1].split(".")[0])
    except ValueError:
        return None


def _load_first(candidates: Sequence[str]) -> tuple[ctypes.CDLL, str]:
    errors = []
    for cand in candidates:
        try:
            return ctypes.CDLL(cand), cand
        except OSError as exc:
            errors.append(f"{cand}: {exc}")
    raise NvrtcUnavailableError(
        "CUDA runtime compiler (libnvrtc) not found. It ships with CUDA builds of "
        "torch on Linux (the nvidia-cuda-nvrtc wheel); set "
        f"{NVRTC_LIBRARY_ENV}=/path/to/libnvrtc.so to point at one explicitly. "
        f"Tried: {'; '.join(errors[-6:]) or 'nothing'}"
    )


# =============================================================================
# Compile target and cache keying (pure, testable without a GPU)
# =============================================================================


@dataclass(frozen=True)
class CompileTarget:
    """What NVRTC emits: SASS for an exact arch, or PTX for the driver to JIT."""

    kind: str  # "cubin" or "ptx"
    arch: int  # e.g. 120 for sm_120 / compute_120

    @property
    def gpu_architecture(self) -> str:
        prefix = "sm" if self.kind == "cubin" else "compute"
        return f"{prefix}_{self.arch}"


def select_target(device_arch: int, supported_archs: Sequence[int]) -> CompileTarget:
    """SASS when NVRTC knows ``device_arch``; else PTX for the best lower arch.

    PTX JIT-compiles forward onto any newer GPU, never backward, so the PTX
    target is the highest supported arch that does not exceed the device.
    """
    if device_arch in supported_archs:
        return CompileTarget("cubin", device_arch)
    lower = [a for a in supported_archs if a <= device_arch]
    if not lower:
        known = ", ".join(f"sm_{a}" for a in sorted(supported_archs)) or "none"
        raise NvrtcUnavailableError(
            f"GPU architecture sm_{device_arch} is older than every architecture this "
            f"NVRTC can target ({known}); CUDA kernels cannot be compiled for it"
        )
    return CompileTarget("ptx", max(lower))


def cache_root(environ: dict[str, str] | None = None) -> Path:
    """``$XDG_CACHE_HOME/luxar/cuda`` (``~/.cache`` when unset or relative)."""
    env = os.environ if environ is None else environ
    xdg = env.get("XDG_CACHE_HOME", "")
    base = Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".cache"
    return base / "luxar" / "cuda"


def cache_key(
    source: str,
    include_dir: Path | None,
    options: Sequence[str],
    name_expressions: Sequence[str],
    nvrtc_version: tuple[int, int],
    target: CompileTarget,
) -> str:
    """Content hash of everything that determines the compiled binary.

    Every header in ``include_dir`` is hashed (not only the ones the program
    happens to include), so the key can only be conservative.
    """
    h = hashlib.sha256()

    def feed(label: str, data: bytes) -> None:
        h.update(label.encode())
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)

    feed("schema", str(_CACHE_SCHEMA).encode())
    feed("source", source.encode())
    if include_dir is not None:
        for header in sorted(include_dir.glob("*.cuh")) + sorted(
            include_dir.glob("*.h")
        ):
            feed("header:" + header.name, header.read_bytes())
    feed("options", "\0".join(options).encode())
    feed("names", "\0".join(name_expressions).encode())
    feed("nvrtc", f"{nvrtc_version[0]}.{nvrtc_version[1]}".encode())
    feed("target", f"{target.kind}:{target.arch}".encode())
    return h.hexdigest()[:24]


def cache_paths(
    root: Path, nvrtc_version: tuple[int, int], device_arch: int, key: str, kind: str
) -> tuple[Path, Path]:
    """(binary, sidecar) paths for one cached program."""
    folder = root / f"nvrtc-{nvrtc_version[0]}.{nvrtc_version[1]}" / f"sm_{device_arch}"
    return folder / f"{key}.{kind}", folder / f"{key}.json"


@dataclass
class CompiledProgram:
    """A compiled program plus the mangled names of its kernels."""

    binary: bytes
    kind: str  # "cubin" or "ptx"
    lowered_names: dict[str, str]
    log: str = ""


def read_cached_program(
    binary_path: Path, sidecar_path: Path
) -> CompiledProgram | None:
    """Return the cached program, or None when absent, stale, or corrupt."""
    try:
        meta = json.loads(sidecar_path.read_text())
        data = binary_path.read_bytes()
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or meta.get("schema") != _CACHE_SCHEMA:
        return None
    names = meta.get("lowered_names")
    if (
        not isinstance(names, dict)
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in names.items())
        or meta.get("sha256") != hashlib.sha256(data).hexdigest()
        or meta.get("kind") != binary_path.suffix.lstrip(".")
    ):
        return None
    return CompiledProgram(binary=data, kind=str(meta["kind"]), lowered_names=names)


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_cached_program(
    binary_path: Path, sidecar_path: Path, program: CompiledProgram
) -> bool:
    """Persist ``program``; returns False (never raises) when the cache is unusable.

    The binary is written first and the sidecar -- which carries the binary's
    digest -- last, each atomically, so a reader never trusts a half-written
    pair.
    """
    meta = {
        "schema": _CACHE_SCHEMA,
        "kind": program.kind,
        "lowered_names": program.lowered_names,
        "sha256": hashlib.sha256(program.binary).hexdigest(),
    }
    try:
        binary_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(binary_path, program.binary)
        _atomic_write(sidecar_path, json.dumps(meta, indent=1).encode())
    except OSError:
        return False
    return True


# =============================================================================
# NVRTC
# =============================================================================


class Nvrtc:
    """Thin ctypes wrapper over the NVRTC C API."""

    def __init__(self, lib: ctypes.CDLL, path: str) -> None:
        self.lib = lib
        self.path = path
        lib.nvrtcGetErrorString.restype = ctypes.c_char_p
        major, minor = ctypes.c_int(), ctypes.c_int()
        self._check(
            lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)), "nvrtcVersion"
        )
        self.version = (major.value, minor.value)
        count = ctypes.c_int()
        self._check(
            lib.nvrtcGetNumSupportedArchs(ctypes.byref(count)),
            "nvrtcGetNumSupportedArchs",
        )
        archs = (ctypes.c_int * max(count.value, 1))()
        self._check(lib.nvrtcGetSupportedArchs(archs), "nvrtcGetSupportedArchs")
        self.supported_archs = tuple(archs[i] for i in range(count.value))

    @classmethod
    def load(cls) -> Nvrtc:
        platform: str = sys.platform  # a plain str: no per-platform mypy narrowing
        if not platform.startswith("linux"):
            raise NvrtcUnavailableError(
                f"runtime-compiled CUDA kernels are supported on Linux only (this is {platform})"
            )
        lib, path = _load_first(candidate_nvrtc_libraries(_torch_cuda_major()))
        return cls(lib, path)

    def _check(self, result: int, what: str) -> None:
        if result != 0:
            msg = self.lib.nvrtcGetErrorString(result)
            text = msg.decode(errors="replace") if msg else f"error {result}"
            raise NvrtcCompileError(
                f"NVRTC (CUDA runtime compiler) call {what} failed: {text}"
            )

    def compile(
        self,
        source: str,
        program_name: str,
        options: Sequence[str],
        name_expressions: Sequence[str],
        target: CompileTarget,
    ) -> CompiledProgram:
        """Compile ``source``; raises :class:`NvrtcCompileError` with the log."""
        lib = self.lib
        prog = ctypes.c_void_p()
        self._check(
            lib.nvrtcCreateProgram(
                ctypes.byref(prog),
                source.encode(),
                program_name.encode(),
                0,
                None,
                None,
            ),
            "nvrtcCreateProgram",
        )
        try:
            for name in name_expressions:
                self._check(
                    lib.nvrtcAddNameExpression(prog, name.encode()),
                    "nvrtcAddNameExpression",
                )
            opts = [f"--gpu-architecture={target.gpu_architecture}", *options]
            c_opts = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
            result = lib.nvrtcCompileProgram(prog, len(opts), c_opts)
            log = self._program_log(prog)
            if result != 0:
                err = lib.nvrtcGetErrorString(result)
                raise NvrtcCompileError(
                    f"NVRTC failed to compile CUDA program {program_name} for "
                    f"{target.gpu_architecture}: {err.decode(errors='replace') if err else result}",
                    log,
                )
            lowered: dict[str, str] = {}
            for name in name_expressions:
                ptr = ctypes.c_char_p()
                self._check(
                    lib.nvrtcGetLoweredName(prog, name.encode(), ctypes.byref(ptr)),
                    "nvrtcGetLoweredName",
                )
                lowered[name] = (ptr.value or b"").decode()
            size = ctypes.c_size_t()
            if target.kind == "cubin":
                self._check(
                    lib.nvrtcGetCUBINSize(prog, ctypes.byref(size)), "nvrtcGetCUBINSize"
                )
                buf = ctypes.create_string_buffer(size.value)
                self._check(lib.nvrtcGetCUBIN(prog, buf), "nvrtcGetCUBIN")
                binary = buf.raw
            else:
                self._check(
                    lib.nvrtcGetPTXSize(prog, ctypes.byref(size)), "nvrtcGetPTXSize"
                )
                buf = ctypes.create_string_buffer(size.value)
                self._check(lib.nvrtcGetPTX(prog, buf), "nvrtcGetPTX")
                binary = buf.raw  # NUL-terminated, as cuModuleLoadData expects
            return CompiledProgram(
                binary=binary, kind=target.kind, lowered_names=lowered, log=log
            )
        finally:
            lib.nvrtcDestroyProgram(ctypes.byref(prog))

    def _program_log(self, prog: ctypes.c_void_p) -> str:
        size = ctypes.c_size_t()
        if (
            self.lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size)) != 0
            or size.value <= 1
        ):
            return ""
        buf = ctypes.create_string_buffer(size.value)
        self.lib.nvrtcGetProgramLog(prog, buf)
        return buf.value.decode(errors="replace").strip()


# =============================================================================
# CUDA driver
# =============================================================================


class CudaDriver:
    """Thin ctypes wrapper over the CUDA driver API (``libcuda``)."""

    def __init__(self) -> None:
        try:
            self.lib = ctypes.CDLL("libcuda.so.1")
        except OSError as exc:
            raise NvrtcUnavailableError(
                f"CUDA driver library libcuda.so.1 not loadable: {exc}"
            ) from exc
        self._contexts: dict[int, ctypes.c_void_p] = {}
        self.check(self.lib.cuInit(0), "cuInit")

    def check(self, result: int, what: str) -> None:
        if result == 0:
            return
        name, text = ctypes.c_char_p(), ctypes.c_char_p()
        self.lib.cuGetErrorName(result, ctypes.byref(name))
        self.lib.cuGetErrorString(result, ctypes.byref(text))
        raise CudaDriverError(
            f"CUDA driver call {what} failed: "
            f"{(name.value or b'?').decode()} ({result}): {(text.value or b'').decode()}"
        )

    def primary_context(self, device_index: int) -> ctypes.c_void_p:
        """Retain (once) the device's primary context -- the one torch uses."""
        ctx = self._contexts.get(device_index)
        if ctx is None:
            dev = ctypes.c_int()
            self.check(
                self.lib.cuDeviceGet(ctypes.byref(dev), device_index), "cuDeviceGet"
            )
            ctx = ctypes.c_void_p()
            self.check(
                self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev),
                "cuDevicePrimaryCtxRetain",
            )
            self._contexts[device_index] = ctx
        return ctx

    def push(self, ctx: ctypes.c_void_p) -> None:
        self.check(self.lib.cuCtxPushCurrent_v2(ctx), "cuCtxPushCurrent")

    def pop(self) -> None:
        popped = ctypes.c_void_p()
        self.check(self.lib.cuCtxPopCurrent_v2(ctypes.byref(popped)), "cuCtxPopCurrent")

    def load_module(
        self, device_index: int, program: CompiledProgram
    ) -> ctypes.c_void_p:
        """Load ``program`` into the device's primary context (JIT for PTX)."""
        ctx = self.primary_context(device_index)
        log_size = 16384
        log = ctypes.create_string_buffer(log_size)
        opt_keys = (ctypes.c_int * 2)(
            _CU_JIT_ERROR_LOG_BUFFER, _CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
        )
        opt_vals = (ctypes.c_void_p * 2)(
            ctypes.cast(log, ctypes.c_void_p), ctypes.c_void_p(log_size)
        )
        module = ctypes.c_void_p()
        self.push(ctx)
        try:
            result = self.lib.cuModuleLoadDataEx(
                ctypes.byref(module),
                ctypes.c_char_p(program.binary),
                2,
                opt_keys,
                opt_vals,
            )
        finally:
            self.pop()
        if result != 0:
            jit_log = log.value.decode(errors="replace").strip()
            try:
                self.check(result, f"cuModuleLoadDataEx ({program.kind})")
            except CudaDriverError as exc:
                raise CudaDriverError(
                    f"{exc}\n{jit_log}" if jit_log else str(exc)
                ) from None
        return module

    def get_function(
        self, module: ctypes.c_void_p, lowered_name: str
    ) -> ctypes.c_void_p:
        func = ctypes.c_void_p()
        self.check(
            self.lib.cuModuleGetFunction(
                ctypes.byref(func), module, lowered_name.encode()
            ),
            f"cuModuleGetFunction({lowered_name})",
        )
        return func


@dataclass
class Kernel:
    """One loaded ``__global__`` function, launchable on a torch stream."""

    driver: CudaDriver
    function: ctypes.c_void_p
    context: ctypes.c_void_p
    name: str
    _max_dynamic_shared: int = DEFAULT_MAX_DYNAMIC_SHARED_BYTES

    def ensure_dynamic_shared(self, nbytes: int) -> None:
        """Opt in to more than 48 KB of dynamic shared memory when needed."""
        if nbytes <= self._max_dynamic_shared:
            return
        self.driver.check(
            self.driver.lib.cuFuncSetAttribute(
                self.function, _CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, nbytes
            ),
            f"cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES={nbytes}) for {self.name}",
        )
        self._max_dynamic_shared = nbytes

    def launch(
        self,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: Sequence[Any],
        stream: int,
        shared_bytes: int = 0,
    ) -> None:
        """Launch with ``args`` (ctypes scalars, in kernel-parameter order)."""
        params = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
        drv = self.driver
        drv.push(self.context)
        try:
            self.ensure_dynamic_shared(shared_bytes)
            result = drv.lib.cuLaunchKernel(
                self.function,
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                shared_bytes,
                ctypes.c_void_p(stream),
                params,
                None,
            )  # fmt: skip
        finally:
            drv.pop()
        drv.check(result, f"cuLaunchKernel({self.name})")


# =============================================================================
# Runtime (process-wide) and program building
# =============================================================================


@dataclass(frozen=True)
class KernelProgram:
    """A CUDA program to runtime-compile: source text plus its include dir."""

    label: str  # human name for logs / program name, e.g. "gsplat-splatting"
    source: str
    include_dir: Path | None = None
    options: tuple[str, ...] = ("-std=c++17", "--use_fast_math")


@dataclass
class NvrtcRuntime:
    """Process-wide NVRTC + driver handles, compiled-program memo, and cache."""

    nvrtc: Nvrtc
    driver: CudaDriver
    _modules: dict[tuple[str, int], dict[str, Kernel]] = field(default_factory=dict)
    _loaded: dict[tuple[KernelProgram, tuple[str, ...], int], dict[str, Kernel]] = (
        field(default_factory=dict)
    )
    _failed: dict[tuple[KernelProgram, tuple[str, ...], int], RuntimeError] = field(
        default_factory=dict
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _cache_warned: bool = False

    def kernels(
        self, program: KernelProgram, name_expressions: Sequence[str], device_index: int
    ) -> dict[str, Kernel]:
        """Compile (or load from cache) and load ``name_expressions`` for a device.

        Called on every kernel launch, so the already-loaded case is a single
        dict lookup: the content hash (which reads every header from disk) is
        computed once per (program, names, device) per process.
        """
        fast_key = (program, tuple(name_expressions), device_index)
        loaded = self._loaded.get(fast_key)
        if loaded is not None:
            return loaded
        failure = self._failed.get(fast_key)
        if failure is not None:
            # A kernel that failed to build fails the same way next call; render
            # and tile loops must not pay a recompile per call to find that out.
            # Raise a fresh copy: re-raising the stored instance would grow its
            # __traceback__ every call, pinning each caller's frames (and their
            # GPU output tensors) for the life of the process.
            raise copy.copy(failure)
        with self._lock:
            loaded = self._loaded.get(fast_key)
            if loaded is not None:
                return loaded
            try:
                loaded = self._build(program, name_expressions, device_index)
            except (NvrtcUnavailableError, NvrtcCompileError, CudaDriverError) as exc:
                self._failed[fast_key] = copy.copy(exc)  # no traceback/context
                raise
            except (OSError, AttributeError, RuntimeError) as exc:
                # Unreadable headers, a driver missing an entry point, torch
                # failing to query the device: all mean "this path cannot run
                # here", which auto mode must treat as a fallback, not a crash.
                err = NvrtcUnavailableError(
                    f"CUDA kernels '{program.label}' could not be built: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._failed[fast_key] = copy.copy(err)
                raise err from exc
            self._loaded[fast_key] = loaded
        return loaded

    def _build(
        self, program: KernelProgram, name_expressions: Sequence[str], device_index: int
    ) -> dict[str, Kernel]:
        """Compile (or read from the disk cache) and load into the device context."""
        import torch

        major, minor = torch.cuda.get_device_capability(device_index)
        device_arch = major * 10 + minor
        target = select_target(device_arch, self.nvrtc.supported_archs)
        options = list(program.options)
        if program.include_dir is not None:
            options.append(f"-I{program.include_dir}")
        key = cache_key(
            program.source,
            program.include_dir,
            program.options,
            name_expressions,
            self.nvrtc.version,
            target,
        )
        memo_key = (key, device_index)
        loaded = self._modules.get(memo_key)
        if loaded is None:
            compiled = self._compiled(
                program, options, name_expressions, target, device_arch, key
            )
            loaded = self._load(compiled, device_index)
            self._modules[memo_key] = loaded
        return loaded

    def _compiled(
        self,
        program: KernelProgram,
        options: list[str],
        name_expressions: Sequence[str],
        target: CompileTarget,
        device_arch: int,
        key: str,
    ) -> CompiledProgram:
        try:
            paths: tuple[Path, Path] | None = cache_paths(
                cache_root(), self.nvrtc.version, device_arch, key, target.kind
            )
        except RuntimeError:  # Path.home() with no resolvable home: memory only
            paths = None
        if paths is not None:
            cached = read_cached_program(*paths)
            if cached is not None and set(cached.lowered_names) == set(
                name_expressions
            ):
                return cached
        t0 = time.perf_counter()
        compiled = self.nvrtc.compile(
            program.source, f"{program.label}.cu", options, name_expressions, target
        )
        elapsed = time.perf_counter() - t0
        how = "SASS" if target.kind == "cubin" else "PTX (driver JIT)"
        aprint(
            f"Compiled CUDA kernels '{program.label}' with NVRTC "
            f"{self.nvrtc.version[0]}.{self.nvrtc.version[1]} for "
            f"{target.gpu_architecture} {how} in {elapsed:.2f}s (one-time; cached)"
        )
        written = paths is not None and write_cached_program(*paths, compiled)
        if not written and not self._cache_warned:
            self._cache_warned = True
            where = paths[0].parent if paths is not None else "(no home directory)"
            aprint(
                f"WARNING: CUDA kernel cache {where} is not writable; "
                "kernels stay in memory and will be recompiled in the next process"
            )
        return compiled

    def _load(self, compiled: CompiledProgram, device_index: int) -> dict[str, Kernel]:
        drv = self.driver
        module = drv.load_module(device_index, compiled)
        ctx = drv.primary_context(device_index)
        return {
            name: Kernel(drv, drv.get_function(module, lowered), ctx, name)
            for name, lowered in compiled.lowered_names.items()
        }


_runtime: NvrtcRuntime | None = None
_runtime_error: NvrtcUnavailableError | None = None
_runtime_lock = threading.Lock()


def _create_runtime() -> NvrtcRuntime:
    try:
        return NvrtcRuntime(nvrtc=Nvrtc.load(), driver=CudaDriver())
    except (NvrtcCompileError, CudaDriverError, AttributeError) as exc:
        # AttributeError: a library missing an entry point (too-old NVRTC/driver).
        raise NvrtcUnavailableError(str(exc)) from exc


def get_runtime() -> NvrtcRuntime:
    """The process-wide runtime; raises :class:`NvrtcUnavailableError` (memoized)."""
    global _runtime, _runtime_error  # noqa: PLW0603
    runtime = _runtime
    if runtime is not None:  # hot path: every kernel launch
        return runtime
    with _runtime_lock:
        if _runtime_error is not None:
            raise _runtime_error
        if _runtime is None:
            try:
                _runtime = _create_runtime()
            except NvrtcUnavailableError as exc:
                _runtime_error = exc
                raise
        return _runtime


def reset_runtime_for_tests() -> None:
    """Forget the memoized runtime (tests only)."""
    global _runtime, _runtime_error  # noqa: PLW0603
    with _runtime_lock:
        _runtime = None
        _runtime_error = None


def nvrtc_unavailable_reason() -> str | None:
    """None when the NVRTC path can run, else a one-line reason."""
    try:
        get_runtime()
    except NvrtcUnavailableError as exc:
        return str(exc)
    return None
