# Metal-Accelerated Gaussian Splatting

High-performance Apple Silicon backend for Luxar Gaussian splatting.  The custom
native path currently targets the production-critical case:

```text
3D volumes + MPS tensors + float32 parameters
```

`GaussianSplatModelMetal` still exposes the same parameter-management API as the
base/CUDA models and can be constructed for 2D-8D MPS shapes; non-3D shapes use
Luxar's PyTorch renderer.

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- PyTorch with MPS support (`torch.backends.mps.is_available() == True`),
  **torch >= 2.7** for the runtime-compiled kernels (`torch.mps.compile_shader`
  with `arg_casts`). torch 2.6 works through the private
  `torch.mps._compile_shader`, which has the same kernel call API including
  `arg_casts` (not exercised here: the measurements below are torch 2.14);
  torch <= 2.5 has no shader compiler and falls back to PyTorch.
- **No Xcode.** `pip install "luxar[gsplats]"` is enough.

## Kernel backends

The same `src/kernels.metal` runs through one of three paths, selected once per
process by `backend.py` and logged once:

| `LUXAR_METAL_BACKEND` | What runs | Needs |
| --- | --- | --- |
| `auto` (default) | `aot` if an up-to-date extension imports, else `runtime`, else `torch` (a pip install has no AOT build, so it gets `runtime`) | — |
| `runtime` | `runtime_backend.py`: `kernels.metal` compiled on first use by `torch.mps.compile_shader` (macOS's built-in Metal compiler; the OS caches the binary) | torch >= 2.7 (2.6 private API) |
| `aot` | the ahead-of-time extension `metal_splatting_backend*.so` + `src/default.metallib` | a developer build with full Xcode |
| `torch` | no custom kernels: Luxar's PyTorch renderer on MPS | — |

A forced `aot` / `runtime` that cannot run **raises** (also out of
`fit_gaussian_splats`) instead of silently degrading, so a benchmark never
reports the wrong backend's numbers. In `auto`, a fit on MPS whose kernels are
unavailable warns loudly with the reason and continues on PyTorch (orders of
magnitude slower; see the table below). Importing the package never compiles or
builds anything.

The runtime backend is a drop-in for the extension: `forward_splat_3d`,
`backward_splat_3d` and `compute_conic_metal` take the same arguments, perform
the same validation (ported from `bindings.mm`), and return the same shapes.
Kernels are enqueued on torch's own MPS stream, so there is no per-call
`synchronize()` + `waitUntilCompleted` as in the extension. Two binding details:
a Python int or list binds as **int64**, which `constant uint3& shape_dhw`
would read as `(D, 0, H)`, so every integer is bound as 32-bit; and the
splat-centric dispatch (`dispatchThreadgroups(N, 64)`) is
`threads=N * 64, group_size=64`.

### Optional AOT build (developers only)

```bash
cd packages/luxar/src/luxar/gsplats/models/gsplats/metal
python setup.py build_ext --inplace --force   # needs full Xcode (xcrun metal)
```

The build compiles `src/kernels.metal` -> `src/default.metallib` and
`src/bindings.mm` -> `metal_splatting_backend*.so` (both gitignored, never in the
wheel). It is never triggered automatically. A build older than its sources is
skipped in `auto` (the runtime backend then compiles the current source) and
rejected under `LUXAR_METAL_BACKEND=aot`. `setup.py` always forces the extension
rebuild (`--force` is belt and braces), because setuptools alone compares only
`bindings.mm` and would leave an extension "stale" after a `setup.py` edit.

When built and current, the AOT extension is **preferred in `auto`** (mirroring
the CUDA backend's order). The runtime path is numerically equivalent and equal
or slightly faster (table below); `LUXAR_METAL_BACKEND=runtime` selects it even
with an AOT build present.

## Architecture

```text
Python
  GaussianSplatModelMetal
  MetalSplatFunction
  L factors converted to packed conics once per forward, then handed to the
  native kernels

Kernel backend (backend.py picks one; same three entry points)
  runtime_backend.py  -> torch.mps.compile_shader(kernels.metal)
  metal_splatting_backend (optional AOT C++/Objective-C++ extension)
    forward_splat_3d(centers, conic, amps, ...)
    backward_splat_3d(grad_output, centers, conic, amps, ...)
    compute_conic_metal(Ls)                   # optional Metal L -> conic helper

Metal kernels
  zero_float_buffer                           # AOT only; runtime zeroes with torch.zeros
  rasterize_forward_splat_centric_3d
  rasterize_backward_splat_centric_3d
  compute_conic_from_L_3d                     # helper/validation path, not the hot path
```

### Splat-centric forward

The forward kernel mirrors the optimized CUDA organization:

```text
one Metal threadgroup = one Gaussian splat
thread 0 loads the splat's precomputed conic and derives its AABB
64 threads cooperate over that splat's AABB
output uses atomic float add
```

The old tile-binned pipeline was removed.  The hot path no longer allocates or
passes `tile_counts`, `tile_offsets`, `tile_content`, or `tile_write_heads`, and
no longer performs a PyTorch prefix sum or CPU `.item()` synchronization.

### Splat-centric backward

Backward also uses one threadgroup per splat:

```text
threads accumulate local d_center / d_conic / d_amp
threadgroup reduction combines partials
thread 0 writes that splat's d_center / d_conic / d_amp once
```

The conic -> L and raw-parameter reparameterization VJPs are handled by PyTorch
autograd, not in-kernel.

This removes the previous voxel-centric global CAS atomics into parameter
gradients.  Forward still needs output atomics because different splats can
contribute to the same voxel; backward does not need global gradient atomics
because each threadgroup owns exactly one splat's gradient row.

### Coordinate convention

All custom Metal kernels now use Luxar/PyTorch volume order directly:

```text
centers: [Z, Y, X]
output:  output[z, y, x]
conic:   [c_zz, c_zy, c_zx, c_yy, c_yx, c_xx]
```

There is no `[Z,Y,X] -> [X,Y,Z]` packed-conic reorder in the current hot path.

## Usage

```python
from luxar.gsplats.models.gsplats.metal import (
    GaussianSplatModelMetal,
    is_metal_available,
)

if is_metal_available():
    model = GaussianSplatModelMetal(
        shape=(128, 128, 128),
        centers0=centers,          # (N, 3), [Z,Y,X]
        L0=L,                      # (N, 3, 3), lower triangular
        amps0=amps,                # (N,)
        sigma_min_diag=(0.5, 0.5, 0.5),
        truncate=3.0,
        intensity_floor=1e-5,
        device="mps",
    )

    pred = model()
    loss = loss_fn(pred, target)
    loss.backward()
```

`model.kernel_backend` reports `"aot"` or `"runtime"`. Fitting integration uses
Metal automatically when available and appropriate:

```python
from luxar.gsplats import fit_gaussian_splats

result = fit_gaussian_splats(volume, device="mps", use_metal=True)
```

## Performance Snapshot

Observed on Apple M4 Max, PyTorch 2.11, macOS 15.7.5 (AOT build), workload
`128³ @ 32k splats`, `L = 2I`, `truncate = 3`:

| Path | Time | Effective throughput |
| --- | ---: | ---: |
| Old tile-binned Metal forward | ~2.3-2.5 ms | ~0.85-0.94 GVox/s |
| New splat-centric Metal forward | ~1.5-1.6 ms | ~1.3-1.4 GVox/s |

The backward kernel shares the same splat-centric threadgroup-reduction
structure (replacing the old voxel-centric global atomics), with the conic -> L
and raw-parameter reparameterization VJPs handled by PyTorch autograd.

Large-output GVox/s is only one view of splatting performance because the actual
work scales with splat AABB volume and overlap.  The rewrite's most important
win is backward: global parameter-gradient atomics were replaced by per-splat
threadgroup reductions.

Correctness sanity for the same workload against CPU PyTorch:

```text
max_abs_diff  ≈ 1.8e-5
mean_abs_diff ≈ 7.1e-7
metal_sum     ≈ 3632518.5
cpu_sum       ≈ 3632518.0
```

### Runtime vs AOT vs PyTorch (M4 Max, torch 2.14.1, macOS 15.8.1)

One training step = forward + squared-error loss + backward through
`MetalSplatFunction` (conic + chain rule included); median of 20 steps (2 for
PyTorch). Random anisotropic splats, `truncate = 3`:

| Workload | runtime | AOT | PyTorch MPS |
| --- | ---: | ---: | ---: |
| 128³, 2k splats | 1.46 ms | 1.86 ms | 646 ms |
| 256³, 10k splats | 4.24 ms | 4.78 ms | 1885 ms |

Parity: runtime vs AOT forward max-relative difference 1.2e-7 / 5.6e-6, backward
1.9e-7 / 8.1e-4 (the backward compared under a random-noise `grad_output`, whose
cancelling sums amplify last-ulp differences between the two compilers' builds of
the same source); runtime vs the PyTorch reference forward 4.9e-6 / 5.6e-6.

End to end, `fit_gaussian_splats` on a synthetic 128³ volume with 2,000 seeds and
200 iterations on MPS took 3.5 s (runtime) and 3.5 s (AOT); about 2.4 s of that is
fixed seeding/setup, leaving ~5.7 ms per iteration. The PyTorch fallback costs
~710 ms per iteration on the same fit (measured over 10-20 iterations), so the
200-iteration fit would take roughly 140 s.

## Limitations

- Custom kernels are **3D only**.  2D and 4D-8D MPS models use PyTorch rendering.
- The runtime path accepts at most 67,108,863 splats per call
  (`runtime_backend.MAX_SPLATS`): the splat-centric dispatch launches
  `N * 64` threads in one 32-bit grid and the kernels take no splat offset, so
  it cannot be chunked without a kernel change. The AOT extension counts
  threadgroups and accepts up to 2³² − 1.
- Custom kernels are **float32 only**.  `use_fp16=True`, `.half()`, `.bfloat16()`,
  `.double()`, and non-float32 `.to()` requests are rejected.
- The model is **MPS-only**.  CPU and CUDA tensors are intentionally rejected;
  use `GaussianSplatModel` for CPU or `GaussianSplatModelCUDA` for NVIDIA GPUs.
- CUDA remains much faster for large production workloads; this Metal rewrite is
  a structural performance fix, not CUDA parity.

## Tests

```bash
hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/metal/tests -q
# the whole suite once per backend:
LUXAR_METAL_BACKEND=runtime hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/metal/tests -q
LUXAR_METAL_BACKEND=aot hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/metal/tests -q
hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/tests -q
hatch run pytest packages/luxar/src/luxar/gsplats/fitting/tests/test_initialization.py -q
```

### Verification status (audit A15-03)

Be precise about what is and is not checked automatically, because the honest
answer is "less than you would assume":

| | status |
|---|---|
| **Compiles** | Partially gated. `make check-native` / `hatch run check-native` checks every translation unit when its toolchain is present, using the standards shipped by the build. CI's Linux `python-tests` requires only the host-C++ arm. The Objective-C++ binding and Metal shader are checked manually on Apple silicon before each release with `hatch run check-native --require cxx --require metal`; hosted macOS CI and a dedicated Mac runner are not used today — they are deferred until a Mac runner exists. No GPU is required. |
| **Numeric parity vs the torch reference** | Tests EXIST and are substantial — 105 test functions here, 138 on the CUDA side, including `test_metal_numerical.py` and `test_metal_conic.py`. They are skip-guarded on Metal availability, so **they never run in CI**, which has no GPU (the platform-independent `test_backend_selection.py` does run there). The [native backend release verification](../../../../../../../../docs/guides/developer/BUILD_SYSTEM_SPEC.md#native-backend-release-verification) requires the Metal suite on an Apple-silicon box with `LUXAR_REQUIRE_METAL=1`, which fails rather than accepting MPS-unavailable skips. |
| **Formatting / static analysis** | None. No `.clang-format`, no `clang-tidy`, no CI arm. Deliberately deferred: introducing a format would rewrite ~3,900 lines of code whose only behavioural check is the GPU-gated suite above, so the reformat could not be verified. |

The compile gate exists because the build was, in fact, broken: both native
build paths pinned `-std=c++17` in the flags torch appends *last*, so the last
`-std=` won and `torch/all.h`'s `#error C++20 or later compatible compiler is
required` fired. The Metal extension then compiled itself on first import, so that
landed on users rather than in a build. It no longer builds itself at all; users
get the runtime-compiled kernels, which need neither Xcode nor a C++ toolchain.

This is a second and third independent implementation of the same
splat-rasterization maths as the torch reference — the same 1:1-sync hazard the
project gates for the Rust/TS pair — and a silent divergence produces slightly
wrong splats rather than a crash. The remaining gap is a *runner*, not a test:
giving the existing parity suites a GPU on some cadence is what would close it.

## Troubleshooting

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
python -c "from luxar.gsplats.models.gsplats.metal import is_metal_available; print(is_metal_available())"
python -c "from luxar.gsplats.models.gsplats.metal import get_metal_status; print(get_metal_status())"
```

`get_metal_status()` names the selected backend, or explains why none is usable
(off-macOS, no MPS device, torch too old for `compile_shader`, a compile error, or
a forced backend that cannot run). Upgrading torch fixes the common case.

Only the optional AOT build needs Xcode. If `xcrun --find metal` fails there,
switch to the full Xcode installation:

```bash
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
```
