# Native CPU Backend (numba)

## Purpose

Fused CPU kernels for 3D Gaussian-splat fitting: the CPU sibling of the Metal
(`../metal/`) and CUDA (`../cuda/`) backends. The PyTorch renderer runs a CPU fit
as dense, unfused autograd over per-splat voxel boxes (grouped by box shape, a
serial `index_add_`, a backward ~3x the forward); these kernels do the same math
in one pass per direction, so a CPU-only machine -- a Linux box without a GPU, a
plain `pip install` -- fits many times faster.

They are compiled at runtime by [numba](https://numba.pydata.org/) from its pip
wheel, so users need no C++ toolchain (the same idea as the runtime-compiled
Metal shaders). numba is in the `gsplats` extra; without it, CPU fits use the
PyTorch renderer.

## Key Classes and Functions

| Name | Module | Description |
|------|--------|-------------|
| `GaussianSplatModelCPUNative` | `gsplat_model_cpu.py` | `GaussianSplatModel` whose 3D `forward` runs on the kernels; other dimensionalities use the PyTorch renderer (as the Metal model does). Same constructor, parameters and constraints. |
| `CpuSplatFunction` | `gsplat_model_cpu.py` | `torch.autograd.Function`: numba forward, numba analytic backward, zero-copy numpy views of contiguous CPU tensors. float32 in fitting; float64 works too (gradchecks). |
| `render_gaussians_cpu_native(shape, centers, Ls, amps, ...)` | `gsplat_model_cpu.py` | Differentiable drop-in for `render_gaussians` (3D, CPU). |
| `cpu_render_numpy` / `cpu_render_backward_numpy` | `gsplat_model_cpu.py` | The kernels on plain numpy arrays. |
| `prepare_cpu_native()` | `backend.py` | Select (per `LUXAR_CPU_BACKEND`), set threads, warm up; `None` means "use torch". |
| `forward_3d`, `backward_3d`, `compute_boxes`, `plane_lists` | `kernels.py` | The `@njit(parallel=True, cache=True, fastmath=False)` kernels. |

## Math

Voxel for voxel the PyTorch 3D renderer (`rendering_core._render_gaussians_3d`):
integer-voxel sampling in `[Z, Y, X]`, the same AABB (truncation radius, then the
amplitude-aware intensity-floor shrink), `D^2 = ||L~^-1 (x - mu)||^2` by forward
substitution with the diagonal clamped to `>= 1e-6`, and the shifted Gaussian
`a * scale * max(0, exp(-D^2/2) - C)`. Unlike the Metal/CUDA kernels it does not
additionally drop sub-floor voxels inside the box, so it agrees with the PyTorch
path to float rounding (forward ~1e-7 relative, gradients ~1e-7 relative in the
tests).

Two things make it fast without changing the numbers:

- **Pruning.** A voxel outside the truncated ellipsoid contributes exactly zero
  (value and gradient), so each row's x-range is cut to the ellipsoid
  analytically; `y0` is constant per plane and `y1` per row, leaving one `exp`
  and a few multiply-adds per voxel.
- **No atomics.** The forward splits the *output* into tiles (one z-plane by a
  band of rows); each tile loops over the splats whose box covers its plane
  (a CSR list built per call), so every voxel has one writer and the result is
  deterministic. The backward's per-splat gradients are independent given
  dL/dV, so it is a parallel loop over splats that only reads the gradient
  volume, accumulating per-splat moments `S_y`, `S_yy` in float64; then
  `dmu = -2 L~^-T S_y` and `dL = lower(-2 L~^-T S_yy)` (no conic needed).

## Performance

Median fit iteration (forward + backward + optimizer), synthetic blob volumes,
`fit_gaussian_splats(..., device="cpu")`, M4 Max (load average 5-10 from other
work), torch and numba both given the thread count shown:

| Volume / seeds | Threads | PyTorch | numba | Speed-up |
|----------------|---------|---------|-------|----------|
| 64x128x128 / 2k | 1 | 0.327 s | 0.044 s | 7x |
| 64x128x128 / 2k | 4 | 0.270 s | 0.015 s | 18x |
| 64x128x128 / 2k | 8 | 0.325 s | 0.010 s | 31x |
| 128x256x256 / 4k | 8 | 2.83 s | 0.077 s | 37x |

The render alone (one forward + backward, 4k splats into 128x256x256) is
4.85 s -> 0.44 s at 1 thread and 3.93 s -> 0.061 s at 8. The PyTorch path barely
scales with threads; the kernels scale ~6x from 1 to 8.

## Selection and Environment

`initialize_optimization` tries this backend for a **3D fit on the CPU** after
Metal/CUDA (which need an MPS/CUDA device). The verbose log says
`✓ Model class: GaussianSplatModelCPUNative (numba, N threads, <layer> layer)`
plus the first-call JIT time; a quiet fit's one-line summary names
`(numba kernels, device=cpu)`.

| Variable | Values | Effect |
|----------|--------|--------|
| `LUXAR_CPU_BACKEND` | `auto` (default), `native`, `torch` | `auto` uses the kernels when numba imports; `native` raises if it does not; `torch` always uses the PyTorch renderer. |
| `LUXAR_CPU_THREADS` | positive int or `0` | Kernel thread count (default: physical cores available to the process; `0` uses the default), resolved once per value. numba keeps its own pool; torch's intra-op setting is neither read nor changed here. Multi-worker CPU fits (`fit -j N`, content `-j N`, `batch-fit run --gpus cpu`, Slurm `--parallel`) give each worker its share unless you set a nonzero value. |
| `NUMBA_CACHE_DIR` | path | Where compiled kernels are cached (numba's own variable). |

**Torch threads still matter.** With the render off torch, what remains on
torch per iteration -- the loss over the volume, `_build_L`, the Adam step -- is
now a visible share of a fast iteration, and it runs on torch's intra-op
threads. The hatch env pins those to 1 (`OMP_NUM_THREADS`/`MKL_NUM_THREADS`);
for real work export the width you want with both `OMP_NUM_THREADS` and
`MKL_NUM_THREADS`. `LUXAR_CPU_THREADS` controls only the native kernels.

**Compile and cache.** The first call in a fresh environment JIT-compiles the
kernels (~2 s on an M4 Max); `cache=True` stores the machine code beside
`kernels.py` in a writable checkout, or in the user cache directory
(`~/.cache/numba` or the platform equivalent) when the package sits in a
read-only site-packages, so later processes load it in well under a second.
If neither is writable (read-only site-packages and a read-only or missing
HOME), the kernels cache in a per-user directory under the system temp dir,
and failing that compile without a cache -- they never fall back to torch
over a cache problem.
Editing `kernels.py` invalidates the cache.

**Threading layer.** numba picks TBB, then OpenMP, then its built-in
`workqueue` (the only one on a stock macOS install). `workqueue` aborts the
process on concurrent launches from two Python threads, so kernel launches are
serialized behind a lock.

## Limitations

- 3D only; 2D and 4D+ CPU fits keep the PyTorch renderer.
- MPS and CUDA paths are untouched.
- A splat with non-finite parameters renders nothing and gets zero gradient
  (the PyTorch path would poison the whole volume with NaN); the first
  occurrence per process prints one warning with the count.

## Tests

```bash
hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/cpu/tests
hatch run pytest -m slow packages/luxar/src/luxar/gsplats/models/gsplats/cpu/tests
```

`test_cpu_parity.py` compares forward and gradients with the PyTorch renderer
(typical, sub-voxel, huge, anisotropic, border, empty) and runs a float64
`gradcheck`; `test_cpu_backend_selection.py` covers the env vars, the numba-less
fallback, the fit wiring and the CLI; `test_cpu_fit_parity.py` (slow) checks a
short fit reaches the PyTorch path's PSNR.
