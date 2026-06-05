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
- Full Xcode installation, not only Command Line Tools
- PyTorch with MPS support (`torch.backends.mps.is_available() == True`)

## Installation and Build

The extension auto-builds on first import when the compiled artifacts are missing
or stale.  Manual rebuild for development:

```bash
cd packages/luxar/src/luxar/gsplats/models/gsplats/metal
python setup.py build_ext --inplace
```

The build compiles:

1. `src/kernels.metal` -> `src/default.metallib`
2. `src/bindings.mm` -> `metal_splatting_backend*.so`

The package passes the absolute `default.metallib` path into the native extension,
invalidates stale extension/metallib artifacts when sources change, and touches
outputs after no-op distutils rebuilds so repeated imports do not auto-compile.

## Architecture

```text
Python
  GaussianSplatModelMetal
  MetalSplatFunction
  L factors handed directly to native kernels

C++/Objective-C++ extension
  forward_raw_splat_3d(raw_mu, raw_L_diag, L_off, raw_a, sigma_min_diag, ...)
  backward_raw_splat_3d(grad_output, raw_mu, raw_L_diag, L_off, raw_a, ...)
  forward_splat_3d(centers, Ls, amps, ...)          # constrained/fallback path
  backward_splat_3d(grad_output, centers, Ls, amps, ...)

Metal kernels
  zero_float_buffer
  rasterize_forward_raw_splat_centric_3d      # raw params -> centers/L/amps + L -> conic
  rasterize_backward_raw_splat_centric_3d     # d_output -> raw parameter gradients
  rasterize_forward_splat_centric_3d          # constrained/fallback L path
  rasterize_backward_splat_centric_3d         # constrained/fallback L path
  compute_conic_from_L_3d                     # helper/validation path, not the hot path
```

### Splat-centric forward

The forward kernel mirrors the optimized CUDA organization:

```text
one Metal threadgroup = one Gaussian splat
thread 0 applies raw sigmoid/softplus transforms when eligible, then computes L -> conic and AABB
64 threads cooperate over that splat's AABB
output uses atomic float add
```

The old tile-binned pipeline was removed.  The hot path no longer allocates or
passes `tile_counts`, `tile_offsets`, `tile_content`, or `tile_write_heads`, and
no longer performs a PyTorch prefix sum or CPU `.item()` synchronization.

### Splat-centric backward

Backward also uses one threadgroup per splat. In the unconstrained raw-parameter path, the native kernel writes gradients for `raw_mu`, `raw_L_diag`, `L_off`, and `raw_a` directly, avoiding Python-side `current_params()` autograd work:

```text
threads accumulate local d_center / d_conic / d_amp
threadgroup reduction combines partials
thread 0 applies the analytic d_conic -> d_L VJP
thread 0 applies sigmoid/softplus VJPs for raw parameters when applicable
thread 0 writes that splat's gradients once
```

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

Fitting integration uses Metal automatically when available and appropriate:

```python
from luxar.gsplats import fit_gaussian_splats

result = fit_gaussian_splats(volume, device="mps", use_metal=True)
```

## Performance Snapshot

Observed on Apple M4 Max, PyTorch 2.11, macOS 15.7.5, workload
`128³ @ 32k splats`, `L = 2I`, `truncate = 3`:

| Path | Time | Effective throughput |
| --- | ---: | ---: |
| Old tile-binned Metal forward | ~2.3-2.5 ms | ~0.85-0.94 GVox/s |
| New splat-centric Metal forward | ~1.5-1.6 ms | ~1.3-1.4 GVox/s |
| Old tile-binned Metal fwd+bwd | ~133 ms | ~0.016 GVox/s |
| New raw splat-centric Metal fwd+bwd | ~2.8-3.0 ms | ~0.70-0.75 GVox/s |

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

## Limitations

- Custom kernels are **3D only**.  2D and 4D-8D MPS models use PyTorch rendering.
- Custom kernels are **float32 only**.  `use_fp16=True`, `.half()`, `.bfloat16()`,
  `.double()`, and non-float32 `.to()` requests are rejected.
- The model is **MPS-only**.  CPU and CUDA tensors are intentionally rejected;
  use `GaussianSplatModel` for CPU or `GaussianSplatModelCUDA` for NVIDIA GPUs.
- CUDA remains much faster for large production workloads; this Metal rewrite is
  a structural performance fix, not CUDA parity.

## Tests

```bash
hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/metal/tests -q
hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/tests -q
hatch run pytest packages/luxar/src/luxar/gsplats/fitting/tests/test_initialization.py -q
```

## Troubleshooting

```bash
xcrun --find metal
python -c "import torch; print(torch.backends.mps.is_available())"
python -c "from luxar.gsplats.models.gsplats.metal import is_metal_available; print(is_metal_available())"
python -c "from luxar.gsplats.models.gsplats.metal import get_metal_status; print(get_metal_status())"
```

`get_metal_status()` returns a human-readable explanation of the current backend
state (off-macOS, MPS-interop failure, build failure, or loaded).

If `xcrun --find metal` fails, switch to the full Xcode installation:

```bash
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
```
