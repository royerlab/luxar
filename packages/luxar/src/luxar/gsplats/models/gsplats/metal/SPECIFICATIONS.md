# Metal Backend Specification

**Version**: 2.0.0
**Last Updated**: 2026-05-02

## Overview

The Metal backend provides Apple Silicon acceleration for Luxar Gaussian splats.
The custom native kernel path is intentionally narrow and explicit:

```text
3D volume shape + MPS device + float32 tensors
```

`GaussianSplatModelMetal` is an MPS-only `GaussianSplatModel` subclass.  It
preserves the base/CUDA parameter-management API (`current_params`, `append_`,
`prune_`, `replace_with`, state dicts, constraints) and dispatches to custom
Metal kernels only for 3D MPS tensors.  2D and 4D-8D MPS shapes use Luxar's
PyTorch renderer.

## Native Architecture

```text
Python layer
  GaussianSplatModelMetal.forward()
  MetalSplatFunction.forward/backward()
  passes Cholesky factors directly to native kernels

C++/Objective-C++ extension: src/bindings.mm
  compute_conic_metal(Ls) -> conic  # helper/validation path
  forward_splat_3d(centers, Ls, amps, shape, truncate, floor) -> output
  backward_splat_3d(grad_output, centers, Ls, amps, shape, truncate, floor)
      -> (d_centers, d_Ls, d_amps)

Metal shaders: src/kernels.metal
  zero_float_buffer
  compute_conic_from_L_3d            # helper/validation path
  rasterize_forward_splat_centric_3d # computes L -> conic per splat
  rasterize_backward_splat_centric_3d # computes L -> conic and d_conic -> d_L per splat
```

The previous tile-binned pipeline (`preprocess_3d`, `bin_3d`, tile counts,
tile offsets, tile content, PyTorch prefix sum, CPU `.item()` allocation size)
has been removed from the hot path.

## Coordinate and Packing Conventions

All Metal kernels use the same order as PyTorch/NumPy volumes:

```text
axis 0: Z / depth
axis 1: Y / height
axis 2: X / width
```

For 3D, packed conics are row-major upper-triangle entries in that order:

```text
[c00, c01, c02, c11, c12, c22]
= [c_zz, c_zy, c_zx, c_yy, c_yx, c_xx]
```

No `[Z,Y,X] <-> [X,Y,Z]` conic permutation is used.  This is a deliberate change
from the older tile-binned Metal renderer.

## Forward Algorithm

### Host-side dispatch

`forward_splat_3d` validates:

- `shape == (D, H, W)` and all dimensions are positive.
- `centers.shape == (N, 3)`, `Ls.shape == (N, 3, 3)`, `amps.shape == (N,)`.
- All tensors are contiguous, MPS, and `float32`.
- `N` and `D*H*W` fit in `uint32_t` for the current kernels.

It allocates an MPS float32 output tensor, then encodes in one command buffer:

1. `zero_float_buffer` over `D*H*W` elements.
2. `rasterize_forward_splat_centric_3d` with one threadgroup per splat.

### Kernel ownership model

```text
threadgroup_position_in_grid.x = splat_id
thread_index_in_threadgroup    = worker thread within that splat
THREADGROUP_SIZE               = 64
```

Thread 0 loads the splat into threadgroup memory:

```text
center[3]
L lower-triangular factors
conic[6] computed from L in native [Z,Y,X] order
sigma diagonal derived directly from L for AABB construction
amplitude
host-precomputed shifted-Gaussian constants
AABB lower corner and extent
effective truncation radius
```

All 64 threads then stride over the splat's local AABB.  For each voxel:

```text
d = [z, y, x] - center
D² = dᵀ C d
if D² <= effective_truncate²:
    I = amp * scale * max(exp(-0.5 * D²) - C_shift, 0)
    if I >= intensity_floor:
        atomic_add(output[z, y, x], I)
```

Forward needs output atomics because different splats may contribute to the same
voxel.  The kernel uses Metal `atomic_float` fetch-add.

## Backward Algorithm

### Host-side dispatch

`backward_splat_3d` validates the same Cholesky-factor splat tensors plus a
contiguous MPS float32 `grad_output` of shape `(D, H, W)`.  For nonzero `N`, it
allocates empty MPS gradient tensors:

```text
d_centers: (N, 3)
d_Ls:      (N, 3, 3)
d_amps:    (N,)
```

Every splat row is written exactly once by the kernel, so zero initialization is
not required for nonzero `N`.

### Kernel ownership model

Backward uses the same splat ownership as forward:

```text
one threadgroup = one splat gradient row
```

Each thread accumulates local scalar values in registers:

```text
local_d_amp
local_d_center_z, local_d_center_y, local_d_center_x
local_d_conic_0 ... local_d_conic_5
```

For each voxel in that splat's AABB, the kernel recomputes the forward intensity
and applies the chain rule:

```text
∂I/∂a  = I / max(a, 1e-10)
∂I/∂D² = -0.5 * (I + a * scale * C_shift)
outer  = dLoss/dI * ∂I/∂D²
```

For conic layout `[c00, c01, c02, c11, c12, c22]` and displacement
`d = [dz, dy, dx]`:

```text
D² = c00*dz² + c11*dy² + c22*dx²
   + 2*(c01*dz*dy + c02*dz*dx + c12*dy*dx)

∂D²/∂center = -2 * C * d
∂D²/∂c00 = dz²
∂D²/∂c01 = 2*dz*dy
∂D²/∂c02 = 2*dz*dx
∂D²/∂c11 = dy²
∂D²/∂c12 = 2*dy*dx
∂D²/∂c22 = dx²
```

The threadgroup then performs a tree reduction in threadgroup memory and thread
0 applies the analytic 3D conic-to-Cholesky VJP and writes:

```text
d_centers[splat_id, :]
d_Ls[splat_id, :, :]
d_amps[splat_id]
```

No global parameter-gradient atomics are used.

## AABB and Truncation

The Metal kernels match the optimized CUDA AABB integer-radius strategy, but now
avoid recovering covariance diagonals from the conic determinant. Because the hot
path receives Cholesky factors directly, AABB radii use the covariance diagonal
from `Σ = L @ L.T`:

```text
sigma_z = abs(l00)
sigma_y = sqrt(l10² + l11²)
sigma_x = sqrt(l20² + l21² + l22²)
```

The base truncation is tightened by `intensity_floor`:

```text
C_shift = exp(-0.5 * truncate²)
scale   = 1 / (1 - C_shift)
threshold = intensity_floor / max(amplitude * scale, 1e-10) + C_shift

if threshold >= 1:
    splat contributes nothing
else:
    t_eff = min(truncate, sqrt(-2 * log(threshold)))
```

AABB bounds use the CUDA-compatible integer-radius rule:

```text
radius_i = ceil(t_eff * sigma_i)
lo_i = max(0, floor(center_i) - radius_i)
hi_i = min(shape_i - 1, ceil(center_i) + radius_i)
```

## Python Autograd Boundary

`MetalSplatFunction.forward` saves:

```text
centers
Ls_for_conic
amps
```

The native backward returns gradients with respect to `centers`, Cholesky factors
`Ls`, and `amps` directly. The 3D `d_conic -> d_L` vector-Jacobian product is
implemented inside `rasterize_backward_splat_centric_3d`, so the Python backward
no longer rebuilds a `cholesky_to_conic()` autograd graph.

## Optional Metal L -> Conic

`compute_conic_metal(Ls)` computes the 3D packed conic in native `[Z,Y,X]` order.
It remains available as a helper and validation path for conic tests, but the
custom forward/backward hot path computes `L -> conic` inside the per-splat Metal
threadgroup instead of materializing a separate conic tensor.

## Performance Characteristics

For `128³ @ 32k splats`, `L = 2I`, `truncate = 3`, on Apple M4 Max:

| Implementation | Forward | Forward+Backward |
| --- | ---: | ---: |
| Previous tile-binned Metal | ~2.3-2.5 ms (~0.9 GVox/s) | ~133 ms (~0.016 GVox/s) |
| Current splat-centric Metal | ~1.5-1.6 ms (~1.3-1.4 GVox/s) | ~3.5-3.7 ms (~0.57-0.60 GVox/s) |

The largest improvement is backward because the old voxel-centric kernel used
global CAS atomics for every voxel-splat gradient contribution.  The new kernel
uses local threadgroup reductions and one write per splat gradient.

CUDA remains much faster for large production workloads; CUDA has 2D-8D
specialization, FP16 input paths, more mature occupancy behavior, and highly
optimized NVIDIA atomics.  The current Metal backend is structurally aligned with
CUDA but not yet CUDA-throughput-equivalent.

## Public API

```python
GaussianSplatModelMetal(
    shape=(D, H, W),          # 2D-8D accepted; custom Metal only for 3D
    centers0=centers,
    L0=L,
    amps0=amps,
    sigma_min_diag=(...),
    sigma_max_diag=None,
    amp_max=None,
    max_eccentricity=None,
    truncate=3.0,
    intensity_floor=1e-5,
    use_fp16=False,           # rejected if True
    use_metal_conic=False,       # retained helper flag; hot path computes conics inline
    voxel_size=None,
    device="mps",
)
```

`MetalSplatFunction.apply` arguments:

```python
output = MetalSplatFunction.apply(
    centers,
    Ls,
    amps,
    shape,
    truncate,
    intensity_floor,
    use_metal_conic,
)
```

## Validation and Limitations

- CPU and CUDA devices are rejected explicitly.
- Non-float32 paths are rejected explicitly.
- `use_fp16=True` is rejected.
- `.to("cpu")`, `.cpu()`, `.cuda()`, `.half()`, `.bfloat16()`, `.double()`, and
  non-float32 `.to()` requests are rejected.
- Dynamic `append_`, `replace_with`, and `prune_` require MPS tensors.
- Custom kernels run only for 3D.  2D and 4D-8D MPS shapes use the PyTorch
  renderer with the same model-management API.

## Testing Requirements

Run:

```bash
hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/metal/tests -q
hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/tests -q
hatch run pytest packages/luxar/src/luxar/gsplats/fitting/tests/test_initialization.py -q
```

Important coverage:

- Metal extension loading and stale rebuilds.
- Native `[Z,Y,X]` coordinate and packed-conic convention.
- L->conic correctness.
- Forward output vs PyTorch reference.
- Gradient signs, values, and optimization convergence.
- Interface parity: dynamic splat ops, state dicts, device/dtype rejection,
  zero-splat pruning, and non-3D MPS PyTorch rendering.

## Changelog

- **v2.0.0** (2026-05-02): Splat-centric performance rewrite.
  - Removed tile-binned preprocessing/binning/prefix-sum pipeline from the hot path.
  - Added splat-centric forward with one threadgroup per splat and atomic output accumulation.
  - Added splat-centric backward with threadgroup reductions and no global parameter-gradient atomics.
  - Removed packed-conic coordinate reorder; Metal now uses native `[Z,Y,X]` order.
  - Updated native API to `forward_splat_3d` / `backward_splat_3d`.
- **v1.1.0** (2026-05-02): CUDA-parity and reliability update.
  - Rebuilt `GaussianSplatModelMetal` as an MPS-only `GaussianSplatModel` subclass.
  - Added 2D-8D constructor support with PyTorch rendering outside the custom 3D path.
  - Added stale-extension rebuild detection and explicit `default.metallib` path handoff.
  - Added interface-parity regression tests.
- **v1.0.0** (2025-12-23): Initial documented release.
