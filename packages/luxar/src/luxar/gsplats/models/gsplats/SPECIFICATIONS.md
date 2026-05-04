# luxar.gsplats.models.gsplats — Technical Specification

**Version**: 1.1.1
**Last Updated**: 2026-05-03

## Purpose

Differentiable n-dimensional Gaussian splat model and rasterizer. Each splat is a full-rank multivariate Gaussian with a learnable center, Cholesky-parameterized covariance, and amplitude; the model rasterizes a sum of these into a regular voxel grid. The forward pass is the rendered volume; the backward pass propagates gradients through every parameter so the model can be used as the substrate of a fitting loop (`luxar.gsplats.fitting`) that minimizes a reconstruction loss against a target volume.

This subpackage owns three responsibilities:

1. **`GaussianSplatModel`** — the PyTorch `nn.Module` that holds parameters, applies activations and constraints, and exposes `forward()` returning the rendered volume.
2. **`rendering_core`** — the differentiable splatting kernel: explicit Cholesky forward-substitution for 2D and 3D fast paths, generic-nD via triangular solve, AABB truncation with intensity-floor culling, and grid caching.
3. **`rendering_wrappers` + backends** — runtime dispatch between the pure-PyTorch path, a CUDA C++/CUDA extension (2D–8D, FP32+FP16), and a Metal Shading Language extension (3D, Apple Silicon).

The model layer is consumed by `luxar.gsplats.fitting.fit_gaussian_splats` and exposed at the CLI as `luxar gsplat fit`.

---

## Mathematical Foundation

### Splat density model

Each splat `k` contributes a truncated, normalized Gaussian to the rendered volume:

```
ρ_k(x) = a_k · scale · max(0, exp(−½ ‖y_k‖²) − C)
y_k    = L_k⁻¹ (x − μ_k)
C      = exp(−½ · truncate²)
scale  = 1 / (1 − C)
```

The full rendered volume is the sum: `V(x) = Σ_k ρ_k(x)`.

The truncation `C` and the `1/(1−C)` rescaling together produce a **shifted Gaussian** that's exactly zero at `‖y‖ = truncate` and integrates to the same total mass as the un-truncated Gaussian at the centre, with **C⁰ continuity** at the truncation boundary. This avoids the "ringing" artifacts you'd get from a hard cutoff and keeps the gradient well-defined across the boundary.

### Cholesky covariance parameterization

The covariance `Σ_k` of splat `k` is stored as its lower-triangular Cholesky factor `L_k`:

```
Σ_k = L_k · L_kᵀ
```

`L_k` has `d` diagonal entries and `d(d−1)/2` strictly-lower-triangular entries — the same parameter count as a symmetric `d×d` matrix, but with a guaranteed positive-definite reconstruction (provided diagonals are strictly positive).

The Cholesky form has three concrete benefits:

- **No matrix inversion**: solving `L_k @ y = (x − μ_k)` is forward-substitution, `O(d²)`. We never compute `Σ⁻¹` explicitly.
- **Positive-definiteness for free**: positive diagonals → `L Lᵀ` is PSD. We enforce positivity by passing the diagonals through softplus.
- **Cheap eigenvalue bounds**: the per-axis radius for AABB truncation is `truncate · √Σ_ii = truncate · √Σ_j L_ij²`, no decomposition needed.

### Activations

All learnable parameters are stored in a "raw" pre-activation form so unconstrained autograd updates remain numerically valid:

| Raw parameter | Activation | Output domain | Purpose |
|---|---|---|---|
| `raw_mu` | `sigmoid` × `(shape − 1)` | `[0, shape − 1]` | Centers strictly inside the volume |
| `raw_L_diag` | `sigma_min_diag + softplus` | `[sigma_min_diag, ∞)` | Strictly positive diagonals (PSD invariant) |
| `L_off` | identity (unconstrained) | `(−∞, ∞)` | Strictly-lower-triangular off-diagonals |
| `raw_a` | `softplus` | `[0, ∞)` | Non-negative amplitudes (optional `amp_max` clamp) |

`stable_inverse_softplus` (from `luxar.gsplats.models.utils`) is used to invert these activations when initializing raw parameters from desired post-activation values.

### Constraints (optional)

- **`sigma_max_diag`** — per-axis upper bound on the activated diagonal, applied via clamp before constructing `L`.
- **`max_eccentricity`** — bound on the ratio of largest to smallest eigenvalue of `L Lᵀ`, applied by adjusting off-diagonals to keep splats from going pathologically anisotropic during fitting.
- **`amp_max`** — global ceiling on amplitudes.

These are all soft constraints — they bias the parameter space without breaking differentiability.

### Loss function

Loss lives in `luxar.gsplats.fitting`, not here. The default is L1 (changed from MSE in April 2026 — see `CHANGELOG.md`).

---

## Parameter Layout

```
GaussianSplatModel.{
    raw_mu:    (N, d)   float32  nn.Parameter   # centers, sigmoid-activated
    raw_L_diag: (N, d)  float32  nn.Parameter   # diagonals, softplus-activated + sigma_min
    L_off:      (N, m)  float32  nn.Parameter   # off-diagonals, m = d(d−1)/2
    raw_a:      (N,)    float32  nn.Parameter   # amplitudes, softplus-activated
}

shape:           (d,)   tuple of int   — output volume dimensions
sigma_min_diag:  (d,)   float          — per-axis floor for activated diagonals
sigma_max_diag:  (d,)   float | None   — per-axis ceiling
amp_max:         float | None
max_eccentricity: float | None
truncate:        float                 — Gaussian sigma cutoff (default 3.0)
voxel_size:      (d,) | None           — physical voxel dimensions
```

**Invariants** maintained across `forward()`, `prune_()`, `append_()`, `replace_with()`:

- All raw tensors have a leading dimension `N` that stays consistent across the four parameters.
- `raw_L_diag.softplus_activated() + sigma_min_diag ∈ [sigma_min_diag, sigma_max_diag]` if `sigma_max_diag` is set.
- `L` (built from `raw_L_diag` and `L_off`) is always lower-triangular.
- After `forward()`, the output shape matches `self.shape` exactly.

---

## Algorithms

### `forward()`

```
1. centers, L, amps = self.current_params()        # apply activations + constraints
2. return render_gaussians(self.shape, centers, L, amps,
                           truncate=self.truncate,
                           intensity_floor=1e-5)   # see below
```

Returns a tensor of shape `self.shape` and dtype `float32` on the model's device.

### `current_params()`

```
1. centers = sigmoid(raw_mu) · (shape − 1)
2. diag    = sigma_min_diag + softplus(raw_L_diag)
3. if sigma_max_diag: diag = clamp(diag, max=sigma_max_diag)
4. L       = _build_L(diag, L_off)                 # zero-fills upper triangle
5. if max_eccentricity: L = _enforce_eccentricity(L, max_eccentricity)
6. amps    = softplus(raw_a)
7. if amp_max: amps = clamp(amps, max=amp_max)
8. return centers, L, amps
```

### `render_gaussians(shape, centers, Ls, amps, truncate=3.0, intensity_floor=1e-5, chunk_size=None)`

The differentiable rasterizer. Dispatches to a fast path by dimensionality:

```
match d:
    case 2: y_norm2 = fwd_norm2_2d(L, dx, dy)         # explicit substitution
    case 3: y_norm2 = fwd_norm2_3d(L, dx, dy, dz)     # explicit substitution
    case _: y_norm2 = ‖torch.linalg.solve_triangular(L, dx)‖²   # generic
```

**Per-splat AABB truncation** (the *real* speedup):

1. Compute per-axis radii `r_i = truncate · √(Σⱼ L_ij²)` (cheap; no eigendecomposition).
2. Optionally shrink by intensity-floor: solve for the radius at which `a_k · scale · (exp(...) − C) < intensity_floor`, take the smaller of that and the geometric truncate.
3. AABB box `[centre − r, centre + r]`, clamped to volume bounds.
4. Group splats by AABB *shape* (not position) so all splats with the same box dimensions can share a cached coordinate grid.

For each AABB-group, walk the local grid (offset by the splat's centre), compute `y` via forward substitution, accumulate `a_k · scale · max(0, exp(−½‖y‖²) − C)` into the output volume at the linear indices given by `linear_strides`.

**Memory chunking**: when the AABB group's voxel count × splat count exceeds the configured budget, the group is processed in chunks of `chunk_size` splats (or auto-sized via `calculate_optimal_chunk_size`, which targets 60 % of free CUDA memory).

**Grid cache**: `cached_base_and_offsets` keeps one byte-budgeted LRU cache per device, keyed on `(box_shape, strides, device, dtype)` and returning `(base, lin_offsets)` arrays for the local grid. Groups with the same box shape reuse the same offsets — typical hit rate during fitting is high when the same support shapes recur. The default budget is adaptive (`min(10% free CUDA memory, 2 GiB)` for CUDA, `512 MiB` for MPS, `min(5% available RAM, 4 GiB)` for CPU when available), and can be overridden for HPC jobs with `LUXAR_GSPLAT_GRID_CACHE_MAX_BYTES` / `_MAX_GB` and `LUXAR_GSPLAT_GRID_CACHE_MAX_ENTRY_BYTES` / `_MAX_ENTRY_GB`. Set `LUXAR_GSPLAT_GRID_CACHE_MAX_BYTES=0` to disable caching.

### Forward substitution: 2D fast path

Using 0-indexed entries of the lower-triangular `L` (matching `rendering_core.py` indexing `L[:, i, j]`):

```
y₀ = d₀ / L₀₀
y₁ = (d₁ − L₁₀ · y₀) / L₁₁
‖y‖² = y₀² + y₁²
```

Implemented batched over all splats in a group: `fwd_norm2_2d(L, d0, d1) → (K, P)` returning the squared norm at every (splat, voxel) pair.

### Forward substitution: 3D fast path

Same idea, three rows. Avoids `torch.linalg.solve_triangular` overhead — measured ~3× faster on CUDA for typical splat counts.

### Generic nD path

`y = torch.linalg.solve_triangular(L, d, upper=False)`, then `‖y‖²` by reduction. ~10× slower than the explicit paths but works for arbitrary `d`. Used by the CLI when fitting 4D+ data (e.g., 4D timelapse splats).

### `_build_L(diag, L_off) → (N, d, d)`

Reconstructs the full lower-triangular `L` tensor from the packed off-diagonal vector `L_off` (row-major over the strictly-lower-triangular indices). Upper triangle is zero-filled. Differentiable through both inputs.

### Eccentricity constraint

When `max_eccentricity = E` is set, `current_params()` clamps each
off-diagonal `L[i,j]` (with `i > j`) to a per-pair budget proportional
to the smaller of the two anchoring diagonals:

```
|L[i,j]|  ≤  γ · min(L[i,i], L[j,j])
```

with

```
γ      = √(E − 1) / (k_2d · (d − 1)^0.7)
k_2d   = 2.4 + 0.5 / √(E − 1)        (for E > 1)
k_2d   = 3.0                          (degenerate E = 1 case)
```

(Implementation: `gsplat_model.py::current_params` lines ~294–319.)

When `voxel_size` is provided the bound is applied in physical space
— `min(vs[i]·L[i,i], vs[j]·L[j,j])` replaces `min(L[i,i], L[j,j])` and
the result is rescaled by `vs[i]` before clamping the voxel-space
off-diagonal. This keeps the constraint anisotropy-aware.

**Origin / status.** This is a **bounding heuristic**, not a closed-form
guarantee that `cond(L Lᵀ) ≤ E` for `d > 2`:

- For **d = 2** the 2×2 lower-triangular case admits an analytic
  expression for `cond(L Lᵀ)` and `k_2d = 2.4 + 0.5/√(E − 1)` was
  fitted so the worst-case condition number tracks `E` closely; the
  asymptote `2.4` corresponds to the large-`E` regime, the `0.5/√(E−1)`
  term corrects the small-`E` near-isotropic regime.
- For **d > 2** the `(d − 1)^0.7` falloff is an empirical correction
  that controls how off-diagonal mass accumulates as more `(i,j)`
  pairs share each diagonal anchor; it was tuned by fitting against
  Monte-Carlo evaluations of `cond(L Lᵀ)` on random lower-triangular
  matrices clamped to the per-pair budget.

The constraint therefore *biases* the parameter space against
pathological elongation rather than guaranteeing a hard eigenvalue
ratio. If a workflow needs a strict `cond(L Lᵀ) ≤ E` guarantee,
post-fit eigendecomposition + projection is the appropriate step.

### `prune_(mask)` / `append_(centers, Ls, amps)` / `replace_with(centers, Ls, amps)`

In-place dynamic splat management used by the fitter's prune-and-densify schedule. `append_` and `replace_with` invert the activations to write raw values:

```
raw_mu       = inverse_sigmoid(centers / (shape − 1))
raw_L_diag   = stable_inverse_softplus(diag − sigma_min_diag)
L_off        = pack(L)
raw_a        = stable_inverse_softplus(amps)
```

(See `luxar.gsplats.models.utils.SPECIFICATIONS.md` for the inverse-softplus algorithm.)

---

## Backend Matrix

| Backend | Dimensionality | Forward | Backward | Dtype | Device | Notes |
|---|---|---|---|---|---|---|
| **PyTorch CPU** | nD (2-∞) | ✓ | ✓ | FP32 | CPU | Reference; 2D/3D fast paths apply |
| **PyTorch CUDA** | nD (2-∞) | ✓ | ✓ | FP32 | CUDA | Default GPU path; 2D/3D fast paths apply |
| **PyTorch MPS** | 2-3D | ✓ | ✓ | FP32 | MPS | `group_by_box` falls back to CPU |
| **CUDA C++ extension** | 2-8D | ✓ | ✓ | FP32 + FP16 | CUDA | Splat-centric kernel; 10–100× over PyTorch CUDA |
| **Metal extension** | 3D custom, 2D-8D model API | ✓ | ✓ | FP32 | MPS | Splat-centric 3D MPS kernels; PyTorch rendering for non-3D MPS shapes |

### Dispatch

`render_gaussians_pytorch` always uses the pure-PyTorch path (CPU/CUDA/MPS).

The accelerator backends are opt-in via dedicated model classes:

- `GaussianSplatModelCUDA` (in `cuda/gsplat_model_cuda.py`) — uses `CUDASplatFunction` (CUDA C++ extension if available; falls back to PyTorch CUDA otherwise).
- `GaussianSplatModelMetal` (in `metal/gsplat_model_metal.py`) — MPS-only subclass that uses `MetalSplatFunction` for 3D MPS tensors and the PyTorch renderer for non-3D MPS shapes while preserving the same dynamic splat-management API.

The CLI (`luxar gsplat fit`) selects the backend via the `--device` flag (`auto`, `cuda`, `cpu`, `mps`) plus the build status of the relevant extension.

---

## CUDA backend

Lives under `cuda/`. Key files:

```
cuda/
├── __init__.py                # exposes the build flag CUDA_SPLATTING_AVAILABLE
├── bindings.cpp               # pybind11 wrapper for forward / backward
├── cuda_splatting.cu          # template instantiations for D ∈ {2,...,8}
├── kernel_launchers.cuh       # launch wrappers (templated on dtype)
├── kernels_core.cuh           # the actual splat-centric kernel
├── utils.cuh                  # CUDA helpers
├── gsplat_model_cuda.py       # GaussianSplatModelCUDA + CUDASplatFunction
└── tests/                     # gradcheck, FP16, performance, nD …
```

### CUDA kernel API (pybind11)

```cpp
std::tuple<torch::Tensor, torch::Tensor>
forward_wrapper(
    const torch::Tensor& centers,        // (N, d)
    const torch::Tensor& conic,          // (N, d(d+1)/2) packed upper triangle of Σ⁻¹
    const torch::Tensor& amps,           // (N,)
    const std::vector<int64_t>& shape,   // d-tuple
    double truncate,                     // sigma cutoff
    double intensity_floor,              // amplitude floor
    bool use_fp16,                       // FP16 kernel
    const c10::optional<torch::Tensor>& output_buffer
) -> (output, shape_cached);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
backward_wrapper(
    const torch::Tensor& grad_output,    // (∏shape,)
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const std::vector<int64_t>& shape,
    double truncate,
    double intensity_floor,
    bool use_fp16,
    const c10::optional<torch::Tensor>& shape_tensor_cached,
    const c10::optional<torch::Tensor>& output_to_zero
) -> (grad_centers, grad_conic, grad_amps);
```

### Template instantiations

```cpp
// cuda_splatting.cu — explicit specialization for D ∈ {2, 3, 4, 5, 6, 7, 8}
template void launch_rasterize_forward_splat_centric<D, float >(...);
template void launch_rasterize_forward_splat_centric<D, __half>(...);
template void launch_rasterize_backward_splat_centric<D, float >(...);
template void launch_rasterize_backward_splat_centric<D, __half>(...);
```

The dimension cap of 8 is a compile-time choice driven by the CUDA template-stack budget; raising it requires editing `cuda_splatting.cu` and recompiling.

### FP16

FP16 kernels exist for both forward and backward and are exercised by the CUDA test suite. Gradient outputs are always cast back to FP32 before returning to PyTorch, regardless of the forward dtype, for numerical stability of the optimizer step.

### Build

`make build-cuda` compiles via `setup.py` (`-O3 --use_fast_math`), auto-detecting the host's compute capability from `torch.cuda.get_device_capability()`. Build metadata lands in `cuda_build_info.json` for reproducibility. `make build-cuda SLURM=1` runs the build on a GPU node via Slurm for HPC users without local GPUs.

---

## Metal backend

Lives under `metal/`. Key files:

```
metal/
├── __init__.py                # availability checks, stale-extension rebuilds
├── src/kernels.metal          # 3D Metal Shading Language kernels
├── src/bindings.mm            # C++/Objective-C++ dispatcher and pybind11 API
├── setup.py                   # Metal C++ extension build
├── gsplat_model_metal.py      # GaussianSplatModelMetal + MetalSplatFunction
└── tests/                     # backend, conic, interface, perf, optimizer compatibility, …
```

### Metal kernel pipeline (3D, splat-centric)

```metal
kernel void zero_float_buffer(...);

kernel void compute_conic_from_L_3d(
    device const float* Ls       [[buffer(0)]],   // (N, 3, 3) [Z, Y, X]
    device       float* conic    [[buffer(1)]],   // (N, 6) [Z, Y, X] packed
    constant     uint& n_splats  [[buffer(2)]]
);

kernel void rasterize_forward_splat_centric_3d(...);   // one threadgroup per splat
kernel void rasterize_backward_splat_centric_3d(...);  // one threadgroup per splat gradient
```

The previous Metal tile pipeline was removed.  There are no tile counts, tile offsets, tile-content buffers, PyTorch prefix sums, or CPU `.item()` synchronizations in the hot path.  Forward uses atomic float adds into the output volume because multiple splats can hit the same voxel.  Backward uses threadgroup reductions and writes each splat's gradients once, with no global parameter-gradient atomics.

### Coordinate convention

PyTorch / NumPy use `[Z, Y, X]` order (first dim = depth).  Current Metal kernels use the same order directly.  The 3D conic is packed as `[c_zz, c_zy, c_zx, c_yy, c_yx, c_xx]`, matching `cholesky_to_conic()` and the CUDA row-major upper-triangle convention.  No `[Z,Y,X] <-> [X,Y,Z]` conic reorder remains in the Metal hot path.

### Build

`pip install -e packages/luxar/src/luxar/gsplats/models/gsplats/metal/` (driven by `setup.py`), Apple Silicon only. Runtime requires macOS 12+ for the Metal Performance Shaders (MPS) PyTorch backend.

---

## Public API Surface

```python
class GaussianSplatModel(torch.nn.Module):
    def __init__(
        self,
        shape: Sequence[int],
        centers0: np.ndarray,                                # (N, d)
        L0: np.ndarray,                                      # (N, d, d) lower-triangular
        amps0: np.ndarray,                                   # (N,)
        sigma_min_diag: Sequence[float],                     # (d,)
        sigma_max_diag: Sequence[float] | None = None,
        amp_max: float | None = None,
        max_eccentricity: float | None = None,
        truncate: float = 3.0,
        voxel_size: np.ndarray | None = None,
        device: torch.device | None = None,
    ) -> None: ...

    def forward(self) -> torch.Tensor: ...                   # (h1, ..., hd) float32
    def current_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def prune_(self, mask: torch.Tensor) -> None: ...
    def append_(self, centers, Ls, amps) -> None: ...
    def replace_with(self, centers, Ls, amps) -> None: ...
    def n_splats(self) -> int: ...

    # Properties: shape, dim, truncate, amp_max, max_eccentricity

# Functional rasterizer
def render_gaussians(
    shape: Sequence[int],
    centers: torch.Tensor,                                   # (N, d)
    Ls: torch.Tensor,                                        # (N, d, d)
    amps: torch.Tensor,                                      # (N,)
    truncate: float = 3.0,
    intensity_floor: float = 1e-5,
    chunk_size: int | None = None,
) -> torch.Tensor

# Convenience wrappers (no_grad)
def render_gaussians_numpy(shape, result, truncate=3.0, chunk_size=None) -> np.ndarray
def render_gaussians_pytorch(shape, result, truncate=3.0, device=None, chunk_size=None) -> torch.Tensor
def render_gaussians_batched(shape, centers, Ls, amps, ...) -> torch.Tensor

# Internal but documented for backend implementers
fwd_norm2_2d(L, d0, d1) -> torch.Tensor                       # (K, P)
fwd_norm2_3d(L, d0, d1, d2) -> torch.Tensor                   # (K, P)
compute_aabb_with_intensity_floor(centers, Ls, amps, shape, truncate, floor, device)
linear_strides(shape, device) -> torch.Tensor
group_by_box(lo, hi) -> dict
cached_base_and_offsets(box_shape, strides, device, dtype) -> tuple
calculate_optimal_chunk_size(K, d, device, dtype) -> int
clear_grid_cache() -> None
get_grid_cache_stats() -> dict
```

---

## Validation Rules

- `centers0.shape == (N, d)`, `L0.shape == (N, d, d)`, `amps0.shape == (N,)` with consistent `N` and `d`.
- `L0` lower-triangular for each splat; the constructor takes the lower triangle as authoritative.
- `sigma_min_diag` length must equal `d`; same for `sigma_max_diag` and `voxel_size` when provided.
- All initial diagonals strictly positive (`L0[k, i, i] > sigma_min_diag[i]` for the post-activation invariant to hold).
- `amps0 >= 0`.
- `truncate > 0`; values < 1.0 are accepted but produce most of the splat being clipped (rarely useful).

Violations raise `ValueError` from the constructor.

---

## Performance Notes

Relative ordering across backends (3D, typical splat counts):

```
CUDA C++ extension  ≪  Metal splat-centric (Apple Silicon)  ≪  PyTorch CPU
```

Concrete throughput depends heavily on splat count, volume shape, and GPU model — see `make benchmark-cuda` (CUDA backend) and the regression suite under `cuda/tests/test_performance.py` / `metal/tests/test_performance.py` for current numbers on the host. The CUDA C++ extension remains the fastest path; the Metal extension now follows the same splat-centric ownership model for 3D MPS FP32 tensors but does not yet reach CUDA-class throughput. The generic-nD path is order-of-magnitude slower and is intended for 4D+ workloads where there is no custom backend; CPU is reserved for development and testing.

Memory: `calculate_optimal_chunk_size` targets 60 % of free GPU memory by default. The support-grid cache is bounded by bytes per device rather than by entry count, skips entries larger than the configured per-entry cap, and can be cleared explicitly via `clear_grid_cache()` between large jobs. HPC users can dedicate more memory to repeated large support shapes with `LUXAR_GSPLAT_GRID_CACHE_MAX_GB=<n>` and `LUXAR_GSPLAT_GRID_CACHE_MAX_ENTRY_GB=<n>`.

The `group_by_box` step is O(N log N) due to a sort+group; `group_by_box_gpu` uses `torch.unique` for a fully-on-GPU alternative on CUDA. MPS lacks `torch.unique` for some dtypes, so the CPU fallback path is used there.

---

## Cross-Language Compatibility

1. **Coordinate convention** — all PyTorch/NumPy paths use `[Z, Y, X]` order (NumPy convention). CUDA and current Metal kernels both consume packed conics in row-major upper-triangle order for that same axis order; for 3D this is `[c_zz, c_zy, c_zx, c_yy, c_yx, c_xx]`.
2. **Cholesky packing** — `L_off` is row-major over strictly-lower-triangular indices `(i, j) with i > j`; this convention is implicit in `_build_L` and locked in by the regression suite rather than documented in the source comments. The CUDA backend takes the conic (`Σ⁻¹`) packed as `(N, d(d+1)/2)`; the bindings layer (`cuda/bindings.cpp` and `cuda/gsplat_model_cuda.py`) derives the conic from `L` before launch, so callers using the model class don't see the packing convention directly.
3. **FP16 promotion rule** — CUDA kernels accept FP16 inputs but always emit FP32 gradients to keep the optimizer step stable.
4. **Truncation rescaling** — the `1 / (1 − exp(−½ · truncate²))` factor is computed identically on every backend; the regression suite locks the per-splat output to within 1 ULP across PyTorch / CUDA / Metal at FP32.
5. **Atomic accumulation** — CUDA uses `atomicAdd(float)`; current Metal uses `atomic_float` fetch-add for forward output accumulation and avoids global parameter-gradient atomics in backward.

---

## Related Specifications

- `luxar.gsplats.models.utils` — `stable_inverse_softplus` (parameter initialization) and `solve_lower_triangular` (used by the generic-nD path). (See `../utils/SPECIFICATIONS.md`.)
- `luxar.gsplats.fitting` — gradient-descent loop, prune-and-densify schedule, loss functions. (See `../../fitting/SPECIFICATIONS.md`.)
- `luxar.gsplats.preprocessing` — denoising preprocess that runs before fitting. (See `../../preprocessing/SPECIFICATIONS.md`.)
- `luxar.gsplats.seeds` — initial center/L/amp values for the constructor. (See `../../seeds/SPECIFICATIONS.md`.)
- CLI surface: `packages/luxar/src/luxar/cli/SPECIFICATIONS.md` documents `luxar gsplat fit`.

---

## Changelog

- **v1.1.1** (2026-05-03): Documented adaptive byte-budgeted support-grid caching, per-device accounting, cache stats, and HPC environment-variable overrides.
- **v1.1.0** (2026-05-02): Updated the Metal backend description for the splat-centric performance rewrite: no tile-binned hot path, native `[Z,Y,X]` conic packing, atomic output accumulation, and threadgroup-reduced backward gradients.
- **v1.0.0** (2026-04-28): Initial specification
  - Documented the full Cholesky-parameterized splat density model, the truncated Gaussian rescaling, and the per-axis AABB radii.
  - Documented the four-tensor parameter layout (`raw_mu`, `raw_L_diag`, `L_off`, `raw_a`) and the activations / constraints applied at `forward()` time.
  - Documented the rendering pipeline: 2D/3D explicit forward-substitution fast paths, generic nD via `solve_triangular`, AABB grouping, intensity-floor culling, LRU grid cache, memory chunking.
  - Documented the CUDA backend (D ∈ {2,…,8}, FP32 + FP16, `forward_wrapper` / `backward_wrapper` pybind API) and the then-current Metal backend.
