# luxar.gsplats.preprocessing — Technical Specification

**Version**: 1.0.0
**Last Updated**: 2026-04-28

## Purpose

Volume denoising, denoising-strength calibration, and intensity normalization for the Gaussian-splat fitting pipeline. The fitter is sensitive to noise — fitting splats to noise produces noise-shaped splats — so volumes are typically denoised before fitting, with the denoising strength `h` calibrated automatically using the Noise2Self (J-invariant) self-supervised loss. Three backends are available: a hand-tuned CUDA kernel (fastest), a pure-PyTorch implementation (any GPU including MPS, plus CPU), and `scikit-image` (CPU only). All are wired behind a single `denoise_nlm()` entry point with auto-backend selection.

This subpackage is also reachable from the CLI as `luxar gsplat denoise` (`packages/luxar/src/luxar/cli/gsplat_commands.py`) and is invoked as a precompute step by `luxar gsplat batch plan` for HPC array jobs.

---

## Mathematical Foundation

### Non-Local Means (NLM)

For each voxel `x` in the input volume, the denoised value is a weighted average of all other voxels `y`, weighted by the similarity of small patches `P(x)` and `P(y)` centered on each:

```
NLM(x)   =  Σ_y w(x, y) · I(y)  /  Σ_y w(x, y)
w(x, y)  =  exp( −‖P(x) − P(y)‖²  /  (V_patch · h²) )
```

where:
- `I(y)` is the intensity at voxel `y`
- `P(x)` is a patch (cube of side `patch_size`) centered at `x`
- `V_patch = patch_size^d` is the number of voxels in the patch (`d = 2` or `3`)
- `h` is the filtering strength — small `h` ⇒ averages only very-similar patches (preserves features); large `h` ⇒ averages aggressively (oversmooths)
- `y` ranges over a search window of side `2·search_distance + 1` around `x`, NOT the full volume (the "non-local" name is a misnomer for the bounded-window variant used in practice)

The weights are normalized per-voxel so the output is a true convex combination.

### Noise2Self calibration

Because we have no clean ground truth, we choose `h` by **self-supervision**: pick the `h` that minimizes a J-invariant loss. The idea is that for a denoiser `f_h`, if the noise at different voxels is independent, then `f_h(I)` evaluated at a held-out voxel — using a version of `I` where that voxel has been replaced by its neighbors' mean — should be close to `I` at that voxel iff `h` is well-tuned.

Concretely, build a complementary partition `{M_1, …, M_K}` of the voxel index set (a stripe pattern with spacing `stride`; for `stride=2` this gives 4 masks in 2D, 8 in 3D). For each mask `M_k`:
1. Replace voxels in `M_k` by the mean of their immediate neighbors → `I_k`.
2. Denoise: `D_k = f_h(I_k)`.
3. Sum squared error on the held-out voxels: `L_k(h) = Σ_{x ∈ M_k} (D_k(x) − I(x))²`.

Total Noise2Self loss:

```
Loss(h)  =  Σ_k L_k(h)
```

Sweep `h` over a grid (default `[0.005, 0.08]` step `0.005`) and pick the minimum. For 3D volumes this is computed on a single central z-slice by default for speed; the resulting `h` transfers to the full volume because NLM weights are scale-invariant in that dimension.

### Normalization

Internally, NLM and calibration both expect intensity in `[0, 1]`. Volumes are normalized via `(I − v_min) / (v_max − v_min)` and denormalized identically before return; `vmin == vmax` (constant volume) short-circuits to zero output and a no-op denormalize.

---

## Algorithms

### `denoise_nlm(volume, h, patch_size=3, search_distance=5, backend='auto', device=None, chunk_size=None)`

**Inputs**:
- `volume: torch.Tensor` of shape `(H, W)` or `(D, H, W)`. Any floating dtype; computation runs in float32, output preserves input dtype.
- `h: float` — filtering strength. Typical range `[0.005, 0.08]` for normalized volumes. Calibrate via `calibrate_nlm_h()`.
- `patch_size: int` — odd integer, default 3. The CUDA backend supports `{3, 5}`; PyTorch and skimage are unconstrained.
- `search_distance: int` — half-width of the search window. CUDA supports `{5, 7, 9, 11, 13, 15}` per shared-memory budget.
- `backend: str` — `'auto'` (default), `'cuda'`, `'pytorch'`, or `'skimage'`. See backend selection below.
- `device: torch.device | None` — destination device; inferred from `volume.device` when `None`.
- `chunk_size: int | None` — for PyTorch 3D path, chunk axis-0 with halo overlap to bound peak memory. `None` disables chunking.

**Outputs**:
- `torch.Tensor` of the same shape, dtype, and device as `volume`.

**Algorithm**:
1. Validate: 2D or 3D tensor; `patch_size` odd; `h > 0`.
2. Resolve backend (see below).
3. Cast to float32 internally.
4. Invoke the selected backend; for chunked PyTorch 3D, slice the volume into overlapping slabs of size `(chunk_size + 2·halo)` where `halo = search_distance + patch_size // 2`, denoise each, stitch by discarding halo regions.
5. Cast back to input dtype.

**Edge cases**:
- CUDA `ValueError` ("shared memory exceeded" on 3D + large `search_distance`): warning emitted, falls back to PyTorch backend automatically.
- Constant volume after normalization: skips the kernel, returns zeros (denormalize restores `vmin`).
- Non-floating dtype: PyTorch raises; the CLI normalizes to `float32` upstream.

### `calibrate_nlm_h(volume, h_range=None, patch_size=3, search_distance=5, stride=2, backend='auto', device=None, use_2d_slice=True, slice_index=None)`

**Inputs**:
- `volume: torch.Tensor`, 2D or 3D.
- `h_range: Sequence[float] | None` — candidate `h` values. Default `np.arange(0.005, 0.08, 0.005)` (15 values from `0.005` to `0.075`).
- `stride: int` — mask spacing. Larger `stride` ⇒ fewer masks ⇒ faster but noisier loss.
- `use_2d_slice: bool` — for 3D, calibrate on a single z-slice (default; ~50× faster) instead of the full volume.
- `slice_index: int | None` — explicit slice for 3D calibration; defaults to the central slice.

**Outputs**:
- `float` — the `h` value with minimum Noise2Self loss.

**Algorithm**:
1. Normalize the (2D-extracted, if 3D + use_2d_slice) volume to `[0, 1]`.
2. Build complementary J-invariant masks via stripe pattern with the given `stride`.
3. For each `h` in `h_range` and each mask `M_k`:
   a. Interpolate masked pixels with neighbor mean → `I_k`.
   b. Run `denoise_nlm(I_k, h, ...)`.
   c. Accumulate `Σ_{x ∈ M_k} (D_k(x) − I(x))²`.
4. Return `argmin_h Σ_k L_k(h)`.

**Edge cases**:
- Empty or all-NaN result (e.g., backend failure): warn, fall back to `h = 0.04` (mid-range default known to be a safe broadband value).
- 3D `use_2d_slice=False`: full-volume calibration is honest but `O(D)` slower; use only when `h` is suspected to vary materially with `z`.

### `denoise_volume_array(volume, h, patch_size=3, search_distance=5, backend='auto', device=None, use_2d=False, chunk_size=None)`

High-level orchestrator: numpy in, numpy out. Wraps the normalize → denoise → denormalize sequence and adds a `use_2d` flag for slice-by-slice processing of 3D volumes (independent denoising per z).

Memory heuristic: if no `chunk_size` is set and the PyTorch 3D path is chosen, picks a chunk size targeting ~16 GB peak GPU memory.

### Multi-channel batch helpers

- `calibrate_h_for_channel(input_path, channel, sample_timepoints, ...)` — load a multi-channel volume, sample equidistant timepoints, calibrate per timepoint, return the median `h`. Robust to outlier frames.
- `calibrate_all_channels(input_path, n_timepoints, n_channels, ...)` — same idea applied across all channels; returns `{channel_index: h}`.

These exist to feed `luxar gsplat batch plan`'s denoise-precompute step, where one calibration run produces the per-channel `h` table consumed by the per-(timepoint, channel) Slurm tasks.

---

## Backend Selection

`nlm_core._resolve_backend(backend, device)` chooses the implementation per the following table:

| Requested | CUDA device | MPS device | CPU |
|---|---|---|---|
| `'auto'` | `cuda` if extension built, else `pytorch` | `pytorch` | `skimage` |
| `'cuda'` | `cuda` (or `ValueError` if extension missing) | `ValueError` | `ValueError` |
| `'pytorch'` | `pytorch` | `pytorch` | `pytorch` |
| `'skimage'` | `skimage` (CPU transfer) | `skimage` | `skimage` |

`skimage` is preferred over `pytorch` on CPU because skimage's `denoise_nl_means` uses optimized C with `fast_mode=True` and beats the unoptimized PyTorch loop on CPU. On any GPU, the PyTorch backend is the GPU-agnostic fallback when CUDA isn't available.

The auto-fallback from CUDA → PyTorch on `ValueError("shared memory exceeded")` triggers when 3D + `search_distance ≥ 9` would exceed the shared-memory budget on the target GPU. The exact limit varies by architecture (Ampere is more constrained than Ada/Hopper/Blackwell); the wrapper queries the device's actual shared-memory capacity at runtime via `torch.cuda.get_device_properties` rather than hard-coding per-architecture numbers, so the fallback is correct on every supported card.

When this happens, a `RuntimeWarning` is emitted with the suggestion to either reduce `search_distance` or switch backends explicitly.

---

## CUDA backend

Lives under `cuda/`. Layout:

```
cuda/
├── __init__.py             # imports nlm_cuda_wrapper, exposes NLM_CUDA_AVAILABLE flag
├── nlm_cuda_wrapper.py     # Python-level validation + dispatch
├── build.py                # nvcc compilation script with arch detection
├── src/
│   ├── nlm_cuda.cu         # 2D + 3D kernels
│   ├── nlm_cuda.h          # kernel signatures
│   └── bindings.cpp        # PyTorch C++ extension bindings
└── nlm_build_info.json     # auto-generated build metadata
```

**Constraints** (enforced by the wrapper before launching the kernel):
- `patch_size ∈ {3, 5}` (kernel uses fixed-size loops for shared-memory packing)
- `search_distance ∈ {5, 7, 9, 11, 13, 15}` (kernel block size derived from this; values outside this set are not compiled)
- 3D + large `search_distance` requires sufficient shared memory; the wrapper raises `ValueError` upfront with a fallback hint

**Kernel design highlights**:
- Each thread block loads its search window into shared memory once, so all threads in the block reuse the same patches without global memory reads.
- Patch-distance computation is unrolled at compile time per `patch_size`.
- Compiled with `-O3 --use_fast_math`; built for the host GPU's compute capability via `build.py` arch detection.

`make build-nlm-cuda` triggers the build; results land in `cuda/`; `make test-nlm-cuda` runs the regression suite; `make clean-nlm-cuda` removes the .so.

---

## Public API Surface

```python
# Top-level entry points re-exported from luxar.gsplats.preprocessing
denoise_nlm(
    volume: torch.Tensor,
    h: float,
    patch_size: int = 3,
    search_distance: int = 5,
    backend: str = "auto",
    device: torch.device | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor

calibrate_nlm_h(
    volume: torch.Tensor,
    h_range: Sequence[float] | None = None,
    patch_size: int = 3,
    search_distance: int = 5,
    stride: int = 2,
    backend: str = "auto",
    device: torch.device | None = None,
    use_2d_slice: bool = True,
    slice_index: int | None = None,
) -> float

denoise_volume_array(
    volume: np.ndarray,
    h: float,
    patch_size: int = 3,
    search_distance: int = 5,
    backend: str = "auto",
    device: torch.device | None = None,
    use_2d: bool = False,
    chunk_size: int | None = None,
) -> np.ndarray

normalize_volume(volume: np.ndarray) -> tuple[np.ndarray, float, float]
denormalize_volume(volume: np.ndarray, vmin: float, vmax: float) -> np.ndarray

calibrate_all_channels(input_path, n_timepoints, n_channels, ...) -> dict[int, float]
calibrate_h_for_channel(input_path, channel, sample_timepoints, ...) -> float
```

---

## Validation Rules

- `volume` must be a 2D or 3D tensor or array; other ranks raise `ValueError`.
- `patch_size` must be odd (NLM patches are centered on a single voxel).
- `h > 0` strictly; `h == 0` would divide by zero in the weight formula.
- For the CUDA backend: `(patch_size, search_distance)` tuple must be in the supported set listed above.
- `h_range` for calibration must contain at least one value.

---

## Performance Notes

| Backend | Relative speed (3D, 256³, sd=5) | Memory profile |
|---|---|---|
| CUDA   | 1× (baseline, ~0.4 s on RTX 4090) | Bounded by shared memory per block |
| PyTorch GPU | ~5–20× slower | Peak ~16 GB (default chunk target) |
| PyTorch MPS | ~5–20× slower | Bounded by unified memory |
| skimage CPU | ~50–200× slower | Streamed, low peak |

The CLI's `--device auto` selection (CUDA → MPS → CPU) combined with backend `'auto'` (CUDA-extension → PyTorch → skimage) gives the right answer in nearly all cases without explicit tuning.

---

## Cross-Language Compatibility

The denoised volume is consumed downstream by `luxar.gsplats.fitting`, all in-process Python; no cross-language serialization. The CUDA extension compiles per-machine via `make build-nlm-cuda` and is not portable across GPU architectures (`build.py` writes the chosen `sm_XX` into `nlm_build_info.json`). End users on different GPUs need to rebuild, which the `make` target does idempotently.

---

## Related Specifications

- `luxar.gsplats.fitting` — consumer of the denoised volume during `fit_gaussian_splats`. (See `../fitting/SPECIFICATIONS.md`.)
- `luxar.gsplats.models.gsplats` — the differentiable model the fitter optimizes against. (See `../models/gsplats/SPECIFICATIONS.md`.)
- `luxar.gsplats.clahe` — sibling preprocessing module for contrast-limited adaptive histogram equalization (different problem: contrast enhancement, not denoising). (See `../clahe/SPECIFICATIONS.md`.)
- CLI surface: `packages/luxar/src/luxar/cli/SPECIFICATIONS.md` documents `luxar gsplat denoise` and `luxar gsplat batch plan --denoise`.

---

## Changelog

- **v1.0.0** (2026-04-28): Initial specification
  - Documented the NLM weight formula, J-invariant Noise2Self loss, and the calibration-by-grid-search procedure.
  - Documented the three-backend selection logic (CUDA / PyTorch / skimage) and the auto-fallback on CUDA shared-memory overflow.
  - Documented the CUDA kernel constraints (`patch_size ∈ {3, 5}`, `search_distance ∈ {5, 7, 9, 11, 13, 15}`) and the runtime-queried shared-memory fallback path.
  - Documented the multi-channel batch helpers used by `luxar gsplat batch plan`.
