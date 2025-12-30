# Seed Generation for Gaussian Splatting

**Version**: 1.0.0
**Last Updated**: 2025-11-27

## Overview

This subpackage provides two complementary methods for generating seed Gaussian splat locations from n-dimensional images. Seeds serve as initial positions for Gaussian splat fitting, determining where splats should be placed to approximate image features.

**Primary Use Cases:**
1. **Gaussian Splat Initialization**: Seeding the fitting process with high-quality seed positions
2. **Multi-Scale Feature Detection**: Identifying important features across spatial scales
3. **Sparse Representation**: Finding optimal locations for efficient image approximation

**Two Complementary Approaches:**
1. **Multiscale Gaussian Method**: Comprehensive multiscale Gaussian-blurred peak detection with optional CLAHE preprocessing
2. **Decomposition Method**: Hierarchical scale-based detection via optimized image decomposition

**Shared Utilities:**
- Peak detection with L∞ neighborhoods
- Spatial deduplication with farthest-first selection
- Seed combination and merging

## Architectural Overview

### Package Structure

```
seeds/
├── __init__.py                    # Public API exports
├── multiscale_gaussian.py         # Multiscale Gaussian seed generation
├── decomposition.py               # Decomposition-based candidate generation
├── utils.py                       # Shared utilities (peak detection, deduplication)
├── README.md                      # User-facing documentation
└── SPECIFICATIONS.md              # This file (technical specification)
```

### Design Philosophy

**Two Complementary Philosophies:**

1. **Multiscale Gaussian (Overcomplete)**: Cast a wide net by detecting features at multiple scales using Gaussian filtering. Generates many candidates with some redundancy, letting downstream fitting select the best subset.

2. **Decomposition (Hierarchical)**: Use optimized scale decomposition to explicitly separate features by scale, then select candidates from each scale layer. Generates fewer, more principled candidates with natural coarse-to-fine ordering.

**Integration Philosophy:**
- Both methods use shared utilities for consistency
- Both produce identical output format: `(N, ndim)` float arrays
- Both integrate seamlessly with `fit_gaussian_splats()`
- Methods can be combined for comprehensive coverage

## Method 1: Multiscale Gaussian Seed Generation

### Mathematical Foundation

#### Scale-Space Theory

The multiscale Gaussian approach is based on **scale-space theory**: convolving an image with Gaussian kernels at different scales creates a scale-space representation where features at different sizes become apparent at corresponding scales.

**Gaussian Scale-Space:**
```
L(x, σ) = G(x, σ) * I(x)
```

where:
- `I(x)`: Input image
- `G(x, σ)`: Gaussian kernel with standard deviation σ
- `L(x, σ)`: Scale-space representation at scale σ
- `*`: Convolution operator

**Blob Detection:**
Peaks (local maxima) in `L(x, σ)` correspond to blob-like structures of size ~σ in the original image.

#### CLAHE Preprocessing (Optional)

CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances local contrast, making features in dim regions as detectable as features in bright regions.

**Purpose:**
- Balanced detection across heterogeneous data
- Reveal dim structures that would be missed by global thresholding
- Prevent bright regions from dominating candidate selection

**Integration:**
When `apply_clahe=True`, all processing uses the CLAHE-enhanced image for consistency between detection and refinement.

### Algorithm: `seed_from_gaussian()`

#### High-Level Algorithm

```
Input:
  - V: n-dimensional image/volume
  - scales: sequence of Gaussian scales (in voxels)
  - peaks_per_scale: max peaks per scale (optional)
  - percentile_thresh: intensity percentile threshold (0-100)
  - min_distance: minimum distance between candidates
  - apply_clahe: enable CLAHE preprocessing
  - clahe_*: CLAHE parameters

Steps:
1. Optional CLAHE preprocessing
   - Convert to torch tensor
   - Apply CLAHE with specified parameters
   - Convert back to numpy for processing
   - All subsequent operations use CLAHE-enhanced image

2. Multiscale Gaussian-blurred peak detection
   - For each scale σ in scales (coarsest to finest):
     a. Apply Gaussian filter: img = gaussian_filter(V_work, σ)
     b. Compute threshold: thr = percentile(img, percentile_thresh)
     c. Set neighborhood radius: radius = max(1, round(1.5 × σ))
     d. Find local maxima: peaks = local_maxima(img, radius, thr, peaks_per_scale)
     e. Collect peak coordinates
   
3. Combine all candidates from all detection methods
   - Stack coordinate arrays vertically
   - Filter out empty arrays

4. Spatial deduplication
   - Remove candidates closer than min_distance
   - Use farthest-first selection for maximum spatial diversity

5. Sub-voxel refinement
   - For each candidate position:
     a. Extract 3×3×...×3 neighborhood (clamped at borders)
     b. Compute intensity-weighted centroid
     c. Use relative weights: w = patch - patch.min()
     d. Calculate: mu = Σ(w × coords) / Σ(w)
   - Return refined float coordinates

Output:
  - candidates: np.ndarray of shape (N, ndim) with float coordinates
```

#### Detailed Algorithm Steps

**Step 1: CLAHE Preprocessing (Optional)**

```python
if apply_clahe:
    import torch
    from luxar.gsplats.clahe import apply_clahe as apply_clahe_torch
    
    # Convert to torch
    V_torch = torch.tensor(V, dtype=torch.float32)
    
    # Apply CLAHE
    V_clahe_torch = apply_clahe_torch(
        V_torch,
        tile_size=clahe_tile_size,
        clip_limit=clahe_clip_limit,
        nbins=clahe_nbins,
    )
    
    # Convert back to numpy
    V_work = V_clahe_torch.cpu().numpy()
else:
    V_work = V
```

**Step 2: Multiscale Gaussian Peak Detection**

Process scales from coarsest to finest (reversed order) for farthest-first priority:

```python
# Pre-compute base image threshold (cache for efficiency)
V_percentile_thresh = np.percentile(V_work, percentile_thresh)

for s in reversed(scales):  # Coarsest to finest
    # Apply Gaussian smoothing
    img = ndi.gaussian_filter(V_work, sigma=s, mode='nearest')
    
    # Adaptive threshold selection
    if s <= min(scales) * 2.0:  # Fine scales: use base threshold
        thr = V_percentile_thresh
    else:  # Coarse scales: compute specific threshold
        thr = np.percentile(img, percentile_thresh)
    
    # Neighborhood radius scales with filter size
    # Factor 1.5 ensures peaks are well-separated relative to blob size
    radius = int(max(1, round(1.5 * s)))
    
    # Find local maxima
    coords = local_maxima(img, radius=radius, thresh=thr, top_k=peaks_per_scale)
    all_coords.append(coords)
```

**Rationale for Coarse-to-Fine Processing:**
- Farthest-first deduplication processes candidates in order
- Coarse-scale features detected first get priority
- Ensures large structures are represented before fine details
- Matches hierarchical decomposition philosophy

**Step 3: Spatial Deduplication**

```python
# Combine all coordinate arrays
coords = np.vstack([c for c in all_coords if c.size > 0])

# Deduplicate with farthest-first selection
coords = dedupe_farthest_first(coords, min_distance=min_distance)
```

**Step 4: Sub-Voxel Refinement**

Refine integer peak positions to sub-voxel precision:

```python
centers = []
for c in coords:
    # Extract 3×3×...×3 neighborhood (clamped at image borders)
    slices = []
    for ax in range(d):
        lo = max(0, int(c[ax] - 1))
        hi = min(V_work.shape[ax], int(c[ax] + 2))
        slices.append(slice(lo, hi))
    
    # Extract intensity patch
    patch = V_work[tuple(slices)]
    
    # Create coordinate grids
    grids = np.meshgrid(*[np.arange(s.start, s.stop) for s in slices], indexing='ij')
    
    # Compute intensity-weighted centroid
    w = patch - patch.min()  # Relative intensities (non-negative)
    W = w.sum() + 1e-12      # Avoid division by zero
    
    # Weighted average across all axes
    mu = np.array([float((w * grids[ax]).sum() / W) for ax in range(d)], dtype=float)
    centers.append(mu)

centers = np.array(centers, float)
```

**Rationale:**
- Intensity weighting improves localization accuracy
- Uses same image (V_work) for consistency with detection
- Relative weighting (patch - patch.min()) handles varying backgrounds
- Sub-voxel precision improves downstream fitting

### API Specification

```python
def seed_from_gaussian(
    V: np.ndarray,
    spacing: Optional[Sequence[float]] = None,
    scales: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
    peaks_per_scale: Optional[int] = None,
    percentile_thresh: float = 75.0,
    min_distance: float = 2.0,
    apply_clahe: bool = True,
    clahe_tile_size: int = 32,
    clahe_clip_limit: float = 16.0,
    clahe_nbins: int = 256,
) -> np.ndarray
```

**Parameters:**

- `V` (np.ndarray): Input n-dimensional image/volume
- `spacing` (Optional[Sequence[float]]): Physical voxel spacing (reserved for future use)
- `scales` (Sequence[float]): Standard deviations for Gaussian filtering (in voxels)
  - Default: `(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)`
  - Should cover range of expected feature sizes
- `peaks_per_scale` (Optional[int]): Maximum peaks to detect at each scale
  - Default: `None` (unlimited)
  - Limits computational cost and memory usage
- `percentile_thresh` (float): Intensity percentile threshold (0-100)
  - Default: `75.0` (75th percentile)
  - Higher values → fewer, stronger candidates
  - Lower values → more, weaker candidates
- `min_distance` (float): Minimum Euclidean distance between candidates (in voxels)
  - Default: `2.0`
  - Used for spatial deduplication
- `apply_clahe` (bool): Enable CLAHE preprocessing
  - Default: `True`
  - Recommended for heterogeneous data
- `clahe_tile_size` (int): Tile size for CLAHE (in voxels)
  - Default: `32`
  - Smaller tiles → more local enhancement
- `clahe_clip_limit` (float): Contrast limiting factor for CLAHE
  - Default: `16.0`
  - Range: `1.0` (no enhancement) to `40.0` (aggressive)
  - Typical: `2.0-16.0`
- `clahe_nbins` (int): Number of histogram bins for CLAHE
  - Default: `256`

**Returns:**
- `np.ndarray`: Candidate coordinates of shape `(N, ndim)` with float values

**Raises:**
- `ValueError`: If input array is empty or 0-dimensional
- `ValueError`: If scales is empty or contains non-positive values
- `ValueError`: If peaks_per_scale is non-positive
- `ValueError`: If percentile_thresh is not in [0, 100]
- `ValueError`: If min_distance is non-positive

### Performance Characteristics

**Time Complexity:**
- CLAHE preprocessing: O(n_voxels × nbins × ndim)
- Gaussian filtering per scale: O(n_voxels × kernel_size^ndim)
- Peak detection per scale: O(n_voxels)
- Deduplication: O(N × M × log M) where M = number of selected candidates
- Sub-voxel refinement: O(N × 3^ndim)
- **Total**: O(n_scales × n_voxels) + O(N × M × log M)

**Space Complexity:**
- Working images: O(n_voxels) per scale
- Candidate storage: O(N × ndim)
- **Peak usage**: ~2-3× input image size

**Typical Performance (3D volume, 512³, 7 scales):**
- CLAHE preprocessing: ~1-2 seconds
- Gaussian filtering: ~5-10 seconds
- Peak detection: <1 second
- Deduplication (1000 candidates): ~0.1 seconds
- **Total**: ~10-15 seconds

### Edge Cases and Robustness

1. **Empty input**: Raises ValueError
2. **Uniform image**: Returns empty array (no peaks above threshold)
3. **Single voxel**: Returns that voxel if above threshold
4. **Anisotropic voxels**: Currently treats voxels as isotropic (spacing parameter reserved for future)
5. **No peaks found**: Returns empty array with shape `(0, ndim)`
6. **Border handling**: Sub-voxel refinement clamps neighborhoods at image borders

## Method 2: Decomposition-Based Seed Generation

### Mathematical Foundation

#### Multi-Scale Image Decomposition

The decomposition method builds on the multi-scale decomposition framework:

```
V = Σₖ upsample(Vₖ)
```

where:
- `Vₖ`: Image at scale k with shape `(s₀/rₖ, s₁/rₖ, ..., sₙ₋₁/rₖ)`
- `rₖ`: Scale factor (e.g., 1, 2, 4, 8, 16, 32)
- `upsample()`: Interpolation function (nearest, linear, cubic)

**Key Properties:**
1. **Energy Separation**: Optimization explicitly distributes energy across scales
2. **Non-Overlapping**: Each scale captures distinct frequency bands
3. **Hierarchical**: Natural coarse-to-fine ordering
4. **Sparse**: Energy concentrated at important features per scale

For detailed mathematical formulation, see [multiscale/SPECIFICATIONS.md](../multiscale/SPECIFICATIONS.md).

#### Noise Suppression via Scale Filtering

**Problem**: Finest scales (especially scale=1, full resolution) capture high-frequency noise along with genuine detail.

**Solution**: Ignore the finest k scales for candidate generation.

**Rationale:**
1. **Noise suppression**: Finest scales dominated by noise, not signal
2. **Overfitting prevention**: Too many fine candidates lead to overfitting
3. **Computational efficiency**: Fewer candidates → faster convergence
4. **Coarse-to-fine bias**: Start with global structure, add detail via dynamic seeding

**Default**: `ignore_finest_k=1` (skip full-resolution scale only)

### Algorithm: `seed_from_decomposition()`

#### High-Level Algorithm

```
Input:
  - V: n-dimensional image/volume
  - scales: scale factors for decomposition [1, 2, 4, 8, ...]
  - ignore_finest_k: number of finest scales to ignore
  - peaks_per_scale: max peaks per scale (optional)
  - min_distance: minimum distance between candidates
  - threshold_rel: relative intensity threshold (0.0-1.0)
  - decompose_kwargs: additional args for decompose_image()
  - verbose: print progress information

Steps:
1. Input validation
   - Check array dimensions and values
   - Validate scale parameters
   - Validate ignore_finest_k (adjust if too large)

2. Multi-scale decomposition
   - Call decompose_image(V, scales, **decompose_kwargs)
   - Returns: scale_images (list of arrays), stats (dict)
   - Typical: 500 iterations, 6 scales → 10-60 seconds

3. Peak detection per scale (excluding finest k)
   - Determine scales to process: range(ignore_finest_k, len(scales))
   - Process from coarse to fine (reversed order)
   - For each scale:
     a. Get scale image and scale factor
     b. Compute threshold: thresh = threshold_rel × max(scale_img)
     c. Find local maxima with radius=1 (3×3×...×3 neighborhood)
     d. Map peak coordinates to full resolution:
        peak_full_res = peak_scale × scale_factor + scale_factor/2
     e. Record energies: energies = scale_img[peak_positions]

4. Combine candidates from all scales
   - Stack all coordinate arrays
   - Stack all energy arrays

5. Spatial deduplication
   - Use dedupe_farthest_first with energy priority
   - Remove candidates closer than min_distance
   - Keep higher-energy peaks

6. Return sorted candidates
   - Already sorted by energy (from deduplication)

Output:
  - candidates: np.ndarray of shape (N, ndim) with float coordinates
```

#### Detailed Algorithm Steps

**Step 1: Input Validation**

```python
# Convert to float array
V = np.asarray(V, dtype=float)
if V.size == 0:
    raise ValueError("Input array V cannot be empty")
if V.ndim == 0:
    raise ValueError("Input array V must have at least 1 dimension")

ndim = V.ndim

# Validate scales
if not scales or len(scales) == 0:
    raise ValueError("scales must be a non-empty list")
if any(s <= 0 for s in scales):
    raise ValueError("All scale values must be positive")

# Validate and adjust ignore_finest_k
if ignore_finest_k < 0:
    raise ValueError("ignore_finest_k must be non-negative")
if ignore_finest_k >= len(scales):
    warnings.warn(
        f"ignore_finest_k={ignore_finest_k} >= len(scales)={len(scales)}. "
        f"Using ignore_finest_k={len(scales) - 1} instead."
    )
    ignore_finest_k = len(scales) - 1
```

**Step 2: Multi-Scale Decomposition**

```python
# Lazy import to avoid circular dependency
from luxar.gsplats.multiscale.decompose import decompose_image

# Prepare kwargs
decompose_kwargs = decompose_kwargs or {}
if "verbose" not in decompose_kwargs:
    decompose_kwargs["verbose"] = verbose

# Decompose image
scale_images, stats = decompose_image(V, scales=scales, **decompose_kwargs)

# scale_images[i] corresponds to scales[i]
# scale_images[0] is finest resolution (scale factor = scales[0], typically 1)
# scale_images[-1] is coarsest resolution
```

**Step 3: Peak Detection Per Scale**

```python
all_candidates = []
all_energies = []

# Determine which scales to process (skip finest k)
scales_to_process = list(range(ignore_finest_k, len(scales)))

# Process scales from coarse to fine (reversed order)
for scale_idx in reversed(scales_to_process):
    scale_factor = scales[scale_idx]
    scale_img = scale_images[scale_idx]
    
    # Compute threshold for this scale
    max_intensity = scale_img.max()
    if max_intensity <= 0:
        continue  # Skip empty scales
    
    threshold = threshold_rel * max_intensity
    
    # Find local maxima in scale image
    # Use radius=1 for finest resolution in scale image (3×3×...×3 neighborhood)
    radius = 1
    peaks = local_maxima(scale_img, radius=radius, thresh=threshold, top_k=peaks_per_scale)
    
    if len(peaks) == 0:
        continue  # No peaks in this scale
    
    # Map peak coordinates to full resolution
    # Peak at position (i, j, ...) in scale image corresponds to
    # position (i×scale_factor + scale_factor/2, ...) in full resolution
    candidates_full_res = peaks.astype(float) * scale_factor + scale_factor / 2.0
    
    # Get energy (intensity) at each peak location
    energies = scale_img[tuple(peaks.T)]
    
    all_candidates.append(candidates_full_res)
    all_energies.append(energies)
```

**Coordinate Mapping Rationale:**
- Scale image at scale factor `r` has shape `(H/r, W/r, ...)`
- A voxel at integer position `(i, j, ...)` in scale image represents region `[i×r, (i+1)×r) × [j×r, (j+1)×r) × ...` in full resolution
- Center of this region: `(i×r + r/2, j×r + r/2, ...)`
- Provides sub-voxel precision through geometric mapping

**Step 4: Combine Candidates**

```python
if len(all_candidates) == 0:
    return np.zeros((0, ndim), dtype=float)

candidates = np.vstack(all_candidates)
energies = np.concatenate(all_energies)
```

**Step 5: Spatial Deduplication**

```python
# Deduplicate spatially close candidates
# Use farthest-first selection with energy priority
candidates_dedup = dedupe_farthest_first(
    candidates, 
    min_distance=min_distance, 
    intensities=energies
)

# Output is already sorted by energy (descending)
return candidates_dedup
```

### API Specification

```python
def seed_from_decomposition(
    V: np.ndarray,
    scales: List[int] = [1, 2, 4, 8, 16, 32, 64],
    ignore_finest_k: int = 1,
    peaks_per_scale: Optional[int] = None,
    min_distance: float = 2.0,
    threshold_rel: float = 0.1,
    decompose_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> np.ndarray
```

**Parameters:**

- `V` (np.ndarray): Input n-dimensional image/volume
- `scales` (List[int]): Scale factors for decomposition
  - Default: `[1, 2, 4, 8, 16, 32, 64]`
  - Scale 1 = full resolution, scale 2 = half resolution, etc.
  - Should cover range of feature sizes
- `ignore_finest_k` (int): Number of finest scales to ignore
  - Default: `1` (ignore full-resolution scale)
  - `0` = use all scales (no filtering)
  - `2` = aggressive noise suppression
- `peaks_per_scale` (Optional[int]): Maximum peaks per scale
  - Default: `None` (unlimited)
  - Limits candidates from each scale
- `min_distance` (float): Minimum Euclidean distance between candidates (in voxels)
  - Default: `2.0`
  - Used for spatial deduplication
- `threshold_rel` (float): Relative threshold for peak detection
  - Default: `0.1` (peaks must be ≥10% of scale maximum)
  - Range: `0.0-1.0`
  - Higher values → fewer, stronger candidates
- `decompose_kwargs` (Optional[Dict[str, Any]]): Additional arguments for `decompose_image()`
  - Common options:
    - `n_iters` (int): Optimization iterations (default 500)
    - `energy_weight` (float): Hierarchical energy penalty (default 0.01)
    - `loss_type` (str): "l1" (default), "mse", or "poisson"
    - `lr` (float): Learning rate (default 0.01)
  - See [multiscale/SPECIFICATIONS.md](../multiscale/SPECIFICATIONS.md) for full details
- `verbose` (bool): Print progress information
  - Default: `False`

**Returns:**
- `np.ndarray`: Candidate coordinates of shape `(N, ndim)` with float values, sorted by energy (descending)

**Raises:**
- `ValueError`: If input array is empty or 0-dimensional
- `ValueError`: If scales is empty or contains non-positive values
- `ValueError`: If ignore_finest_k is negative
- `ValueError`: If peaks_per_scale is non-positive
- `ValueError`: If min_distance is non-positive
- `ValueError`: If threshold_rel is not in [0, 1]
- `UserWarning`: If ignore_finest_k >= len(scales) (auto-adjusted)

### Performance Characteristics

**Time Complexity:**
- Decomposition: O(n_iters × n_scales × n_voxels) - dominant cost
- Peak detection: O(n_scales × n_voxels_per_scale)
- Deduplication: O(N × M × log M) where M = number of selected candidates
- **Total**: Dominated by decomposition (95%+ of time)

**Space Complexity:**
- Scale images: O(Σₖ n_voxels / rₖ^ndim) ≈ O(n_voxels × (1 + 1/2^d + 1/4^d + ...))
- For d=3: ≈ 1.14× input size
- Candidate storage: O(N × ndim)
- **Peak usage**: ~2× input image size

**Typical Performance (3D volume, 512³, 7 scales):**
- Decomposition (500 iterations): ~30-60 seconds
- Peak detection: <1 second
- Deduplication (500 candidates): ~0.05 seconds
- **Total**: ~30-60 seconds

### Edge Cases and Robustness

1. **Empty input**: Raises ValueError
2. **Uniform image**: May return empty array (depends on decomposition convergence)
3. **No peaks found**: Returns empty array with shape `(0, ndim)`
4. **ignore_finest_k too large**: Automatically adjusted to `len(scales) - 1` with warning
5. **Scale with zero energy**: Skipped automatically
6. **Decomposition convergence failure**: Still produces candidates, but quality may be reduced (check `stats['converged']`)

## Shared Utilities

### Peak Detection: `local_maxima()`

Identifies local maxima in n-dimensional images using L∞ (Chebyshev) neighborhoods.

#### Mathematical Definition

A point `p` is a local maximum if:
```
V(p) = max{V(q) : q ∈ N_∞(p, r)} AND V(p) ≥ threshold
```

where:
- `N_∞(p, r)`: L∞ neighborhood (hypercube) of radius r around p
- `N_∞(p, r) = {q : max_i |q_i - p_i| ≤ r}`
- Hypercube has side length `2r + 1` in each dimension
- Total neighborhood size: `(2r + 1)^ndim`

**L∞ vs L₂ (Euclidean) Neighborhood:**
- L∞: Hypercube (e.g., 3×3×3 for r=1 in 3D)
- L₂: Hypersphere (e.g., 7 voxels for r=1 in 3D: center + 6 face neighbors)
- L∞ is more permissive (includes edge and corner neighbors)
- L∞ is faster to compute (no distance calculations)
- L∞ matches natural image grid topology

#### Algorithm Implementation

```python
def local_maxima(
    img: np.ndarray,
    radius: int,
    thresh: float,
    top_k: Optional[int]
) -> np.ndarray:
    """
    Find local maxima in n-dimensional image.
    
    Returns: np.ndarray of shape (N, ndim) with integer coordinates
    """
    # Ensure minimum radius
    if radius < 1:
        radius = 1
    
    # Define kernel size (avoid creating large footprint arrays)
    kernel_size = [2 * radius + 1] * img.ndim
    
    # Apply maximum filter
    max_f = ndi.maximum_filter(img, size=kernel_size, mode='nearest')
    
    # Peak criteria: equals neighborhood maximum AND exceeds threshold
    peaks_mask = (img == max_f) & (img >= thresh)
    
    # Extract coordinates
    coords = np.argwhere(peaks_mask)
    
    if coords.size == 0:
        return coords
    
    # Optionally limit to top_k strongest peaks
    if top_k is not None and len(coords) > top_k:
        vals = img[tuple(coords.T)]
        keep = np.argsort(vals)[-top_k:]  # Indices of top k
        coords = coords[keep]
    
    return coords
```

#### Implementation Details

**Maximum Filter:**
- Uses `scipy.ndimage.maximum_filter` with `size` parameter
- `size=[2r+1, 2r+1, ...]` defines hypercube kernel
- `mode='nearest'` handles boundaries by replicating border values
- Time complexity: O(n_voxels × kernel_size^ndim) with optimized sliding window

**Peak Identification:**
- `img == max_f`: Finds voxels equal to neighborhood maximum
- Multiple voxels can satisfy this if they have identical maximum value
- Threshold filter reduces false positives from noise

**Top-k Selection:**
- Sorts peaks by intensity (descending)
- Selects strongest k peaks
- Time complexity: O(N log N) for sorting

**Boundary Handling:**
- `mode='nearest'` replicates border values
- Prevents border voxels from becoming spurious peaks due to padding
- Alternative modes: 'constant', 'reflect', 'wrap' (not used)

#### API Specification

```python
def local_maxima(
    img: np.ndarray,
    radius: int,
    thresh: float,
    top_k: Optional[int]
) -> np.ndarray
```

**Parameters:**
- `img` (np.ndarray): Input n-dimensional image
- `radius` (int): Half-width of L∞ neighborhood (minimum 1)
- `thresh` (float): Minimum intensity threshold
- `top_k` (Optional[int]): Maximum number of peaks to return (None = unlimited)

**Returns:**
- `np.ndarray`: Integer coordinates of shape `(N, ndim)`

### Spatial Deduplication: `dedupe_farthest_first()`

Removes spatially redundant candidates using farthest-first selection with quality priority.

#### Algorithm: Farthest-First Greedy Selection

The farthest-first algorithm ensures **maximum spatial diversity** while respecting quality priorities:

```
Input:
  - coords: (N, ndim) candidate coordinates
  - min_distance: minimum distance constraint
  - intensities: (N,) optional quality values

Steps:
1. Sort candidates by intensity (descending) if provided
   - Ensures high-quality candidates are considered first

2. Initialize selected set with first (highest quality) candidate
   - S = {coords[0]}

3. Build KD-tree from selected set for efficient queries
   - tree = KDTree(S)

4. For each remaining candidate in order:
   a. Query KD-tree for nearest distance to selected set
      dist = tree.query(candidate, k=1)
   
   b. If dist >= min_distance:
      - Candidate satisfies constraint
      - Add to candidate pool
   
5. Among valid candidates, select FARTHEST one
   - candidate* = argmax(distances)
   - Add to selected set: S = S ∪ {candidate*}
   - Rebuild KD-tree with updated S

6. Repeat until no valid candidates remain

Output:
  - Selected candidates with maximum spatial diversity
```

#### Why Farthest-First?

**Comparison with Greedy Sequential:**

**Greedy Sequential (naive):**
```python
for candidate in sorted_by_quality:
    if all(distance(candidate, s) >= min_distance for s in selected):
        selected.append(candidate)
```
- Problem: Can create spatial clusters
- Early selections can "block" entire regions
- No guarantee of spatial coverage

**Farthest-First:**
```python
while candidates_remain:
    distances = [min_distance_to_selected(c) for c in candidates]
    farthest = argmax(distances)
    if distances[farthest] >= min_distance:
        selected.append(farthest)
```
- Actively maximizes spatial spread
- Each selection considers global spatial distribution
- Guaranteed maximum diversity given constraints

**Visual Example (2D):**
```
Greedy Sequential:        Farthest-First:
    ×  ×  ×  ×                 ×     ×
    ×  ×  ×  ×                 
                               ×     ×
    ×  ×  ×  ×                 
                               ×     ×
(clustered)                (well-distributed)
```

#### KD-Tree Optimization

**Naive Implementation:** O(N³)
```python
for i in range(N):                          # O(N)
    for s in selected:                      # O(M) where M ≤ N
        distance = sqrt(sum((c[i] - s)**2)) # O(ndim)
    # Total: O(N × M × ndim) ≈ O(N² × ndim) for greedy
    # For farthest-first: need all distances → O(N × M × ndim) per iteration → O(N² × M × ndim) ≈ O(N³)
```

**KD-Tree Implementation:** O(N × M × log M)
```python
tree = KDTree(selected)                     # O(M log M)
for i in range(remaining):                  # O(N)
    distance = tree.query(c[i], k=1)        # O(log M)
    # Total per iteration: O(remaining × log M)
    # Total: O(N × M × log M) where M is final selection size
```

**Performance Gain:**
- For N=1000, M=500: Naive ≈ 10⁹ operations, KD-tree ≈ 10⁶ operations
- Speedup: ~1000× for large datasets

**Small Dataset Optimization:**
For N < 50, KD-tree overhead isn't worth it → use simple greedy O(N²) fallback.

#### Algorithm Implementation

```python
def dedupe_farthest_first(
    coords: np.ndarray,
    min_distance: float,
    intensities: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Remove duplicate candidates using farthest-first selection.
    
    Returns: np.ndarray of shape (M, ndim) with M ≤ N (float coordinates)
    """
    # Handle empty input
    if len(coords) == 0:
        return coords.astype(float)
    
    # Small dataset: use simple O(N²) greedy
    if len(coords) < 50:
        return _dedupe_simple(coords, min_distance, intensities)
    
    # Sort by intensity if provided (highest first)
    if intensities is not None:
        sort_indices = np.argsort(intensities)[::-1]
        coords_sorted = coords[sort_indices].astype(float)
    else:
        coords_sorted = coords.astype(float)
    
    # Initialize with first (strongest) candidate
    selected = [coords_sorted[0]]
    selected_array = np.array(selected)
    tree = cKDTree(selected_array)
    
    # Track remaining candidates
    remaining_mask = np.ones(len(coords_sorted), dtype=bool)
    remaining_mask[0] = False
    
    # Farthest-first selection loop
    while True:
        remaining_indices = np.where(remaining_mask)[0]
        
        if len(remaining_indices) == 0:
            break  # No more candidates
        
        # Query KD-tree for all remaining candidates (vectorized)
        remaining_coords = coords_sorted[remaining_indices]
        distances, _ = tree.query(remaining_coords, k=1)
        
        # Find candidates satisfying min_distance
        valid_mask = distances >= min_distance
        
        if not np.any(valid_mask):
            break  # No more valid candidates
        
        # Among valid candidates, pick the FARTHEST one
        valid_distances = distances[valid_mask]
        valid_indices_in_remaining = np.where(valid_mask)[0]
        farthest_idx_in_valid = np.argmax(valid_distances)
        farthest_idx_in_remaining = valid_indices_in_remaining[farthest_idx_in_valid]
        farthest_idx_global = remaining_indices[farthest_idx_in_remaining]
        
        # Add farthest candidate to selected set
        selected.append(coords_sorted[farthest_idx_global])
        remaining_mask[farthest_idx_global] = False
        
        # Rebuild KD-tree with all selected points
        selected_array = np.array(selected)
        tree = cKDTree(selected_array)
    
    return np.array(selected, dtype=float)
```

#### API Specification

```python
def dedupe_farthest_first(
    coords: np.ndarray,
    min_distance: float,
    intensities: Optional[np.ndarray] = None
) -> np.ndarray
```

**Parameters:**
- `coords` (np.ndarray): Candidate coordinates of shape `(N, ndim)`
- `min_distance` (float): Minimum Euclidean distance strictly enforced
- `intensities` (Optional[np.ndarray]): Quality values of shape `(N,)` for priority sorting

**Returns:**
- `np.ndarray`: Deduplicated coordinates of shape `(M, ndim)` where M ≤ N (float dtype)

**Guarantees:**
- All pairwise distances ≥ min_distance: `∀i,j: ||coords[i] - coords[j]|| ≥ min_distance`
- Maximum spatial diversity among valid selections
- Quality ordering respected (if intensities provided)

### Candidate Combination: `combine_seeds()`

Merges candidate arrays from multiple detection methods.

#### Algorithm

```python
def combine_seeds(
    *candidate_arrays: np.ndarray,
    min_distance: Optional[float] = None,
    method: str = "union",
) -> np.ndarray:
    """
    Combine candidates from multiple detection methods.
    
    Returns: np.ndarray of shape (M, ndim)
    """
    if method != "union":
        raise ValueError(f"Unknown method: {method}")
    
    # Filter out empty arrays
    non_empty = [arr for arr in candidate_arrays if len(arr) > 0]
    
    if len(non_empty) == 0:
        # Return empty array with correct shape
        for arr in candidate_arrays:
            if arr is not None:
                return np.zeros((0, arr.shape[1]), dtype=float)
        return np.zeros((0, 2), dtype=float)  # Default to 2D
    
    # Concatenate all non-empty arrays
    combined = np.vstack(non_empty)
    
    # Optionally deduplicate
    if min_distance is not None:
        combined = dedupe_farthest_first(combined, min_distance=min_distance)
    
    return combined
```

#### API Specification

```python
def combine_seeds(
    *candidate_arrays: np.ndarray,
    min_distance: Optional[float] = None,
    method: str = "union",
) -> np.ndarray
```

**Parameters:**
- `*candidate_arrays` (np.ndarray): Variable number of candidate arrays, each shape `(N_i, ndim)`
- `min_distance` (Optional[float]): If provided, deduplicate merged candidates
- `method` (str): Combination method (currently only "union" supported)

**Returns:**
- `np.ndarray`: Combined candidates of shape `(M, ndim)`

**Future Extensions:**
- `method="intersection"`: Keep only candidates appearing in multiple methods
- `method="weighted"`: Weight candidates by detection confidence
- Priority ordering for deduplication

## Algorithmic Comparisons

### Philosophical Differences

| Aspect | Multiscale Gaussian | Decomposition |
|--------|-------------------|---------------|
| **Philosophy** | Overcomplete detection | Hierarchical selection |
| **Scale Treatment** | Independent Gaussian filtering | Optimized energy separation |
| **Feature Overlap** | Redundant across scales | Non-overlapping by design |
| **Noise Handling** | CLAHE preprocessing | Scale filtering (ignore_finest_k) |
| **Candidate Count** | Many (1000-10000+) | Fewer (100-1000) |
| **Quality Metric** | Coverage completeness | Energy-based importance |
| **Computational Cost** | Fast (1-15 seconds) | Slower (30-60 seconds) |

### Performance Trade-offs

#### Speed vs Quality

**Multiscale Gaussian:**
- **Speed**: Fast Gaussian filtering + peak detection
- **Quality**: High coverage, some redundancy
- **Use case**: When speed matters, comprehensive coverage needed

**Decomposition:**
- **Speed**: Slow decomposition + fast peak detection
- **Quality**: Principled selection, sparse representation
- **Use case**: When quality matters, computational budget allows

#### Coverage vs Sparsity

**Multiscale Gaussian:**
- Dense candidate coverage
- Multiple candidates per feature (at different scales)
- Downstream fitting selects best subset
- Higher memory usage

**Decomposition:**
- Sparse candidate distribution
- One candidate per feature per scale
- Pre-filtered by decomposition quality
- Lower memory usage

### When to Use Which Method

**Use Multiscale Gaussian When:**
1. **Speed is critical**: Need candidates quickly (<15 seconds)
2. **Comprehensive coverage**: Want to ensure no features missed
3. **Downstream selection**: Fitting process will prune redundancy
4. **Heterogeneous data**: CLAHE preprocessing valuable
5. **Complex scenes**: Mixed feature types benefit from multiple detection methods

**Use Decomposition When:**
1. **Quality matters**: Computational budget allows thorough decomposition
2. **Noisy data**: Scale filtering provides robust noise suppression
3. **Hierarchical structure**: Want explicit coarse-to-fine ordering
4. **Sparse representation**: Prefer fewer, higher-quality seeds
5. **Energy-based priority**: Want seeds sorted by importance
6. **Principled approach**: Value mathematically-motivated feature separation

**Use Both (Combined) When:**
1. **Critical application**: Maximum robustness required
2. **Unknown data characteristics**: Hedge against method weaknesses
3. **Exploration**: Compare and evaluate both approaches
4. **Research**: Studying method differences and complementarity

### Complementarity

The two methods are **complementary**, not competing:

**Multiscale Gaussian strengths address Decomposition weaknesses:**
- Fast when decomposition is slow
- Dense coverage when decomposition is sparse
- Multiple detection methods vs single decomposition approach

**Decomposition strengths address Multiscale Gaussian weaknesses:**
- Principled scale separation vs overlapping Gaussian scales
- Energy-based quality vs simple intensity thresholding
- Explicit noise filtering vs preprocessing

**Combining Both:**
```python
# Generate seeds from both methods (returns GSplatData)
seeds_gaussian = seed_from_gaussian(image, min_distance=3.0)
seeds_decomp = seed_from_decomposition(image, scales=[1,2,4,8])

# Combine centers with deduplication
combined_centers = combine_seeds(
    seeds_gaussian.centers,
    seeds_decomp.centers,
    min_distance=3.0
)

# Use for fitting (positions only - fitter initializes shapes)
result = fit_gaussian_splats(image, seeds=combined_centers)
```

**Result:**
- Coarse structure from decomposition (high energy, principled)
- Fine detail from multiscale Gaussian (comprehensive coverage)
- Robust to failures of either individual method

## Testing Strategy

### Unit Tests

#### Multiscale Gaussian Tests

**Basic Functionality:**
```python
def test_multiscale_gaussian_2d():
    """Test basic 2D seed generation."""
    # Create synthetic 2D image with known features
    image = create_synthetic_blobs_2d(n_blobs=10, size=128)

    # Returns GSplatData with scale-informed shapes
    seeds = seed_from_gaussian(
        image,
        scales=(1.0, 2.0, 4.0),
        min_distance=3.0,
        apply_clahe=False,
    )

    # Should find approximately n_blobs seeds (±tolerance)
    assert 5 <= len(seeds.centers) <= 15
    assert seeds.centers.shape[1] == 2  # 2D coordinates
    assert seeds.centers.dtype == float
```

**Parameter Validation:**
```python
def test_multiscale_gaussian_parameter_validation():
    """Test parameter validation."""
    image = np.random.rand(64, 64)
    
    # Empty scales
    with pytest.raises(ValueError, match="scales must be a non-empty"):
        seed_from_gaussian(image, scales=[])
    
    # Negative scale
    with pytest.raises(ValueError, match="All scale values must be positive"):
        seed_from_gaussian(image, scales=[1.0, -2.0])
    
    # Invalid percentile
    with pytest.raises(ValueError, match="percentile_thresh must be between 0 and 100"):
        seed_from_gaussian(image, percentile_thresh=150.0)
```

**Edge Cases:**
```python
def test_multiscale_gaussian_edge_cases():
    """Test edge cases."""
    # Uniform image → no peaks
    uniform = np.ones((64, 64))
    seeds = seed_from_gaussian(uniform)
    assert len(seeds.centers) == 0

    # Single peak → one seed
    single_peak = np.zeros((64, 64))
    single_peak[32, 32] = 1.0
    seeds = seed_from_gaussian(single_peak, min_distance=1.0)
    assert len(seeds.centers) >= 1
    assert np.allclose(seeds.centers[0], [32, 32], atol=2.0)
```

#### Decomposition Tests

**Basic Functionality:**
```python
def test_decomposition_2d():
    """Test decomposition-based seed generation in 2D."""
    image = create_synthetic_blobs_2d(n_blobs=10, size=128)

    # Returns GSplatData with scale-informed shapes
    seeds = seed_from_decomposition(
        image,
        scales=[1, 2, 4],
        ignore_finest_k=1,
        min_distance=3.0,
        decompose_kwargs={'n_iters': 200},  # Fast for testing
    )

    # Should find seeds (count depends on decomposition quality)
    assert len(seeds.centers) > 0
    assert seeds.centers.shape[1] == 2
    assert seeds.centers.dtype == float
```

**ignore_finest_k Validation:**
```python
def test_decomposition_ignore_finest_k():
    """Test ignore_finest_k parameter."""
    image = create_synthetic_blobs_2d(n_blobs=5, size=64)

    # ignore_finest_k = 0: use all scales
    seeds_all = seed_from_decomposition(
        image, scales=[1, 2, 4], ignore_finest_k=0
    )

    # ignore_finest_k = 1: skip finest scale
    seeds_skip1 = seed_from_decomposition(
        image, scales=[1, 2, 4], ignore_finest_k=1
    )

    # ignore_finest_k = 2: skip two finest scales
    seeds_skip2 = seed_from_decomposition(
        image, scales=[1, 2, 4], ignore_finest_k=2
    )

    # More filtering → fewer or equal seeds
    assert len(seeds_skip1.centers) <= len(seeds_all.centers)
    assert len(seeds_skip2.centers) <= len(seeds_skip1.centers)
    
    # Validate warning for ignore_finest_k >= len(scales)
    with pytest.warns(UserWarning):
        seed_from_decomposition(image, scales=[1, 2], ignore_finest_k=3)
```

#### Utility Tests

**local_maxima Tests:**
```python
def test_local_maxima_basic():
    """Test basic peak detection."""
    # Create image with 4 peaks at known locations
    img = np.zeros((32, 32))
    peaks_true = [(8, 8), (8, 24), (24, 8), (24, 24)]
    for y, x in peaks_true:
        img[y, x] = 1.0
    
    # Detect peaks
    peaks = local_maxima(img, radius=1, thresh=0.5, top_k=None)
    
    # Should find all 4 peaks
    assert len(peaks) == 4
    
    # Should match known locations (order may vary)
    peaks_set = set(map(tuple, peaks))
    assert peaks_set == set(peaks_true)

def test_local_maxima_top_k():
    """Test top-k limiting."""
    # Create image with intensity gradient
    img = np.random.rand(32, 32)
    
    peaks_all = local_maxima(img, radius=1, thresh=0.0, top_k=None)
    peaks_10 = local_maxima(img, radius=1, thresh=0.0, top_k=10)
    
    assert len(peaks_10) == 10
    assert len(peaks_10) <= len(peaks_all)
    
    # Top 10 should be strongest peaks
    vals_10 = img[tuple(peaks_10.T)]
    vals_all = img[tuple(peaks_all.T)]
    assert np.min(vals_10) >= np.percentile(vals_all, 50)  # At least median
```

**dedupe_farthest_first Tests:**
```python
def test_dedupe_farthest_first_basic():
    """Test basic deduplication."""
    # Create candidates with known distances
    coords = np.array([
        [0, 0],
        [1, 0],  # Close to first (distance 1)
        [10, 0], # Far from first (distance 10)
    ], dtype=float)
    
    # min_distance = 5.0 should keep only [0,0] and [10,0]
    deduped = dedupe_farthest_first(coords, min_distance=5.0)
    
    assert len(deduped) == 2
    assert np.allclose(deduped[0], [0, 0])
    assert np.allclose(deduped[1], [10, 0])

def test_dedupe_with_intensities():
    """Test deduplication with quality priority."""
    coords = np.array([
        [0, 0],
        [1, 0],
        [10, 0],
    ], dtype=float)
    
    intensities = np.array([0.5, 1.0, 0.3])  # Middle one is strongest
    
    deduped = dedupe_farthest_first(coords, min_distance=5.0, intensities=intensities)
    
    # Strongest candidate [1,0] should be selected first
    # Then [10,0] is far enough away
    assert len(deduped) == 2
    assert np.allclose(deduped[0], [1, 0])  # Highest intensity selected first
```

**combine_seeds Tests:**
```python
def test_combine_seeds_basic():
    """Test basic candidate combination."""
    cand1 = np.array([[0, 0], [10, 10]], dtype=float)
    cand2 = np.array([[5, 5], [15, 15]], dtype=float)
    
    # No deduplication
    combined = combine_seeds(cand1, cand2, min_distance=None)
    assert len(combined) == 4
    
    # With deduplication
    combined = combine_seeds(cand1, cand2, min_distance=8.0)
    # [0,0], [10,10], [5,5], [15,15] → pairwise distances vary
    # Should keep well-separated subset
    assert len(combined) <= 4
```

### Integration Tests

**Integration with fit_gaussian_splats:**
```python
def test_seeds_integration_with_fitting():
    """Test seeds integrate properly with fitting pipeline."""
    image = create_synthetic_blobs_2d(n_blobs=10, size=128)

    # Generate seeds (returns GSplatData)
    seeds = seed_from_gaussian(image, min_distance=3.0)

    # Fit Gaussian splats - passes GSplatData directly
    result = fit_gaussian_splats(
        image,
        seeds=seeds,  # GSplatData with scale-informed shapes
        n_iters=100,
        enable_dynamic_ops=False,  # Use only provided seeds
    )

    # Should have fitted splats
    assert len(result.amplitudes) > 0
    assert len(result.amplitudes) <= len(seeds.centers)  # May prune some
```

**Cross-Method Comparison:**
```python
def test_methods_comparison():
    """Compare multiscale Gaussian vs decomposition on same image."""
    image = create_synthetic_blobs_2d(n_blobs=20, size=256)

    # Both methods return GSplatData with scale-informed shapes
    seeds_gaussian = seed_from_gaussian(
        image,
        scales=(1.0, 2.0, 4.0, 8.0),
        min_distance=3.0,
    )

    seeds_decomp = seed_from_decomposition(
        image,
        scales=[1, 2, 4, 8],
        ignore_finest_k=1,
        min_distance=3.0,
        decompose_kwargs={'n_iters': 200},
    )

    # Both should find seeds
    assert len(seeds_gaussian.centers) > 0
    assert len(seeds_decomp.centers) > 0

    # Gaussian typically finds more (overcomplete)
    # But not guaranteed (depends on parameters)
    print(f"Gaussian: {len(seeds_gaussian.centers)}, Decomposition: {len(seeds_decomp.centers)}")
```

### Dimension-Agnostic Tests

**1D, 2D, 3D, 4D:**
```python
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_multiscale_gaussian_ndim(ndim):
    """Test multiscale Gaussian in various dimensions."""
    shape = (32,) * ndim
    image = create_synthetic_blobs_nd(n_blobs=5, shape=shape)

    seeds = seed_from_gaussian(
        image,
        scales=(2.0, 4.0),
        min_distance=3.0,
    )

    assert seeds.centers.shape[1] == ndim
    assert len(seeds.centers) > 0

@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_decomposition_ndim(ndim):
    """Test decomposition in various dimensions."""
    shape = (32,) * ndim
    image = create_synthetic_blobs_nd(n_blobs=5, shape=shape)

    seeds = seed_from_decomposition(
        image,
        scales=[1, 2, 4],
        ignore_finest_k=1,
        decompose_kwargs={'n_iters': 100},
    )

    assert seeds.centers.shape[1] == ndim
    assert len(seeds.centers) > 0
```

## Implementation Design Decisions

### Coordinate Precision: Integer vs Float

**Decision**: All candidate functions return **float coordinates**.

**Rationale:**
1. **Sub-voxel precision**: Refinement steps produce non-integer positions
2. **Downstream compatibility**: Gaussian fitting expects float positions
3. **Consistency**: Uniform interface across all methods
4. **Future extensibility**: Allows arbitrary precision improvements

**Impact:**
- All utilities (`local_maxima`, `dedupe_farthest_first`) handle integer coordinates internally but return float
- Coordinate mapping in decomposition produces float coordinates naturally

### CLAHE Integration: Preprocessing vs Detection

**Decision**: When `apply_clahe=True`, **all processing** uses CLAHE-enhanced image.

**Rationale:**
1. **Consistency**: Detection and refinement see same features
2. **Effectiveness**: Sub-voxel refinement should match detected features
3. **Simplicity**: Single working image avoids dual-path complexity

**Alternative Considered:**
- Apply CLAHE only for detection, refine on original image
- Rejected: Sub-voxel refinement would not match detected peaks

### Deduplication: Farthest-First vs Greedy

**Decision**: Use **farthest-first** selection for spatial deduplication.

**Rationale:**
1. **Spatial diversity**: Actively maximizes spatial spread
2. **Global optimization**: Considers entire selected set
3. **Quality**: Avoids spatial clustering artifacts
4. **Empirical performance**: Better coverage in practice

**Alternative Considered:**
- Simple greedy: Keep candidates passing min_distance check in order
- Rejected: Creates spatial clusters, poor coverage

### Scale Processing Order: Coarse-to-Fine

**Decision**: Process scales from **coarse to fine** (reversed order).

**Rationale:**
1. **Priority ordering**: Coarse features selected first in deduplication
2. **Hierarchical philosophy**: Large structures before fine details
3. **Consistency**: Matches decomposition energy hierarchy
4. **Robustness**: Coarse features more stable, less noise-sensitive

**Impact:**
- Both methods process scales in same order
- Farthest-first deduplication prioritizes earlier (coarser) candidates

### Decomposition: ignore_finest_k Default

**Decision**: Default `ignore_finest_k=1` (skip full-resolution scale).

**Rationale:**
1. **Noise suppression**: Full-resolution scale dominated by noise
2. **Overfitting prevention**: Too many fine candidates harm convergence
3. **Empirical validation**: Testing shows k=1 optimal for most data
4. **Adjustability**: Users can override for clean synthetic data (k=0)

**Alternative Defaults Considered:**
- `k=0`: Use all scales - rejected (too noisy)
- `k=2`: Aggressive filtering - rejected (loses too much detail)

### API Design: Shared Parameters

**Decision**: Use **consistent parameter names** across both methods.

**Common Parameters:**
- `min_distance`: Same meaning in both methods
- `peaks_per_scale`: Same limiting behavior
- Return type: Always `np.ndarray` of shape `(N, ndim)`

**Method-Specific Parameters:**
- Multiscale Gaussian: `scales` (floats, Gaussian σ), `apply_clahe`, etc.
- Decomposition: `scales` (ints, scale factors), `ignore_finest_k`, etc.

**Rationale:**
1. **Learnability**: Similar concepts use same names
2. **Interchangeability**: Easy to switch between methods
3. **Clarity**: Method-specific parameters clearly distinguished

## Future Extensions

### Hybrid Approaches

**Motivation**: Combine strengths of both methods.

**Proposal 1: Two-Stage Detection**
```python
def seed_hybrid(V, **kwargs):
    # Stage 1: Decomposition for coarse structure (scales 8, 16, 32)
    coarse = seed_from_decomposition(
        V, scales=[8, 16, 32], ignore_finest_k=0
    )

    # Stage 2: Gaussian for fine detail (scales 1.0, 2.0, 4.0)
    fine = seed_from_gaussian(
        V, scales=(1.0, 2.0, 4.0)
    )

    # Combine centers with deduplication
    combined_centers = combine_seeds(
        coarse.centers, fine.centers, min_distance=kwargs['min_distance']
    )
    # Note: Loses shape info - consider preserving GSplatData
    return combined_centers
```

**Proposal 2: Adaptive Method Selection**
```python
def seed_adaptive(V, **kwargs):
    # Analyze image characteristics
    snr = estimate_snr(V)
    complexity = estimate_complexity(V)

    # All methods return GSplatData with scale-informed shapes
    if snr < 5.0:  # Noisy
        return seed_from_decomposition(V, ignore_finest_k=2)
    elif complexity > 0.7:  # Complex
        return seed_from_gaussian(V, apply_clahe=True)
    else:  # Default
        return seed_from_decomposition(V, ignore_finest_k=1)
```

### Adaptive Parameters

**Motivation**: Automatically tune parameters based on data characteristics.

**Proposal 1: Adaptive ignore_finest_k**
```python
def auto_ignore_finest_k(scale_images, scales):
    """
    Determine optimal ignore_finest_k based on energy distribution.
    
    Idea: Skip scales where energy is dominated by noise (high frequency, low structure).
    """
    energies = [scale_img.sum() for scale_img in scale_images]
    
    # Compute normalized gradient of energy distribution
    energy_gradient = np.diff(energies) / energies[:-1]
    
    # Find first scale where gradient stabilizes
    # (transition from noise-dominated to structure-dominated)
    threshold_gradient = 0.1
    for k, grad in enumerate(energy_gradient):
        if abs(grad) < threshold_gradient:
            return k
    
    return 1  # Default
```

**Proposal 2: Adaptive min_distance**
```python
def auto_min_distance(V, target_density=0.001):
    """
    Determine min_distance to achieve target candidate density.
    
    target_density: candidates per voxel (e.g., 0.001 → 1 candidate per 1000 voxels)
    """
    n_voxels = V.size
    n_candidates_target = int(n_voxels * target_density)
    
    # Estimate required min_distance via binary search
    # (requires iterative candidate generation - expensive)
    # Simplified: use heuristic based on image size
    ndim = V.ndim
    voxels_per_candidate = 1 / target_density
    min_distance = (voxels_per_candidate) ** (1 / ndim)
    
    return float(min_distance)
```

### Multi-Channel Support

**Motivation**: Handle multi-channel images (e.g., RGB, multi-fluorescence).

**Proposal: Per-Channel Decomposition + Merging**
```python
def seed_multichannel(V_channels, **kwargs):
    """
    Generate seeds from multi-channel image.

    V_channels: list of nD arrays (one per channel)
    Returns: combined centers (positions only - shape info lost)
    """
    all_centers = []

    for channel_idx, V in enumerate(V_channels):
        # Decompose each channel independently (returns GSplatData)
        seeds = seed_from_decomposition(V, **kwargs)
        all_centers.append(seeds.centers)

    # Combine centers across channels with deduplication
    combined = combine_seeds(*all_centers, min_distance=kwargs['min_distance'])

    return combined
```

**Alternative: Joint Decomposition**
- Decompose multi-channel image as single tensor
- Requires extension of decomposition algorithm
- More principled but more complex

### Confidence Scores

**Motivation**: Provide quality metrics for each seed.

**Proposal: Multi-Factor Confidence**
```python
def seed_with_confidence(V, **kwargs):
    """
    Return seeds with confidence scores.

    Returns: (GSplatData, confidences)
    """
    seeds = seed_from_gaussian(V, **kwargs)
    centers = seeds.centers

    confidences = []
    for c in centers:
        # Factor 1: Peak intensity
        intensity = V[tuple(c.astype(int))]
        
        # Factor 2: Local contrast (peak vs neighborhood mean)
        neighborhood = extract_neighborhood(V, c, radius=5)
        contrast = intensity / (neighborhood.mean() + 1e-6)
        
        # Factor 3: Peak sharpness (second derivative magnitude)
        sharpness = compute_laplacian_magnitude(V, c)
        
        # Combine factors (weighted)
        confidence = 0.5 * intensity + 0.3 * contrast + 0.2 * sharpness
        confidences.append(confidence)
    
    confidences = np.array(confidences)
    
    return candidates, confidences
```

**Use Case:**
- Prioritize high-confidence candidates in fitting
- Threshold low-confidence candidates for speed
- Visualize confidence for diagnostics

### GPU Acceleration

**Motivation**: Speed up candidate generation for large volumes.

**Proposal 1: GPU Gaussian Filtering**
```python
def seed_from_gaussian_gpu(V, **kwargs):
    """
    GPU-accelerated multiscale Gaussian candidate generation.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter
    
    # Transfer to GPU
    V_gpu = cp.asarray(V)
    
    all_coords = []
    for s in kwargs['scales']:
        # GPU Gaussian filtering
        img_gpu = gpu_gaussian_filter(V_gpu, sigma=s)
        
        # GPU peak detection (custom kernel)
        peaks_gpu = gpu_local_maxima(img_gpu, ...)
        
        # Transfer back to CPU
        peaks = cp.asnumpy(peaks_gpu)
        all_coords.append(peaks)
    
    # Rest of processing on CPU (deduplication, refinement)
    ...
```

**Expected Speedup**: 5-10× for large 3D volumes (512³+)

**Proposal 2: GPU Decomposition**
- Already supported via PyTorch (decompose_image uses torch)
- Automatically uses GPU if available

### Learned Candidate Generation

**Motivation**: Use machine learning to predict optimal candidate locations.

**Proposal: CNN-Based Detection**
```python
class CandidateNetworkModel(nn.Module):
    """
    U-Net style network that predicts candidate probability maps.
    
    Input: Image/volume
    Output: Probability map (same size as input)
    """
    def __init__(self, ndim):
        # ... define U-Net architecture for ndim dimensions
        pass
    
    def forward(self, x):
        # ... forward pass producing probability map
        pass

def seed_learned(V, model, threshold=0.5):
    """
    Use trained neural network to predict seed locations.
    Returns centers only (positions without shape info).
    """
    # Predict probability map
    prob_map = model(torch.tensor(V))

    # Threshold and find peaks
    centers = local_maxima(prob_map.numpy(), radius=2, thresh=threshold)

    return centers
```

**Training:**
- Supervised: Train on images with known optimal candidate locations
- Unsupervised: Train to minimize reconstruction error in fitting pipeline
- Transfer learning: Pre-train on large dataset, fine-tune for specific modality

**Challenges:**
- Requires large training dataset
- Generalization across data types
- Computational cost of inference

## References

**Internal References:**
- **Multiscale Decomposition**: [multiscale/SPECIFICATIONS.md](../multiscale/SPECIFICATIONS.md)
- **CLAHE Preprocessing**: [clahe/SPECIFICATIONS.md](../clahe/SPECIFICATIONS.md)
- **Gaussian Splat Fitting**: [fitting/SPECIFICATIONS.md](../fitting/SPECIFICATIONS.md)
- **Main Fitting API**: [SPECIFICATIONS.md](../SPECIFICATIONS.md)

**External References:**
- **Scale-Space Theory**: Lindeberg, T. (1993). "Scale-space theory: A basic tool for analyzing structures at different scales." Journal of Applied Statistics.
- **CLAHE**: Pizer, S. M., et al. (1987). "Adaptive histogram equalization and its variations." Computer Vision, Graphics, and Image Processing.
- **KD-Tree**: Bentley, J. L. (1975). "Multidimensional binary search trees used for associative searching." Communications of the ACM.
- **Farthest-First Traversal**: Hochbaum, D. S., & Shmoys, D. B. (1985). "A best possible heuristic for the k-center problem." Mathematics of Operations Research.

## Changelog

- **v1.0 (2025-11-27)**: Initial comprehensive specification
  - Documented both multiscale Gaussian and decomposition methods
  - Detailed shared utilities (peak detection, deduplication, combination)
  - Algorithmic comparisons and performance trade-offs
  - Mathematical foundations and implementation details
  - Testing strategy and future extensions
  - Replaces previous single-method SPECIFICATION.md (backed up as SPECIFICATION.md.backup)
