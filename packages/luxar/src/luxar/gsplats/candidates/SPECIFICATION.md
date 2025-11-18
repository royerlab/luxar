# Specification: Decomposition-Based Candidate Generation

## Overview

This specification describes a new method for generating candidate Gaussian splat locations using multi-scale image decomposition. The approach leverages the `decompose_image()` function to decompose an image into multiple scales, then identifies local maxima across scales as candidate splat centers.

## Motivation

### Current Approach Limitations
The existing `find_candidates_multiscale_gaussian()` function uses:
1. Multiscale Gaussian-blurred peaks
2. Difference of Gaussians (DoG)
3. Intensity-weighted grid sampling
4. CLAHE preprocessing

While effective, this approach:
- Detects features at predefined Gaussian scales
- May oversample in homogeneous regions
- Doesn't explicitly leverage hierarchical energy decomposition

### Decomposition-Based Advantages
Using `decompose_image()` offers:
1. **Principled scale separation**: Energy is explicitly distributed across scales
2. **Sparse representation**: Each scale captures distinct features (not redundant blurred versions)
3. **Coarse-to-fine hierarchy**: Natural ordering from global structure to fine detail
4. **Adaptive complexity**: Number of candidates reflects intrinsic image complexity
5. **Energy-weighted sampling**: High-energy regions naturally generate more candidates

## Proposed Approach

### Algorithm

```
Input:
  - V: n-dimensional image/volume
  - scales: scale factors for decomposition (default [1, 2, 4, 8, 16, 32])
  - ignore_finest_k: number of finest scales to ignore (default 1)
  - other decompose_image parameters

Steps:
1. Decompose image: [V_1, V_2, ..., V_k] = decompose_image(V, scales)
2. For scales at indices [ignore_finest_k, ..., len(scales)-1]:
   a. Find local maxima in V_i using peak detection
   b. Map peak locations to full resolution coordinates
   c. Record intensity/energy at each peak
3. Deduplicate spatially close peaks (min_dist criterion)
4. Sort by energy/intensity (descending)
5. Return top N candidates or all above threshold

Output:
  - candidates: np.ndarray of shape (N, ndim) with candidate centers in voxel coordinates
```

### Rationale for Ignoring Finest Scales

**Why ignore the finest k scales?**
1. **Noise suppression**: Finest scales (especially scale=1) capture high-frequency noise
2. **Overfitting prevention**: Too many fine-detail candidates lead to overfitting
3. **Computational efficiency**: Fewer candidates = faster convergence
4. **Coarse-to-fine bias**: Start with global structure, refine with dynamic seeding

**Default k=1**: Skip only the finest (full resolution) scale, balancing detail capture and noise robustness.

## API Design

### Primary Function

```python
def find_candidates_from_decomposition(
    V: np.ndarray,
    scales: List[int] = [1, 2, 4, 8, 16, 32],
    ignore_finest_k: int = 1,
    peaks_per_scale: Optional[int] = None,
    min_distance: float = 2.0,
    threshold_rel: float = 0.1,
    decompose_kwargs: Optional[dict] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Generate candidate Gaussian splat locations using multi-scale decomposition.

    This method decomposes the input image into multiple scales, finds local maxima
    in each scale (excluding the finest k scales), and returns their positions as
    candidate splat centers.

    Parameters
    ----------
    V : np.ndarray
        Input n-dimensional image/volume. Shape: (s_0, s_1, ..., s_{n-1}).
    scales : List[int], optional
        Scale factors for decomposition. Default: [1, 2, 4, 8, 16, 32].
        Scale 1 = full resolution, scale 2 = half resolution, etc.
    ignore_finest_k : int, optional
        Number of finest scales to ignore for peak detection. Default: 1.
        Setting k=1 ignores the full-resolution scale to suppress noise.
    peaks_per_scale : int or None, optional
        Maximum number of peaks to extract per scale. If None, extract all peaks
        above threshold. Default: None.
    min_distance : float, optional
        Minimum Euclidean distance between candidates (in voxels). Default: 2.0.
        Closer candidates are deduplicated, keeping the higher-energy peak.
    threshold_rel : float, optional
        Relative threshold for peak detection (0.0 to 1.0). Default: 0.1.
        Peaks must be at least threshold_rel * max_intensity to be considered.
    decompose_kwargs : dict or None, optional
        Additional keyword arguments passed to decompose_image().
        Common options:
        - n_iters: optimization iterations (default 500)
        - energy_weight: hierarchical energy penalty (default 0.01)
        - loss_type: "l1" (default), "mse", or "poisson"
    verbose : bool, optional
        Print progress information. Default: False.

    Returns
    -------
    candidates : np.ndarray, shape (N, ndim)
        Candidate center coordinates in voxel units (float).
        Sorted by energy (descending).

    Notes
    -----
    - Ignoring the finest k scales (default k=1) suppresses noise and overfitting
    - Scales are processed from coarse to fine; coarser scales contribute first
    - Candidates are deduplicated spatially using farthest-first with min_distance
    - Peak positions are mapped from scale resolution to full resolution
    - Energy-based sorting ensures high-quality candidates are prioritized

    Examples
    --------
    >>> from skimage import data
    >>> import numpy as np
    >>> from luxar.gsplats.candidates import find_candidates_from_decomposition
    >>>
    >>> # Load example image
    >>> image = data.cell().astype(np.float32)
    >>>
    >>> # Generate candidates (ignore finest scale to suppress noise)
    >>> candidates = find_candidates_from_decomposition(
    ...     image,
    ...     scales=[1, 2, 4, 8],
    ...     ignore_finest_k=1,
    ...     min_distance=3.0,
    ...     verbose=True
    ... )
    >>> print(f"Generated {len(candidates)} candidate locations")

    See Also
    --------
    decompose_image : Multi-scale image decomposition
    find_candidates_multiscale_gaussian : Alternative overcomplete candidate generation
    """
```

### Integration with fit_gaussian_splats

The new function seamlessly integrates with existing workflow:

```python
from luxar.gsplats import fit_gaussian_splats
from luxar.gsplats.candidates import find_candidates_from_decomposition

# Generate candidates using decomposition
candidates = find_candidates_from_decomposition(
    image,
    scales=[1, 2, 4, 8, 16],
    ignore_finest_k=1,
)

# Use candidates to seed Gaussian splat fitting
params, amps, stats = fit_gaussian_splats(
    image,
    seeds=candidates,  # Pass explicit candidate positions
    n_iters=1000,
)
```

## Implementation Details

### Peak Detection Strategy

For each scale image V_scale:

1. **Local maximum detection**:
   - Use `scipy.ndimage.maximum_filter` with footprint size ~3x3x... (ndim-dependent)
   - Peak condition: `V_scale[pos] == max_filter[pos] AND V_scale[pos] > threshold`

2. **Coordinate mapping**:
   - Scale image has shape: `(H/scale, W/scale, ...)`
   - Map peak position `(i, j, ...)` to full resolution: `(i*scale, j*scale, ...)`
   - Add scale/2 offset to map to region center: `(i*scale + scale/2, j*scale + scale/2, ...)`

3. **Energy recording**:
   - Record intensity value at peak: `energy = V_scale[i, j, ...]`
   - Use for sorting and deduplication priority

### Deduplication Algorithm

Use spatial distance-based deduplication:

```python
def deduplicate_candidates(
    candidates: np.ndarray,  # shape (N, ndim)
    energies: np.ndarray,    # shape (N,)
    min_distance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove spatially close candidates, keeping higher-energy peaks.

    Uses farthest-first greedy selection:
    1. Sort candidates by energy (descending)
    2. Keep first (highest energy)
    3. For each subsequent candidate:
       - If distance to all kept candidates >= min_distance: keep
       - Else: discard
    """
```

### Scale Processing Order

Process scales from **coarse to fine**:
- Start with largest scale factors (most downsampled)
- Progress to finer scales
- Rationale: Coarse scales capture global structure; fine scales add detail
- Matches energy hierarchy from decomposition

### Edge Case Handling

1. **No peaks found**: Return empty array with shape (0, ndim)
2. **Single scale**: Still apply threshold and deduplication
3. **ignore_finest_k >= len(scales)**: Warning + use all scales
4. **Anisotropic data**: min_distance applies in voxel coordinates (assumes isotropic)

## Testing Strategy

### Unit Tests

1. **Basic functionality**:
   - 2D synthetic image with known features at multiple scales
   - Verify correct number of candidates detected
   - Verify candidate locations match expected peaks

2. **Scale ignoring**:
   - Test ignore_finest_k parameter
   - Verify finest k scales are actually excluded
   - Test edge case: ignore_finest_k >= num_scales

3. **Deduplication**:
   - Test min_distance parameter
   - Verify spatially close candidates are merged
   - Verify higher-energy peak is retained

4. **Edge cases**:
   - Empty image (all zeros)
   - Uniform image (no peaks)
   - Single pixel image
   - 1D, 2D, 3D, 4D images

### Integration Tests

1. **With fit_gaussian_splats**:
   - Generate candidates using decomposition
   - Fit Gaussian splats using these candidates
   - Verify convergence and reconstruction quality

2. **Comparison with overcomplete method**:
   - Run both methods on same image
   - Compare number of candidates, spatial distribution
   - Compare final reconstruction quality

### Visual Tests (Demos)

1. **Mitosis demo**:
   - Show decomposition scales
   - Overlay candidate locations on original image
   - Color-code by scale of origin
   - Compare reconstruction with overcomplete seeding

2. **Scaling analysis**:
   - Test on images of varying complexity
   - Plot: num_candidates vs image complexity
   - Plot: reconstruction error vs ignore_finest_k

## Performance Considerations

### Computational Cost

**Decomposition**: O(n_iters * n_scales * n_voxels)
- Bottleneck: Iterative optimization in decompose_image()
- Typical: 500 iterations, 6 scales, 1-10M voxels → 1-30 seconds

**Peak Detection**: O(n_scales * n_voxels_per_scale)
- Fast: Maximum filtering is optimized in scipy
- Typical: <1 second

**Deduplication**: O(N³) naive farthest-first, O(N * M * log M) with KD-tree
  where M is number of selected candidates (worst case: O(N² log N))
- Typical: 1000-10000 candidates → 1-10 seconds with KD-tree optimization

**Total**: Dominated by decomposition (95%+ of time)

### Memory Usage

- Decomposition returns K scale images (all at different resolutions)
- Peak detection uses temporary arrays for max filtering
- Candidate storage: ~8 bytes per coordinate * ndim * N_candidates
- Typical: 10000 candidates in 3D → ~240KB

### Optimization Opportunities

1. **Caching**: Cache decomposition results for multiple candidate generation trials
2. **Parallel peak detection**: Process scales independently (trivially parallel)
3. **KD-tree optimization**: Already implemented for O(N * M * log M) deduplication
4. **Coarse-scale-only mode**: For rapid prototyping, use only coarsest 2-3 scales

## Future Extensions

1. **Adaptive ignore_finest_k**:
   - Automatically determine k based on SNR estimation
   - Skip scales where energy distribution is noise-dominated

2. **Energy-weighted sampling**:
   - Instead of taking all peaks, sample proportional to energy
   - Provides more candidates in high-energy regions

3. **Hybrid approach**:
   - Combine decomposition candidates (global structure) with overcomplete candidates (local features)
   - Best of both worlds

4. **Scale-specific parameters**:
   - Different threshold_rel per scale
   - Different peaks_per_scale per scale
   - Allows fine-tuning trade-off between coarse and fine structure

5. **Multi-channel support**:
   - Decompose each channel independently
   - Merge candidates across channels
   - Useful for multi-color/multi-modal data

## References

- `decompose_image()`: `/packages/luxar/src/luxar/gsplats/multiscale/decompose.py`
- `find_candidates_multiscale_gaussian()`: `/packages/luxar/src/luxar/gsplats/candidates.py`
- `fit_gaussian_splats()`: `/packages/luxar/src/luxar/gsplats/fit_gsplats.py`
- Multiscale decomposition paper: [TODO - add reference]

## Revision History

- 2025-01-16: Initial specification (v0.1)
