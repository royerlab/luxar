# CLAHE Subpackage Specification

**Version**: 1.0.0
**Last Updated**: 2025-11-27

## Overview

This subpackage provides a PyTorch-based implementation of CLAHE (Contrast Limited Adaptive Histogram Equalization) for arbitrary-dimensional tensors. CLAHE is a computer vision technique that enhances local contrast by performing histogram equalization on small tiles while preventing noise amplification through contrast limiting.

**Primary Use Cases:**
1. **Dynamic Operations**: Generating perceptually-balanced sampling distributions for coverage seeding
2. **Image Enhancement**: Preprocessing for visualization and feature detection
3. **Heterogeneous Data**: Handling images/volumes with varying background levels

## Mathematical Foundation

### Standard Histogram Equalization

Given an image with intensity values in range [I_min, I_max]:

1. **Compute histogram**: `H(i) = count of pixels with intensity i`
2. **Compute CDF**: `CDF(i) = Σ(j=0 to i) H(j)`
3. **Normalize CDF**: `CDF_norm(i) = (CDF(i) - CDF_min) / (CDF_max - CDF_min)`
4. **Map intensities**: `I_out = CDF_norm(I_in) × (I_max - I_min) + I_min`

**Result**: Histogram is "flattened" toward uniform distribution, enhancing global contrast.

**Problem**: Global approach fails for images with varying local characteristics.

### CLAHE Algorithm

**Key Innovation**: Perform histogram equalization **locally** on tiles, then apply **contrast limiting** to prevent noise amplification.

#### Step 1: Tile Decomposition

Divide volume into non-overlapping tiles of size `tile_size^d`:
- 2D: tiles of size `(tile_size, tile_size)`
- 3D: tiles of size `(tile_size, tile_size, tile_size)`
- nD: hypercubes of size `tile_size^d`

Number of tiles per dimension: `n_tiles_i = ceil(shape_i / tile_size)`

#### Step 2: Local Histogram Computation

For each tile `T`:
1. Compute local histogram `H_T(i)` using `nbins` bins spanning [V_min, V_max]
2. Bin width: `bin_width = (V_max - V_min) / nbins`

#### Step 3: Contrast Limiting

**Purpose**: Prevent over-amplification of noise in uniform regions.

**Algorithm**:
1. Calculate uniform height: `uniform_height = n_pixels_per_tile / nbins`
2. Calculate clip height: `clip_height = clip_limit × uniform_height`
3. Clip histogram: `H_clipped(i) = min(H_T(i), clip_height)`
4. Redistribute excess: `excess = Σ max(0, H_T(i) - clip_height)`
5. Add back uniformly: `H_final(i) = H_clipped(i) + excess / nbins`

**Effect**:
- `clip_limit = 1.0`: No enhancement (preserves original)
- `clip_limit = 2.0`: Moderate enhancement (recommended)
- `clip_limit = 4.0`: Aggressive enhancement (may amplify noise)

#### Step 4: Local CDF Computation

For each tile with clipped histogram `H_final`:
1. Compute CDF: `CDF_T(i) = Σ(j=0 to i) H_final(j)`
2. Normalize: `CDF_norm(i) = (CDF_T(i) - CDF_T_min) / (CDF_T_max - CDF_T_min)`

#### Step 5: Intensity Mapping

For each pixel `p` in tile `T`:
1. Find bin: `bin_idx = searchsorted(bin_edges[1:], intensity_value)`
2. Clamp to valid range: `bin_idx = clamp(bin_idx, 0, nbins-1)`
3. Apply mapping: `I_out = CDF_norm(bin_idx)`
4. Rescale: `I_out = I_out × (V_max - V_min) + V_min`

**Result**: Each tile has locally-equalized contrast, adapted to local intensity distribution.

### Optional: Interpolation (Not Implemented)

Standard CLAHE interpolates between neighboring tile CDFs for smooth transitions. For our use case (sampling), discontinuities are acceptable and we skip interpolation for speed.

## Implementation Details

### Function: `apply_clahe(V, tile_size=16, clip_limit=2.0, nbins=256)`

**Input**:
- `V`: PyTorch tensor of shape `(d1, d2, ..., dn)` (any dimensionality)
- `tile_size`: Size of tiles in voxels (int)
- `clip_limit`: Contrast limiting factor (float, range: 1.0-4.0)
- `nbins`: Number of histogram bins (int, typically 256)

**Output**:
- `V_clahe`: CLAHE-equalized tensor, same shape and device as `V`, dtype preserved

**Detailed Algorithm**:
```python
def apply_clahe(V, tile_size, clip_limit, nbins):
    shape = V.shape
    device = V.device
    dtype = V.dtype
    
    # Calculate tile grid dimensions
    n_tiles = tuple((s + tile_size - 1) // tile_size for s in shape)
    
    # Initialize output
    V_clahe = torch.zeros_like(V)
    
    # Get global min/max for consistent binning across all tiles
    V_min, V_max = V.min().item(), V.max().item()
    
    # Early exit: uniform image (tolerance: 1e-12)
    if V_max - V_min < 1e-12:
        return V.clone()
    
    # Iterate over all tiles using itertools.product
    for tile_idx in itertools.product(*[range(n) for n in n_tiles]):
        # Compute tile boundaries (handle edge tiles that may be smaller)
        tile_slice = tuple(
            slice(t * tile_size, min((t + 1) * tile_size, s))
            for t, s in zip(tile_idx, shape)
        )
        
        # Extract tile and flatten
        tile_data = V[tile_slice]
        tile_flat = tile_data.reshape(-1)
        
        # Compute histogram over [V_min, V_max]
        hist = torch.histc(tile_flat, bins=nbins, min=V_min, max=V_max)
        
        # Apply contrast limiting
        uniform_height = tile_flat.numel() / nbins
        clip_height = clip_limit * uniform_height
        excess = torch.clamp(hist - clip_height, min=0).sum()
        hist = torch.clamp(hist, max=clip_height)
        hist += excess / nbins  # Redistribute excess uniformly
        
        # Compute CDF
        cdf = torch.cumsum(hist, dim=0)
        
        # Normalize CDF (robust to edge cases)
        cdf_min = cdf[cdf > 0].min() if (cdf > 0).any() else 0
        cdf_range = cdf[-1] - cdf_min
        
        if cdf_range > 0:
            cdf_normalized = (cdf - cdf_min) / cdf_range
        else:
            cdf_normalized = cdf  # Degenerate case: no normalization
        
        # Map intensities through CDF
        # Create bin edges for searchsorted
        bin_edges = torch.linspace(V_min, V_max, nbins + 1, device=device)
        
        # Digitize tile values (find which bin each value falls into)
        # Uses searchsorted on bin_edges[1:] (right edges of bins)
        bin_indices = torch.searchsorted(bin_edges[1:], tile_flat.contiguous())
        bin_indices = torch.clamp(bin_indices, 0, nbins - 1)
        
        # Apply CDF mapping
        tile_equalized = cdf_normalized[bin_indices]
        
        # Reshape and store in output
        V_clahe[tile_slice] = tile_equalized.reshape(tile_data.shape)
    
    # Rescale to original intensity range
    V_clahe = V_clahe * (V_max - V_min) + V_min
    
    # Preserve dtype
    return V_clahe.to(dtype=dtype)
```

**Key Implementation Details**:

1. **Tile Iteration**: Uses `itertools.product()` to generate all tile index combinations
2. **Edge Tiles**: Boundary tiles are clipped using `min((t+1)*tile_size, s)` to handle non-divisible dimensions
3. **Uniform Image Handling**: Returns clone unchanged if `V_max - V_min < 1e-12`
4. **Binning Method**: Uses `torch.searchsorted()` with `bin_edges[1:]` (right edges) for accurate digitization
5. **Contiguous Requirement**: `tile_flat.contiguous()` required for `searchsorted()`
6. **CDF Normalization**: Robust handling of degenerate cases (zero range, all-zero CDF)
7. **Device Preservation**: All operations inherit device from input tensor
8. **Dtype Preservation**: Final `.to(dtype=dtype)` ensures output dtype matches input

### Function: `compute_clahe_sampling_probabilities(V, tile_size=16, clip_limit=2.0, nbins=256)`

**Purpose**: Convenience function for using CLAHE output as sampling probabilities.

**Output**:
- `probabilities`: 1D tensor of shape `(V.numel(),)` summing to 1.0
- `V_clahe`: CLAHE-equalized volume (for visualization/debugging)

**Detailed Algorithm**:
```python
def compute_clahe_sampling_probabilities(V, tile_size, clip_limit, nbins):
    # Apply CLAHE
    V_clahe = apply_clahe(V, tile_size=tile_size, clip_limit=clip_limit, nbins=nbins)
    
    # Normalize to [0, 1] for probability distribution
    V_min, V_max = V_clahe.min(), V_clahe.max()
    
    if V_max - V_min < 1e-12:
        # Uniform case: use uniform probabilities
        V_norm = torch.ones_like(V_clahe)
    else:
        V_norm = (V_clahe - V_min) / (V_max - V_min)
    
    # Flatten and normalize to valid probability distribution
    V_flat = V_norm.reshape(-1)
    prob_sum = V_flat.sum()
    
    if prob_sum < 1e-12:
        # Degenerate case: uniform probabilities
        probabilities = torch.ones_like(V_flat) / V_flat.numel()
    else:
        probabilities = V_flat / prob_sum
    
    return probabilities, V_clahe
```

**Key Implementation Details**:
1. **Dual Output**: Returns both probabilities and CLAHE result for visualization
2. **Degenerate Handling**: Falls back to uniform distribution if input is uniform or sum is zero
3. **Numerical Stability**: Uses tolerance `1e-12` for zero checks

## Parameter Selection Guidelines

### `tile_size` (voxels)

**Effect**: Controls spatial scale of adaptation

| Value | Effect | Use Case |
|-------|--------|----------|
| < 8 | Over-local adaptation, noise amplification | Avoid |
| 8-16 | Moderate local adaptation | General purpose, small features |
| 16-32 | Balanced local/global | **Recommended**, typical features |
| > 32 | Weak local adaptation, approaches global | Large-scale structures only |

**Rule of Thumb**: `tile_size ≈ 2 × typical_feature_diameter`

**Examples**:
- DAPI nuclei (diameter ~8 voxels) → `tile_size = 16`
- Large structures (diameter ~20 voxels) → `tile_size = 32-40`

### `clip_limit` (contrast limiting factor)

**Effect**: Controls degree of enhancement vs noise amplification

| Value | Effect | Use Case |
|-------|--------|----------|
| 1.0 | No enhancement (preserves original) | Baseline / debugging |
| 1.5 | Gentle enhancement, minimal noise | Very noisy data |
| 2.0 | Moderate enhancement | **Recommended**, general purpose |
| 3.0 | Aggressive enhancement | Clean data, need strong contrast |
| 4.0 | Very aggressive | Extreme cases, expect noise |

**Rule of Thumb**: Start with 2.0, decrease if too noisy, increase if insufficient enhancement.

### `nbins` (histogram bins)

**Effect**: Granularity of intensity mapping

| Value | Effect | Use Case |
|-------|--------|----------|
| 64 | Coarse quantization, fast | Low-resolution images |
| 128 | Moderate quantization | 8-bit images |
| 256 | Fine quantization | **Recommended**, 8-16 bit images |
| 512 | Very fine quantization | High bit-depth images |

**Rule of Thumb**: 256 is standard and works well for most cases.

## Edge Cases and Robustness

### Handled Edge Cases

1. **Uniform Images**: Return unchanged clone (threshold: `V_max - V_min < 1e-12`)
2. **Near-Uniform Images**: Graceful handling via CDF normalization fallback
3. **Small Images**: Works correctly when `image_size < tile_size` (single tile)
4. **Non-Divisible Dimensions**: Boundary tiles automatically clipped to image bounds
5. **Zero-Range CDF**: No normalization applied if `cdf_range == 0`
6. **Negative Values**: Fully supported (histogram uses actual min/max)
7. **Extreme Values**: No assumptions about intensity range
8. **Mixed Sign Values**: Works correctly with positive and negative values
9. **Very Small Ranges**: Numerical stability via epsilon comparisons (`1e-12`)
10. **Non-Square Shapes**: Works with arbitrary rectangular/cuboid/hypercuboid shapes

### Numerical Stability Features

- **Tolerance Checks**: Uses `1e-12` for zero comparisons to avoid division by zero
- **Conditional Normalization**: Skips normalization when range is zero
- **Robust CDF Minimum**: Handles all-zero CDF via `cdf[cdf > 0].min() if (cdf > 0).any() else 0`
- **Clamping**: Bin indices clamped to `[0, nbins-1]` to prevent out-of-bounds access

## Performance Characteristics

### Computational Complexity

**Per tile**: `O(n_pixels_per_tile × log(nbins))`
- Histogram computation: `O(n_pixels_per_tile)` via `torch.histc`
- Contrast limiting: `O(nbins)`
- CDF computation: `O(nbins)` via `torch.cumsum`
- Intensity mapping: `O(n_pixels_per_tile × log(nbins))` via `torch.searchsorted`

**Total**: `O(V.numel() × log(nbins) + n_tiles × nbins)`
- Dominated by intensity mapping (searchsorted)
- Approximately linear in image size

### Memory Usage

**Peak memory**: `O(V.numel() + nbins × n_tiles)`
- Input volume: `V.numel()` elements
- Output volume: `V.numel()` elements (allocated upfront)
- Histograms: `nbins` elements per tile (ephemeral, not all stored simultaneously)
- Intermediate tensors: `tile_flat`, `bin_edges`, `cdf_normalized` (reused per tile)

**Typical**: For 256×256 image, tile_size=16, nbins=256:
- n_tiles = 16×16 = 256
- Memory ≈ 65K (input) + 65K (output) + 256 (histogram) ≈ 130K elements ≈ 0.52 MB (float32)

### GPU Acceleration

- **Histogram computation**: GPU-accelerated via `torch.histc`
- **Tensor operations**: All operations (`cumsum`, `searchsorted`, `clamp`, etc.) GPU-accelerated
- **Tile iteration**: Sequential (no parallelization across tiles - potential optimization opportunity)
- **Expected speedup**: 5-10× on GPU vs CPU for large volumes
- **Memory transfer**: Minimal - only `V_min`, `V_max` transferred to CPU via `.item()`

## Testing Requirements

### Unit Tests

**Test Categories** (all implemented in `test_clahe.py`):

1. **Basic Functionality** (`TestCLAHEBasic`):
   - Uniform image unchanged
   - Range preservation (min/max)
   - Shape preservation (1D, 2D, 3D, 4D)
   - Dtype preservation (float32, float64)
   - Device preservation (CPU, CUDA)

2. **Contrast Enhancement** (`TestCLAHEContrastEnhancement`):
   - Low-contrast region enhancement
   - Heterogeneous image balancing
   - Clip limit effect

3. **Dimensionality** (`TestCLAHEDimensionality`):
   - 1D support
   - 2D support
   - 3D support
   - 4D support
   - Non-square shapes

4. **Edge Cases** (`TestCLAHEEdgeCases`):
   - Small images (< tile_size)
   - Single tile
   - Exact tile division
   - Inexact tile division
   - Near-uniform images
   - Zero images

5. **Sampling Probabilities** (`TestCLAHESamplingProbabilities`):
   - Probability properties (sum=1, non-negative)
   - Sampling works (no errors)
   - CLAHE output returned
   - Uniform image probabilities

6. **Parameter Validation** (`TestCLAHEParameterValidation`):
   - Various tile sizes (4, 8, 16, 32, 64)
   - Various clip limits (1.0, 1.5, 2.0, 3.0, 4.0)
   - Various nbins (64, 128, 256, 512)

7. **Numerical Stability** (`TestCLAHENumericalStability`):
   - Extreme values (large magnitudes)
   - Negative values
   - Mixed sign values
   - Very small intensity ranges

### Test Examples

**Test: Uniform Image**
```python
def test_uniform_image_unchanged():
    V = torch.ones(256, 256) * 0.5
    V_clahe = apply_clahe(V, tile_size=16, clip_limit=2.0)
    assert torch.allclose(V_clahe, V, atol=1e-6)
```

**Test: Range Preservation**
```python
def test_range_preservation():
    V = torch.randn(256, 256)
    V_min, V_max = V.min(), V.max()
    V_clahe = apply_clahe(V, tile_size=16, clip_limit=2.0)
    assert torch.allclose(V_clahe.min(), V_min, atol=1e-5)
    assert torch.allclose(V_clahe.max(), V_max, atol=1e-5)
```

**Test: Heterogeneous Enhancement**
```python
def test_heterogeneous_image_balancing():
    V = torch.zeros(256, 256)
    V[0:128, :] = torch.randn(128, 256) * 0.05 + 0.1  # Dark region
    V[128:256, :] = torch.randn(128, 256) * 0.05 + 0.9  # Bright region
    
    V_clahe = apply_clahe(V, tile_size=16, clip_limit=2.0)
    
    # Both regions should have similar local contrast
    dark_std = V_clahe[0:128, :].std()
    bright_std = V_clahe[128:256, :].std()
    assert abs(dark_std.item() - bright_std.item()) < 0.3
```

**Test: Device Preservation**
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_preservation():
    V_cpu = torch.randn(256, 256)
    V_gpu = V_cpu.cuda()
    
    result_cpu = apply_clahe(V_cpu, tile_size=16)
    result_gpu = apply_clahe(V_gpu, tile_size=16)
    
    assert result_cpu.device.type == "cpu"
    assert result_gpu.device.type == "cuda"
```

**Test: Sampling Probabilities**
```python
def test_probability_properties():
    V = torch.randn(256, 256)
    probs, V_clahe = compute_clahe_sampling_probabilities(V, tile_size=16)
    
    assert probs.shape == (256 * 256,)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.all(probs >= 0)
    assert torch.all(probs <= 1)
```

## Usage Examples

### Example 1: Basic Image Enhancement

```python
import torch
from luxar.gsplats.clahe import apply_clahe

# Load image with varying background
image = torch.randn(512, 512)

# Apply CLAHE with default parameters
enhanced = apply_clahe(image, tile_size=16, clip_limit=2.0)

# Result has locally-equalized contrast
```

### Example 2: 3D Volume Enhancement

```python
import torch
from luxar.gsplats.clahe import apply_clahe

# Load 3D microscopy volume
volume = torch.randn(128, 128, 128)

# Apply CLAHE with parameters tuned for microscopy
enhanced = apply_clahe(
    volume,
    tile_size=16,  # ~2× nucleus diameter
    clip_limit=2.0,  # Moderate enhancement
    nbins=256  # Standard
)
```

### Example 3: Importance Sampling

```python
import torch
from luxar.gsplats.clahe import compute_clahe_sampling_probabilities

# Heterogeneous image (dim and bright regions)
image = torch.randn(256, 256)

# Compute sampling probabilities
probs, enhanced = compute_clahe_sampling_probabilities(
    image, tile_size=16, clip_limit=2.0
)

# Sample k locations proportionally to local importance
k = 100
sampled_indices = torch.multinomial(probs, k, replacement=True)

# Convert flat indices to 2D coordinates
coords = torch.unravel_index(sampled_indices, image.shape)
```

### Example 4: Parameter Tuning

```python
import torch
from luxar.gsplats.clahe import apply_clahe

image = torch.randn(256, 256)

# Conservative (noisy data)
enhanced_conservative = apply_clahe(image, tile_size=16, clip_limit=1.5)

# Moderate (recommended)
enhanced_moderate = apply_clahe(image, tile_size=16, clip_limit=2.0)

# Aggressive (clean data)
enhanced_aggressive = apply_clahe(image, tile_size=16, clip_limit=3.0)
```

### Example 5: GPU Acceleration

```python
import torch
from luxar.gsplats.clahe import apply_clahe

# Move data to GPU
image_gpu = torch.randn(512, 512, device='cuda')

# CLAHE automatically runs on GPU
enhanced_gpu = apply_clahe(image_gpu, tile_size=16, clip_limit=2.0)

# Result stays on GPU
assert enhanced_gpu.device.type == 'cuda'
```

## References

1. Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization." *Graphics Gems IV*, Academic Press, 474-485.

2. Pizer, S. M., et al. (1987). "Adaptive Histogram Equalization and Its Variations." *Computer Vision, Graphics, and Image Processing*, 39(3), 355-368.

3. Pisano, E. D., et al. (1998). "Contrast Limited Adaptive Histogram Equalization Image Processing to Improve the Detection of Simulated Spiculations in Dense Mammograms." *Journal of Digital Imaging*, 11(4), 193-200.

## Changelog

- **v1.0.0** (January 2025): Initial implementation
  - nD support for arbitrary dimensions
  - PyTorch-native implementation with GPU acceleration
  - Contrast limiting for noise robustness
  - Convenience function for sampling probabilities
  - Comprehensive edge case handling
  - Extensive test coverage (7 test classes, 35+ tests)
