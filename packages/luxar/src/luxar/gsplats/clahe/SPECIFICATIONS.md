# CLAHE Subpackage Specification

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
1. Find bin: `bin_idx = floor((I_p - V_min) / bin_width)`
2. Apply mapping: `I_out = CDF_norm(bin_idx)`
3. Rescale: `I_out = I_out × (V_max - V_min) + V_min`

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
- `V_clahe`: CLAHE-equalized tensor, same shape as `V`

**Algorithm**:
```python
def apply_clahe(V, tile_size, clip_limit, nbins):
    # Get global min/max for consistent binning
    V_min, V_max = V.min(), V.max()

    # Calculate tile grid
    n_tiles = tuple((s + tile_size - 1) // tile_size for s in V.shape)

    # Initialize output
    V_clahe = torch.zeros_like(V)

    # Process each tile
    for tile_idx in all_tile_combinations(n_tiles):
        # Extract tile
        tile = V[tile_slice(tile_idx, tile_size)]

        # Compute histogram
        hist = torch.histc(tile.reshape(-1), bins=nbins, min=V_min, max=V_max)

        # Apply contrast limiting
        uniform_height = tile.numel() / nbins
        clip_height = clip_limit * uniform_height
        excess = sum(max(0, hist - clip_height))
        hist_clipped = min(hist, clip_height) + excess / nbins

        # Compute CDF
        cdf = cumsum(hist_clipped)
        cdf_norm = (cdf - cdf.min()) / (cdf.max() - cdf.min())

        # Map intensities
        bin_indices = digitize(tile, nbins, V_min, V_max)
        tile_equalized = cdf_norm[bin_indices]

        # Store result
        V_clahe[tile_slice(tile_idx, tile_size)] = tile_equalized.reshape(tile.shape)

    # Rescale to original range
    V_clahe = V_clahe * (V_max - V_min) + V_min

    return V_clahe
```

### Function: `compute_clahe_sampling_probabilities(V, ...)`

**Purpose**: Convenience function for using CLAHE output as sampling probabilities.

**Algorithm**:
```python
def compute_clahe_sampling_probabilities(V, tile_size, clip_limit, nbins):
    # Apply CLAHE
    V_clahe = apply_clahe(V, tile_size, clip_limit, nbins)

    # Normalize to [0, 1]
    V_norm = (V_clahe - V_clahe.min()) / (V_clahe.max() - V_clahe.min())

    # Flatten and normalize to probabilities
    probabilities = V_norm.reshape(-1)
    probabilities = probabilities / probabilities.sum()

    return probabilities, V_clahe
```

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

## Performance Characteristics

### Computational Complexity

**Per tile**: `O(n_pixels_per_tile × nbins)`
- Histogram computation: `O(n_pixels_per_tile)`
- Contrast limiting: `O(nbins)`
- CDF computation: `O(nbins)`
- Intensity mapping: `O(n_pixels_per_tile × log(nbins))` (binary search)

**Total**: `O(V.numel() × nbins / tile_size^d + V.numel() × log(nbins))`
- Dominated by intensity mapping when `tile_size` is small
- Approximately linear in image size

### Memory Usage

**Peak memory**: `O(V.numel() + nbins × n_tiles)`
- Input volume: `V.numel()` elements
- Output volume: `V.numel()` elements
- Histograms: `nbins × n_tiles` elements (one histogram per tile)

**Typical**: For 256×256 image, tile_size=16, nbins=256:
- n_tiles = 16×16 = 256
- Peak memory ≈ 65K + 65K + 65K = 195K elements ≈ 0.8 MB (float32)

### GPU Acceleration

- Histogram computation: GPU-accelerated via `torch.histc`
- Tensor operations: GPU-accelerated
- Tile iteration: Sequential (no parallelization across tiles)
- Expected speedup: 5-10× on GPU vs CPU for large volumes

## Testing Requirements

### Unit Tests

**Test: Uniform Image**
```python
def test_uniform_image():
    V = torch.ones(256, 256) * 0.5
    V_clahe = apply_clahe(V, tile_size=16, clip_limit=2.0)
    assert torch.allclose(V_clahe, V)  # Unchanged
```

**Test: Range Preservation**
```python
def test_range_preservation():
    V = torch.randn(256, 256)
    V_min, V_max = V.min(), V.max()
    V_clahe = apply_clahe(V, tile_size=16, clip_limit=2.0)
    assert torch.allclose(V_clahe.min(), V_min, atol=1e-6)
    assert torch.allclose(V_clahe.max(), V_max, atol=1e-6)
```

**Test: Contrast Enhancement**
```python
def test_contrast_enhancement():
    # Create image with low-contrast region
    V = torch.zeros(256, 256)
    V[64:192, 64:192] = torch.linspace(0.4, 0.6, 128).unsqueeze(1)

    V_clahe = apply_clahe(V, tile_size=16, clip_limit=2.0)

    # Check that local contrast increased
    region = V_clahe[64:192, 64:192]
    assert region.std() > V[64:192, 64:192].std()
```

**Test: Dimensionality**
```python
def test_nd_support():
    # 1D
    V_1d = torch.randn(256)
    assert apply_clahe(V_1d, tile_size=16).shape == (256,)

    # 2D
    V_2d = torch.randn(256, 256)
    assert apply_clahe(V_2d, tile_size=16).shape == (256, 256)

    # 3D
    V_3d = torch.randn(64, 64, 64)
    assert apply_clahe(V_3d, tile_size=16).shape == (64, 64, 64)

    # 4D
    V_4d = torch.randn(32, 32, 32, 32)
    assert apply_clahe(V_4d, tile_size=8).shape == (32, 32, 32, 32)
```

**Test: Device Preservation**
```python
def test_device_preservation():
    if torch.cuda.is_available():
        V_cpu = torch.randn(256, 256)
        V_gpu = V_cpu.cuda()

        result_cpu = apply_clahe(V_cpu, tile_size=16)
        result_gpu = apply_clahe(V_gpu, tile_size=16)

        assert result_cpu.device.type == 'cpu'
        assert result_gpu.device.type == 'cuda'
```

**Test: Sampling Probabilities**
```python
def test_sampling_probabilities():
    V = torch.randn(256, 256)
    probs, V_clahe = compute_clahe_sampling_probabilities(V, tile_size=16)

    # Check probability properties
    assert probs.shape == (256 * 256,)
    assert torch.allclose(probs.sum(), torch.tensor(1.0))
    assert torch.all(probs >= 0)
    assert torch.all(probs <= 1)
```

### Integration Tests

**Test: Heterogeneous Image Enhancement**
```python
def test_heterogeneous_enhancement():
    # Create image with varying background
    V = torch.zeros(256, 256)

    # Dark region with dim features
    V[0:128, :] = torch.randn(128, 256) * 0.05 + 0.1

    # Bright region with bright features
    V[128:256, :] = torch.randn(128, 256) * 0.05 + 0.9

    V_clahe = apply_clahe(V, tile_size=16, clip_limit=2.0)

    # Check that both regions have similar dynamic range
    dark_std = V_clahe[0:128, :].std()
    bright_std = V_clahe[128:256, :].std()
    assert abs(dark_std - bright_std) < 0.2  # Similar local contrast
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

## References

1. Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization." *Graphics Gems IV*, Academic Press, 474-485.

2. Pizer, S. M., et al. (1987). "Adaptive Histogram Equalization and Its Variations." *Computer Vision, Graphics, and Image Processing*, 39(3), 355-368.

3. Pisano, E. D., et al. (1998). "Contrast Limited Adaptive Histogram Equalization Image Processing to Improve the Detection of Simulated Spiculations in Dense Mammograms." *Journal of Digital Imaging*, 11(4), 193-200.

## Version History

- **v1.0.0** (January 2025): Initial implementation
  - nD support for arbitrary dimensions
  - PyTorch-native implementation with GPU acceleration
  - Contrast limiting for noise robustness
  - Convenience function for sampling probabilities
