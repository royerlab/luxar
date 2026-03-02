# Gaussian Splat Models

PyTorch model implementations and rendering engines for n-dimensional oriented Gaussian splatting.

## Overview

This package provides the core rendering engine and model classes for fitting collections of oriented Gaussian functions to reconstruct images and volumes. The key components are:

- **GaussianSplatModel**: PyTorch `nn.Module` for optimizable Gaussian splat parameters
- **Rendering Engine**: High-performance rendering with 2D/3D fast paths
- **Wrapper Functions**: Convenient NumPy and PyTorch interfaces
- **Numerical Utilities**: Stable inverse transformations for parameter initialization

## Key Features

- **nD support**: Works with 1D, 2D, 3D, 4D, and higher-dimensional data
- **Full covariance**: Oriented Gaussians via Cholesky parameterization
- **Generalized Gaussians**: Per-splat sharpness control for flexible falloff
- **Fast paths**: Explicit forward substitution for 2D/3D (10-50x faster)
- **Memory efficient**: Automatic chunking based on device memory
- **Dynamic operations**: Add, remove, or replace splats during optimization
- **Multi-device**: CUDA, CPU, and MPS support with auto-detection

## Quick Start

### Basic 2D Rendering

```python
from luxar.gsplats.models import GaussianSplatModel
import numpy as np

# Define output shape
shape = (256, 256)
d = 2

# Initialize splat parameters
N = 100  # Number of splats
centers0 = np.random.rand(N, d) * 200 + 28  # Random centers
L0 = np.zeros((N, d, d))
L0[:, 0, 0] = 2.0  # σx = 2.0
L0[:, 1, 1] = 2.0  # σy = 2.0
amps0 = np.ones(N) * 0.5  # Uniform amplitudes

# Create model
model = GaussianSplatModel(
    shape=shape,
    centers0=centers0,
    L0=L0,
    amps0=amps0,
    sigma_min_diag=[0.5, 0.5],  # Minimum width
    sigma_max_diag=[5.0, 5.0],  # Maximum width
    truncate=3.0,
)

# Render
rendered = model()  # Returns torch.Tensor of shape (256, 256)
```

### Basic 3D Rendering

```python
# 3D volume
shape = (64, 64, 64)
d = 3
N = 50

centers0 = np.random.rand(N, d) * 50 + 7
L0 = np.zeros((N, d, d))
L0[:, 0, 0] = 3.0
L0[:, 1, 1] = 3.0
L0[:, 2, 2] = 3.0
amps0 = np.ones(N) * 0.3

model = GaussianSplatModel(
    shape=shape,
    centers0=centers0,
    L0=L0,
    amps0=amps0,
    sigma_min_diag=[0.8, 0.8, 0.8],
)

volume = model()  # Returns torch.Tensor of shape (64, 64, 64)
```

## Core Components

### GaussianSplatModel

The main PyTorch model class for optimizable Gaussian splats.

**Parameters:**
- `shape`: Target image/volume dimensions
- `centers0`: Initial center positions (N, d) in voxel coordinates
- `L0`: Initial lower-triangular Cholesky factors (N, d, d)
- `amps0`: Initial amplitudes (N,)
- `sigma_min_diag`: Minimum diagonal values (prevents degeneracy)
- `sigma_max_diag`: Maximum diagonal values (prevents over-smoothing)
- `truncate`: Truncation radius in standard deviations (default: 3.0)
- `device`: PyTorch device (auto-detects if None)

**Key Methods:**

```python
# Get current transformed parameters
centers, Ls, amps, sharpness = model.current_params()

# Render to image/volume
output = model()  # Same as model.forward()

# Dynamic operations
model.prune_(keep_mask)  # Remove splats by boolean mask
model.append_(centers_new, Ls_new, amps_new, sharpness_new)  # Add new splats
model.replace_with(centers, Ls, amps, sharpness)  # Replace all splats
n = model.n_splats()  # Get current number of splats
```

**Parameterization Details:**

The model uses constrained parameterizations to ensure valid parameters:

1. **Centers**: Sigmoid parameterization keeps centers within image bounds
   - Raw parameter: `raw_mu` (unconstrained)
   - Transformed: `centers = sigmoid(raw_mu) * (shape - 1)`

2. **Cholesky factors**: Softplus ensures positive diagonal elements
   - Raw diagonal: `raw_L_diag` (unconstrained)
   - Transformed diagonal: `L_diag = softplus(raw_L_diag) + sigma_min`
   - Off-diagonal: `L_off` (unconstrained)

3. **Amplitudes**: Softplus ensures non-negativity
   - Raw: `raw_a` (unconstrained)
   - Transformed: `amps = softplus(raw_a)`

4. **Sharpness**: Exponential mapping for generalized Gaussians
   - Raw: `sharpness_offsets_raw` (clamped to [-2.5, 2.5])
   - Transformed: `sharpness = 2.0 * exp(sharpness_offsets_raw)`
   - Default: `s = 2.0` (standard Gaussian)

### Rendering Functions

The package provides multiple rendering interfaces:

#### Core Rendering (Advanced)

```python
from luxar.gsplats.models import render_gaussians
import torch

# Prepare parameters
centers = torch.tensor([[5.0, 5.0]], dtype=torch.float32)
Ls = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
amps = torch.tensor([1.0], dtype=torch.float32)
sharpness = torch.tensor([2.0], dtype=torch.float32)  # Required

# Render
output = render_gaussians(
    shape=(11, 11),
    centers=centers,
    Ls=Ls,
    amps=amps,
    sharpness=sharpness,
    truncate=3.0,
    intensity_floor=1e-5,  # Amplitude-aware culling
    chunk_size=None,  # Auto-detect
)
```

#### NumPy Wrapper (Convenient)

```python
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
from luxar.gsplats.gsplat_data import GSplatData
import numpy as np

# Create a GSplatData (typically from fit_gaussian_splats())
result = GSplatData(
    centers=np.array([[5.0, 5.0]], dtype=np.float32),
    amplitudes=np.array([1.0], dtype=np.float32),
    cholesky_factors=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),  # Packed L for 2D
    sharpnesses=np.array([2.0], dtype=np.float32),
    stats={}
)

# Render directly with result - clean and simple!
output = render_gaussians_numpy(
    shape=(11, 11),
    result=result,
    truncate=3.0,
)
```

#### PyTorch Wrapper

```python
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_pytorch

# Render with GSplatData on any device
output = render_gaussians_pytorch(
    shape=(11, 11),
    result=result,
    truncate=3.0,
    device='cuda',  # Specify device
)
```

#### Batched Interface

```python
from luxar.gsplats.models import render_gaussians_batched

output = render_gaussians_batched(
    shape=(11, 11),
    centers=centers_tensor,
    Ls=Ls_tensor,
    amps=amps_tensor,
    sharpness=sharpness_tensor,  # Optional
    truncate=3.0,
)
```

## Usage Examples

### Device Selection

```python
import torch

# Auto-detect best device (CUDA → CPU)
model = GaussianSplatModel(shape, centers0, L0, amps0, sigma_min_diag)
print(f"Using device: {next(model.parameters()).device}")

# Explicit CUDA
model = GaussianSplatModel(..., device=torch.device('cuda'))

# Explicit CPU
model = GaussianSplatModel(..., device=torch.device('cpu'))

# Explicit MPS (Metal Performance Shaders on macOS)
model = GaussianSplatModel(..., device=torch.device('mps'))
# Note: MPS is supported but currently slower than CPU for typical workloads
```

### Parameter Manipulation

```python
# Get current parameters
centers, Ls, amps, sharpness = model.current_params()

# Modify parameters (e.g., increase all amplitudes)
amps_new = amps * 1.5
model.replace_with(centers, Ls, amps_new, sharpness)

# Remove low-amplitude splats
keep_mask = amps > 0.1
model.prune_(keep_mask)

# Add new splats
centers_add = torch.tensor([[10.0, 10.0]], device=model.raw_mu.device)
Ls_add = torch.eye(2, device=model.raw_mu.device).unsqueeze(0)
amps_add = torch.tensor([0.5], device=model.raw_mu.device)
sharpness_add = torch.tensor([2.0], device=model.raw_mu.device)
model.append_(centers_add, Ls_add, amps_add, sharpness_add)

print(f"Number of splats: {model.n_splats()}")
```

### Optimization with Gradient Descent

```python
import torch.optim as optim

# Create model
model = GaussianSplatModel(
    shape=(64, 64),
    centers0=centers0,
    L0=L0,
    amps0=amps0,
    sigma_min_diag=[0.5, 0.5],
)

# Target image
target = torch.tensor(target_image, dtype=torch.float32, device=model.raw_mu.device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for i in range(1000):
    optimizer.zero_grad()

    # Render
    rendered = model()

    # Loss
    loss = torch.nn.functional.mse_loss(rendered, target)

    # Backward
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss.item():.6f}")
```

### Sharpness Control (Generalized Gaussians)

```python
# Create splats with different sharpness values
N = 10
d = 2
centers0 = np.random.rand(N, d) * 200 + 28
L0 = np.eye(d)[None, :, :].repeat(N, axis=0) * 2.0
amps0 = np.ones(N) * 0.5

model = GaussianSplatModel(
    shape=(256, 256),
    centers0=centers0,
    L0=L0,
    amps0=amps0,
    sigma_min_diag=[0.5, 0.5],
)

# Get parameters and modify sharpness
centers, Ls, amps, sharpness = model.current_params()

# Vary sharpness: sharp to soft
sharpness_varied = torch.linspace(1.0, 4.0, N, device=sharpness.device)
# s=1.0: Very soft, heavy tails
# s=2.0: Standard Gaussian
# s=4.0: Very sharp, compact

model.replace_with(centers, Ls, amps, sharpness_varied)
output = model()

# Effect of sharpness on appearance:
# - Low sharpness (s < 2): Softer edges, wider spread
# - Standard (s = 2): Classic Gaussian falloff
# - High sharpness (s > 2): Sharper edges, more compact
```

### Memory Management with Chunking

```python
# Automatic chunk size calculation (recommended)
output = render_gaussians(
    shape=large_shape,
    centers=many_centers,
    Ls=many_Ls,
    amps=many_amps,
    sharpness=many_sharpness,
    chunk_size=None,  # Auto-detect based on available memory
)

# Manual chunk size (for memory-constrained devices)
output = render_gaussians(
    shape=large_shape,
    centers=many_centers,
    Ls=many_Ls,
    amps=many_amps,
    sharpness=many_sharpness,
    chunk_size=50000,  # Process 50k voxels at a time
)

# Larger chunks = faster but more memory
# Smaller chunks = slower but more memory-efficient
```

### Numerical Stability Features

```python
# The model includes several numerical stability features:

# 1. Minimum diagonal constraints prevent degeneracy
model = GaussianSplatModel(
    ...,
    sigma_min_diag=[0.5, 0.5],  # Prevents zero-width splats
)

# 2. Maximum diagonal constraints prevent over-smoothing
model = GaussianSplatModel(
    ...,
    sigma_min_diag=[0.5, 0.5],
    sigma_max_diag=[10.0, 10.0],  # Prevents infinite-width splats
)

# 3. Center bounds via sigmoid prevent out-of-bounds
# (Automatically applied, no user configuration needed)

# 4. Stable inverse softplus for initialization
from luxar.gsplats.models.utils import stable_inverse_softplus

# Convert initial values to raw parameters safely
raw_amps = stable_inverse_softplus(amps0)
# Handles large values and edge cases without numerical issues

# 5. Clamped forward substitution prevents division by zero
# (Automatically applied in rendering, no user configuration needed)

# 6. Power function clamping prevents log(0) in gradients
# (Automatically applied when using sharpness feature)
```

## API Reference

For comprehensive technical details, see [SPECIFICATIONS.md](SPECIFICATIONS.md), which includes:

- Complete mathematical formulation
- Parameter transformation details
- Rendering algorithm with AABB truncation
- Fast path implementations (2D/3D)
- Memory management strategies
- Grid caching system
- Sharpness feature specification
- Testing requirements

### Main Classes and Functions

**GaussianSplatModel**
- `__init__(shape, centers0, L0, amps0, sigma_min_diag, sigma_max_diag, truncate, device)`
- `forward() -> torch.Tensor`
- `current_params() -> Tuple[Tensor, Tensor, Tensor, Tensor]`
- `prune_(keep_mask: Tensor) -> None`
- `append_(centers, Ls, amps, sharpness) -> None`
- `replace_with(centers, Ls, amps, sharpness) -> None`
- `n_splats() -> int`

**Rendering Functions**
- `render_gaussians(shape, centers, Ls, amps, sharpness, truncate, intensity_floor, chunk_size)`
- `render_gaussians_numpy(shape, result: GSplatData, truncate, chunk_size)`
- `render_gaussians_pytorch(shape, result: GSplatData, truncate, device, chunk_size)`
- `render_gaussians_batched(shape, centers, Ls, amps, sharpness, truncate, intensity_floor, chunk_size)`

**Utilities**
- `stable_inverse_softplus(y, beta) -> np.ndarray` - Stable inverse softplus for initialization
- `solve_lower_triangular(L, B) -> torch.Tensor` - Cross-version triangular solver

## Device Compatibility

### CUDA (NVIDIA GPUs)

**Best Performance**: CUDA provides the fastest rendering for large datasets.

```python
model = GaussianSplatModel(..., device=torch.device('cuda'))
```

**Memory**: Auto-detected using `torch.cuda.mem_get_info()` (60% of available memory for chunking)

**Fast Paths**: 2D/3D explicit forward substitution provides 10-50× speedup

### CPU

**Good Performance**: CPU rendering is efficient for small-to-medium datasets.

```python
model = GaussianSplatModel(..., device=torch.device('cpu'))
```

**Memory**: Conservative 8GB estimate for chunking

**Fast Paths**: Same 2D/3D optimizations as CUDA

### MPS (Apple Silicon / Metal)

**Supported but Slower**: MPS (Metal Performance Shaders) is fully supported but currently slower than CPU for typical workloads.

```python
model = GaussianSplatModel(..., device=torch.device('mps'))
```

**Memory**: Conservative 4GB estimate (unified memory)

**Note**: Some operations fall back to CPU (e.g., `torch.unique` with dim argument)

**Recommendation**: Use CPU on Apple Silicon unless you have specific reasons to use MPS

### Auto-Detection

```python
# Auto-detects best device: CUDA → CPU
model = GaussianSplatModel(...)  # device=None uses auto-detection

# Check which device was selected
device = next(model.parameters()).device
print(f"Using: {device}")
```

## Performance Considerations

### Rendering Speed

**Fast Paths (2D/3D)**:
- Explicit forward substitution: 10-50× faster than generic nD solver
- Automatically dispatched based on dimensionality
- No user configuration needed

**Typical Performance**:
- 1K splats, 2D (512×512): ~100ms on GPU
- 10K splats, 3D (128×128×128): ~500ms on GPU
- Performance scales with `N_splats × avg_box_volume`

**Optimization Tips**:
1. Use 2D/3D when possible (automatic fast paths)
2. Adjust `truncate` parameter (smaller = faster)
3. Enable `intensity_floor` for amplitude-aware culling
4. Use appropriate `chunk_size` for your device memory
5. Batch similar operations when modifying splats dynamically

### Memory Usage

**Model Parameters**: `O(N × (d + d*(d+1)/2 + 2))`
- Centers: `N × d`
- Cholesky: `N × (d + d*(d+1)/2)`
- Amplitudes: `N`
- Sharpness: `N`

**Rendering Memory**: `O(K × P_chunk)`
- Automatically managed via chunk size calculation
- Accounts for intermediate tensors and overhead

**Grid Cache**: `O(unique_box_shapes × max_box_volume × d)`
- Process-wide cache, reused across forward passes
- Typically small compared to other memory usage

### Computational Complexity

**AABB Calculation**: `O(N × d)` - diagonal computation and radius calculation

**Grouping**: `O(N × log(N))` - GPU-friendly grouping by AABB shape

**Rendering per Group**: `O(K_group × P_box)` - dominant cost
- `K_group`: Number of splats in group
- `P_box`: Number of voxels in AABB

**Total Forward Pass**: `O(N × avg_P_box)`

## Mathematical Background

### Gaussian Splat Formulation

Each splat `k` contributes to the output:

```
I_k(x) = a_k × exp(-0.5 × ||y_k||^s_k)
```

where:
- `a_k`: Amplitude (non-negative)
- `y_k`: Solved from `L_k × y_k = (x - μ_k)`
- `L_k`: Lower-triangular Cholesky factor where `Σ_k = L_k × L_k^T`
- `s_k`: Sharpness parameter (generalized Gaussian exponent)

**Standard Gaussian** (`s = 2`):
```
I_k(x) = a_k × exp(-0.5 × (x - μ_k)^T × Σ_k^{-1} × (x - μ_k))
```

**Generalized Gaussian** (`s ≠ 2`):
- `s > 2`: Sharper edges, more compact support
- `s < 2`: Softer edges, heavier tails
- `s → ∞`: Approaches box function
- `s → 0`: Approaches uniform

### Cholesky Parameterization

**Why Cholesky?**
- Ensures positive definiteness: `Σ = L × L^T`
- Fewer parameters: `d × (d+1)/2` instead of `d × d`
- Numerically stable: Avoids matrix inversion via triangular solve
- Natural parameterization: Diagonal elements directly relate to standard deviations

**Positive Definiteness Guarantee**:
```
Σ = L × L^T
```
where `L` is lower-triangular with positive diagonal elements.

**Softplus Activation**:
```
L_diag = softplus(raw_L_diag) + sigma_min
```
ensures `L_diag > sigma_min > 0` for all optimization steps.

### AABB Truncation

**Standard Gaussian** (`s = 2`):
```
radius = truncate × sqrt(diag(Σ))
AABB = [center - radius, center + radius]
```

**Generalized Gaussian** (`s ≠ 2`):
```
effective_truncate = truncate^(2/s)
radius = effective_truncate × sqrt(diag(Σ))
AABB = [center - radius, center + radius]
```

**Sharpness Effect**:
- Sharp splats (`s > 2`): Smaller AABB (more compact)
- Soft splats (`s < 2`): Larger AABB (wider spread)

**Amplitude-Aware Shrinking** (optional):
When `intensity_floor > 0`, compute radius where intensity drops to threshold:
```
t = (2 × log(a / intensity_floor))^(1/s)
radius = min(truncate_radius, t × sqrt(diag(Σ)))
```

This prevents large AABBs for low-amplitude splats, improving performance.

## Testing

### Running Tests

```bash
# Run all model tests
hatch run pytest packages/luxar/src/luxar/gsplats/models/ -v

# Run specific test file
hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/tests/test_gsplat_model.py -v

# Run with coverage
hatch run pytest packages/luxar/src/luxar/gsplats/models/ --cov=luxar.gsplats.models
```

### Test Coverage

**GaussianSplatModel Tests**:
- Parameter initialization correctness
- Forward pass rendering
- Dynamic operations (prune, append, replace)
- State dict serialization
- Sharpness parameter handling
- Device placement (CUDA, CPU, MPS)
- Bounds constraints

**Rendering Tests**:
- 2D/3D fast paths vs generic nD
- Memory chunking correctness
- Grid caching functionality
- AABB calculation accuracy
- Sharpness-adjusted AABB
- Amplitude-aware shrinking
- Edge cases (empty input, boundary splats)

**Wrapper Tests**:
- Parameter unpacking (with/without sharpness)
- Format detection and validation
- Backward compatibility
- NumPy vs PyTorch consistency

**Utility Tests**:
- Inverse softplus numerical stability
- Round-trip consistency
- Triangular solver correctness
- Cross-version compatibility

## References

**Related Documentation**:
- [SPECIFICATIONS.md](SPECIFICATIONS.md) - Complete technical specification
- [Main Gaussian Splats Package](../README.md) - Higher-level fitting API
- [Utils Package](../utils/SPECIFICATIONS.md) - Matrix utilities (pack/unpack)
- [Optimization Package](../optim/SPECIFICATIONS.md) - Optimizer integration

**See Also**:
- [Multiscale Decomposition](../multiscale/README.md) - Multi-scale image decomposition
- [Candidate Generation](../candidates/README.md) - Seeding methods for fitting

## Version History

**v1.0.0** (January 2025): Initial implementation
- nD Gaussian splatting with full covariance
- 2D/3D fast paths with explicit forward substitution
- Per-splat sharpness (generalized Gaussians)
- Memory-optimized chunking and grid caching
- Dynamic operations support (prune/append/replace)
- Cross-device support (CUDA, CPU, MPS)
- Comprehensive test suite
