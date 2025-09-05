# Luxar GSplats Package

This package provides Gaussian Splat fitting and rendering functionality for Luxar. It implements n-dimensional oriented Gaussian splats with full covariance matrices for high-quality compression and reconstruction of volumetric data.

## Overview

The GSplats package enables fitting Gaussian splat models to images and volumes, providing an efficient compression method that maintains high visual quality. It uses PyTorch for optimization and supports both 2D and 3D data.

## Core Components

### Fitting (`fit_gsplats.py`)
- **`fit_gaussian_splats()`** - Main fitting function that optimizes Gaussian splat parameters
- Supports n-dimensional data with full covariance matrices
- Uses Cholesky decomposition for positive-definite covariance matrices
- Implements both MSE and Poisson deviance loss functions
- Includes amplitude regularization for sparsity control

### Models (`models/`)
- **`gsplat_model.py`** - PyTorch model implementation for Gaussian splats
- **`gsplats_render.py`** - Rendering functions for both PyTorch and NumPy
- **`gsplats_batched_render.py`** - Efficient batched rendering for large datasets
- **Utilities** - Supporting functions for triangular matrix solving and inverse softplus

### Candidate Generation (`candidates.py`)
- **`find_candidates_overcomplete_nd()`** - Multi-scale candidate center detection
- Supports various detection methods including local maxima and intensity grids
- Handles n-dimensional data with configurable parameters

### Utilities (`utils/`)
- **`trils.py`** - Triangular matrix packing/unpacking for covariance storage
- Helper functions for working with Cholesky factors

### Demo Scripts (`demo/`)
- **2D Demos**: Basic 2D image compression examples
- **3D Demos**: Volumetric data compression with Napari visualization
- **Health check scripts**: PyTorch and MPS compatibility verification

## Key Features

### Mathematical Foundation
- **Oriented Gaussian Splats**: Full covariance matrices for anisotropic features
- **Cholesky Parameterization**: Ensures positive-definite covariance matrices
- **Stable Numerics**: Uses triangular solvers instead of matrix inversion
- **Multi-scale Fitting**: Supports different scales and orientations

### Optimization
- **PyTorch Backend**: GPU-accelerated optimization with automatic differentiation
- **Multiple Loss Functions**: MSE and Poisson deviance for different data types
- **Regularization**: L1 amplitude penalties for sparsity control
- **Bounded Parameters**: Sigmoid parameterization keeps centers within bounds

### Rendering
- **Efficient Evaluation**: Fast Gaussian evaluation using triangular solves
- **Batched Processing**: Optimized for large numbers of splats
- **Truncation Support**: Configurable truncation radius for performance
- **Cross-platform**: Works with both NumPy and PyTorch tensors

## Usage Examples

### Basic 2D Fitting
```python
from luxar.gsplats import fit_gaussian_splats
from luxar.gsplats.candidates import find_candidates_overcomplete_nd

# Find candidate centers
centers = find_candidates_overcomplete_nd(
    image, scales=[1.0, 2.0], peaks_per_scale=100
)

# Fit Gaussian splats
params, amplitudes = fit_gaussian_splats(
    image, centers, n_iters=300, loss_type="mse"
)
```

### 3D Volume Compression
```python
# For 3D volumes
centers_3d = find_candidates_overcomplete_nd(
    volume, scales=[0.8, 1.5, 2.5], peaks_per_scale=200
)

params_3d, amps_3d = fit_gaussian_splats(
    volume, centers_3d, 
    init_sigma_vox=1.2,
    loss_type="poisson",
    n_iters=400
)
```

### Rendering Reconstructions
```python
from luxar.gsplats.models.gsplats.gsplats_render import render_gaussians_full_numpy

# Render reconstruction
reconstruction = render_gaussians_full_numpy(
    image.shape, params, amplitudes, truncate=3.0
)
```

## Performance Considerations

- **Memory Usage**: Scales with number of candidate centers and dimensions
- **Computation**: GPU acceleration recommended for large datasets
- **Convergence**: Typically requires 200-500 iterations depending on complexity
- **Truncation**: Use appropriate truncation radius to balance quality vs speed

## Dependencies

- **Required**: NumPy, PyTorch
- **Optional**: SciPy (for advanced candidate detection), Napari (for 3D visualization)
- **GPU Support**: CUDA or MPS (Metal Performance Shaders) for acceleration

## Testing

The package includes comprehensive tests for:
- Model initialization and parameter handling
- Rendering consistency between PyTorch and NumPy
- Numerical stability and edge cases
- Gradient flow for optimization
- Cross-device compatibility (CPU/GPU)

Run tests with:
```bash
hatch run pytest packages/luxar/src/luxar/gsplats/
```

## Architecture Notes

### Internal Organization
- **Models**: Core PyTorch implementations
- **Utils**: Mathematical utilities and helpers  
- **Tests**: Comprehensive test suite
- **Demo**: Example scripts and applications

### Design Principles
- **Modularity**: Clean separation between fitting, rendering, and utilities
- **Flexibility**: Supports arbitrary dimensions and loss functions  
- **Performance**: Optimized for both small experiments and large datasets
- **Stability**: Robust numerical implementations with proper error handling

### Future Extensions
- Support for additional loss functions
- Advanced regularization techniques
- Integration with other Luxar compression methods
- Multi-GPU training support