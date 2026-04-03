# fitting Tests

Tests for the Gaussian splat fitting pipeline.

## Test Files

- **test_fitting_config.py** - `FitConfig`, `PreprocessedData`, `OptimizationResults`, `ModelComponents` dataclass creation and defaults.
- **test_fitting_validation.py** - Input validation in `prepare_fit_config()`: array dimensions, numeric ranges, type constraints, parameter consistency.
- **test_fitting_preprocessing.py** - Data normalization, seed generation, L1 regularization defaults, convergence thresholds.
- **test_initialization.py** - Model, optimizer, and scheduler initialization; hardware acceleration selection (Metal, CUDA, PyTorch fallback).
- **test_losses.py** - MSE, Poisson, and L1 loss functions; asymmetric penalty; L1 regularization on amplitudes and diagonals; boundary penalty.
- **test_optimization.py** - Training loop, convergence criteria, early stopping, dynamic operations integration, best state tracking.
- **test_results.py** - Result finalization, amplitude rescaling, voxel footprint correction, clip-to-bounds, quality metrics.
- **test_visualization.py** - Compression analysis display, napari optimization movie (mocked).
- **test_sorting.py** - Z-order Morton code sorting of splats and optimizer state.
- **test_downscale.py** - Volume downscaling, coordinate rescaling, packed Cholesky rescaling.

## Running

```bash
hatch run pytest packages/luxar/src/luxar/gsplats/fitting/tests/ -v
```
