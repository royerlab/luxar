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
- **test_warm_start_round_trip.py** - A declared `seeds=GSplatData` warm start's iteration-0 model reproduces the seed (parameters and render) for voxel and physical-unit fits, an explicit `seed_space`, and the `lod --refine volume` basis wiring; sheared (large off-diagonal) seeds; `volume_refine_splats` called directly; the warm-start sigma floor (incl. 1 ulp below it), uncapped background-relative amplitudes, the `amp_max` clip that keeps them trainable, and the `seed_space` warnings.
- **test_floor_denoise_basis.py** - Background-floor resolution on the denoised basis: the two probe regimes (exact whole-volume probe vs bounded crop), the percentile/`auto` regime rule, graceful degradation, probe sampling.

## Running

```bash
hatch run pytest packages/luxar/src/luxar/gsplats/fitting/tests/ -v
```
